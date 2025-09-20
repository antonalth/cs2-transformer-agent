#!/usr/bin/env python3
"""
embed.py — Offline Feature Pre-computation for Video and Audio

This script processes the entire dataset of recordings to pre-compute and cache
features, significantly accelerating the training data loading pipeline.

Features:
  - Video Embeddings: Extracts raw, pre-projection features from the DINOv3
    vision backbone for every frame of every video at a consistent 32 FPS.
  - Audio Embeddings: Generates stereo log-mel spectrograms for every audio
    file, perfectly aligned to the 32 FPS video time base.
  - Multi-GPU Parallelism: Utilizes all available GPUs to process files in
    parallel using `torch.multiprocessing`.
  - Robust & Idempotent: Skips already processed files (resumable), writes
    outputs atomically to prevent corruption, and handles errors gracefully.
  - Consistency with Training: Reuses DALI pipelines, model configurations,
    and processing logic directly from `train2.py` and `model.py` to
    guarantee that cached features are identical to those generated on-the-fly
    during training.

Workflow:
  1. Discover all .mp4 and .wav files in the `recordings/` directory.
  2. Filter out files that already have corresponding embeddings, unless
     --overwrite is specified.
  3. A pool of worker processes (one per GPU) pulls jobs (file pairs).
  4. Each worker configures a DALI pipeline to read the full media file.
  5. For video, frames are processed in batches through the DINOv3 backbone.
  6. For audio, the full waveform is converted to a mel spectrogram on the GPU.
  7. The resulting feature tensors are aligned to the same frame count (T)
     and saved atomically as .npy files in `vit_embed/` and `aud_embed/`.
"""
import os
import argparse
import logging
import time
import tempfile
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

# Reuse components from the training scripts
from train2 import DaliConfig, TICK_RATE, FPS
from model import CS2Config, DINOv3VisualEncoder

# --- Try to import DALI ---
try:
    from nvidia.dali import fn, pipeline_def, types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from nvidia.dali.plugin.base_iterator import LastBatchPolicy
    DALI_AVAILABLE = True
except ImportError as e:
    DALI_AVAILABLE = False
    DALI_IMPORT_ERROR = e


# ---------------------------
# Core Utilities
# ---------------------------

def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Pre-compute video and audio embeddings.")
    parser.add_argument("--data-root", type=str, default=os.environ.get("DATA_ROOT", "data"),
                        help="Root directory for the dataset containing recordings/, vit_embed/, etc.")
    parser.add_argument("--mode", type=str, choices=["video", "audio", "both"], default="both",
                        help="Which modality to process.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing embedding files.")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Number of video frames to process at a time in DALI/ViT.")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of GPUs to use. Defaults to all available GPUs.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of files to process for debugging.")
    args = parser.parse_args()
    return args

def atomic_save_npy(path: Path, array: np.ndarray):
    """Saves a NumPy array atomically to prevent corruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent, delete=False) as f:
        np.save(f, array)
        temp_path = f.name
    # Ensure data is written to disk before renaming
    if hasattr(os, 'fsync'):
        with open(temp_path, 'rb') as f:
            os.fsync(f.fileno())
    os.replace(temp_path, path)

# ---------------------------
# Video Processing
# ---------------------------

@pipeline_def
def create_video_pipeline(video_path: str, batch_size: int, device_id: int):
    """DALI pipeline to decode a full video at 32 FPS."""
    # DALI's video reader is powerful. By giving it a single file and a sequence_length
    # equal to the batch_size, it will iterate through the video in chunks.
    video, _ = fn.readers.video(
        device="gpu",
        filenames=[video_path],
        sequence_length=batch_size,
        step=batch_size, # Advance by a full batch each time
        normalized=False,
        image_type=types.RGB,
        dtype=types.UINT8,
        pad_last_batch=True, # Ensure the last partial batch is processed
        name=f"VideoReader_{os.path.basename(video_path)}",
    )
    return video

def process_video(video_path: Path, model: DINOv3VisualEncoder, batch_size: int, gpu_id: int) -> Optional[torch.Tensor]:
    """Processes a single video file and returns raw DINOv3 features."""
    if not video_path.exists():
        logging.warning(f"[GPU {gpu_id}] Video file not found: {video_path}")
        return None

    pipe = create_video_pipeline(
        video_path=str(video_path),
        device_id=gpu_id,
        batch_size=batch_size,
        num_threads=4,
        device_id=gpu_id,
    )
    pipe.build()

    dali_iter = DALIGenericIterator(
        [pipe],
        ['frames'],
        reader_name=f"VideoReader_{video_path.name}",
        auto_reset=True,
        last_batch_policy=LastBatchPolicy.PARTIAL
    )

    all_features = []
    try:
        with torch.no_grad():
            for batch in dali_iter:
                frames_uint8 = batch[0]['frames'] # [B, H, W, C]
                if frames_uint8.shape[0] == 0: continue

                # DINOv3 encoder expects [B, C, H, W]
                frames_uint8 = frames_uint8.permute(0, 3, 1, 2)

                # --- Extract RAW backbone features (pre-projection) ---
                # This logic is adapted from DINOv3VisualEncoder.forward to get
                # intermediate results without modifying the original class.
                x = model._normalize_chunk(frames_uint8, from_uint8=True)
                feats = model._forward_backbone_no_grad(x) # [B, D_raw]
                all_features.append(feats.cpu())
    except Exception as e:
        logging.error(f"[GPU {gpu_id}] Failed processing video {video_path}: {e}")
        return None
    finally:
        dali_iter.reset() # Clean up DALI resources
        del pipe, dali_iter

    if not all_features:
        return None

    return torch.cat(all_features, dim=0)

# ---------------------------
# Audio Processing
# ---------------------------

@pipeline_def
def create_audio_pipeline(wav_path: str, dali_cfg: DaliConfig, device_id: int):
    """DALI pipeline to convert a full .wav file to a stereo mel spectrogram."""
    audio_raw, _ = fn.readers.file(files=[wav_path], name=f"AudioReader_{os.path.basename(wav_path)}")
    decoded, _ = fn.decoders.audio(audio_raw, sample_rate=dali_cfg.sample_rate, downmix=False)

    left = decoded[:, 0]
    right = decoded[:, 1]

    def to_mel_db(channel_1d):
        spec = fn.spectrogram(
            channel_1d, nfft=dali_cfg.nfft,
            window_length=dali_cfg.window_length, window_step=dali_cfg.hop_length,
            center_windows=False
        )
        mel = fn.mel_filter_bank(
            spec, sample_rate=dali_cfg.sample_rate,
            nfilter=dali_cfg.mel_bins, freq_high=dali_cfg.mel_fmax
        )
        db = fn.to_decibels(mel, cutoff_db=dali_cfg.db_cutoff)
        return fn.transpose(db, perm=[1, 0]) # -> [time, n_mels]

    mel_left = to_mel_db(left)
    mel_right = to_mel_db(right)
    # Stack to [2, time, n_mels] to match training pipeline intermediate shape
    mel_stereo = fn.stack(mel_left, mel_right, axis=0)
    return mel_stereo.gpu()

def process_audio(wav_path: Path, dali_cfg: DaliConfig, gpu_id: int) -> Optional[torch.Tensor]:
    """Processes a single audio file and returns a stereo mel spectrogram."""
    if not wav_path.exists():
        logging.warning(f"[GPU {gpu_id}] Audio file not found: {wav_path}")
        return None

    pipe = create_audio_pipeline(
        wav_path=str(wav_path),
        dali_cfg=dali_cfg,
        device_id=gpu_id,
        batch_size=1, # One file at a time
        num_threads=2,
        device_id=gpu_id,
    )
    pipe.build()

    dali_iter = DALIGenericIterator(
        [pipe], ['mel'], reader_name=f"AudioReader_{wav_path.name}",
        auto_reset=True, last_batch_policy=LastBatchPolicy.FILL
    )

    try:
        # The pipeline processes the entire file and returns one item
        mel_batch = next(iter(dali_iter))
        mel_tensor = mel_batch[0]['mel'].squeeze(0) # [2, T, Mels]
        # Permute to spec: [T, Mels, 2] for [T, n_mels, channels]
        mel_tensor = mel_tensor.permute(1, 2, 0)
    except Exception as e:
        logging.error(f"[GPU {gpu_id}] Failed processing audio {wav_path}: {e}")
        return None
    finally:
        dali_iter.reset()
        del pipe, dali_iter

    return mel_tensor

# ---------------------------
# Worker & Main Logic
# ---------------------------

def worker(rank: int, world_size: int, jobs: List[Tuple], args: argparse.Namespace, stats: dict):
    """The main worker process function, assigned to a specific GPU."""
    gpu_id = rank
    torch.cuda.set_device(gpu_id)
    
    # --- Setup models and configs once per worker ---
    dali_cfg = DaliConfig(fps=FPS)
    cs2_cfg = CS2Config()
    
    video_model = None
    if args.mode in ["video", "both"]:
        try:
            video_model = DINOv3VisualEncoder(cs2_cfg)
            video_model.to(torch.bfloat16).to(gpu_id).eval()
        except Exception as e:
            logging.error(f"[GPU {gpu_id}] Failed to initialize DINOv3 model: {e}")
            return

    # Process assigned jobs
    num_jobs = len(jobs)
    for i in range(rank, num_jobs, world_size):
        job = jobs[i]
        md5, name, video_in_path, audio_in_path, video_out_path, audio_out_path = job
        
        start_time = time.time()
        
        try:
            video_tensor, audio_tensor = None, None
            
            # --- Process Video ---
            if args.mode in ["video", "both"]:
                video_tensor = process_video(video_in_path, video_model, args.batch_size, gpu_id)
            
            # --- Process Audio ---
            if args.mode in ["audio", "both"]:
                audio_tensor = process_audio(audio_in_path, dali_cfg, gpu_id)

            if video_tensor is None and args.mode in ["video", "both"]:
                stats['failed'] += 1
                continue
            if audio_tensor is None and args.mode in ["audio", "both"]:
                stats['failed'] += 1
                continue

            # --- Align and Save ---
            if args.mode == "both":
                T_vid = video_tensor.shape[0]
                T_aud = audio_tensor.shape[0]
                T_final = min(T_vid, T_aud)
                
                if T_final > 0:
                    atomic_save_npy(video_out_path, video_tensor[:T_final].numpy())
                    atomic_save_npy(audio_out_path, audio_tensor[:T_final].numpy())
                    stats['completed'] += 1
                    logging.info(f"[GPU {gpu_id}] Completed {md5}/{name} -> T={T_final} in {time.time()-start_time:.2f}s")
                else:
                    logging.warning(f"[GPU {gpu_id}] Skipped {md5}/{name} due to zero-length output.")
                    stats['failed'] += 1

            elif args.mode == "video":
                atomic_save_npy(video_out_path, video_tensor.numpy())
                stats['completed'] += 1
                logging.info(f"[GPU {gpu_id}] Completed {md5}/{name} -> T={video_tensor.shape[0]} in {time.time()-start_time:.2f}s")
            
            elif args.mode == "audio":
                atomic_save_npy(audio_out_path, audio_tensor.numpy())
                stats['completed'] += 1
                logging.info(f"[GPU {gpu_id}] Completed {md5}/{name} -> T={audio_tensor.shape[0]} in {time.time()-start_time:.2f}s")

        except Exception as e:
            logging.error(f"[GPU {gpu_id}] Unhandled error on job {md5}/{name}: {e}", exc_info=True)
            stats['failed'] += 1

def main():
    """Main function to discover files and spawn worker processes."""
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if not DALI_AVAILABLE:
        logging.error("NVIDIA DALI is not available or failed to import. This script cannot run.")
        logging.error(f"Import Error: {DALI_IMPORT_ERROR}")
        return

    data_root = Path(args.data_root)
    recordings_dir = data_root / "recordings"
    vit_embed_dir = data_root / "vit_embed"
    aud_embed_dir = data_root / "aud_embed"

    if not recordings_dir.is_dir():
        logging.error(f"Recordings directory not found at: {recordings_dir}")
        return

    # --- 1. Discover all potential jobs ---
    logging.info("Discovering media files...")
    all_files = defaultdict(dict)
    for p in tqdm(list(recordings_dir.rglob("*.mp4")) + list(recordings_dir.rglob("*.wav"))):
        md5 = p.parent.name
        name = p.stem
        key = (md5, name)
        if p.suffix == '.mp4':
            all_files[key]['video'] = p
        elif p.suffix == '.wav':
            all_files[key]['audio'] = p
    
    # --- 2. Create and filter job list ---
    jobs = []
    skipped_count = 0
    for (md5, name), paths in all_files.items():
        video_in = paths.get('video')
        audio_in = paths.get('audio')
        video_out = vit_embed_dir / md5 / f"{name}.npy"
        audio_out = aud_embed_dir / md5 / f"{name}.npy"

        # Check if job should be skipped
        if not args.overwrite:
            video_done = (args.mode != "video" and args.mode != "both") or video_out.exists()
            audio_done = (args.mode != "audio" and args.mode != "both") or audio_out.exists()
            if video_done and audio_done:
                skipped_count += 1
                continue
        
        # Check if required source files exist for the mode
        if args.mode in ["video", "both"] and not video_in:
            continue
        if args.mode in ["audio", "both"] and not audio_in:
            continue
            
        jobs.append((md5, name, video_in, audio_in, video_out, audio_out))

    if args.limit:
        jobs = jobs[:args.limit]
    
    logging.info(f"Found {len(jobs)} jobs to process. Skipped {skipped_count} already completed jobs.")
    if not jobs:
        logging.info("Nothing to do.")
        return

    # --- 3. Spawn workers ---
    num_gpus = torch.cuda.device_count()
    world_size = args.num_workers if args.num_workers is not None else num_gpus
    if world_size == 0:
        logging.error("No CUDA-enabled GPUs found.")
        return
        
    logging.info(f"Spawning {world_size} worker processes...")

    with mp.Manager() as manager:
        stats = manager.dict({'completed': 0, 'failed': 0})
        
        # Use spawn context for CUDA safety
        mp.set_start_method("spawn", force=True)
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=worker, args=(rank, world_size, jobs, args, stats))
            p.start()
            processes.append(p)

        # Monitor progress with tqdm
        with tqdm(total=len(jobs), desc="Processing files") as pbar:
            completed_last = 0
            failed_last = 0
            while any(p.is_alive() for p in processes):
                completed_now = stats['completed']
                failed_now = stats['failed']
                pbar.update((completed_now - completed_last) + (failed_now - failed_last))
                completed_last = completed_now
                failed_last = failed_now
                time.sleep(1)
            
            # Final update
            completed_now = stats['completed']
            failed_now = stats['failed']
            pbar.update((completed_now - completed_last) + (failed_now - failed_last))


        for p in processes:
            p.join()

        logging.info("--- Processing Complete ---")
        logging.info(f"Successfully processed: {stats['completed']} items")
        logging.info(f"Failed to process: {stats['failed']} items")
        logging.info(f"Skipped (pre-existing): {skipped_count} items")


if __name__ == "__main__":
    main()