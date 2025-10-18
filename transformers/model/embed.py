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
import sys
import time
import tempfile
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

# Reuse components from the training scripts
from train3 import DaliConfig, FPS
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


# --- Script Path Resolution ---
SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------
# Logging
# ---------------------------

def setup_logging(log_filename):
    """Configures logging for the main process and workers."""
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_formatter)
        root_logger.addHandler(console_handler)

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
    if hasattr(os, 'fsync'):
        with open(temp_path, 'rb') as f:
            os.fsync(f.fileno())
    os.replace(temp_path, path)

# ---------------------------
# Video Processing
# ---------------------------

def process_video(video_path: Path, model: DINOv3VisualEncoder, batch_size: int, gpu_id: int) -> Optional[torch.Tensor]:
    if not video_path.exists():
        logging.warning(f"[GPU {gpu_id}] Video file not found: {video_path}")
        return None

    # --- OPTIMIZATION: Increase prefetch queue depth ---
    @pipeline_def(batch_size=1, num_threads=4, device_id=gpu_id, prefetch_queue_depth=3)
    def create_video_pipeline():
        video = fn.readers.video(
            device="gpu",
            filenames=[str(video_path)],
            sequence_length=batch_size,
            step=batch_size,
            normalized=False,
            image_type=types.RGB,
            dtype=types.UINT8,
            pad_last_batch=True,
            file_list_include_preceding_frame=False,
            name=f"VideoReader_{video_path.name}",
        )
        return video

    pipe = create_video_pipeline()
    pipe.build()
    dali_iter = DALIGenericIterator(
        [pipe], ['frames'], reader_name=f"VideoReader_{video_path.name}",
        auto_reset=True, last_batch_policy=LastBatchPolicy.PARTIAL
    )
    all_features = []
    try:
        with torch.no_grad():
            for batch in dali_iter:
                frames_5d = batch[0]['frames']
                if frames_5d.nelement() == 0: continue
                frames_4d = frames_5d.squeeze(0)
                if frames_4d.nelement() == 0: continue
                frames_for_model = frames_4d.permute(0, 3, 1, 2)
                x = model._normalize_chunk(frames_for_model, from_uint8=True)
                feats = model._forward_backbone_no_grad(x)
                all_features.append(feats) # Keep features on GPU for now
    except Exception as e:
        logging.error(f"[GPU {gpu_id}] Failed processing video {video_path}: {e}", exc_info=True)
        return None
    finally:
        del pipe, dali_iter
    if not all_features: return None
    return torch.cat(all_features, dim=0)

# ---------------------------
# Audio Processing
# ---------------------------

def process_audio(wav_path: Path, dali_cfg: DaliConfig, gpu_id: int) -> Optional[torch.Tensor]:
    if not wav_path.exists():
        logging.warning(f"[GPU {gpu_id}] Audio file not found: {wav_path}")
        return None
        
    # --- OPTIMIZATION: Increase prefetch queue depth ---
    @pipeline_def(batch_size=1, num_threads=2, device_id=gpu_id, prefetch_queue_depth=3)
    def create_audio_pipeline():
        audio_raw, _ = fn.readers.file(files=[str(wav_path)], name=f"AudioReader_{wav_path.name}")
        decoded, _ = fn.decoders.audio(audio_raw, sample_rate=dali_cfg.sample_rate, downmix=False)
        left, right = decoded[:, 0], decoded[:, 1]
        def to_mel_db(channel_1d):
            spec = fn.spectrogram(channel_1d, nfft=dali_cfg.nfft, window_length=dali_cfg.window_length, window_step=dali_cfg.hop_length, center_windows=False)
            mel = fn.mel_filter_bank(spec, sample_rate=dali_cfg.sample_rate, nfilter=dali_cfg.mel_bins, freq_high=dali_cfg.mel_fmax)
            db = fn.to_decibels(mel, cutoff_db=dali_cfg.db_cutoff)
            return fn.transpose(db, perm=[1, 0])
        mel_left, mel_right = to_mel_db(left), to_mel_db(right)
        mel_stereo = fn.stack(mel_left, mel_right, axis=0)
        return mel_stereo.gpu()
        
    pipe = create_audio_pipeline()
    pipe.build()
    dali_iter = DALIGenericIterator([pipe], ['mel'], reader_name=f"AudioReader_{wav_path.name}", auto_reset=True, last_batch_policy=LastBatchPolicy.FILL)
    try:
        mel_batch = next(iter(dali_iter))
        # Original shape [1, C, T, Mel] -> Squeeze to [C, T, Mel] -> Permute to [T, Mel, C] for saving
        mel_tensor = mel_batch[0]['mel'].squeeze(0).permute(1, 2, 0)
    except Exception as e:
        logging.error(f"[GPU {gpu_id}] Failed processing audio {wav_path}: {e}", exc_info=True)
        return None
    finally:
        del pipe, dali_iter
    return mel_tensor

# ---------------------------
# Worker & Main Logic
# ---------------------------

def save_results_background(future):
    """Callback to check for exceptions in the background save thread."""
    try:
        future.result()
    except Exception as e:
        logging.error(f"Error during background save: {e}", exc_info=True)

def worker(rank: int, world_size: int, jobs: List[Tuple], args: argparse.Namespace, stats: dict, log_filename: str):
    setup_logging(log_filename)
    gpu_id = rank
    torch.cuda.set_device(gpu_id)
    dali_cfg, cs2_cfg = DaliConfig(fps=FPS), CS2Config()
    video_model = None
    if args.mode in ["video", "both"]:
        try:
            video_model = DINOv3VisualEncoder(cs2_cfg).to(torch.bfloat16).to(gpu_id).eval()
        except Exception as e:
            logging.error(f"[GPU {gpu_id}] Failed to initialize DINOv3 model: {e}")
            return

    # --- OPTIMIZATION: Thread pool for overlapping CPU-bound saving ---
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = None
        for i in range(rank, len(jobs), world_size):
            md5, name, video_in_path, audio_in_path, video_out_path, audio_out_path = jobs[i]
            
            # Before starting new work, ensure the previous save job is finished
            if future:
                future.result() # Wait for the last save to complete

            start_time = time.time()
            try:
                logging.info(f"[GPU {gpu_id}] Starting job {md5}/{name}...")
                video_tensor, audio_tensor = None, None
                if args.mode in ["video", "both"]: video_tensor = process_video(video_in_path, video_model, args.batch_size, gpu_id)
                if args.mode in ["audio", "both"]: audio_tensor = process_audio(audio_in_path, dali_cfg, gpu_id)
                
                if (video_tensor is None and args.mode in ["video", "both"]) or \
                   (audio_tensor is None and args.mode in ["audio", "both"]):
                    stats['failed'] += 1
                    continue
                
                # --- Submit save job to background thread ---
                def save_job():
                    if args.mode == "both":
                        T_final = min(video_tensor.shape[0], audio_tensor.shape[0])
                        if T_final > 0:
                            # FIX: Cast to float16 for smaller files and DALI compatibility
                            video_np = video_tensor[:T_final].cpu().to(torch.float16).numpy()
                            audio_np = audio_tensor[:T_final].cpu().to(torch.float16).numpy()
                            atomic_save_npy(video_out_path, video_np)
                            atomic_save_npy(audio_out_path, audio_np)
                        else:
                            logging.warning(f"[GPU {gpu_id}] Skipped {name} due to zero-length output.")
                            # This path doesn't increment stats to avoid double counting
                            return
                    elif args.mode == "video":
                        # FIX: Cast to float16
                        video_np = video_tensor.cpu().to(torch.float16).numpy()
                        atomic_save_npy(video_out_path, video_np)
                    elif args.mode == "audio":
                        # FIX: Cast to float16
                        audio_np = audio_tensor.cpu().to(torch.float16).numpy()
                        atomic_save_npy(audio_out_path, audio_np)
                    
                    stats['completed'] += 1
                    logging.info(f"[GPU {gpu_id}] Completed {md5}/{name} in {time.time()-start_time:.2f}s")
                
                future = executor.submit(save_job)
                future.add_done_callback(save_results_background)

            except Exception as e:
                logging.error(f"[GPU {gpu_id}] Unhandled error on job {md5}/{name}: {e}", exc_info=False)
                stats['failed'] += 1
        
        # Wait for the very last job to finish saving
        if future:
            future.result()

def main():
    args = get_args()

    # --- Logging Setup ---
    log_dir = SCRIPT_DIR / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_filename = log_dir / f"embed_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    setup_logging(log_filename)
    
    if not DALI_AVAILABLE:
        logging.error("NVIDIA DALI not available. Error: %s", DALI_IMPORT_ERROR)
        return
    data_root, recordings_dir = Path(args.data_root), Path(args.data_root) / "recordings"
    vit_embed_dir, aud_embed_dir = data_root / "vit_embed", data_root / "aud_embed"
    if not recordings_dir.is_dir():
        logging.error(f"Recordings directory not found: {recordings_dir}")
        return

    logging.info("Discovering media files...")
    all_files = defaultdict(dict)
    media_files = list(recordings_dir.rglob("*.mp4")) + list(recordings_dir.rglob("*.wav"))
    for p in tqdm(media_files, desc="Scanning files"):
        md5, name = p.parent.name, p.stem
        all_files[(md5, name)]['video' if p.suffix == '.mp4' else 'audio'] = p

    jobs, skipped_count = [], 0
    for (md5, name), paths in all_files.items():
        video_in, audio_in = paths.get('video'), paths.get('audio')
        video_out, audio_out = vit_embed_dir / md5 / f"{name}.npy", aud_embed_dir / md5 / f"{name}.npy"
        if not args.overwrite:
            video_done = (args.mode not in ["video", "both"]) or video_out.exists()
            audio_done = (args.mode not in ["audio", "both"]) or audio_out.exists()
            if video_done and audio_done:
                skipped_count += 1
                continue
        if (args.mode in ["video", "both"] and not video_in) or \
           (args.mode in ["audio", "both"] and not audio_in):
            continue
        jobs.append((md5, name, video_in, audio_in, video_out, audio_out))
    if args.limit: jobs = jobs[:args.limit]
    
    logging.info(f"Found {len(jobs)} jobs to process. Skipped {skipped_count} already completed jobs.")
    if not jobs:
        logging.info("Nothing to do.")
        return

    world_size = args.num_workers if args.num_workers is not None else torch.cuda.device_count()
    if world_size == 0:
        logging.error("No CUDA-enabled GPUs found.")
        return
    logging.info(f"Spawning {world_size} worker processes...")
    with mp.Manager() as manager:
        stats = manager.dict({'completed': 0, 'failed': 0})
        mp.set_start_method("spawn", force=True)
        processes = [mp.Process(target=worker, args=(rank, world_size, jobs, args, stats, log_filename)) for rank in range(world_size)]
        for p in processes: p.start()
        with tqdm(total=len(jobs), desc="Processing files") as pbar:
            completed_last, failed_last = 0, 0
            while any(p.is_alive() for p in processes):
                completed_now, failed_now = stats['completed'], stats['failed']
                pbar.update((completed_now - completed_last) + (failed_now - failed_last))
                completed_last, failed_last = completed_now, failed_now
                time.sleep(1)
            completed_now, failed_now = stats['completed'], stats['failed']
            pbar.update((completed_now - completed_last) + (failed_now - failed_last))
        for p in processes: p.join()
        logging.info("--- Processing Complete ---")
        logging.info(f"Successfully processed: {stats['completed']} items")
        logging.info(f"Failed to process: {stats['failed']} items")
        logging.info(f"Skipped (pre-existing): {skipped_count} items")

if __name__ == "__main__":
    main()