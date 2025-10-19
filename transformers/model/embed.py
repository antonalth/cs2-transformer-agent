#!/usr/bin/env python3
"""
embed.py — Offline Feature Pre-computation for Video and Audio (with --save-full, CLS KEPT)

This script processes recordings to pre-compute and cache features.
Default behavior matches the previous pipeline (pooled video embeddings to vit_embed/,
audio mel-spectrograms to aud_embed/).

When --save-full is provided, the script saves the **full final hidden states** from the
vision backbone (INCLUDING the CLS/global token) under vid_full_embed/, and mirrors
the audio save path to aud_full_embed/.

Outputs:
  - Default:
        <data_root>/vit_embed/<md5>/<name>.npy   (video pooled, [T, D])
        <data_root>/aud_embed/<md5>/<name>.npy   (audio mel,   [T, Mel, 2])
  - With --save-full:
        <data_root>/vid_full_embed/<md5>/<name>.npy  (video tokens incl. CLS, [T, L, D])
        <data_root>/aud_full_embed/<md5>/<name>.npy  (audio mel,              [T, Mel, 2])

Notes:
  - Uses DALI if available for fast IO; otherwise, you may adapt loaders.
  - Relies on DINOv3VisualEncoder from model.py and DaliConfig from train3.py.
  - Saves float16 .npy to reduce disk usage.
"""

import os
import argparse
import logging
import sys
import time
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

# External components from your codebase
from train3 import DaliConfig, FPS
from model import CS2Config, DINOv3VisualEncoder

# --- Optional DALI ---
try:
    from nvidia.dali import fn, pipeline_def, types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from nvidia.dali.plugin.base_iterator import LastBatchPolicy
    DALI_AVAILABLE = True
except Exception as e:
    DALI_AVAILABLE = False
    DALI_IMPORT_ERROR = e


# ---------------------------
# Logging
# ---------------------------

def setup_logging(log_filename: str):
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
# Args
# ---------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Pre-compute video and audio embeddings.")
    parser.add_argument("--data-root", type=str, default=os.environ.get("DATA_ROOT", "data"),
                        help="Root directory containing recordings/, vit_embed/, aud_embed/, etc.")
    parser.add_argument("--mode", type=str, choices=["video", "audio", "both"], default="both",
                        help="Which modality to process.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing embedding files.")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Number of video frames per DALI chunk.")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of GPUs to use. Defaults to all available GPUs.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of files for debugging.")
    parser.add_argument("--save-full", action="store_true",
                        help="If set, save full vision hidden states (INCLUDING CLS) under vid_full_embed/ and mirror audio to aud_full_embed/.")
    return parser.parse_args()


# ---------------------------
# Utils
# ---------------------------

def atomic_save_npy(path: Path, array: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent, delete=False) as f:
        np.save(f, array)
        tmp = f.name
    try:
        if hasattr(os, 'fsync'):
            with open(tmp, 'rb') as ff:
                os.fsync(ff.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass


# ---------------------------
# Video
# ---------------------------

def process_video(video_path: Path, model: DINOv3VisualEncoder, batch_size: int, gpu_id: int, save_full: bool=False) -> Optional[torch.Tensor]:
    if not DALI_AVAILABLE:
        logging.error("DALI is not available. Install NVIDIA DALI or adjust the loader.")
        return None
    if not video_path.exists():
        logging.warning(f"[GPU {gpu_id}] Video file not found: {video_path}")
        return None

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
    dali_iter = DALIGenericIterator([pipe], ['data'], reader_name=f"VideoReader_{video_path.name}",
                                    auto_reset=True, last_batch_policy=LastBatchPolicy.PARTIAL)

    all_features = []
    try:
        for batch in dali_iter:
            frames = batch[0]['data']  # [1, T, H, W, C]
            frames = frames.squeeze(0).permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]
            x = frames.to(device=gpu_id, dtype=torch.uint8)

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                norm = (x.float() / 255.0 - model.img_mean) / model.img_std

                if save_full:
                    # Return full final hidden states (INCLUDING CLS token)
                    out = model.backbone(pixel_values=norm)
                    seq = out.last_hidden_state  # [T, L, D] — L includes CLS for ViT
                    feats = seq
                else:
                    # Pooled representation as in your existing encoder helper
                    feats = model._forward_backbone_no_grad(norm)  # [T, D]

            all_features.append(feats.cpu())
    except Exception as e:
        logging.error(f"[GPU {gpu_id}] Failed processing video {video_path}: {e}", exc_info=True)
        return None
    finally:
        del pipe, dali_iter

    if not all_features:
        return None
    return torch.cat(all_features, dim=0)  # [T, D] or [T, L, D]


# ---------------------------
# Audio
# ---------------------------

def process_audio(wav_path: Path, dali_cfg: DaliConfig, gpu_id: int) -> Optional[torch.Tensor]:
    if not DALI_AVAILABLE:
        logging.error("DALI is not available. Install NVIDIA DALI or adjust the loader.")
        return None
    if not wav_path.exists():
        logging.warning(f"[GPU {gpu_id}] Audio file not found: {wav_path}")
        return None

    @pipeline_def(batch_size=1, num_threads=2, device_id=gpu_id, prefetch_queue_depth=3)
    def create_audio_pipeline():
        audio_raw, _ = fn.readers.file(files=[str(wav_path)], name=f"AudioReader_{wav_path.name}")
        decoded, _ = fn.decoders.audio(audio_raw, sample_rate=dali_cfg.sample_rate, downmix=False)
        left, right = decoded[:, 0], decoded[:, 1]

        def to_mel_db(channel_1d):
            spec = fn.spectrogram(channel_1d,
                                  nfft=dali_cfg.nfft, window_length=dali_cfg.win_length,
                                  window_step=dali_cfg.hop_length, center_windows=False)
            mel = fn.mel_filter_bank(spec, sample_rate=dali_cfg.sample_rate,
                                     nfilter=dali_cfg.mel_bins, freq_high=dali_cfg.mel_fmax)
            db = fn.to_decibels(mel, cutoff_db=dali_cfg.db_cutoff)
            return fn.transpose(db, perm=[1, 0])  # [T, Mel]

        mel_left, mel_right = to_mel_db(left), to_mel_db(right)
        mel_stereo = fn.stack(mel_left, mel_right, axis=0)  # [2, T, Mel]
        return mel_stereo.gpu()

    pipe = create_audio_pipeline()
    pipe.build()
    dali_iter = DALIGenericIterator([pipe], ['mel'], reader_name=f"AudioReader_{wav_path.name}",
                                    auto_reset=True, last_batch_policy=LastBatchPolicy.FILL)
    try:
        mel_batch = next(iter(dali_iter))
        mel_tensor = mel_batch[0]['mel'].squeeze(0).permute(1, 2, 0)  # [T, Mel, 2]
        return mel_tensor
    except Exception as e:
        logging.error(f"[GPU {gpu_id}] Failed processing audio {wav_path}: {e}", exc_info=True)
        return None
    finally:
        del pipe, dali_iter


# ---------------------------
# Worker
# ---------------------------

def save_results_background(fut):
    try:
        fut.result()
    except Exception as e:
        logging.error(f"Background save failed: {e}", exc_info=True)

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

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = None
        for md5, name, video_in_path, audio_in_path, video_out_path, audio_out_path in jobs[rank::world_size]:
            try:
                start_time = time.time()

                video_tensor = None
                audio_tensor = None

                if args.mode in ["video", "both"] and video_in_path is not None:
                    video_tensor = process_video(video_in_path, video_model, args.batch_size, gpu_id, save_full=args.save_full)
                    if video_tensor is None:
                        raise RuntimeError("Video tensor is None")

                if args.mode in ["audio", "both"] and audio_in_path is not None:
                    audio_tensor = process_audio(audio_in_path, dali_cfg, gpu_id)
                    if audio_tensor is None:
                        raise RuntimeError("Audio tensor is None")

                # By accepting the tensors as arguments, we avoid a race condition where the main
                # loop could set video_tensor/audio_tensor to None before this thread runs.
                def save_job(video_tensor, audio_tensor):
                    if args.mode == "both":
                        if video_tensor is None or audio_tensor is None:
                            logging.warning(f"[GPU {gpu_id}] Skipped {name} due to missing modality outputs.")
                            return
                        T_final = min(video_tensor.shape[0], audio_tensor.shape[0])
                        if T_final <= 0:
                            logging.warning(f"[GPU {gpu_id}] Skipped {name} due to zero-length output.")
                            return
                        video_np = video_tensor[:T_final].cpu().to(torch.float16).numpy()
                        audio_np = audio_tensor[:T_final].cpu().to(torch.float16).numpy()
                        atomic_save_npy(video_out_path, video_np)
                        atomic_save_npy(audio_out_path, audio_np)
                    elif args.mode == "video":
                        if video_tensor is None:
                            logging.warning(f"[GPU {gpu_id}] Cannot save None video tensor for {name}.")
                            return
                        video_np = video_tensor.cpu().to(torch.float16).numpy()
                        atomic_save_npy(video_out_path, video_np)
                    else:  # audio
                        if audio_tensor is None:
                            logging.warning(f"[GPU {gpu_id}] Cannot save None audio tensor for {name}.")
                            return
                        audio_np = audio_tensor.cpu().to(torch.float16).numpy()
                        atomic_save_npy(audio_out_path, audio_np)

                    stats['completed'] += 1
                    logging.info(f"[GPU {gpu_id}] Completed {md5}/{name} in {time.time()-start_time:.2f}s")

                future = executor.submit(save_job, video_tensor, audio_tensor)
                future.add_done_callback(save_results_background)

            except Exception as e:
                logging.error(f"[GPU {gpu_id}] Unhandled error on job {md5}/{name}: {e}", exc_info=False)
                stats['failed'] += 1

        if future:
            future.result()


# ---------------------------
# Main
# ---------------------------

def main():
    args = get_args()

    # Setup logging here so the log file is created early
    log_dir = Path(args.data_root)
    log_dir.mkdir(exist_ok=True)
    log_filename = str(log_dir / f"embed_{int(time.time())}.log")
    setup_logging(log_filename)


    data_root = Path(args.data_root)
    recordings_dir = data_root / "recordings"

    # Output dirs based on flag
    if args.save_full:
        vid_dir = data_root / "vid_full_embed"
        aud_dir = data_root / "aud_full_embed"
    else:
        vid_dir = data_root / "vit_embed"
        aud_dir = data_root / "aud_embed"

    if not recordings_dir.is_dir():
        logging.error(f"Recordings directory not found: {recordings_dir}")
        return

    # Discover files
    mp4s = sorted(recordings_dir.rglob("*.mp4"))
    wavs = sorted(recordings_dir.rglob("*.wav"))
    media = {}

    for p in mp4s:
        md5 = p.parent.name
        name = p.stem
        media.setdefault((md5, name), {})['video'] = p
    for p in wavs:
        md5 = p.parent.name
        name = p.stem
        media.setdefault((md5, name), {})['audio'] = p

    jobs: List[Tuple] = []
    skipped_count = 0

    for (md5, name), d in media.items():
        v_in = d.get('video')
        a_in = d.get('audio')

        v_out = vid_dir / md5 / f"{name}.npy"
        a_out = aud_dir / md5 / f"{name}.npy"

        need_v = (args.mode in ["video", "both"]) and (v_in is not None) and (args.overwrite or not v_out.exists())
        need_a = (args.mode in ["audio", "both"]) and (a_in is not None) and (args.overwrite or not a_out.exists())

        if not need_v and not need_a:
            skipped_count += 1
            continue

        jobs.append((md5, name, v_in, a_in, v_out, a_out))

        if args.limit is not None and len(jobs) >= args.limit:
            break

    if not jobs:
        logging.info("Nothing to do.")
        return

    world_size = args.num_workers if args.num_workers is not None else torch.cuda.device_count()
    if world_size is None or world_size <= 0:
        logging.error("No CUDA-enabled GPUs found.")
        return

    logging.info(f"Spawning {world_size} worker processes...")
    with mp.Manager() as manager:
        stats = manager.dict({'completed': 0, 'failed': 0})
        mp.set_start_method("spawn", force=True)

        procs = [mp.Process(target=worker, args=(rank, world_size, jobs, args, stats, log_filename))
                 for rank in range(world_size)]
        for p in procs:
            p.start()

        with tqdm(total=len(jobs), desc="Processing files") as pbar:
            completed_last, failed_last = 0, 0
            while any(p.is_alive() for p in procs):
                completed_now, failed_now = stats['completed'], stats['failed']
                pbar.update((completed_now - completed_last) + (failed_now - failed_last))
                completed_last, failed_last = completed_now, failed_now
                time.sleep(1)
            completed_now, failed_now = stats['completed'], stats['failed']
            pbar.update((completed_now - completed_last) + (failed_now - failed_last))

        for p in procs:
            p.join()

        logging.info("--- Processing Complete ---")
        logging.info(f"Successfully processed: {stats['completed']} items")
        logging.info(f"Failed to process: {stats['failed']} items")
        logging.info(f"Skipped (pre-existing): {skipped_count} items")


if __name__ == "__main__":
    main()