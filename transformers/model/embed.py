#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
embed.py — Offline precompute of video ViT embeddings and audio mel specs (32 Hz), multi-GPU.

Outputs:
  vit_embed/<md5>/<basename>.npy  -> float16 [T, HIDDEN]  (default: HIDDEN = backbone hidden, e.g., 768)
  aud_embed/<md5>/<basename>.npy  -> float16 [T, n_mels, 2]

Key features:
- EXACT 32 fps alignment across modalities.
- Multi-GPU: spawn workers per GPU and distribute jobs across them.
- Atomic writes (.tmp -> rename), idempotent resume (skip existing unless --overwrite).
- No project imports; uses torchvision/torchaudio/transformers only (no DALI).

Video:
- Decoded via torchvision (falls back to decord if installed).
- Backbone: DINOv3-B/16 frozen (facebook/dinov3-vitb16-pretrain-lvd1689m), pooled features.
- Optional fixed projection to --project-d-model (default: 0 = off; keep projection learnable at train time).

Audio:
- Stereo; resample to 24 kHz; hop=win=750 -> exactly 32 steps/sec; n_mels=128; Mel->dB (top_db=80).

Usage:
  python embed.py --data-root /mnt/trainingdata/sampledata --gpus all --workers-per-gpu 1 \
                  --batch-frames 256 --dtype fp16 --overwrite

You can also restrict to a subset:
  --gpus 0,1       # use only GPU 0 and 1
  --only video     # or audio
"""

from __future__ import annotations
import os
import sys
import math
import argparse
import logging
import pathlib
import traceback
from typing import List, Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp

# TorchAudio for WAV I/O + mels
try:
    import torchaudio
    import torchaudio.functional as AF
    import torchaudio.transforms as AT
except Exception:
    torchaudio = None

# Prefer torchvision for decoding (widely available). Fallback to decord if present.
TV_AVAILABLE = False
try:
    import torchvision
    from torchvision.io import read_video
    TV_AVAILABLE = True
except Exception:
    TV_AVAILABLE = False

DEC_AVAILABLE = False
try:
    import decord
    from decord import VideoReader
    DEC_AVAILABLE = True
except Exception:
    DEC_AVAILABLE = False

# HF Transformers for DINOv3 backbone
try:
    from transformers import AutoModel
except Exception:
    AutoModel = None


# -----------------------------
# CLI
# -----------------------------
def get_args():
    p = argparse.ArgumentParser(description="Precompute ViT + audio features (multi-GPU).")
    p.add_argument("--data-root", required=True, type=str,
                   help="Root containing recordings/, vit_embed/, aud_embed/, etc.")
    p.add_argument("--gpus", type=str, default="all",
                   help="'all' or comma-separated GPU indices, e.g., '0,1,2,3'.")
    p.add_argument("--workers-per-gpu", type=int, default=1,
                   help="Processes per GPU. Usually 1 is best to avoid oversubscription.")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--batch-frames", type=int, default=256, help="Frames per ViT forward.")
    p.add_argument("--overwrite", action="store_true", help="Recompute even if outputs exist.")
    p.add_argument("--only", type=str, default="both", choices=["both", "video", "audio"])
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--model-name", type=str,
                   default="facebook/dinov3-vitb16-pretrain-lvd1689m",
                   help="HF model id for DINOv3 backbone.")
    p.add_argument("--project-d-model", type=int, default=0,
                   help="If >0, project to this dim before saving (NOT recommended).")
    p.add_argument("--target-fps", type=float, default=32.0)

    # Audio params (32 Hz grid via 24kHz / 750)
    p.add_argument("--audio-sr", type=int, default=24000)
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--win-length", type=int, default=750)
    p.add_argument("--hop-length", type=int, default=750)
    p.add_argument("--db-cutoff", type=float, default=80.0)

    # Advanced
    p.add_argument("--shuffle", action="store_true", help="Shuffle job order before dispatch.")
    return p.parse_args()


# -----------------------------
# Utilities
# -----------------------------
def atomic_save_npy(final_path: str, array: np.ndarray):
    tmp = final_path + ".tmp"
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    np.save(tmp, array)
    os.replace(tmp, final_path)


def list_recordings(root: str) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """Returns lists of (md5, stem, fullpath) for mp4s and wavs."""
    rec_dir = os.path.join(root, "recordings")
    mp4s, wavs = [], []
    if not os.path.isdir(rec_dir):
        return mp4s, wavs
    for md5 in sorted(os.listdir(rec_dir)):
        p = os.path.join(rec_dir, md5)
        if not os.path.isdir(p):
            continue
        for fname in sorted(os.listdir(p)):
            stem, ext = os.path.splitext(fname)
            fpath = os.path.join(p, fname)
            if ext.lower() == ".mp4":
                mp4s.append((md5, stem, fpath))
            elif ext.lower() == ".wav":
                wavs.append((md5, stem, fpath))
    return mp4s, wavs


def build_jobs(root: str, overwrite: bool, only: str):
    vids, auds = list_recordings(root)
    jobs = []
    for md5, stem, src in vids:
        out = os.path.join(root, "vit_embed", md5, f"{stem}.npy")
        if overwrite or not os.path.exists(out):
            jobs.append({"type": "video", "md5": md5, "stem": stem, "src": src, "out": out})
    for md5, stem, src in auds:
        out = os.path.join(root, "aud_embed", md5, f"{stem}.npy")
        if overwrite or not os.path.exists(out):
            jobs.append({"type": "audio", "md5": md5, "stem": stem, "src": src, "out": out})
    if only == "video":
        jobs = [j for j in jobs if j["type"] == "video"]
    elif only == "audio":
        jobs = [j for j in jobs if j["type"] == "audio"]
    return jobs


def to_torch_dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def parse_gpu_list(arg: str) -> List[int]:
    if not torch.cuda.is_available():
        return []
    if arg.strip().lower() == "all":
        return list(range(torch.cuda.device_count()))
    idxs = []
    for tok in arg.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        i = int(tok)
        if i < 0 or i >= torch.cuda.device_count():
            raise ValueError(f"GPU index {i} out of range (0..{torch.cuda.device_count()-1})")
        idxs.append(i)
    # de-dup, preserve order
    seen = set()
    out = []
    for i in idxs:
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out


# -----------------------------
# DINOv3 backbone (frozen)
# -----------------------------
class DINOv3Backbone(nn.Module):
    """
    Frozen DINOv3 ViT backbone (HF), with on-GPU normalization to ImageNet stats.

    Input:  [N,3,H,W] uint8 or float.
    Output: [N, hidden] pooled features (e.g., 768 for vitb16).
    """
    def __init__(self, model_name: str, compute_dtype: torch.dtype, channels_last: bool = True):
        super().__init__()
        if AutoModel is None:
            raise RuntimeError("transformers not available. pip install transformers accelerate")

        self.compute_dtype = compute_dtype
        self.use_channels_last = channels_last

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("img_mean", mean.to(self.compute_dtype), persistent=False)
        self.register_buffer("img_std", std.to(self.compute_dtype), persistent=False)

        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden = int(getattr(self.backbone.config, "hidden_size", 768))

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def _maybe_channels_last(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_channels_last and not x.is_contiguous(memory_format=torch.channels_last):
            return x.contiguous(memory_format=torch.channels_last)
        return x

    def _normalize(self, x: torch.Tensor, from_uint8: bool) -> torch.Tensor:
        if from_uint8:
            x = x.to(torch.float32).mul_(1.0 / 255.0)
        x = x.to(self.compute_dtype)
        mean = self.img_mean.to(x.device, non_blocking=True)
        std = self.img_std.to(x.device, non_blocking=True)
        x.sub_(mean).div_(std)
        return self._maybe_channels_last(x)

    @torch.no_grad()
    def forward(self, images_nchw: torch.Tensor) -> torch.Tensor:
        if images_nchw.dim() != 4 or images_nchw.shape[1] != 3:
            raise ValueError(f"Expected [N,3,H,W], got {tuple(images_nchw.shape)}")
        from_uint8 = images_nchw.dtype == torch.uint8
        x = self._normalize(images_nchw, from_uint8=from_uint8)
        with torch.autocast(device_type=("cuda" if x.is_cuda else "cpu"),
                            dtype=self.compute_dtype,
                            enabled=(self.compute_dtype != torch.float32)):
            out = self.backbone(pixel_values=x)
            feats = out.pooler_output  # [N, hidden]
        return feats


class OptionalProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# -----------------------------
# Video decoding @ 32 fps
# -----------------------------
def sample_indices_for_fps(n_src: int, fps_src: float, fps_tgt: float) -> np.ndarray:
    if fps_src <= 0 or n_src <= 0:
        return np.zeros((0,), dtype=np.int64)
    dur = n_src / fps_src
    n_tgt = int(math.floor(dur * fps_tgt + 1e-6))
    if n_tgt <= 0:
        return np.zeros((0,), dtype=np.int64)
    idx = np.clip(np.rint(np.arange(n_tgt) * (fps_src / fps_tgt)).astype(np.int64),
                  0, n_src - 1)
    return idx


def load_video_frames_32fps(path: str, target_fps: float) -> np.ndarray:
    """
    Returns uint8 frames [T, H, W, 3] sampled to target_fps.
    Prefers torchvision; falls back to decord if available.
    """
    if TV_AVAILABLE:
        video, _, info = read_video(path, pts_unit="sec")
        fps_src = float(info.get("video_fps", 0.0) or 0.0)
        if fps_src <= 0 or video.numel() == 0:
            return np.zeros((0, 1, 1, 3), dtype=np.uint8)
        T_src = int(video.shape[0])
        idx = sample_indices_for_fps(T_src, fps_src, target_fps)
        if idx.size == 0:
            return np.zeros((0, 1, 1, 3), dtype=np.uint8)
        picked = video.index_select(0, torch.from_numpy(idx))
        return picked.numpy()  # [T, H, W, 3], uint8
    elif DEC_AVAILABLE:
        vr = VideoReader(path)
        fps_src = float(vr.get_avg_fps())
        T_src = len(vr)
        idx = sample_indices_for_fps(T_src, fps_src, target_fps)
        if idx.size == 0:
            return np.zeros((0, 1, 1, 3), dtype=np.uint8)
        frames = vr.get_batch(idx)  # [T, H, W, 3], uint8
        return frames.asnumpy()
    else:
        raise RuntimeError("No video decoder found. Install torchvision (preferred) or decord.")


def compute_vit_embeddings_for_video(
    mp4_path: str,
    out_path: str,
    backbone: DINOv3Backbone,
    device: torch.device,
    compute_dtype: torch.dtype,
    batch_frames: int,
    project_d: int = 0,
    target_fps: float = 32.0,
) -> Tuple[int, Tuple[int, int]]:
    frames = load_video_frames_32fps(mp4_path, target_fps=target_fps)  # [T, H, W, 3] uint8
    T = int(frames.shape[0])
    if T == 0:
        raise RuntimeError("No frames decoded or zero-length after resampling.")

    projector = None
    if project_d and project_d > 0:
        projector = OptionalProjector(backbone.hidden, project_d).to(device)
        projector.eval()

    feats_out = []
    with torch.no_grad():
        for t0 in range(0, T, batch_frames):
            t1 = min(T, t0 + batch_frames)
            chunk = frames[t0:t1]  # [b, H, W, 3]
            nchw = torch.from_numpy(chunk).permute(0, 3, 1, 2).contiguous()  # uint8
            nchw = nchw.to(device, non_blocking=True)
            feats = backbone(nchw)  # [b, hidden]
            if projector is not None:
                with torch.autocast(device_type=("cuda" if feats.is_cuda else "cpu"),
                                    dtype=compute_dtype,
                                    enabled=(compute_dtype != torch.float32)):
                    feats = projector(feats)
            feats_out.append(feats.cpu())

    feats_all = torch.cat(feats_out, dim=0)  # [T, D]
    arr = feats_all.to(torch.float16).numpy()
    atomic_save_npy(out_path, arr)
    return T, arr.shape


# -----------------------------
# Audio @ 32 fps-aligned mels
# -----------------------------
def compute_audio_mels_for_wav(
    wav_path: str,
    out_path: str,
    sr: int = 24000,
    n_mels: int = 128,
    n_fft: int = 1024,
    win_length: int = 750,
    hop_length: int = 750,
    db_cutoff: float = 80.0,
    device: torch.device = torch.device("cuda:0"),
) -> Tuple[int, Tuple[int, int, int]]:
    if torchaudio is None:
        raise RuntimeError("torchaudio not available. pip install torchaudio")

    wav, orig_sr = torchaudio.load(wav_path)  # [C,N]
    if wav.dim() != 2:
        raise RuntimeError("Expected [C,N] waveform.")
    C, N = wav.shape
    if C == 1:
        wav = wav.repeat(2, 1)  # mono→stereo

    if orig_sr != sr:
        wav = AF.resample(wav, orig_sr, sr)

    N = wav.shape[1]
    T = N // hop_length
    if T <= 0:
        raise RuntimeError("Too-short audio.")
    N_use = T * hop_length
    wav = wav[:, :N_use]  # [2, N_use]

    dev = device if torch.cuda.is_available() else torch.device("cpu")
    mel = AT.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=0.0,
        f_max=sr / 2.0,
        pad=0,
        power=2.0,
        normalized=False,
        center=False,
        norm=None,
        n_mels=n_mels,
        mel_scale="htk",
    ).to(dev)
    to_db = AT.AmplitudeToDB(stype="power", top_db=db_cutoff).to(dev)

    x = wav.to(dev)
    with torch.no_grad():
        S = mel(x)             # [2, n_mels, T]
        S_db = to_db(S)        # [2, n_mels, T]
        S_db = S_db.permute(2, 1, 0).contiguous()  # [T, n_mels, 2]
        arr = S_db.to(torch.float16).cpu().numpy()

    atomic_save_npy(out_path, arr)
    return T, arr.shape


# -----------------------------
# Job building & dispatch
# -----------------------------
def build_job_list(root: str, overwrite: bool, only: str, shuffle: bool) -> List[dict]:
    import random
    jobs = build_jobs(root, overwrite, only)
    if shuffle:
        random.shuffle(jobs)
    else:
        # Interleave by type to keep GPUs fed (simple zipper)
        vids = [j for j in jobs if j["type"] == "video"]
        auds = [j for j in jobs if j["type"] == "audio"]
        jobs = []
        i = j = 0
        while i < len(vids) or j < len(auds):
            if i < len(vids):
                jobs.append(vids[i]); i += 1
            if j < len(auds):
                jobs.append(auds[j]); j += 1
    return jobs


def worker_main(
    rank: int,
    gpu_id: Optional[int],
    job_queue: mp.Queue,
    result_queue: mp.Queue,
    args_dict: dict,
):
    # Basic per-worker logging
    def log(level, msg):
        print(f"[W{rank}|GPU{gpu_id if gpu_id is not None else 'cpu'}][{level}] {msg}", flush=True)

    # Set device
    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    # Perf knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    torch.backends.cudnn.benchmark = True

    # Unpack args
    model_name = args_dict["model_name"]
    compute_dtype = to_torch_dtype(args_dict["dtype"])
    batch_frames = args_dict["batch_frames"]
    project_d = args_dict["project_d_model"]
    target_fps = args_dict["target_fps"]

    audio_sr = args_dict["audio_sr"]
    n_mels = args_dict["n_mels"]
    n_fft = args_dict["n_fft"]
    win_length = args_dict["win_length"]
    hop_length = args_dict["hop_length"]
    db_cutoff = args_dict["db_cutoff"]

    # Lazy-load backbone on first video job to avoid model init cost if audio-only
    backbone = None

    while True:
        job = job_queue.get()
        if job is None:
            job_queue.task_done()
            break

        jtype = job["type"]
        src = job["src"]
        out = job["out"]

        try:
            if jtype == "video":
                if backbone is None:
                    log("INFO", f"Loading DINOv3 backbone on {device} (dtype={args_dict['dtype']})...")
                    backbone = DINOv3Backbone(
                        model_name=model_name,
                        compute_dtype=compute_dtype,
                        channels_last=True,
                    ).to(device)
                    backbone.eval()
                log("INFO", f"VIDEO {src} -> {out}")
                T, shape = compute_vit_embeddings_for_video(
                    mp4_path=src,
                    out_path=out,
                    backbone=backbone,
                    device=device,
                    compute_dtype=compute_dtype,
                    batch_frames=batch_frames,
                    project_d=project_d,
                    target_fps=target_fps,
                )
                log("OK", f"wrote {out} shape={shape} T={T}")
                result_queue.put(("video_ok", 1))
            else:
                log("INFO", f"AUDIO {src} -> {out}")
                T, shape = compute_audio_mels_for_wav(
                    wav_path=src,
                    out_path=out,
                    sr=audio_sr,
                    n_mels=n_mels,
                    n_fft=n_fft,
                    win_length=win_length,
                    hop_length=hop_length,
                    db_cutoff=db_cutoff,
                    device=device,
                )
                log("OK", f"wrote {out} shape={shape} T={T}")
                result_queue.put(("audio_ok", 1))

        except Exception as e:
            log("ERR", f"FAILED {jtype} {src}: {e}")
            # Clean any tmp residue
            tmp = out + ".tmp"
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            result_queue.put((f"{jtype}_fail", 1))
        finally:
            job_queue.task_done()

    # Cleanup
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    log("INFO", "Worker exiting.")
def multi_gpu_dispatch(args):
    import random, time
    import multiprocessing as mp
    from queue import Empty

    logging.basicConfig(
        level=(logging.DEBUG if args.verbose else logging.INFO),
        format="[%(levelname)s] %(message)s",
    )

    if args.only in ("both", "audio") and torchaudio is None:
        raise RuntimeError("torchaudio is required for audio features. Install torchaudio or use --only video.")

    jobs = build_job_list(args.data_root, args.overwrite, args.only, shuffle=args.shuffle)
    logging.info("Total jobs to run: %d", len(jobs))
    if not jobs:
        logging.info("Nothing to do.")
        return

    gpu_ids = parse_gpu_list(args.gpus)
    if len(gpu_ids) == 0:
        logging.warning("No CUDA GPUs detected or requested; running on CPU with a single worker.")
        num_workers = 1
        gpu_map = [None]
    else:
        num_workers = max(1, args.workers_per_gpu) * len(gpu_ids)
        gpu_map = [gpu_ids[i % len(gpu_ids)] for i in range(num_workers)]
        logging.info("Using GPUs: %s (%d workers total)", ",".join(map(str, gpu_ids)), num_workers)

    # Use spawn (safer across libs) and an **unbounded** JoinableQueue
    ctx = mp.get_context("spawn")
    job_queue = ctx.JoinableQueue(maxsize=0)  # unbounded avoids feeder blocking
    result_queue = ctx.Queue()

    args_dict = {
        "model_name": args.model_name,
        "dtype": args.dtype,
        "batch_frames": args.batch_frames,
        "project_d_model": args.project_d_model,
        "target_fps": args.target_fps,
        "audio_sr": args.audio_sr,
        "n_mels": args.n_mels,
        "n_fft": args.n_fft,
        "win_length": args.win_length,
        "hop_length": args.hop_length,
        "db_cutoff": args.db_cutoff,
    }

    # 1) Spawn workers **before** enqueuing any jobs/sentinels
    procs = []
    for rank in range(num_workers):
        p = ctx.Process(
            target=worker_main,
            args=(rank, gpu_map[rank], job_queue, result_queue, args_dict),
            daemon=True,
        )
        p.start()
        procs.append(p)

    # 2) Enqueue jobs (producer)
    try:
        for j in jobs:
            job_queue.put(j)

        # 3) Enqueue sentinels to stop workers after jobs drain
        for _ in range(num_workers):
            job_queue.put(None)

        # Progress accounting
        counts = {"video_ok": 0, "video_fail": 0, "audio_ok": 0, "audio_fail": 0}
        remaining = len(jobs)

        # 4) Collect results; tolerant to transient timeouts
        while remaining > 0:
            try:
                key, val = result_queue.get(timeout=1.0)
                if key in counts:
                    counts[key] += val
                remaining -= 1
            except Empty:
                # keep UI responsive; also break if all workers are gone unexpectedly
                if all(not p.is_alive() for p in procs):
                    break

        # 5) Wait for queue drain + worker exit
        job_queue.join()
        for p in procs:
            p.join(timeout=5.0)

        logging.info("DONE. Video ok/failed: %d/%d, Audio ok/failed: %d/%d",
                     counts["video_ok"], counts["video_fail"], counts["audio_ok"], counts["audio_fail"])

    except KeyboardInterrupt:
        logging.warning("KeyboardInterrupt received — stopping workers...")
        # Push sentinels so workers break promptly (if not already pushed)
        try:
            for _ in range(num_workers):
                job_queue.put_nowait(None)
        except Exception:
            pass
        # Best-effort drain + join
        try:
            job_queue.join()
        except Exception:
            pass
        # Terminate any stragglers
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=2.0)
        logging.info("Shutdown complete.")


# -----------------------------
# Entry
# -----------------------------
def main():
    args = get_args()
    multi_gpu_dispatch(args)


if __name__ == "__main__":
    main()
