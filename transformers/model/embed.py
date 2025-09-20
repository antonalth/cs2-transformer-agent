#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
embed.py — Offline precompute of ViT frame embeddings + stereo log-mels (32 Hz) with multi-GPU.
Audio uses **NVIDIA DALI** (mirrors train2.py), no torchaudio anywhere.

Outputs
  vit_embed/<md5>/<name>.npy  -> float16 [T, D]      (D = backbone hidden, e.g., 768)
  aud_embed/<md5>/<name>.npy  -> float16 [T, n_mels, 2]  (stereo, 32 Hz alignment)

Highlights
- EXACT 32 fps alignment for both modalities (audio: 24 kHz, nfft=1024, win=750, hop=750).
- Audio: DALI graph = decoders.audio → spectrogram(window_step=hop) → mel_filter_bank → to_decibels.
- Video: same robust ViT path you already have (HF AutoModel, frozen, pooled features).
- Multi-GPU: workers-first queue, unbounded JoinableQueue, clean Ctrl-C.
- Atomic .npy writes via temp file handle (no “.npy.tmp.npy” bug).
- Robust HF imports even if your CWD is named `transformers/`.

Run
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python embed.py --data-root /mnt/trainingdata/sampledata \
    --gpus all --workers-per-gpu 1 \
    --batch-frames 256 --dtype fp16 --shuffle
"""

from __future__ import annotations
import os, sys, math, argparse, logging, tempfile
from typing import List, Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp

# ---------------------------
# Video decode (as before)
# ---------------------------
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

# ---------------------------
# DALI (audio) — mirror train2.py
# ---------------------------
DALI_AVAILABLE = False
_DALI_IMPORT_ERROR = None
try:
    from nvidia.dali import fn, types
    from nvidia.dali.pipeline import Pipeline
    DALI_AVAILABLE = True
except Exception as _e:
    DALI_AVAILABLE = False
    _DALI_IMPORT_ERROR = _e

# =========================
# CLI
# =========================
def get_args():
    p = argparse.ArgumentParser(description="Precompute ViT + audio (DALI), multi-GPU.")
    p.add_argument("--data-root", required=True, type=str,
                   help="Root containing recordings/, vit_embed/, aud_embed/, ...")
    p.add_argument("--gpus", type=str, default="all",
                   help="'all' or comma-separated GPU indices (e.g., '0,1,2,3').")
    p.add_argument("--workers-per-gpu", type=int, default=1,
                   help="Processes per GPU; 1 is usually best.")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--batch-frames", type=int, default=256, help="Frames per ViT forward.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--only", type=str, default="both", choices=["both", "video", "audio"])
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--model-name", type=str,
                   default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    p.add_argument("--target-fps", type=float, default=32.0)

    # Audio params (mirrors train2.py DaliConfig)
    p.add_argument("--audio-sr", type=int, default=24000)
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--win-length", type=int, default=750)
    p.add_argument("--hop-length", type=int, default=750)
    p.add_argument("--db-cutoff", type=float, default=80.0)

    p.add_argument("--shuffle", action="store_true")
    return p.parse_args()


# =========================
# Utils
# =========================
def atomic_save_npy(final_path: str, array: np.ndarray):
    """Atomic .npy write using a real temp file (no '.npy.tmp.npy')."""
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    dirpath = os.path.dirname(final_path) or "."
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".npy.tmp", dir=dirpath, delete=False) as tmpf:
        tmp_path = tmpf.name
        np.save(tmpf, array)
        tmpf.flush()
        os.fsync(tmpf.fileno())
    os.replace(tmp_path, final_path)

def list_recordings(root: str) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    rec_dir = os.path.join(root, "recordings")
    mp4s, wavs = [], []
    if not os.path.isdir(rec_dir):
        return mp4s, wavs
    for md5 in sorted(os.listdir(rec_dir)):
        d = os.path.join(rec_dir, md5)
        if not os.path.isdir(d): continue
        for fname in sorted(os.listdir(d)):
            stem, ext = os.path.splitext(fname)
            fpath = os.path.join(d, fname)
            if ext.lower() == ".mp4": mp4s.append((md5, stem, fpath))
            elif ext.lower() == ".wav": wavs.append((md5, stem, fpath))
    return mp4s, wavs

def build_jobs(root: str, overwrite: bool, only: str) -> List[dict]:
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
    out, seen = [], set()
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok: continue
        i = int(tok)
        if i < 0 or i >= torch.cuda.device_count():
            raise ValueError(f"GPU index {i} out of range (0..{torch.cuda.device_count()-1})")
        if i not in seen:
            out.append(i); seen.add(i)
    return out


# =========================
# Robust HF import (avoid local `transformers/` shadowing)
# =========================
def import_hf_safely():
    import importlib, site
    site_paths = []
    try: site_paths.extend(site.getsitepackages())
    except Exception: pass
    try:
        user_site = site.getusersitepackages()
        if user_site: site_paths.append(user_site)
    except Exception: pass
    site_paths = [p for p in site_paths if p and os.path.isdir(p)]
    for p in reversed(site_paths):
        if p not in sys.path:
            sys.path.insert(0, p)
    if "transformers" in sys.modules:
        mod = sys.modules["transformers"]
        mfile = getattr(mod, "__file__", "") or ""
        if mfile and "site-packages" not in mfile and "dist-packages" not in mfile:
            del sys.modules["transformers"]
    tr = importlib.import_module("transformers")
    AutoModel = getattr(tr, "AutoModel")
    return tr, AutoModel


# =========================
# ViT backbone (model.py style; pooled features)
# =========================
class VisualBackbone(nn.Module):
    """
    Frozen ViT backbone (HF AutoModel), pooled features like model.py.
    - Tries requested model (e.g., DINOv3); if unavailable, falls back to ViT-Base-16.
    - Manual center-crop + bilinear resize to config.image_size, ImageNet normalize on device.
    """
    def __init__(self, model_name: str, compute_dtype: torch.dtype, channels_last: bool = True):
        super().__init__()
        try:
            _, AutoModel = import_hf_safely()
        except Exception as e:
            raise RuntimeError(f"Could not import Hugging Face transformers from site-packages: {e}")

        used = model_name
        try:
            bb = AutoModel.from_pretrained(model_name)
        except Exception:
            used = "google/vit-base-patch16-224-in21k"
            bb = AutoModel.from_pretrained(used)

        self.backbone = bb
        self.hidden = int(getattr(self.backbone.config, "hidden_size", 768))
        self.img_size = int(getattr(self.backbone.config, "image_size", 224))
        self.used_model_name = used

        self.compute_dtype = compute_dtype
        self.channels_last = channels_last

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("img_mean", mean.to(self.compute_dtype), persistent=False)
        self.register_buffer("img_std",  std.to(self.compute_dtype),  persistent=False)

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    @torch.no_grad()
    def _maybe_channels_last(self, x: torch.Tensor) -> torch.Tensor:
        if self.channels_last and not x.is_contiguous(memory_format=torch.channels_last):
            return x.contiguous(memory_format=torch.channels_last)
        return x

    @torch.no_grad()
    def _center_crop_resize(self, x: torch.Tensor, size: int) -> torch.Tensor:
        import torch.nn.functional as F
        N, C, H, W = x.shape
        s = min(H, W)
        y0 = (H - s) // 2
        x0 = (W - s) // 2
        x = x[:, :, y0:y0+s, x0:x0+s]
        if s != size:
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        return x

    @torch.no_grad()
    def _normalize(self, x: torch.Tensor, from_uint8: bool) -> torch.Tensor:
        if from_uint8:
            x = x.to(self.compute_dtype)
            x.mul_(1.0 / 255.0)
        else:
            if x.dtype != self.compute_dtype:
                x = x.to(self.compute_dtype)
        x = x.sub(self.img_mean.to(x.device)).div(self.img_std.to(x.device))
        return self._maybe_channels_last(x)

    @torch.no_grad()
    def embed_nchw(self, images_nchw: torch.Tensor, device: torch.device, chunk: int = 256) -> torch.Tensor:
        if images_nchw.dim() != 4 or images_nchw.shape[1] != 3:
            raise ValueError(f"Expected [N,3,H,W], got {tuple(images_nchw.shape)}")
        N = images_nchw.shape[0]
        outs = []
        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            x = images_nchw[s:e].to(device, non_blocking=True)
            x = self._center_crop_resize(x, self.img_size)
            x = self._normalize(x, from_uint8=(x.dtype == torch.uint8))
            with torch.autocast(device_type=("cuda" if x.is_cuda else "cpu"),
                                dtype=self.compute_dtype,
                                enabled=(x.is_cuda and self.compute_dtype != torch.float32)):
                out = self.backbone(pixel_values=x)
                feats = out.pooler_output  # [b, hidden]
            outs.append(feats.cpu())
        return torch.cat(outs, dim=0)


# =========================
# Video decode @ 32 fps (same as before)
# =========================
def sample_indices_for_fps(n_src: int, fps_src: float, fps_tgt: float) -> np.ndarray:
    if fps_src <= 0 or n_src <= 0:
        return np.zeros((0,), dtype=np.int64)
    dur = n_src / fps_src
    n_tgt = int(math.floor(dur * fps_tgt + 1e-6))
    if n_tgt <= 0:
        return np.zeros((0,), dtype=np.int64)
    idx = np.clip(
        np.rint(np.arange(n_tgt) * (fps_src / fps_tgt)).astype(np.int64),
        0, n_src - 1
    )
    return idx

def load_video_frames_32fps(path: str, target_fps: float) -> np.ndarray:
    """Return uint8 frames [T, H, W, 3] resampled to target_fps."""
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
        return picked.numpy()
    elif DEC_AVAILABLE:
        vr = VideoReader(path)
        fps_src = float(vr.get_avg_fps())
        T_src = len(vr)
        idx = sample_indices_for_fps(T_src, fps_src, target_fps)
        if idx.size == 0:
            return np.zeros((0, 1, 1, 3), dtype=np.uint8)
        frames = vr.get_batch(idx)
        return frames.asnumpy()
    else:
        raise RuntimeError("No video decoder found. Install torchvision (preferred) or decord.")

def compute_vit_embeddings_for_video(
    mp4_path: str,
    out_path: str,
    backbone: VisualBackbone,
    device: torch.device,
    batch_frames: int,
    target_fps: float = 32.0,
) -> Tuple[int, Tuple[int, int]]:
    frames = load_video_frames_32fps(mp4_path, target_fps=target_fps)  # [T,H,W,3] uint8
    T = int(frames.shape[0])
    if T == 0:
        raise RuntimeError("No frames decoded or zero-length after resampling.")

    outs = []
    with torch.no_grad():
        for t0 in range(0, T, batch_frames):
            t1 = min(T, t0 + batch_frames)
            chunk = frames[t0:t1]
            nchw = torch.from_numpy(chunk).permute(0, 3, 1, 2).contiguous()
            feats = backbone.embed_nchw(nchw, device=device, chunk=batch_frames)
            outs.append(feats.cpu())

    feats_all = torch.cat(outs, dim=0)  # [T, D]
    arr = feats_all.to(torch.float16).numpy()
    atomic_save_npy(out_path, arr)
    return T, arr.shape


# =========================
# DALI audio (mirror train2.py)
# =========================
class DaliAudioPipeline(Pipeline):
    """
    Single-file audio → stereo log-mel [T, n_mels, 2] at 32 Hz.
    Mirrors train2.py: decoders.audio → spectrogram(window_step=hop) → mel_filter_bank → to_decibels.
    """
    def __init__(
        self,
        file_list: str,
        device_id: int,
        batch_size: int = 1,
        num_threads: int = 2,
        seed: int = 42,
        sample_rate: int = 24000,
        n_mels: int = 128,
        nfft: int = 1024,
        win_length: int = 750,
        hop_length: int = 750,
        db_cutoff: float = 80.0,
    ):
        super().__init__(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=seed,
                         prefetch_queue_depth={"cpu_size": 2, "gpu_size": 1})
        self.file_list = file_list
        self.sample_rate = float(sample_rate)
        self.n_mels = int(n_mels)
        self.nfft = int(nfft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.db_cutoff = float(db_cutoff)

    def define_graph(self):
        if not DALI_AVAILABLE:
            raise RuntimeError(f"NVIDIA DALI not available: {_DALI_IMPORT_ERROR}")
        # read raw bytes from a 1-line file list
        audio_raw, _ = fn.readers.file(
            file_list=self.file_list,
            random_shuffle=False,
            name="A0",
        )
        decoded, _ = fn.decoders.audio(audio_raw, sample_rate=self.sample_rate, downmix=False)  # [N, C?]
        x = decoded.gpu()  # -> GPU

        # Ensure 2 channels by slicing channel axis with padding (if mono)
        # time axis = 0, channel axis = 1
        # shape=[-1,2] isn't allowed, so slice per axis:
        left  = fn.slice(x, start=[0, 0], shape=[-1, 1], axes=[0, 1], out_of_bounds_policy="pad")
        right = fn.slice(x, start=[0, 1], shape=[-1, 1], axes=[0, 1], out_of_bounds_policy="pad")
        left  = fn.squeeze(left, axes=[1])   # [N]
        right = fn.squeeze(right, axes=[1])  # [N]

        def to_mel_db(ch_1d):
            spec = fn.spectrogram(
                ch_1d,
                nfft=self.nfft,
                window_length=self.win_length,
                window_step=self.hop_length,       # <— match train2.py
                center_windows=False,
            )
            mel = fn.mel_filter_bank(
                spec,
                sample_rate=self.sample_rate,
                nfilter=self.n_mels,
                freq_high=self.sample_rate / 2.0,
            )
            db = fn.to_decibels(mel, cutoff_db=self.db_cutoff)
            db = fn.cast(db, dtype=types.FLOAT16)   # keep small, matches training
            return fn.transpose(db, perm=[1, 0])    # [time, n_mels]

        mL = to_mel_db(left)
        mR = to_mel_db(right)
        out = fn.stack(mL, mR, axis=2)              # [time, n_mels, 2]
        return out

def compute_audio_mels_dali(
    wav_path: str,
    out_path: str,
    device_id: int,
    sr: int = 24000,
    n_mels: int = 128,
    n_fft: int = 1024,
    win_length: int = 750,
    hop_length: int = 750,
    db_cutoff: float = 80.0,
) -> Tuple[int, Tuple[int, int, int]]:
    if not DALI_AVAILABLE:
        raise RuntimeError(f"NVIDIA DALI not available: {_DALI_IMPORT_ERROR}")

    # Build 1-line filelist for readers.file
    tmp_dir = os.path.dirname(out_path) or "."
    os.makedirs(tmp_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lst", dir=tmp_dir, delete=False) as fl:
        # label is unused; any int is fine
        fl.write(f"{os.path.abspath(wav_path)} 0\n")
        file_list_path = fl.name

    try:
        pipe = DaliAudioPipeline(
            file_list=file_list_path,
            device_id=device_id,
            batch_size=1,
            num_threads=2,
            seed=42,
            sample_rate=sr,
            n_mels=n_mels,
            nfft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            db_cutoff=db_cutoff,
        )
        pipe.build()
        # single iteration → one sample
        outs = pipe.run()
        mel_gpu = outs[0]  # TensorListGPU
        mel_cpu = mel_gpu.as_cpu()
        # Fetch the single sample as numpy
        arr = mel_cpu.at(0)  # numpy array [T, n_mels, 2], float16
        atomic_save_npy(out_path, arr)
        T = int(arr.shape[0])
        return T, arr.shape
    finally:
        try: os.remove(file_list_path)
        except Exception: pass


# =========================
# Job building & dispatch
# =========================
def build_job_list(root: str, overwrite: bool, only: str, shuffle: bool) -> List[dict]:
    import random
    jobs = build_jobs(root, overwrite, only)
    if shuffle:
        random.shuffle(jobs)
    else:
        vids = [j for j in jobs if j["type"] == "video"]
        auds = [j for j in jobs if j["type"] == "audio"]
        jobs = []
        i = j = 0
        while i < len(vids) or j < len(auds):
            if i < len(vids): jobs.append(vids[i]); i += 1
            if j < len(auds): jobs.append(auds[j]); j += 1
    return jobs

def worker_main(rank: int, gpu_id: Optional[int], job_queue: mp.Queue, result_queue: mp.Queue, args_dict: dict):
    def log(level, msg):
        print(f"[W{rank}|GPU{gpu_id if gpu_id is not None else 'cpu'}][{level}] {msg}", flush=True)

    # Device
    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        dali_device_id = gpu_id
    else:
        device = torch.device("cpu")
        dali_device_id = 0  # still required by DALI, but won't be used if CUDA absent

    # Perf
    torch.backends.cuda.matmul.allow_tf32 = True
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass
    torch.backends.cudnn.benchmark = True

    # Args
    model_name = args_dict["model_name"]
    compute_dtype = to_torch_dtype(args_dict["dtype"])
    batch_frames = args_dict["batch_frames"]
    target_fps = args_dict["target_fps"]
    audio_sr = args_dict["audio_sr"]
    n_mels = args_dict["n_mels"]
    n_fft = args_dict["n_fft"]
    win_length = args_dict["win_length"]
    hop_length = args_dict["hop_length"]
    db_cutoff = args_dict["db_cutoff"]

    backbone = None

    while True:
        job = job_queue.get()
        if job is None:
            job_queue.task_done()
            break

        jtype, src, out = job["type"], job["src"], job["out"]
        try:
            if jtype == "video":
                if backbone is None:
                    log("INFO", f"Loading ViT backbone on {device} (dtype={args_dict['dtype']})...")
                    backbone = VisualBackbone(
                        model_name=model_name,
                        compute_dtype=compute_dtype,
                        channels_last=True,
                    ).to(device)
                    backbone.eval()
                    log("INFO", f"Using backbone: {backbone.used_model_name}")
                log("INFO", f"VIDEO {src} -> {out}")
                T, shape = compute_vit_embeddings_for_video(
                    mp4_path=src, out_path=out, backbone=backbone,
                    device=device, batch_frames=batch_frames, target_fps=target_fps
                )
                log("OK", f"wrote {out} shape={shape} T={T}")
                result_queue.put(("video_ok", 1))
            else:
                if not DALI_AVAILABLE:
                    raise RuntimeError(f"NVIDIA DALI is required for audio, but not available: {_DALI_IMPORT_ERROR}")
                log("INFO", f"AUDIO {src} -> {out}")
                T, shape = compute_audio_mels_dali(
                    wav_path=src, out_path=out,
                    device_id=dali_device_id,
                    sr=audio_sr, n_mels=n_mels, n_fft=n_fft,
                    win_length=win_length, hop_length=hop_length, db_cutoff=db_cutoff
                )
                log("OK", f"wrote {out} shape={shape} T={T}")
                result_queue.put(("audio_ok", 1))

        except Exception as e:
            log("ERR", f"FAILED {jtype} {src}: {e}")
            # cleanup temp if any
            tmp = out + ".tmp"
            try:
                if os.path.exists(tmp): os.remove(tmp)
            except Exception: pass
            result_queue.put((f"{jtype}_fail", 1))
        finally:
            job_queue.task_done()

    try:
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception: pass
    log("INFO", "Worker exiting.")


def multi_gpu_dispatch(args):
    from queue import Empty

    logging.basicConfig(
        level=(logging.DEBUG if args.verbose else logging.INFO),
        format="[%(levelname)s] %(message)s",
    )

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

    if args.only in ("both", "audio") and not DALI_AVAILABLE:
        raise RuntimeError(f"NVIDIA DALI is required for audio (torchaudio removed). Import error: {_DALI_IMPORT_ERROR}")

    ctx = mp.get_context("spawn")
    job_queue = ctx.JoinableQueue(maxsize=0)  # unbounded to avoid producer blocking
    result_queue = ctx.Queue()

    args_dict = {
        "model_name": args.model_name,
        "dtype": args.dtype,
        "batch_frames": args.batch_frames,
        "target_fps": args.target_fps,
        "audio_sr": args.audio_sr,
        "n_mels": args.n_mels,
        "n_fft": args.n_fft,
        "win_length": args.win_length,
        "hop_length": args.hop_length,
        "db_cutoff": args.db_cutoff,
    }

    # 1) Start workers first
    procs = []
    for rank in range(num_workers):
        p = ctx.Process(
            target=worker_main,
            args=(rank, gpu_map[rank], job_queue, result_queue, args_dict),
            daemon=True,
        )
        p.start()
        procs.append(p)

    # 2) Enqueue jobs, then sentinels
    try:
        for j in jobs:
            job_queue.put(j)
        for _ in range(num_workers):
            job_queue.put(None)

        # 3) Collect results
        counts = {"video_ok": 0, "video_fail": 0, "audio_ok": 0, "audio_fail": 0}
        remaining = len(jobs)
        while remaining > 0:
            try:
                key, val = result_queue.get(timeout=1.0)
                if key in counts: counts[key] += val
                remaining -= 1
            except Empty:
                if all(not p.is_alive() for p in procs):
                    break

        job_queue.join()
        for p in procs:
            p.join(timeout=5.0)

        logging.info("DONE. Video ok/failed: %d/%d, Audio ok/failed: %d/%d",
                     counts["video_ok"], counts["video_fail"], counts["audio_ok"], counts["audio_fail"])

    except KeyboardInterrupt:
        logging.warning("KeyboardInterrupt received — stopping workers...")
        try:
            for _ in range(num_workers): job_queue.put_nowait(None)
        except Exception: pass
        try: job_queue.join()
        except Exception: pass
        for p in procs:
            if p.is_alive(): p.terminate()
        for p in procs:
            p.join(timeout=2.0)
        logging.info("Shutdown complete.")


# =========================
# Entry
# =========================
def main():
    args = get_args()
    multi_gpu_dispatch(args)


if __name__ == "__main__":
    main()
