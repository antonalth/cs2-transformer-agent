#!/usr/bin/env python3
"""
embed.py — Offline precompute of visual (ViT/DINOv3) embeddings and stereo log-mel features
for CS2 recordings.

- Scans:   <data_root>/recordings/<md5>/<name>.mp4 and .wav
- Writes:  <data_root>/vit_embed/<md5>/<name>.npy      # [T, D]
           <data_root>/aud_embed/<md5>/<name>.npy      # [T, n_mels, 2]
- Time base: exactly 32 fps for both modalities. Audio uses sr=24000 with hop=750 (24000/32).

Key properties:
- Atomic writes (.tmp + os.replace) to avoid partial files.
- Idempotent resume (skip existing unless --overwrite).
- GPU-accelerated where possible (HF DINOv3 on CUDA; torchaudio MelSpectrogram on CUDA).
- No imports from your repo: this file is self-contained. Concepts mirror train2.py (DALI params,
  audio mel config, atomic writing). DALI is optional; we default to decord/pyav/cv2 for video decode.

Usage:
  python embed.py --data-root /mnt/trainingdata/sampledata --device cuda:0 --dtype fp16

"""
import argparse
import contextlib
import dataclasses
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# Torch & optional libs
import torch
import torch.nn as nn

# Optional decoders: prefer decord, then PyAV, then OpenCV
_DECORD = None
try:
    import decord
    _DECORD = decord
except Exception:
    _DECORD = None

_AV = None
try:
    import av
    _AV = av
except Exception:
    _AV = None

_CV2 = None
try:
    import cv2
    _CV2 = cv2
except Exception:
    _CV2 = None

# Optional torchaudio for audio features
_TORCHAUDIO = None
try:
    import torchaudio
    _TORCHAUDIO = torchaudio
except Exception:
    _TORCHAUDIO = None

# -------------------------------
# Simple timer (from train2.py)
# -------------------------------
class Timer:
    """Simple timer context for instrumentation."""
    def __init__(self, name: str): self.name = name
    def __enter__(self): self.t0 = time.time(); return self
    def __exit__(self, *args): self.dt = time.time() - self.t0

# -------------------------------
# Atomic save helper
# -------------------------------
def atomic_save_npy(array: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        np.save(tmp, array)
        os.replace(tmp, out_path)
    except Exception:
        # clean up tmp if replace failed
        with contextlib.suppress(Exception):
            if tmp.exists(): tmp.unlink()
        raise

# -------------------------------
# Video readers (32 fps sampling)
# -------------------------------
def _video_info(path: Path) -> Tuple[int, float]:
    """Return (num_frames, fps) using any available backend."""
    if _DECORD is not None:
        vr = _DECORD.VideoReader(str(path))
        try:
            fps = float(vr.get_avg_fps())
        except Exception:
            # Some containers may not report fps; best-effort from frame count and stream duration if available
            fps = float(vr.get_avg_fps())  # will throw again if not available
        return int(len(vr)), fps
    if _AV is not None:
        with _AV.open(str(path)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else float(stream.base_rate)
            # Count frames (may be slow but reliable)
            num = int(stream.frames) if stream.frames else None
            if num is None or num <= 0:
                num = 0
                for frame in container.decode(stream):
                    num += 1
            return num, fps
    if _CV2 is not None:
        cap = _CV2.VideoCapture(str(path))
        num = int(cap.get(_CV2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(cap.get(_CV2.CAP_PROP_FPS)) or 30.0
        cap.release()
        return num, fps
    raise RuntimeError("No video backend available (decord, PyAV, or OpenCV required).")

def _read_frames_32fps(path: Path, target_fps: float = 32.0, chunk: int = 512) -> Iterable[np.ndarray]:
    """Yield batches of frames as numpy uint8 arrays of shape [B, H, W, 3] (RGB), sampled at ~32 fps.
    Uses decord if available; else PyAV; else OpenCV. The final batch may be smaller than 'chunk'.
    """
    if _DECORD is not None:
        vr = _DECORD.VideoReader(str(path))
        orig_fps = float(vr.get_avg_fps())
        N = int(len(vr))
        if N == 0:
            return
        duration = N / orig_fps
        T = int(math.floor(duration * target_fps + 1e-6))
        idx = (np.floor(np.arange(T) * (orig_fps / target_fps))).astype(np.int64)
        idx = np.clip(idx, 0, N - 1)
        # Iterate in chunks
        for i in range(0, len(idx), chunk):
            sel = idx[i:i+chunk]
            batch = vr.get_batch(sel).asnumpy()  # [B, H, W, 3], RGB
            yield batch.astype(np.uint8)
        return
    if _AV is not None:
        # Decode all frames then resample indices
        with _AV.open(str(path)) as container:
            stream = container.streams.video[0]
            frames = []
            for frame in container.decode(stream):
                arr = frame.to_ndarray(format='rgb24')  # H W 3
                frames.append(arr)
            if not frames:
                return
            frames = np.stack(frames, axis=0)
            N = frames.shape[0]
            orig_fps = float(stream.average_rate) if stream.average_rate else 30.0
            duration = N / orig_fps
            T = int(math.floor(duration * target_fps + 1e-6))
            idx = (np.floor(np.arange(T) * (orig_fps / target_fps))).astype(np.int64)
            idx = np.clip(idx, 0, N - 1)
            for i in range(0, len(idx), chunk):
                sel = idx[i:i+chunk]
                yield frames[sel]
        return
    if _CV2 is not None:
        cap = _CV2.VideoCapture(str(path))
        N = int(cap.get(_CV2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(cap.get(_CV2.CAP_PROP_FPS)) or 30.0
        if N <= 0:
            cap.release(); return
        duration = N / fps
        T = int(math.floor(duration * target_fps + 1e-6))
        idx = (np.floor(np.arange(T) * (fps / target_fps))).astype(np.int64)
        idx = np.clip(idx, 0, N - 1)
        cur = -1
        buf = []
        # Sequentially read and collect required frames
        for j in range(N):
            ok, frame = cap.read()
            if not ok: break
            cur += 1
            # BGR->RGB
            frame = frame[:, :, ::-1]
            # Append duplicates for any index equal to current frame
            while len(idx) and idx[0] == cur:
                buf.append(frame.copy())
                idx = idx[1:]
                if len(buf) == chunk:
                    yield np.stack(buf, axis=0); buf = []
            if len(idx) == 0:
                break
        if buf:
            yield np.stack(buf, axis=0)
        cap.release()
        return
    raise RuntimeError("No video backend available (decord, PyAV, or OpenCV required).")

# -------------------------------
# Visual encoder (borrowed from model.py: DINOv3VisualEncoder)
# -------------------------------
class DINOv3VisualEncoder(nn.Module):
    """
    Hugging Face DINOv3 encoder with chunked, just-in-time normalization and
    chunked backbone forward to reduce peak VRAM.

    Contract:
      - Input:  [B, T, P, 3, H, W] uint8 in [0,255] or float in [0,1]
      - Output: [B, T, P, d_model]
    """
    def __init__(self, cfg):
        super().__init__()
        self.d_model = int(getattr(cfg, "d_model", 2048))
        self.model_name = str(getattr(cfg, "hf_model_name", "facebook/dinov3-vitb16-pretrain-lvd1689m"))
        self.use_processor = bool(getattr(cfg, "hf_use_processor", True))
        self.channels_last = bool(getattr(cfg, "hf_channels_last", True))
        self.norm_chunk = int(getattr(cfg, "hf_norm_chunk", 64))
        self.forward_chunk = int(getattr(cfg, "hf_forward_chunk", self.norm_chunk))
        compute_dtype = str(getattr(cfg, "compute_dtype", "fp16"))
        self.compute_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[compute_dtype]

        # Processor (for resize & normalization)
        self.processor = None
        if self.use_processor:
            try:
                from transformers import AutoImageProcessor as _AutoImageProcessor
                self.processor = _AutoImageProcessor.from_pretrained(self.model_name)
            except Exception as e:
                raise RuntimeError(f"Failed to load AutoImageProcessor for '{self.model_name}': {e}")

        # Backbone
        try:
            from transformers import AutoModel as _AutoModel
            self.backbone = _AutoModel.from_pretrained(self.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load DINOv3 model '{self.model_name}': {e}")

        # Project hidden size to d_model
        self.hidden = int(getattr(self.backbone.config, "hidden_size", 768))
        self.proj = nn.Linear(self.hidden, self.d_model)

    def _normalize_chunk(self, x: torch.Tensor, from_uint8: bool) -> torch.Tensor:
        """x: [N, 3, H, W] -> normalized pixel_values for backbone."""
        if self.processor is None:
            # Manual: convert to float in [0,1], then normalize with ImageNet means/stds
            if from_uint8:
                x = x.to(torch.float32) / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            x = (x - mean) / std
            return x
        else:
            # Use HF processor to resize+normalize on CPU, then move back
            xs = x.detach().cpu()
            imgs = [xs[i].permute(1,2,0).numpy() for i in range(xs.shape[0])]  # HWC in [0,1] or [0,255]
            inputs = self.processor(images=imgs, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(x.device)
            return pixel_values

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, T, P, 3, H, W] uint8 or float -> [B, T, P, d_model]"""
        assert images.ndim == 6, f"Expected 6D input [B,T,P,3,H,W], got {images.shape}"
        device = images.device
        B, T, P, C, H, W = images.shape
        from_uint8 = images.dtype == torch.uint8

        if self.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        chunks = []
        t_chunk = int(self.forward_chunk)
        for t0 in range(0, T, t_chunk):
            t1 = min(T, t0 + t_chunk)
            img_chunk = images[:, t0:t1]  # [B, t, P, 3, H, W]
            N_chunk = B * (t1 - t0) * P
            x = img_chunk.reshape(N_chunk, C, H, W)

            # Normalize
            x = self._normalize_chunk(x, from_uint8=from_uint8)

            # Backbone forward with autocast when on CUDA (unless compute_dtype is fp32)
            use_amp = (device.type == "cuda" and self.compute_dtype != torch.float32)
            with torch.autocast(device_type="cuda", dtype=self.compute_dtype, enabled=use_amp):
                out = self.backbone(pixel_values=x)
                feats = out.last_hidden_state[:, 0]  # CLS token [N_chunk, hidden]

            # Project
            vis = self.proj(feats).view(B, (t1 - t0), P, self.d_model)
            chunks.append(vis)

            # free ASAP
            del x, img_chunk, out, feats, vis

        return torch.cat(chunks, dim=1)  # [B, T, P, d_model]

# -------------------------------
# Audio mel @ 32 fps (sr=24000)
# -------------------------------
def compute_logmel_32fps_stereo(wav_path: Path, device: torch.device,
                                n_mels: int = 128, sample_rate: int = 24000,
                                n_fft: int = 1024, win_length: int = 750, hop_length: int = 750,
                                dtype: str = "fp16") -> np.ndarray:
    if _TORCHAUDIO is None:
        raise RuntimeError("torchaudio is required for audio feature extraction.")
    wav, sr = _TORCHAUDIO.load(str(wav_path))  # [C, N]
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    if sr != sample_rate:
        wav = _TORCHAUDIO.functional.resample(wav, sr, sample_rate)
    # Trim to full hops
    T = wav.shape[1] // hop_length
    N = T * hop_length
    wav = wav[:, :N]
    if T == 0:
        raise RuntimeError("Audio too short after resampling to 24kHz.")
    # Mel on GPU if available
    mel = _TORCHAUDIO.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
        f_min=20.0, f_max=sample_rate/2.0, n_mels=n_mels, center=False, power=2.0
    ).to(device)
    x = wav.to(device)
    S = mel(x)  # [2, n_mels, T]
    S = torch.log1p(S)  # log-mel
    # [T, n_mels, 2] for easy per-frame indexing
    out = S.permute(2, 1, 0).contiguous()
    if dtype == "fp16":
        out = out.half()
    return out.cpu().numpy()

# -------------------------------
# Video embedding
# -------------------------------
@dataclass
class VisionCfg:
    d_model: int = 2048
    hf_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    hf_use_processor: bool = True
    hf_channels_last: bool = True
    hf_norm_chunk: int = 64
    hf_forward_chunk: int = 64
    compute_dtype: str = "fp16"

def compute_vit_embeddings_32fps(mp4_path: Path, device: torch.device, dtype: str,
                                 batch_chunk: int, encoder: DINOv3VisualEncoder) -> np.ndarray:
    """Return [T, D] numpy array."""
    frames_total = 0
    feats: List[np.ndarray] = []
    # Iterate batches of [B, H, W, 3]
    for batch in _read_frames_32fps(mp4_path, target_fps=32.0, chunk=batch_chunk):
        # Convert to torch [B, 3, H, W], then to [1, B, 1, 3, H, W] treating B as time
        b = torch.from_numpy(batch).to(device, non_blocking=True)  # [B, H, W, 3]
        b = b.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
        # Make [1, T, 1, 3, H, W]
        b = b.unsqueeze(0).unsqueeze(2)  # [1, B, 1, 3, H, W]
        with torch.no_grad():
            out = encoder(b)  # [1, B, 1, D]
        arr = out.squeeze(0).squeeze(1).contiguous()  # [B, D]
        if dtype == "fp16":
            arr = arr.half()
        feats.append(arr.detach().cpu().numpy())
        frames_total += arr.shape[0]
        # free
        del b, out, arr
        torch.cuda.empty_cache() if device.type == "cuda" else None
    if frames_total == 0:
        raise RuntimeError(f"No frames decoded from {mp4_path}.")
    return np.concatenate(feats, axis=0)  # [T, D]

# -------------------------------
# Job enumeration
# -------------------------------
def enumerate_jobs(root: Path, overwrite: bool, only: Optional[str]) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    vids, auds = [], []
    rec_root = root / "recordings"
    for md5_dir in sorted([p for p in rec_root.iterdir() if p.is_dir()]):
        for f in sorted(md5_dir.iterdir()):
            if f.suffix.lower() == ".mp4":
                out = root / "vit_embed" / md5_dir.name / (f.stem + ".npy")
                if overwrite or (not out.exists()):
                    vids.append((f, out))
            elif f.suffix.lower() == ".wav":
                out = root / "aud_embed" / md5_dir.name / (f.stem + ".npy")
                if overwrite or (not out.exists()):
                    auds.append((f, out))
    if only == "video":
        auds = []
    elif only == "audio":
        vids = []
    return vids, auds

# -------------------------------
# Main
# -------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, choices=["fp16", "fp32", "bf16"], default="fp16")
    p.add_argument("--batch-frames", type=int, default=256, help="Frames per forward chunk.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--only", choices=["video", "audio"], default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    root = Path(args.data_root)
    if not root.exists():
        logging.error("Data root %s does not exist.", root); sys.exit(1)

    device = torch.device(args.device)
    # dtype note: embeddings are saved as fp16 by default
    if args.dtype == "fp16" and device.type == "cpu":
        logging.warning("fp16 on CPU not supported; saving as fp32.")
        save_dtype = "fp32"
    else:
        save_dtype = args.dtype

    vids, auds = enumerate_jobs(root, args.overwrite, args.only)
    logging.info("Discovered %d video and %d audio jobs (overwrite=%s).", len(vids), len(auds), args.overwrite)

    # Prepare visual encoder once
    encoder = None
    if len(vids) > 0:
        vcfg = VisionCfg(d_model=2048, compute_dtype=args.dtype)
        encoder = DINOv3VisualEncoder(vcfg).to(device).eval()
        logging.info("Loaded DINOv3 visual encoder '%s' -> d_model=%d",
                     vcfg.hf_model_name, vcfg.d_model)

    # Process video jobs
    ok_v, fail_v = 0, 0
    for src, dst in vids:
        try:
            logging.info("[video] %s -> %s", src.relative_to(root), dst.relative_to(root))
            with Timer("video") as t:
                arr = compute_vit_embeddings_32fps(src, device=device, dtype=save_dtype,
                                                   batch_chunk=args.batch_frames, encoder=encoder)
            logging.info("[video] T=%d, D=%d, elapsed=%.2fs", arr.shape[0], arr.shape[1], t.dt)
            # Ensure dtype
            if save_dtype == "fp16":
                arr = arr.astype(np.float16, copy=False)
            atomic_save_npy(arr, dst)
            ok_v += 1
        except Exception as e:
            logging.exception("Failed video %s: %s", src, e)
            fail_v += 1

    # Process audio jobs
    ok_a, fail_a = 0, 0
    for src, dst in auds:
        try:
            logging.info("[audio] %s -> %s", src.relative_to(root), dst.relative_to(root))
            with Timer("audio") as t:
                arr = compute_logmel_32fps_stereo(src, device=device, dtype=save_dtype)
            logging.info("[audio] T=%d, mels=%d, elapsed=%.2fs", arr.shape[0], arr.shape[1], t.dt)
            if save_dtype == "fp16":
                arr = arr.astype(np.float16, copy=False)
            atomic_save_npy(arr, dst)
            ok_a += 1
        except Exception as e:
            logging.exception("Failed audio %s: %s", src, e)
            fail_a += 1

    logging.info("DONE. Video: %d ok, %d failed. Audio: %d ok, %d failed.", ok_v, fail_v, ok_a, fail_a)

if __name__ == "__main__":
    main()
