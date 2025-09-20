#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
embed.py — Offline precompute of ViT frame embeddings + stereo log-mels (32 Hz), multi-GPU.
TorchCodec-only decoding (no torchaudio/torchvision/decord).

Outputs
  vit_embed/<md5>/<name>.npy  -> float16 [T, D]      (D = backbone hidden, e.g., 768)
  aud_embed/<md5>/<name>.npy  -> float16 [T, n_mels, 2]

Run
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python embed.py --data-root data \
    --gpus all --workers-per-gpu 1 \
    --batch-frames 256 --dtype fp16 --shuffle
"""

from __future__ import annotations
import os, sys, math, argparse, logging, traceback, tempfile, random
from typing import List, Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F

# ----------------------------
# TorchCodec decoders (audio+video)
# ----------------------------
try:
    from torchcodec.decoders import VideoDecoder, AudioDecoder
except Exception as e:
    raise RuntimeError(
        "torchcodec is required. Install a CUDA-enabled build if you want GPU video decoding:\n"
        "  pip install -U torchcodec\n"
        "  # or conda: conda install -c conda-forge torchcodec\n"
        f"Original error: {e}"
    )

# =========================
# CLI
# =========================
def get_args():
    p = argparse.ArgumentParser(description="Precompute ViT + audio (TorchCodec decode, multi-GPU).")
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

    # Audio params (24 kHz, hop=win=750 -> exactly 32 steps/sec)
    p.add_argument("--audio-sr", type=int, default=24000)
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--win-length", type=int, default=750)  # samples
    p.add_argument("--hop-length", type=int, default=750)  # samples
    p.add_argument("--db-cutoff", type=float, default=80.0)

    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--video-threads", type=int, default=1, help="num_ffmpeg_threads for VideoDecoder (per worker).")
    p.add_argument("--video-seek-mode", type=str, default="exact", choices=["exact","approximate"],
                   help="TorchCodec seek_mode (exact = safe; approximate = faster on some files).")
    return p.parse_args()

# =========================
# Small utils
# =========================
def atomic_save_npy(final_path: str, array: np.ndarray):
    """Atomic .npy write using a real temp file; avoids race/partial files."""
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    dirpath = os.path.dirname(final_path) or "."
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".npy.tmp", dir=dirpath, delete=False) as tmpf:
        tmp_path = tmpf.name
        np.save(tmpf, array)
        tmpf.flush()
        os.fsync(tmpf.fileno())
    os.replace(tmp_path, final_path)

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

def build_jobs(root: str, overwrite: bool, only: str, shuffle: bool) -> List[dict]:
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
    if shuffle:
        random.shuffle(jobs)
    else:
        # interleave v/a to keep GPUs busy
        vids = [j for j in jobs if j["type"] == "video"]
        auds = [j for j in jobs if j["type"] == "audio"]
        jobs = []
        i = j = 0
        while i < len(vids) or j < len(auds):
            if i < len(vids): jobs.append(vids[i]); i += 1
            if j < len(auds): jobs.append(auds[j]); j += 1
    return jobs

# =========================
# Robust HF import (avoid local `transformers/` shadowing)
# =========================
def import_hf_safely():
    import importlib, site
    site_paths = []
    try:
        site_paths.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        user_site = site.getusersitepackages()
        if user_site: site_paths.append(user_site)
    except Exception:
        pass
    site_paths = [p for p in site_paths if p and os.path.isdir(p)]
    for p in reversed(site_paths):
        if p not in sys.path:
            sys.path.insert(0, p)
    # If a wrong 'transformers' is already imported, unload it
    if "transformers" in sys.modules:
        mod = sys.modules["transformers"]
        mfile = getattr(mod, "__file__", "") or ""
        if mfile and "site-packages" not in mfile and "dist-packages" not in mfile:
            del sys.modules["transformers"]
    tr = importlib.import_module("transformers")
    AutoModel = getattr(tr, "AutoModel")
    return tr, AutoModel

def try_import_timm():
    try:
        import timm
        return timm
    except Exception:
        return None

# =========================
# ViT backbone (model.py style, manual preprocess)
# =========================
class VisualBackboneModelPy(nn.Module):
    """
    Frozen ViT backbone, pooled features, manual center-crop/resize + ImageNet normalize.
    Tries requested HF model (trust_remote_code=True), falls back to ViT-Base-16; finally TIMM if needed.
    """
    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        compute_dtype: torch.dtype = torch.bfloat16,
        channels_last: bool = True,
    ):
        super().__init__()
        # Import HF safely
        try:
            _, AutoModel = import_hf_safely()
        except Exception as e:
            AutoModel = None
            self._hf_import_error = e
        used = model_name
        backend = "hf"
        bb = None
        if AutoModel is not None:
            try:
                bb = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            except Exception:
                try:
                    used = "google/vit-base-patch16-224-in21k"
                    bb = AutoModel.from_pretrained(used, trust_remote_code=True)
                except Exception:
                    bb = None
        if bb is None:
            timm = try_import_timm()
            if timm is None:
                raise RuntimeError(
                    f"Failed to load ViT from transformers (err={getattr(self,'_hf_import_error',None)}) "
                    "and timm is not installed. Try `pip install timm`."
                )
            used = "timm/vit_base_patch16_224"
            bb = timm.create_model("vit_base_patch16_224", pretrained=True)
            class _Cfg: pass
            bb.config = _Cfg()
            bb.config.hidden_size = getattr(bb, "num_features", 768)
            bb.config.image_size = 224
            bb.reset_classifier(0)
            backend = "timm"

        self.backbone = bb
        self.hidden = int(getattr(self.backbone.config, "hidden_size", 768))
        self.img_size = int(getattr(self.backbone.config, "image_size", 224))
        self.used_model_name = used
        self._backend = backend

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
                if self._backend == "timm":
                    feats = self.backbone.forward_features(x)
                    if isinstance(feats, (list, tuple)): feats = feats[-1]
                    if feats.dim() == 4: feats = feats.mean(dim=(2,3))
                else:
                    out = self.backbone(pixel_values=x)
                    feats = out.pooler_output
            outs.append(feats.cpu())
        return torch.cat(outs, dim=0)

# =========================
# Video via TorchCodec
# =========================
def sample_indices_for_fps(n_src: int, fps_src: float, fps_tgt: float) -> np.ndarray:
    if fps_src <= 0 or n_src <= 0 or fps_tgt <= 0:
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

def compute_vit_embeddings_for_video(
    mp4_path: str,
    out_path: str,
    backbone: VisualBackboneModelPy,
    device: torch.device,
    batch_frames: int,
    target_fps: float,
    num_ffmpeg_threads: int = 1,
    seek_mode: str = "exact",
) -> Tuple[int, Tuple[int, int]]:
    # Decode on assigned GPU to keep frames on-device if CUDA/NVDEC is available
    dec_device = device if device.type == "cuda" else "cpu"
    decoder = VideoDecoder(
        mp4_path,
        num_ffmpeg_threads=num_ffmpeg_threads,
        device=dec_device,
        seek_mode=seek_mode
    )
    n_src = len(decoder)
    fps_src = float(decoder.metadata.average_fps)
    if n_src == 0 or fps_src <= 0:
        raise RuntimeError("No frames or invalid FPS in video.")
    idx = sample_indices_for_fps(n_src, fps_src, target_fps)
    T = int(idx.size)
    if T == 0:
        raise RuntimeError("Resampled to 0 frames.")

    outs = []
    for s in range(0, T, batch_frames):
        e = min(T, s + batch_frames)
        batch_idx = idx[s:e].tolist()
        # get_frames_at returns a FrameBatch with .data [N,3,H,W] uint8
        fb = decoder.get_frames_at(batch_idx)
        frames = fb.data  # torch.uint8 on CPU or CUDA depending on decoder device
        # embed expects NCHW
        feats = backbone.embed_nchw(frames, device=device, chunk=batch_frames)
        outs.append(feats.cpu())
        del fb, frames
        if device.type == "cuda":
            torch.cuda.empty_cache()

    feats_all = torch.cat(outs, dim=0)  # [T, D]
    arr = feats_all.to(torch.float16).numpy()
    atomic_save_npy(out_path, arr)
    return T, arr.shape

# =========================
# Audio via TorchCodec  (32 Hz aligned log-mels, no torchaudio)
# =========================
def _hz_to_mel_htk(freq: torch.Tensor) -> torch.Tensor:
    # HTK mel: mel = 2595 * log10(1 + f/700)
    return 2595.0 * torch.log10(1.0 + freq / 700.0)

def _mel_to_hz_htk(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0**(mel / 2595.0) - 1.0)

def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: Optional[float] = None, device="cpu"):
    if fmax is None: fmax = sr / 2.0
    # FFT bins
    n_freqs = n_fft // 2 + 1
    freqs = torch.linspace(0.0, sr/2.0, n_freqs, dtype=torch.float64, device=device)
    m_min = _hz_to_mel_htk(torch.tensor(fmin, dtype=torch.float64, device=device))
    m_max = _hz_to_mel_htk(torch.tensor(fmax, dtype=torch.float64, device=device))
    m_pts = torch.linspace(m_min, m_max, n_mels + 2, dtype=torch.float64, device=device)
    f_pts = _mel_to_hz_htk(m_pts)

    # Bin numbers
    bins = torch.floor((n_fft + 1) * f_pts / sr).long()
    fb = torch.zeros((n_mels, n_freqs), dtype=torch.float64, device=device)
    for m in range(1, n_mels + 1):
        f_left = bins[m - 1].item()
        f_center = bins[m].item()
        f_right = bins[m + 1].item()
        if f_center == f_left:  f_center += 1
        if f_right <= f_center: f_right = f_center + 1
        # rising
        if f_center > f_left:
            fb[m - 1, f_left:f_center] = torch.linspace(0.0, 1.0, f_center - f_left, device=device)
        # falling
        if f_right > f_center:
            fb[m - 1, f_center:f_right] = torch.linspace(1.0, 0.0, f_right - f_center, device=device)
    # Normalize (Slaney-style area norm)
    enorm = 2.0 / (f_pts[2:n_mels+2] - f_pts[:n_mels])
    fb *= enorm.view(-1, 1)
    return fb.to(dtype=torch.float32)

def amplitude_to_db_power(mel_power: torch.Tensor, top_db: float = 80.0, eps: float = 1e-10) -> torch.Tensor:
    """
    mel_power: [..., n_mels, T] nonnegative
    Return: log10 scaled, max=0 dB per-sample, clamped at -top_db.
    """
    x = torch.clamp(mel_power, min=eps)
    x = 10.0 * torch.log10(x)
    x = x - x.amax(dim=-2, keepdim=True)  # per time step max across mels -> 0 dB
    if top_db is not None:
        x = torch.clamp(x, min=-float(top_db))
    return x

def compute_audio_mels_for_wav(
    wav_path: str,
    out_path: str,
    sr: int = 24000,
    n_mels: int = 128,
    win_length: int = 750,
    hop_length: int = 750,
    db_cutoff: float = 80.0,
    device: torch.device = torch.device("cuda:0"),
) -> Tuple[int, Tuple[int, int, int]]:
    # TorchCodec decode (stereo resampled)
    adec = AudioDecoder(wav_path, sample_rate=sr, num_channels=2)
    samples = adec.get_all_samples()  # AudioSamples with .data [C,N] float in [-1,1]
    wav = samples.data  # [2, N] float32 (usually)
    # Force stereo
    if wav.dim() != 2:
        raise RuntimeError("AudioSamples.data must be [C, N].")
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)

    N = wav.shape[1]
    T = N // hop_length
    if T <= 0:
        raise RuntimeError("Too-short audio.")
    N_use = T * hop_length
    wav = wav[:, :N_use].to(device)

    # STFT with center=False so frames align perfectly to hops.
    n_fft = win_length  # choose n_fft == win_length -> exactly T frames
    window = torch.hann_window(win_length, periodic=True, device=device, dtype=wav.dtype)
    # stft returns [C, F, T] complex
    spec = torch.stft(
        wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, center=False, onesided=True, return_complex=True
    )
    power = (spec.real**2 + spec.imag**2)  # [C, F, T]

    # Mel filterbank
    fb = _mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=sr/2.0, device=device)  # [M,F]
    mel = torch.einsum("mf,cft->mct", fb, power)  # [M, C, T]
    mel = mel.permute(2, 0, 1).contiguous()       # [T, M, C]

    # Log dB scaled with top_db, then cast to float16 on CPU
    mel_db = amplitude_to_db_power(mel, top_db=db_cutoff)  # [T, M, C]
    arr = mel_db.to(torch.float16).cpu().numpy()
    atomic_save_npy(out_path, arr)
    return T, arr.shape

# =========================
# Worker
# =========================
def worker_main(rank: int, gpu_id: Optional[int], job_queue: mp.JoinableQueue, result_queue: mp.Queue, args_dict: dict):
    def log(level, msg):
        print(f"[W{rank}|GPU{gpu_id if gpu_id is not None else 'cpu'}][{level}] {msg}", flush=True)

    # Device
    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

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
    win_length = args_dict["win_length"]
    hop_length = args_dict["hop_length"]
    db_cutoff = args_dict["db_cutoff"]
    v_threads = args_dict["video_threads"]
    v_seek = args_dict["video_seek_mode"]

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
                    backbone = VisualBackboneModelPy(
                        model_name=model_name,
                        compute_dtype=compute_dtype,
                        channels_last=True,
                    ).to(device)
                    backbone.eval()
                    log("INFO", f"Using backbone: {backbone.used_model_name}")
                log("INFO", f"VIDEO {src} -> {out}")
                T, shape = compute_vit_embeddings_for_video(
                    mp4_path=src, out_path=out, backbone=backbone,
                    device=device, batch_frames=batch_frames, target_fps=target_fps,
                    num_ffmpeg_threads=v_threads, seek_mode=v_seek
                )
                log("OK", f"wrote {out} shape={shape} T={T}")
                result_queue.put(("video_ok", 1))
            else:
                log("INFO", f"AUDIO {src} -> {out}")
                T, shape = compute_audio_mels_for_wav(
                    wav_path=src, out_path=out,
                    sr=audio_sr, n_mels=n_mels,
                    win_length=win_length, hop_length=hop_length, db_cutoff=db_cutoff,
                    device=device
                )
                log("OK", f"wrote {out} shape={shape} T={T}")
                result_queue.put(("audio_ok", 1))
        except Exception as e:
            log("ERR", f"FAILED {jtype} {src}: {e}")
            # best-effort cleanup
            try:
                if os.path.exists(out + ".tmp"): os.remove(out + ".tmp")
            except Exception:
                pass
            result_queue.put((f"{jtype}_fail", 1))
        finally:
            job_queue.task_done()

    try:
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception: pass
    log("INFO", "Worker exiting.")

# =========================
# Dispatch
# =========================
def multi_gpu_dispatch(args):
    from queue import Empty

    logging.basicConfig(
        level=(logging.DEBUG if args.verbose else logging.INFO),
        format="[%(levelname)s] %(message)s",
    )

    jobs = build_jobs(args.data_root, args.overwrite, args.only, shuffle=args.shuffle)
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
        logging.info("Using GPUs: %s (%%d workers total)", ",".join(map(str, gpu_ids)), num_workers)

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
        "win_length": args.win_length,
        "hop_length": args.hop_length,
        "db_cutoff": args.db_cutoff,
        "video_threads": args.video_threads,
        "video_seek_mode": args.video_seek_mode,
    }

    # Start workers first
    procs = []
    for rank in range(num_workers):
        p = ctx.Process(
            target=worker_main,
            args=(rank, gpu_map[rank], job_queue, result_queue, args_dict),
            daemon=True,
        )
        p.start()
        procs.append(p)

    # Enqueue jobs, then sentinels
    try:
        for j in jobs:
            job_queue.put(j)
        for _ in range(num_workers):
            job_queue.put(None)

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
