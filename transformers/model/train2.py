#!/usr/bin/env python3
"""
train2.py — Data loading and Loss Calculation (Steps 1–11)

MODIFIED: This version includes a dual-path data pipeline.
It can be run in two modes, controlled by the `--use-precomputed-embeddings` flag:

1. On-the-fly (default):
   - DALI reads raw .mp4 and .wav files.
   - Video frames are decoded and audio is converted to mel spectrograms on the fly.

2. Pre-computed (--use-precomputed-embeddings):
   - `embed.py` must be run first to generate .npy feature files.
   - DALI reads these .npy files (one for ViT features, one for spectrograms).
   - DALI slices the required frame window from the full-round tensor on the GPU.
   - This significantly reduces CPU/GPU load during training, allowing for
     larger batch sizes or faster iteration.

The BatchAssembler and downstream model logic are designed to handle both
input types transparently.
"""
from __future__ import annotations

import os
import re
import json
import math
import time
import random
import logging
import argparse
import contextlib
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

import lmdb  # pip install lmdb
import msgpack  # pip install msgpack
import numpy as np
import msgpack_numpy as mpnp  # pip install msgpack-numpy

# Torch is used for device placement and to hand CUDA tensors to the model later
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity # <-- PROFILER: Added imports

# --- Try to import DALI; provide a friendly message if missing ---
try:
    from nvidia.dali import fn, types, pipeline_def
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from nvidia.dali.plugin.base_iterator import LastBatchPolicy
    DALI_AVAILABLE = True
except Exception as _e:
    DALI_AVAILABLE = False
    _DALI_IMPORT_ERROR = _e


# -------------------------------
# Constants & utilities
# -------------------------------

TICK_RATE = 64     # demo ticks per second
FPS = 32           # video frames per second
TICKS_PER_FRAME = TICK_RATE // FPS  # == 2
LABEL_SCALE = 1_000_000 # pack sample_id + start_f into one int; ensure start_f << LABEL_SCALE


TICK_RE = re.compile(r"_([0-9]+)_([0-9]+)\.(mp4|wav)$")


def ticks_from_filename(path: str) -> tuple[int, int] | None:
    """Extracts start and end ticks from a media filename."""
    m = TICK_RE.search(os.path.basename(path))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def ticks_to_frames(start_tick: int, end_tick: int) -> int:
    """Inclusive frame count derived from ticks."""
    if end_tick < start_tick:
        return 0
    return ((end_tick - start_tick) // TICKS_PER_FRAME) + 1


def clamp_window_to_pov(req_start_f: int, T_frames: int, pov_start_tick: int, pov_end_tick: int) -> tuple[int, int]:
    """
    Clamps the requested frame window to the actual bounds of a specific POV video.
    Returns a (start, end_exclusive) tuple for the filelist, as DALI's end frame is exclusive.
    """
    pov_frame_count = ticks_to_frames(pov_start_tick, pov_end_tick)
    if pov_frame_count <= 0:
        return -1, -1

    pov_last_f = pov_frame_count - 1
    start_f = min(req_start_f, pov_last_f)
    end_f_exclusive = min(start_f + T_frames, pov_frame_count)

    return start_f, end_f_exclusive

# =============================================================================
# STEP 10 Utilities: Heatmap Target Generation
# =============================================================================

@dataclass
class MapBoundaries:
    """
    Defines the game world coordinate boundaries.
    NOTE: These values are placeholders and MUST be determined by analyzing
    the actual range of player coordinates in your dataset to ensure all
    positions are contained.
    """
    # Shape: (x, y, z)
    WORLD_MIN: tuple[float, float, float] = (-3000.0, -3500.0, -500.0)
    WORLD_MAX: tuple[float, float, float] = (3500.0, 2500.0, 1000.0)


class CoordinateMapper:
    """Utility to map continuous world coordinates to a discrete grid."""
    def __init__(self, grid_dims: tuple[int, int, int] = (64, 64, 8)): # (X, Y, Z)
        self.grid_dims = torch.tensor(grid_dims, dtype=torch.long)
        self.world_min = torch.tensor(MapBoundaries.WORLD_MIN, dtype=torch.float32)
        self.world_max = torch.tensor(MapBoundaries.WORLD_MAX, dtype=torch.float32)
        self.world_range = self.world_max - self.world_min

    def discretize_world_to_grid(self, world_coords: torch.Tensor) -> torch.Tensor:
        """Converts a batch of continuous world coords to discrete grid indices."""
        # Ensure the mapper's internal tensors are on the same device as the input.
        device = world_coords.device
        self.world_min = self.world_min.to(device)
        self.world_max = self.world_max.to(device) # This line was missing
        self.world_range = self.world_range.to(device)
        self.grid_dims = self.grid_dims.to(device)

        coords = torch.max(torch.min(world_coords, self.world_max), self.world_min)
        normalized_coords = (coords - self.world_min) / self.world_range
        grid_indices = (normalized_coords * (self.grid_dims - 1e-6)).long()
        return grid_indices

def create_gaussian_heatmap_target(
    grid_indices: torch.Tensor,
    grid_dims: tuple[int, int, int],
    sigma: float = 1.5
) -> torch.Tensor:
    """Creates a batch of smooth target heatmaps using a Gaussian kernel. (Memory-Efficient Version)"""
    X, Y, Z = grid_dims
    device = grid_indices.device
    num_targets = grid_indices.shape[0]
    
    # Create the coordinate grid once
    zz, yy, xx = torch.meshgrid(
        torch.arange(Z, device=device),
        torch.arange(Y, device=device),
        torch.arange(X, device=device),
        indexing='ij'
    )
    grid_coords = torch.stack((xx, yy, zz), dim=-1).float() # Shape: [Z, Y, X, 3]
    
    heatmaps = []
    # Loop over each target instead of broadcasting all at once
    for i in range(num_targets):
        target_idx = grid_indices[i].float() # Shape: [3]
        
        # Broadcasting here is cheap, as the temp tensor is only [Z, Y, X, 3]
        distance_sq = torch.sum((grid_coords - target_idx)**2, dim=-1) # Shape: [Z, Y, X]
        heatmap = torch.exp(-distance_sq / (2 * sigma**2))
        heatmaps.append(heatmap)
    
    if not heatmaps:
        # Handle the edge case where there are no targets
        return torch.empty(0, Z, Y, X, device=device)
        
    return torch.stack(heatmaps, dim=0) # Shape: [num_targets, Z, Y, X]

# =============================================================================

@dataclass
class TeamRound:
    """Canonical record for a single team-round as discovered from `<demoname>_INFO`."""
    demoname: str
    lmdb_path: str
    round_num: int
    team: str  # "T" or "CT"
    start_tick: int
    end_tick: int
    pov_videos: List[str]
    pov_audio: List[str]
    fps: int = FPS
    tick_rate: int = TICK_RATE

    @property
    def parity(self) -> int:
        return self.start_tick % 2

    @property
    def frame_count(self) -> int:
        return ((self.end_tick - self.start_tick) // TICKS_PER_FRAME) + 1


@dataclass
class SampleRecord:
    """A sampled fixed-length window inside a TeamRound for a given epoch."""
    sample_id: int
    demoname: str
    lmdb_path: str
    round_num: int
    team: str
    pov_videos: List[str]
    pov_audio: List[str]
    start_f: int
    start_tick_win: int
    T_frames: int
    parity: int


# -------------------------------
# Step 1: Manifest + `_INFO` reader
# -------------------------------

class Manifest:
    """Loads `manifest.json` and enumerates games per split."""
    def __init__(self, data_root: str, manifest_path: str):
        self.data_root = os.path.abspath(data_root)
        self.manifest_path = manifest_path
        with open(manifest_path, "r", encoding="utf-8") as f:
            self._data = json.load(f)

    def get_games(self, split: str) -> List[Tuple[str, str]]:
        """Return list of (demoname, lmdb_path) for the given split."""
        entries = self._data.get(split, [])
        out: List[Tuple[str, str]] = []
        for e in entries:
            if isinstance(e, str):
                demoname = e
                lmdb_path = os.path.join(self.data_root, "lmdb", f"{demoname}.lmdb")
            elif isinstance(e, dict):
                demoname = e.get("demoname") or e.get("name")
                if demoname is None:
                    raise ValueError(f"Manifest entry missing 'demoname': {e}")
                lmdb_path = e.get("lmdb_path") or os.path.join(self.data_root, "lmdb", f"{demoname}.lmdb")
            else:
                raise ValueError(f"Unsupported manifest entry: {e}")
            out.append((demoname, os.path.abspath(lmdb_path)))
        return out


class LmdbStore:
    """Opens and caches LMDB environments. Also reads `<demoname>_INFO` once."""
    def __init__(self, max_readers: int = 512, map_size: int = 0, readahead: bool = True):
        self._envs: Dict[str, lmdb.Environment] = {}
        self._info_cache: Dict[str, Dict[str, Any]] = {}
        self._max_readers = max_readers
        self._map_size = map_size
        self._readahead = readahead

    def open(self, lmdb_path: str) -> lmdb.Environment:
        env = self._envs.get(lmdb_path)
        if env is None:
            env = lmdb.open(
                lmdb_path, readonly=True, lock=False, max_readers=self._max_readers,
                map_size=self._map_size or 1 << 30, readahead=self._readahead,
            )
            self._envs[lmdb_path] = env
        return env

    def read_info(self, demoname: str, lmdb_path: str) -> Dict[str, Any]:
        """Read and cache `<demoname>_INFO` from the LMDB."""
        key = f"{demoname}_INFO".encode("utf-8")
        cache_key = (demoname, lmdb_path)
        if cache_key in self._info_cache:
            return self._info_cache[cache_key]
        env = self.open(lmdb_path)
        with env.begin(write=False) as txn:
            blob = txn.get(key)
            if blob is None:
                raise FileNotFoundError(f"Missing _INFO entry for {demoname} in {lmdb_path}")
            info = json.loads(blob.decode("utf-8"))
        self._info_cache[cache_key] = info
        return info


def build_team_rounds(data_root: str, games: List[Tuple[str, str]], store: LmdbStore) -> List[TeamRound]:
    """Enumerate all TeamRounds for the split by reading each game's `<demoname>_INFO`."""
    team_rounds: List[TeamRound] = []
    for demoname, lmdb_path in games:
        info = store.read_info(demoname, lmdb_path)
        for r in info["rounds"]:
            pov_videos = r.get("pov_videos", [])
            pov_audio = r.get("pov_audio", [])
            if len(pov_videos) != 5: continue
            if len(pov_audio) != 5: continue

            def _resolve_media_path(p: str) -> str:
                return os.path.abspath(os.path.join(data_root, "recordings", demoname, p))
            
            pov_videos_abs = [_resolve_media_path(pv) for pv in pov_videos]
            pov_audio_abs = [_resolve_media_path(pa) for pa in pov_audio]
            all_media_paths_exist = True
            for p in pov_videos_abs + pov_audio_abs:
                if not os.path.exists(p):
                    logging.warning(f"Skipping round {demoname}/{r['round_num']} due to missing media file: {p}")
                    all_media_paths_exist = False
                    break
            if not all_media_paths_exist:
                continue

            tr = TeamRound(
                demoname=demoname, lmdb_path=os.path.abspath(lmdb_path),
                round_num=int(r["round_num"]), team=str(r["team"]).upper(),
                start_tick=int(r["start_tick"]), end_tick=int(r["end_tick"]),
                pov_videos=pov_videos_abs, pov_audio=pov_audio_abs,
            )
            team_rounds.append(tr)
    logging.info("Discovered %d team-rounds across %d games.", len(team_rounds), len(games))
    return team_rounds


# -------------------------------
# Step 3: EpochIndex (window sampler)
# -------------------------------

class EpochIndex:
    """Samples a fixed-length frame window inside each TeamRound for a given epoch."""
    def __init__(self, T_frames: int, seed: int):
        self.T_frames = T_frames
        self.seed = seed
        self.records: List[SampleRecord] = []
        self.id_to_sample: Dict[int, SampleRecord] = {}

    def build(self, team_rounds: List[TeamRound], epoch: int, allow_padding: bool = True) -> Tuple[List[SampleRecord], Dict[int, SampleRecord]]:
        rnd = random.Random(self.seed + epoch)
        self.records.clear(); self.id_to_sample.clear()
        sid = 0
        for tr in team_rounds:
            start_f = rnd.randint(0, tr.frame_count - self.T_frames) if tr.frame_count >= self.T_frames else 0
            rec = SampleRecord(
                sample_id=sid, demoname=tr.demoname, lmdb_path=tr.lmdb_path,
                round_num=tr.round_num, team=tr.team, pov_videos=tr.pov_videos,
                pov_audio=tr.pov_audio, start_f=start_f,
                start_tick_win=tr.start_tick + TICKS_PER_FRAME * start_f,
                T_frames=self.T_frames, parity=tr.parity,
            )
            self.records.append(rec); self.id_to_sample[sid] = rec; sid += 1
        logging.info("EpochIndex built: %d samples (T=%d)", len(self.records), self.T_frames)
        return self.records, self.id_to_sample


# -------------------------------
# Step 4: Filelist writer (10 aligned lists)
# -------------------------------

class FilelistWriter:



    """Writes ten aligned DALI filelists, supporting both raw media and pre-computed .npy files."""
    def __init__(self, out_dir: str, use_precomputed: bool = False, data_root: str = ""):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.use_precomputed = use_precomputed
        if use_precomputed and not data_root:
            raise ValueError("data_root must be provided when use_precomputed is True.")
        self.data_root = data_root

    def write(self, records: List[SampleRecord]) -> Tuple[List[str], List[str]]:
        vid_paths = [os.path.join(self.out_dir, f"pov{k}_video.txt") for k in range(5)]
        aud_paths = [os.path.join(self.out_dir, f"pov{k}_audio.txt") for k in range(5)]

        def _embed_path(kind: str, demo: str, base: str, data_root: str) -> Path:
            """
            kind: 'vit' or 'aud'
            demo: e.g. '8dda4a92df5d014c57a64f6eba2937ff'
            base: basename without extension used by your code, e.g. '01_CT_--Mokuj1n_4353_7897'
            """
            root = Path(data_root).resolve()
            sub = "vit_embed" if kind == "vit" else "aud_embed"
            return root / sub / demo / f"{base}.npy"

        def _assert_exists(p: Path):
            if not p.is_file():
                raise FileNotFoundError(f"Missing precomputed embedding: {p}")
        
        with contextlib.ExitStack() as stack:
            files = [stack.enter_context(open(p, "w", encoding="utf-8")) for p in vid_paths + aud_paths]
            vid_files, aud_files = files[:5], files[5:]
            for rec in records:
                if self.use_precomputed:
                    # PATH 1: Pre-computed .npy files
                    # Label encodes sample_id and the start frame for slicing in DALI
                    packed_label = rec.sample_id * LABEL_SCALE + rec.start_f
                    for k in range(5):
                        # FIX: Use the specific filename stem for each POV's video and audio
                        vid_base = os.path.splitext(os.path.basename(rec.pov_videos[k]))[0]
                        aud_base = os.path.splitext(os.path.basename(rec.pov_audio[k]))[0]

                        vid_npy_path = _embed_path("vit", rec.demoname, vid_base, self.data_root)
                        aud_npy_path = _embed_path("aud", rec.demoname, aud_base, self.data_root)

                        _assert_exists(vid_npy_path)
                        _assert_exists(aud_npy_path)

                        vid_files[k].write(f"{vid_npy_path} {packed_label}\n")
                        aud_files[k].write(f"{aud_npy_path} {packed_label}\n")
                else:
                    # PATH 2: Original logic for raw media files
                    for k in range(5):
                        pov_path = rec.pov_videos[k]
                        ticks = ticks_from_filename(pov_path)
                        if not ticks: raise ValueError(f"Could not parse ticks from: {pov_path}")
                        pov_start_tick, pov_end_tick = ticks
                        start, end_exclusive = clamp_window_to_pov(rec.start_f, rec.T_frames, pov_start_tick, pov_end_tick)
                        # FIX: DALI's file_list_frame_num=True expects: filename label start_frame frame_count
                        frame_num = max(0, end_exclusive - start)
                        vid_files[k].write(f"{pov_path} {rec.sample_id} {start} {frame_num}\n")
                        
                        packed_audio_label = rec.sample_id * LABEL_SCALE + start
                        aud_files[k].write(f"{rec.pov_audio[k]} {packed_audio_label}\n")

        logging.info("Wrote DALI filelists to %s (N=%d, Precomputed=%s)", self.out_dir, len(records), self.use_precomputed)
        return vid_paths, aud_paths

# -------------------------------
# Step 5: DALI pipeline (video + audio)
# -------------------------------

@dataclass
class DaliConfig:
    sequence_length: int = 128
    fps: float = 32.0; sample_rate: float = 24000.0; n_mels: int = 128
    n_fft: int = 1024; win_length: int = 750; hop_length: int = 750
    batch_size: int = 1; num_threads: int = 4; device_id: int = 0
    shard_id: int = 0; num_shards: int = 1; seed: int = 42
    # DALI aliases and runtime params
    mel_bins: int = field(init=False); nfft: int = field(init=False)
    window_length: int = field(init=False); mel_fmax: float = field(init=False)
    db_cutoff: float = 80.0; shuffle: bool = False
    
    def __post_init__(self):
        self.mel_bins, self.nfft, self.window_length = self.n_mels, self.n_fft, self.win_length
        self.mel_fmax = self.sample_rate / 2.0


class DaliInputPipeline:
    """Builds a DALI pipeline for either raw media or pre-computed .npy files."""
    def __init__(self, video_filelists: List[str], audio_filelists: List[str], cfg: DaliConfig, use_precomputed: bool, video_pathlists: Optional[List[str]] = None, audio_pathlists: Optional[List[str]] = None):
        if not DALI_AVAILABLE: raise ImportError(f"NVIDIA DALI not available: {_DALI_IMPORT_ERROR}")
        
        reader_name_for_iterator = None
        if use_precomputed:
            if video_pathlists is None or audio_pathlists is None:
                raise ValueError("video_pathlists and audio_pathlists must be provided for precomputed mode.")
            self.pipeline = self._build_npy_pipeline(video_filelists, audio_filelists, video_pathlists, audio_pathlists, cfg)
            # The output map must match the order of tensors returned by the pipeline.
            out_map = [f"video_embed{k}" for k in range(5)]
            out_map.extend([f"audio_embed{k}" for k in range(5)])
            out_map.append("labels0") # Single shared label for the sample
            reader_name_for_iterator = "VidReader0" # Name of one reader for epoch size
        else:
            self.pipeline = self._build_media_pipeline(video_filelists, audio_filelists, cfg)
            out_map = []
            for k in range(5):
                out_map.extend([f"pov{k}", f"labels{k}", f"mel{k}"])
            reader_name_for_iterator = "V0"

        self.iterator = DALIGenericIterator(
            [self.pipeline], out_map,
            reader_name=reader_name_for_iterator,
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.DROP
        )

    def _build_npy_pipeline(self, vlists, alists, vpaths, apaths, cfg):
        """
        Builds a DALI pipeline for pre-computed .npy files with a focus on
        robustness on older DALI builds and perfect multi-GPU synchronization.
        """
        assert len(vlists) == 5 and len(alists) == 5, "Need 5 video and 5 audio file lists"
        assert len(vpaths) == 5 and len(apaths) == 5, "Need 5 video and 5 audio path-only lists"

        @pipeline_def(
            batch_size=cfg.batch_size,
            num_threads=cfg.num_threads,
            device_id=cfg.device_id,
            seed=cfg.seed,
            exec_async=True,
            exec_pipelined=True,
            prefetch_queue_depth=2,
            enable_memory_stats=True,
        )
        def pipe():
            def read_and_decode(filelist_lbl: str, filelist_paths: str, reader_name: str):
                common = dict(
                    shard_id=cfg.shard_id,
                    num_shards=cfg.num_shards,
                    stick_to_shard=True,
                    random_shuffle=False,
                    shuffle_after_epoch=False,
                    read_ahead=True,
                )


                _, packed_label = fn.readers.file(name=reader_name, file_list=filelist_lbl, **common)

                # Preferred path on newer DALI: bytes -> decoders.numpy
                if hasattr(fn, "decoders") and hasattr(fn.decoders, "numpy"):
                    bytes_i, _ = fn.readers.file(name=f"{reader_name}_BYTES", file_list=filelist_paths, **common)
                    arr = fn.decoders.numpy(bytes_i, dtype=types.FLOAT16)
                    return arr, packed_label

                # Fallback (older DALI): direct numpy reader from PATHS-ONLY list
                np_args = dict(name=f"{reader_name}_NPY", file_list=filelist_paths, **common)
                arr = fn.readers.numpy(**np_args)
                arr = fn.cast(arr, dtype=types.FLOAT16)  # match downstream expectations
                return arr, packed_label


            # Read all 10 views (5 video, 5 audio). The filelists are aligned,
            # so we only need to extract the label from one of them.
            video_embeds_raw = []
            audio_embeds_raw = []

            # Read the first video view to get the shared label
            v0_raw, packed_label = read_and_decode(vlists[0], vpaths[0], "VidReader0")
            video_embeds_raw.append(v0_raw)

            # Read remaining views, ignoring their redundant labels
            for k in range(1, 5):
                v_raw, _ = read_and_decode(vlists[k], vpaths[k], f"VidReader{k}")
                video_embeds_raw.append(v_raw)
            for k in range(5):
                a_raw, _ = read_and_decode(alists[k], apaths[k], f"AudReader{k}")
                audio_embeds_raw.append(a_raw)

            # --- Unpack label to get sample_id and start_frame ---
            packed_i64   = fn.cast(packed_label, dtype=types.INT64)
            sample_id_i64 = packed_i64 // LABEL_SCALE
            start_f_i64   = packed_i64 - sample_id_i64 * LABEL_SCALE

            # Match DALI's requirement: anchor and shape must share dtype (use INT32)
            start_f_i32 = fn.cast(start_f_i64, dtype=types.INT32)

            # --- Slice the required window from the full tensor ---
            def slice_window(tensor_node):
                return fn.slice(
                    tensor_node,
                    start_f_i32,                 # <-- now INT32
                    cfg.sequence_length,         # DALI treats this as INT32
                    axes=[0],
                    out_of_bounds_policy="pad",
                    fill_values=0.0
                )

            video_embeds = [slice_window(v) for v in video_embeds_raw]
            audio_embeds = [slice_window(a) for a in audio_embeds_raw]

            # --- Move data to GPU and finalize outputs ---
            video_embeds_gpu = [v.gpu() for v in video_embeds]
            audio_embeds_gpu = [a.gpu() for a in audio_embeds]

            label_out = fn.cast(sample_id_i64, dtype=types.INT32)

            # The final flattened tuple of outputs for the DALIGenericIterator
            return (*video_embeds_gpu, *audio_embeds_gpu, label_out)
        
        p = pipe()
        p.build()
        return p

    def _build_media_pipeline(self, vlists, alists, cfg):
        # This is the original _build_pipeline method, unchanged.
        @pipeline_def(
            enable_memory_stats=True,
            batch_size=cfg.batch_size,
            num_threads=cfg.num_threads,
            device_id=cfg.device_id,
            seed=cfg.seed,
            prefetch_queue_depth={ "cpu_size": 2, "gpu_size": 1 },
        )
        def pipe():
            outputs = []
            for k in range(5):
                # ---- VIDEO ----
                video, label_vid = fn.readers.video(
                    name=f"V{k}",
                    device="gpu",
                    file_list=vlists[k],
                    sequence_length=cfg.sequence_length,
                    pad_sequences=True,
                    shard_id=cfg.shard_id,
                    num_shards=cfg.num_shards,
                    stick_to_shard=True,              # FIX: For DDP determinism
                    random_shuffle=cfg.shuffle,
                    shuffle_after_epoch=False,        # FIX: For DDP determinism
                    dtype=types.UINT8,
                    file_list_frame_num=True,
                    file_list_include_preceding_frame=True,
                    additional_decode_surfaces=2,
                )
                # video: [F, H, W, C] uint8 -> [F, C, H, W]
                frames_uint8 = fn.transpose(video, perm=[0, 3, 1, 2])
                frames_uint8 = fn.reinterpret(frames_uint8, layout="FCHW")
                
                # ---- AUDIO ----
                audio_raw, label_cpu = fn.readers.file(
                    name=f"A{k}",
                    file_list=alists[k],
                    shard_id=cfg.shard_id,
                    num_shards=cfg.num_shards,
                    stick_to_shard=True,              # FIX: For DDP determinism
                    random_shuffle=cfg.shuffle,
                    shuffle_after_epoch=False,        # FIX: For DDP determinism
                )
                packed_i64 = fn.cast(label_cpu, dtype=types.INT64)
                sample_id_i64 = packed_i64 // LABEL_SCALE
                start_f_i64 = packed_i64 - sample_id_i64 * LABEL_SCALE

                sample_id_i32 = fn.cast(sample_id_i64, dtype=types.INT32)
                start_f_f32 = fn.cast(start_f_i64, dtype=types.FLOAT)

                decoded, _ = fn.decoders.audio(audio_raw, sample_rate=cfg.sample_rate, downmix=False)
                start_s = start_f_f32 / cfg.fps
                shape_samples = (cfg.sequence_length - 1) * cfg.hop_length + cfg.window_length
                sliced = fn.slice(
                    decoded.gpu(),
                    start=fn.cast(start_s * cfg.sample_rate, dtype=types.INT32),
                    shape=[int(shape_samples)],
                    axes=[0],
                    out_of_bounds_policy="pad",
                )

                left = sliced[:, 0]
                right = sliced[:, 1]

                def to_mel_db(channel_1d):
                    spec = fn.spectrogram(
                        channel_1d,
                        nfft=cfg.nfft,
                        window_length=cfg.window_length,
                        window_step=cfg.hop_length,
                        center_windows=False,
                    )
                    mel = fn.mel_filter_bank(
                        spec,
                        sample_rate=cfg.sample_rate,
                        nfilter=cfg.mel_bins,
                        freq_high=cfg.mel_fmax,
                    )
                    db = fn.to_decibels(mel, cutoff_db=cfg.db_cutoff)
                    db = fn.cast(db, dtype=types.FLOAT16)  # keep small, matches your TODO
                    return fn.transpose(db, perm=[1, 0])   # -> [time, n_mels]

                mel_db_left = to_mel_db(left)
                mel_db_right = to_mel_db(right)
                mel_db = fn.stack(mel_db_left, mel_db_right, axis=0)  # [2, time, n_mels]

                outputs.extend([frames_uint8, label_vid, mel_db])
            return tuple(outputs)

        p = pipe()
        p.build()
        return p

    def __iter__(self): return self
    def __next__(self): return next(self.iterator)[0]
    def reset(self): self.iterator.reset()


# -------------------------------
# Step 6: Tick vector generator
# -------------------------------

def ticks_for_window(start_tick_win: int, T_frames: int) -> np.ndarray:
    """Compute tick indices for a fixed-length window, parity-locked to round start."""
    return start_tick_win + (np.arange(T_frames, dtype=np.int32) * TICKS_PER_FRAME)


# -------------------------------
# Step 7 & 10: LMDB metadata fetch
# -------------------------------

@dataclass
class MetaFetchResult:
    """Holds all ground truth data fetched from LMDB for one training sample."""
    alive_mask: np.ndarray             # [T, 5] (uint8)
    # Player-specific ground truth [T, 5, ...]
    stats: np.ndarray                  # [T, 5, 3] (health, armor, money) float32
    mouse_delta: np.ndarray            # [T, 5, 2] (dx, dy) float32
    position: np.ndarray               # [T, 5, 3] (x, y, z) float32
    keyboard_mask: np.ndarray          # [T, 5] uint32
    eco_mask: np.ndarray               # [T, 5, 4] uint64
    inventory_mask: np.ndarray         # [T, 5, 2] uint64
    active_weapon_idx: np.ndarray      # [T, 5] int32
    # Game-strategy ground truth [T, ...]
    round_number: np.ndarray           # [T] int32
    round_state_mask: np.ndarray       # [T] uint8
    enemy_positions: np.ndarray        # [T, 5, 3] float32

class LmdbMetaFetcher:
    """Fetches per-frame metadata and ground truth labels from LMDB for a sampled window."""
    def __init__(self, store: LmdbStore):
        self.store = store

    @staticmethod
    def _key(demoname: str, round_num: int, team: str, tick: int) -> bytes:
        return f"{demoname}_round_{round_num:03d}_team_{team}_tick_{tick:08d}".encode("utf-8")

    @staticmethod
    def _bitmask_to_weapon_index(mask: np.ndarray) -> int:
        """Converts a [2] uint64 weapon bitmask to a single item index."""
        if mask.sum() == 0: return -1
        for i in range(128):
            if (mask[i // 64] >> np.uint64(i % 64)) & np.uint64(1):
                return i
        return -1

    def fetch(self, rec: SampleRecord) -> MetaFetchResult:
        """Fetches next-frame-prediction targets for the given sample record."""
        env = self.store.open(rec.lmdb_path)
        T = rec.T_frames
        
        # Initialize numpy arrays to hold the results
        alive_mask = np.zeros((T, 5), dtype=np.uint8)
        stats = np.zeros((T, 5, 3), dtype=np.float32)
        mouse_delta = np.zeros((T, 5, 2), dtype=np.float32)
        position = np.zeros((T, 5, 3), dtype=np.float32)
        keyboard_mask = np.zeros((T, 5), dtype=np.uint32)
        eco_mask = np.zeros((T, 5, 4), dtype=np.uint64)
        inventory_mask = np.zeros((T, 5, 2), dtype=np.uint64)
        active_weapon_idx = np.full((T, 5), -1, dtype=np.int32)
        round_number = np.full((T,), rec.round_num, dtype=np.int32)
        round_state_mask = np.zeros((T,), dtype=np.uint8)
        enemy_positions = np.zeros((T, 5, 3), dtype=np.float32)
        
        ticks = ticks_for_window(rec.start_tick_win, T)

        with env.begin(write=False) as txn:
            for f, tick in enumerate(ticks.tolist()):
                target_blob = txn.get(self._key(rec.demoname, rec.round_num, rec.team, int(tick)))
                if not target_blob: continue

                payload = msgpack.unpackb(target_blob, raw=False, object_hook=mpnp.decode)

                gs = payload.get("game_state")
                if gs is None or len(gs) == 0: continue

                gs_data = gs[0]
                mask_bits = int(gs_data['team_alive'])
                
                # Populate alive_mask from the current tick's data
                for slot in range(5):
                    if (mask_bits >> slot) & 1:
                        alive_mask[f, slot] = 1

                # Populate Game State / Strategy Targets
                round_state_mask[f] = gs_data['round_state']
                enemy_positions[f] = gs_data['enemy_pos']

                pdl = payload.get("player_data")
                if pdl:
                    alive_indices = [i for i in range(5) if (mask_bits >> i) & 1]
                    
                    # Sanity check: the number of alive players must match the data list length
                    if len(alive_indices) != len(pdl):
                        continue
                        
                    for p_idx, p_data_arr in zip(alive_indices, pdl):
                        if p_data_arr is not None and len(p_data_arr) > 0:
                            p_data = p_data_arr[0]
                            stats[f, p_idx] = [p_data['health'], p_data['armor'], p_data['money']]
                            mouse_delta[f, p_idx] = p_data['mouse']
                            position[f, p_idx] = p_data['pos']
                            keyboard_mask[f, p_idx] = p_data['keyboard_bitmask']
                            eco_mask[f, p_idx] = p_data['eco_bitmask']
                            inventory_mask[f, p_idx] = p_data['inventory_bitmask']
                            active_weapon_idx[f, p_idx] = self._bitmask_to_weapon_index(p_data['active_weapon_bitmask'])

        return MetaFetchResult(
            alive_mask=alive_mask, stats=stats, mouse_delta=mouse_delta, position=position,
            keyboard_mask=keyboard_mask, eco_mask=eco_mask, inventory_mask=inventory_mask,
            active_weapon_idx=active_weapon_idx, round_number=round_number,
            round_state_mask=round_state_mask, enemy_positions=enemy_positions
        )

# -------------------------------
# Step 8 & 10: Batch assembler
# -------------------------------

class BatchAssembler:
    """Converts a raw DALI batch + LMDB fetch into model-ready tensors."""
    def __init__(self, id_to_sample: Dict[int, SampleRecord], fetcher: LmdbMetaFetcher, device: torch.device):
        self.id_to_sample = id_to_sample
        self.fetcher = fetcher
        self.device = device

    @staticmethod
    def _masks_to_multi_hot(masks: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Convert a tensor of integer bitmasks to a multi-hot float tensor."""
        safe_masks = masks.long()
        powers = torch.arange(num_classes, device=safe_masks.device, dtype=torch.long).view(1, -1)
        return ((safe_masks.unsqueeze(-1) >> powers) & 1).float()


    def assemble(self, dali_batch: Dict[str, torch.Tensor], use_precomputed: bool) -> Dict[str, Any]:
        """Assembles the final batch dictionary, handling both data paths."""

        if use_precomputed:
            # PATH 1: Pre-computed embeddings
            # Video: DALI returns [B, T, D_raw] -> Stack to [B, T, 5, D_raw]
            video_embeddings = torch.stack([dali_batch[f"video_embed{k}"] for k in range(5)], dim=2)
            
            # Audio: DALI returns [B, T, Mel, 2] from [T, Mel, 2] npy file.
            # We need to assemble this into the model's expected [B, T, P, C, Mel, 1] shape.
            audio_embed_list = [dali_batch[f"audio_embed{k}"] for k in range(5)] # List of 5x [B, T, Mel, 2]
            audio_stacked = torch.stack(audio_embed_list, dim=2)      # [B, T, 5, Mel, 2]
            audio_permuted = audio_stacked.permute(0, 1, 2, 4, 3)     # [B, T, 5, 2, Mel]
            mel_spectrogram = audio_permuted.unsqueeze(-1)            # [B, T, 5, 2, Mel, 1]
            
            sample_ids = dali_batch["labels0"].view(-1).cpu().tolist()
            batch = {
                "video_embeddings": video_embeddings,
                "mel_spectrogram": mel_spectrogram, # FIX: Use the key the model expects
            }
        else:
            # PATH 2: On-the-fly processing
            images = torch.stack([dali_batch[f"pov{k}"] for k in range(5)], dim=2)
            mels_list = [dali_batch[f"mel{k}"] for k in range(5)] # [B, 2, T, Mels]
            mel_stacked = torch.stack(mels_list, dim=2) # [B, 2, 5, T, Mels]
            mel_permuted = mel_stacked.permute(0, 3, 2, 1, 4) # [B, T, 5, 2, Mels]
            mel = mel_permuted.unsqueeze(-1) # [B, T, P, C, Mels, 1]
            
            sample_ids = dali_batch["labels0"].view(-1).cpu().tolist()
            batch = {"images": images, "mel_spectrogram": mel}

        # The rest of the assembly (metadata, targets) is IDENTICAL for both paths
        gt_lists = defaultdict(list)
        meta = {"sample_ids": sample_ids, "demonames": [], "round_nums": [], "teams": []}
        for sid in sample_ids:
            rec = self.id_to_sample[int(sid)]
            meta["demonames"].append(rec.demoname)
            meta["round_nums"].append(rec.round_num)
            meta["teams"].append(rec.team)
            gt_result = self.fetcher.fetch(rec)
            for key, value in gt_result.__dict__.items():
                gt_lists[key].append(torch.from_numpy(value))

        gt_tensors = {k: torch.stack(v, dim=0).to(self.device, non_blocking=True) for k, v in gt_lists.items()}
        batch["alive_mask"] = gt_tensors.pop("alive_mask").bool()
        batch["meta"] = meta

        targets = {"player": [{} for _ in range(5)], "game_strategy": {}}
        for i in range(5):
            targets["player"][i]["stats"] = gt_tensors["stats"][:, :, i]
            targets["player"][i]["mouse_delta_deg"] = gt_tensors["mouse_delta"][:, :, i]
            targets["player"][i]["pos_coords"] = gt_tensors["position"][:, :, i] # Raw coords for loss fn
            targets["player"][i]["keyboard_logits"] = self._masks_to_multi_hot(gt_tensors["keyboard_mask"][:, :, i], 31)

            # Eco Logits (4x uint64 -> 256 bits, truncated to 224 classes)
            eco_mask_player = gt_tensors["eco_mask"][:, :, i]  # Shape: [B, T, 4]
            eco_parts = [self._masks_to_multi_hot(eco_mask_player[:, :, k], 64) for k in range(4)]
            full_eco_logits = torch.cat(eco_parts, dim=-1)  # Shape: [B, T, 256]
            targets["player"][i]["eco_logits"] = full_eco_logits[..., :224]

            # Inventory Logits (2x uint64 -> 128 bits)
            inv_mask_player = gt_tensors["inventory_mask"][:, :, i]  # Shape: [B, T, 2]
            inv_parts = [self._masks_to_multi_hot(inv_mask_player[:, :, k], 64) for k in range(2)]
            targets["player"][i]["inventory_logits"] = torch.cat(inv_parts, dim=-1) # Shape: [B, T, 128]

            targets["player"][i]["active_weapon_logits"] = gt_tensors["active_weapon_idx"][:, :, i].long()

        targets["game_strategy"]["enemy_pos_coords"] = gt_tensors["enemy_positions"]
        targets["game_strategy"]["round_state_logits"] = self._masks_to_multi_hot(gt_tensors["round_state_mask"], 5)
        targets["game_strategy"]["round_number"] = gt_tensors["round_number"].float()

        batch["targets"] = targets
        return batch


# =============================================================================
# STEP 11: Composite Loss Function
# =============================================================================

class CompositeLoss(nn.Module):
    """
    Calculates the composite, masked loss across all model prediction heads.

    Efficiency improvements added:
    - Cache 1-D coordinate axes (xs, ys, zs) as buffers and reuse them each step.
    - Generate Gaussian heatmaps via separable 1-D Gaussians (no [Z,Y,X,3] grid).
    - Create targets under torch.no_grad() to avoid autograd bookkeeping.
    - Avoid unsqueeze+expand for enemy heatmaps; use advanced indexing instead.
    - Keep BCE in AMP when global autocast is enabled.
    """
    def __init__(self, weights: dict, grid_dims: tuple[int,int,int]=(64,64,8), sigma: float = 1.5):
        super().__init__()
        self.weights = weights
        self.coord_mapper = CoordinateMapper(grid_dims=grid_dims)

        # Loss fns with no reduction for manual masking
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.ce_loss  = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)

        # Cache axes for separable Gaussian target generation
        X, Y, Z = grid_dims
        self.register_buffer('xs', torch.arange(X, dtype=torch.float32), persistent=False)
        self.register_buffer('ys', torch.arange(Y, dtype=torch.float32), persistent=False)
        self.register_buffer('zs', torch.arange(Z, dtype=torch.float32), persistent=False)
        self.register_buffer(
            "stats_scale",
            torch.tensor([1/100.0, 1/100.0, 1/16000.0], dtype=torch.float32),  # health, armor, money
            persistent=False
        )

        self.grid_dims = grid_dims
        self.sigma = float(sigma)

    @staticmethod
    def _scalar_loss(unmasked_loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply mask and reduce to a single scalar."""
        masked = unmasked_loss * mask
        return masked.sum() / mask.sum().clamp(min=1.0)

    def _gaussian_heatmaps_from_indices(self, centers_xyz_idx: torch.Tensor) -> torch.Tensor:
        """
        Build smooth 3-D heatmaps for N centers using separable 1-D Gaussians.
        centers_xyz_idx: Long/float tensor [N, 3] with (x_idx, y_idx, z_idx).
        Returns: Float tensor [N, Z, Y, X].
        """
        if centers_xyz_idx.numel() == 0:
            Z, Y, X = self.grid_dims[2], self.grid_dims[1], self.grid_dims[0]
            return torch.empty((0, Z, Y, X), device=self.xs.device, dtype=torch.float32)

        centers = centers_xyz_idx.to(dtype=torch.float32, device=self.xs.device)
        gx, gy, gz = centers[:, 0], centers[:, 1], centers[:, 2]  # [N]

        xs, ys, zs = self.xs, self.ys, self.zs
        s2 = 2.0 * (self.sigma ** 2)

        # [N, X], [N, Y], [N, Z]
        hx = torch.exp(-((xs.unsqueeze(0) - gx.unsqueeze(1)) ** 2) / s2)
        hy = torch.exp(-((ys.unsqueeze(0) - gy.unsqueeze(1)) ** 2) / s2)
        hz = torch.exp(-((zs.unsqueeze(0) - gz.unsqueeze(1)) ** 2) / s2)

        # [N, Z, Y, X]
        return hz[:, :, None, None] * hy[:, None, :, None] * hx[:, None, None, :]

    def _build_targets_heatmaps(self, world_xyz: torch.Tensor) -> torch.Tensor:
        """
        Convert world coordinates [N, 3] to grid indices then to heatmaps [N, Z, Y, X].
        Computed under no_grad to avoid autograd overhead.
        """
        with torch.no_grad():
            grid_idx = self.coord_mapper.discretize_world_to_grid(world_xyz).to(self.xs.device)  # [N,3] long
            return self._gaussian_heatmaps_from_indices(grid_idx)

    def forward(self, predictions: dict, targets: dict, alive_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predictions: model outputs dict.
            targets: target tensors dict.
            alive_mask: [B, T, 5] bool tensor indicating alive players.
        Returns:
            A tuple of (total_loss_tensor, detailed_losses_dict).
        """
        device = self.xs.device
        total_loss = torch.tensor(0.0, device=device)
        detailed_losses = {} # This will store the final weighted losses as tensors

        B, T = predictions['player'][0]['stats'].shape[:2]
        alive_mask = alive_mask.to(device=device, non_blocking=True)  # [B,T,5] bool

        # ----------------------------
        # Player heads (MSE/BCE/CE)
        # ----------------------------
        player_loss_keys = ['stats', 'mouse', 'keyboard', 'eco', 'inventory', 'weapon']
        for key in player_loss_keys: # Initialize accumulators for player-specific losses
            detailed_losses[key] = torch.tensor(0.0, device=device)

        for i in range(5):
            p_pred = predictions["player"][i]
            p_targ = targets["player"][i]
            player_alive_mask = alive_mask[:, :, i].float()  # [B,T]

            # Stats (MSE)
            pred_stats_n = p_pred["stats"] * self.stats_scale
            targ_stats_n = p_targ["stats"] * self.stats_scale
            loss_stats = self.mse_loss(pred_stats_n, targ_stats_n).mean(dim=-1)
            loss_component = self.weights['stats'] * self._scalar_loss(loss_stats, player_alive_mask)
            detailed_losses['stats'] += loss_component

            # Mouse (MSE)
            loss_mouse = self.mse_loss(p_pred["mouse_delta_deg"], p_targ["mouse_delta_deg"]).mean(dim=-1)  # [B,T]
            loss_component = self.weights['mouse'] * self._scalar_loss(loss_mouse, player_alive_mask)
            detailed_losses['mouse'] += loss_component

            # Multi-label BCE heads
            for key in ["keyboard_logits", "eco_logits", "inventory_logits"]:
                loss_bce = self.bce_loss(p_pred[key], p_targ[key]).mean(dim=-1)  # [B,T]
                wkey = key.replace('_logits','')
                loss_component = self.weights[wkey] * self._scalar_loss(loss_bce, player_alive_mask)
                detailed_losses[wkey] += loss_component

            # Active weapon (CE), flatten B*T
            pred_flat = p_pred["active_weapon_logits"].view(B * T, -1)
            targ_flat = p_targ["active_weapon_logits"].view(B * T)
            loss_weapon_unmasked = self.ce_loss(pred_flat, targ_flat)  # [B*T]
            loss_component = self.weights['weapon'] * self._scalar_loss(
                loss_weapon_unmasked, player_alive_mask.view(B * T)
            )
            detailed_losses['weapon'] += loss_component

        # Sum up all player-related losses
        for key in player_loss_keys:
            total_loss += detailed_losses[key]

        # ----------------------------
        # Game strategy scalar heads
        # ----------------------------
        gs_pred = predictions["game_strategy"]
        gs_targ = targets["game_strategy"]
        frame_mask = alive_mask.any(dim=-1).float()  # [B,T]

        # Round number (MSE)
        loss_round_num = self.mse_loss(gs_pred["round_number"], gs_targ["round_number"].view(B, T, 1)).squeeze(-1)  # [B,T]
        loss_component = self.weights['round_number'] * self._scalar_loss(loss_round_num, frame_mask)
        detailed_losses['round_number'] = loss_component
        total_loss += loss_component

        # Round state (BCE)
        loss_round_state = self.bce_loss(gs_pred["round_state_logits"], gs_targ["round_state_logits"]).mean(dim=-1)  # [B,T]
        loss_component = self.weights['round_state'] * self._scalar_loss(loss_round_state, frame_mask)
        detailed_losses['round_state'] = loss_component
        total_loss += loss_component

        # ----------------------------
        # Heatmap heads (efficient targets & indexing)
        # ----------------------------
        # Player position
        pred_pos_heatmaps = torch.stack([p["pos_heatmap_logits"] for p in predictions["player"]], dim=2)  # [B,T,5,Z,Y,X]
        targ_pos_coords   = torch.stack([p["pos_coords"]         for p in targets["player"]],     dim=2)  # [B,T,5,3]

        if alive_mask.any():
            alive_flat   = alive_mask.view(-1)  # [B*T*5]
            pred_alive   = pred_pos_heatmaps.view(-1, *pred_pos_heatmaps.shape[3:])[alive_flat]  # [N,Z,Y,X]
            coord_alive  = targ_pos_coords.view(-1, 3)[alive_flat]  # [N,3]
            target_heatmap = self._build_targets_heatmaps(coord_alive).to(dtype=pred_alive.dtype)
            loss_pos = self.bce_loss(pred_alive, target_heatmap).mean()
            loss_component = self.weights['pos_heatmap'] * loss_pos
            detailed_losses['pos_heatmap'] = loss_component
            total_loss += loss_component
        else:
            detailed_losses['pos_heatmap'] = torch.tensor(0.0, device=device)

        # Enemy position (one predicted heatmap per frame; up to 5 valid targets)
        pred_enemy_heatmaps = gs_pred["enemy_pos_heatmap_logits"]           # [B,T,Z,Y,X]
        targ_enemy_coords   = gs_targ["enemy_pos_coords"]                   # [B,T,5,3]
        valid_enemy_mask    = (targ_enemy_coords[..., 0] >= 0)              # [B,T,5]

        if valid_enemy_mask.any():
            b_idx, t_idx, p_idx = valid_enemy_mask.nonzero(as_tuple=True)   # each [N]
            pred_sel  = pred_enemy_heatmaps[b_idx, t_idx]                    # [N,Z,Y,X]
            coord_sel = targ_enemy_coords[b_idx, t_idx, p_idx]               # [N,3]
            target_heatmap = self._build_targets_heatmaps(coord_sel).to(dtype=pred_sel.dtype)
            loss_enemy_pos = self.bce_loss(pred_sel, target_heatmap).mean()
            loss_component = self.weights['enemy_heatmap'] * loss_enemy_pos
            detailed_losses['enemy_heatmap'] = loss_component
            total_loss += loss_component
        else:
            detailed_losses['enemy_heatmap'] = torch.tensor(0.0, device=device)

        return total_loss, {k: v.item() for k, v in detailed_losses.items()}


# -------------------------------
# Step 9: Determinism, DDP helpers, instrumentation
# -------------------------------

class Timer:
    """Simple timer context for instrumentation."""
    def __init__(self, name: str): self.name = name
    def __enter__(self): self.t0 = time.time(); return self
    def __exit__(self, *args): self.dt = time.time() - self.t0

def get_ddp_info() -> Tuple[int, int]:
    """Return (shard_id, num_shards) using torch.distributed if available."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1

# -------------------------------
# Optional: minimal driver for sanity
# -------------------------------

def build_data_iter(args):
    """Convenience function that wires Steps 1→8 and returns components."""
    manifest = Manifest(args.data_root, args.manifest)
    store = LmdbStore()
    team_rounds = build_team_rounds(args.data_root, manifest.get_games(args.split), store)
    index = EpochIndex(T_frames=args.T_frames, seed=args.seed)
    records, id_map = index.build(team_rounds, epoch=0)
    
    fl_dir = os.path.join(args.run_dir, "epoch_0")
    writer = FilelistWriter(fl_dir, use_precomputed=args.use_precomputed_embeddings, data_root=args.data_root)
    video_lists, audio_lists = writer.write(records)
    
    video_path_lists, audio_path_lists = None, None
    if args.use_precomputed_embeddings:
        def make_paths_only_list(src_lst: str) -> str:
            from pathlib import Path
            dst = str(Path(src_lst).with_suffix(".paths"))
            with open(src_lst, "r") as s, open(dst, "w") as d:
                for line in s:
                    d.write(line.rstrip("\n").split()[0] + "\n")
            return dst
        video_path_lists = [make_paths_only_list(p) for p in video_lists]
        audio_path_lists = [make_paths_only_list(p) for p in audio_lists]

    shard_id, num_shards = get_ddp_info()
    device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
    dali_cfg = DaliConfig(
        sequence_length=args.T_frames, batch_size=args.batch_size, num_threads=args.dali_threads,
        device_id=device_id, shard_id=shard_id, num_shards=num_shards
    )
    dali_iter = DaliInputPipeline(
        video_filelists=video_lists,
        audio_filelists=audio_lists,
        cfg=dali_cfg,
        use_precomputed=args.use_precomputed_embeddings,
        video_pathlists=video_path_lists,
        audio_pathlists=audio_path_lists,
    )
    
    fetcher = LmdbMetaFetcher(store)
    assembler_device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    assembler = BatchAssembler(id_map, fetcher, device=assembler_device)
    
    # Return a model config object to be used by the main script
    from model import CS2Config
    model_cfg = CS2Config(context_frames=args.T_frames)
    
    return dali_iter, assembler, model_cfg


def get_args():
    """Parses command-line arguments for the smoke test driver."""
    parser = argparse.ArgumentParser(description="train2.py Data Loader and Loss Calculation Smoke Test")
    parser.add_argument("--data-root", type=str, default=os.environ.get("DATA_ROOT", "data"), help="Root directory for the dataset.")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest.json. Defaults to <data-root>/manifest.json.")
    parser.add_argument("--split", type=str, default="train", help="Data split to use (e.g., 'train', 'val').")
    parser.add_argument("--run-dir", type=str, default="runs/exp1", help="Directory for logs and outputs.")
    parser.add_argument("--T-frames", type=int, default=64, help="Number of frames per sequence.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per GPU.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--dali-threads", type=int, default=4, help="Number of threads for DALI.")
    parser.add_argument("--num-steps", type=int, default=20, help="Number of iterations for the smoke test.")
    parser.add_argument("--profile", action='store_true', help="Enable torch.profiler and save trace to TensorBoard.")
    parser.add_argument("--use-precomputed-embeddings", action='store_true',
                        help="Load pre-computed .npy embeddings instead of processing media on-the-fly.")
    parser.add_argument("--detailed-loss", action='store_true', help="Print a detailed breakdown of all loss components.")
    
    args = parser.parse_args()
    if args.manifest is None:
        args.manifest = os.path.join(args.data_root, "manifest.json")
    return args


def run_step(dali_iter, assembler, model, loss_fn, use_precomputed: bool, detailed_loss: bool):
    """Encapsulates one iteration of the smoke test."""
    try:
        with record_function("dali_fetch"), Timer("dali_fetch") as t:
            batch_raw = next(iter(dali_iter))
        logging.info("DALI fetched data in %.3fs", t.dt)
        
        with record_function("assemble_batch"), Timer("assemble") as t2:
            batch = assembler.assemble(batch_raw, use_precomputed)
        logging.info("Assembled batch in %.3fs", t2.dt)
        
        with record_function("forward_pass"), Timer("forward_pass") as t3:
            predictions = model(batch)
        logging.info("Forward pass in %.3fs", t3.dt)
        
        with record_function("loss_calculation"), Timer("loss_calc") as t4:
            total_loss, detailed_losses_dict = loss_fn(predictions, batch['targets'], batch['alive_mask'])
        logging.info("Calculated loss in %.4fs -> Total Loss: %.4f", t4.dt, total_loss.item())
        
        if detailed_loss:
            logging.info("--- Detailed Loss Breakdown ---")
            max_key_len = max(len(k) for k in detailed_losses_dict.keys()) if detailed_losses_dict else 0
            for name, value in sorted(detailed_losses_dict.items()):
                logging.info(f"  - {name:<{max_key_len}}: {value:12.4f}")
            logging.info("-----------------------------")

        assert total_loss.requires_grad, "Loss must require gradients"
        logging.info("✅ Smoke assertion passed: Loss requires grad.")
        return True
    except StopIteration:
        logging.info("Finished iterator.")
        return False


def main(args):
    """Main driver for the data loader smoke test."""
    # These imports are specific to the driver and model
    from model import CS2Transformer

    if not DALI_AVAILABLE:
        logging.error("DALI is not available: %s", _DALI_IMPORT_ERROR)
        return

    try:
        dali_iter, assembler, model_cfg = build_data_iter(args)
        
        # --- SMOKE TEST SETUP ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Configure the model based on the data loading mode
        logging.info(f"Configuring model for pre-computed embeddings: {args.use_precomputed_embeddings}")
        # This config is now passed to the model, but the model's forward pass
        # makes the final decision based on the batch keys. This is for consistency.
        model = CS2Transformer(model_cfg, use_dummy_vision=False).to(device)
        
        loss_weights = {
            'stats': 1.0, 'mouse': 1.0, 'keyboard': 1.0, 'eco': 1.0,
            'inventory': 1.0, 'weapon': 1.0, 'round_number': 1.0,
            'round_state': 1.0, 'pos_heatmap': 1.0, 'enemy_heatmap': 1.0
        }
        loss_fn = CompositeLoss(weights=loss_weights).to(device)
        # --- END SMOKE TEST SETUP ---

        if args.profile:
            profile_dir = os.path.join(args.run_dir, "profiler_logs")
            os.makedirs(profile_dir, exist_ok=True)
            logging.info(f"Profiler enabled. Traces will be saved to: {profile_dir}")
            logging.info(f"To view, run: tensorboard --logdir {profile_dir}")
            
            # Schedule: wait 1 step, warmup 1 step, then actively record for 3 steps.
            schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
            trace_handler = torch.profiler.tensorboard_trace_handler(profile_dir)

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule,
                on_trace_ready=trace_handler,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                for i in range(args.num_steps):
                    if not run_step(dali_iter, assembler, model, loss_fn, args.use_precomputed_embeddings, args.detailed_loss):
                        break
                    prof.step() # Signal profiler that a step is complete

            logging.info("--- PyTorch Profiler Summary ---")

            # Print top 10 operators by self-CUDA time
            print("\n--- Top 10 Operators by Self CUDA Time ---")
            print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

            # Print top 10 operators by CUDA memory usage
            print("\n--- Top 10 Operators by CUDA Memory Usage ---")
            print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
            print(torch.cuda.max_memory_allocated() / 1024**3, "GiB peak")
        else:
            for i in range(args.num_steps):
                if not run_step(dali_iter, assembler, model, loss_fn, args.use_precomputed_embeddings, args.detailed_loss):
                    break

    except Exception as e:
        logging.exception("Data loader smoke test failed.")
        logging.error("Failed with error: %s", e)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    cli_args = get_args()
    main(cli_args)