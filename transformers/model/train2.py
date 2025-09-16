#!/usr/bin/env python3
"""
train2.py — Data loading and Loss Calculation (Steps 1–11)

This file implements the end-to-end data input subsystem for training, focusing on:
  1) Round discovery from each game's LMDB `_INFO` entry (including video and audio paths)
  2) LMDB environment manager for metadata
  3) Epoch index / window sampling (fixed-length frame windows)
  4) Writing 10 aligned DALI filelists (one per POV for video and audio)
  5) Building a single DALI pipeline with 10 branches (5 video, 5 audio)
  6) Tick-vector generator (64-tick demos → 32 fps, 2 ticks per frame)
  7) LMDB metadata fetch (ground truth for all prediction heads)
  8) Batch assembler (fuse DALI outputs with LMDB features into model-ready batches)
  9) Determinism, DDP sharding, and instrumentation
  10) Ground truth target preparation, including smooth heatmaps
  11) A composite loss function for multi-headed predictions

Notes:
- This module defines the data preparation pipeline into a CS2Batch-like dict.
- Mel spectrograms are now generated ON-THE-FLY by DALI from .wav files.
- LMDB is used for all per-frame metadata and ground truth labels.
- The training task is next-frame prediction: given frame `t`, predict frame `t+1`.
"""
from __future__ import annotations

import os
import re
import json
import math
import time
import random
import logging
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any

import lmdb  # pip install lmdb
import msgpack  # pip install msgpack
import numpy as np
import msgpack_numpy as mpnp  # pip install msgpack-numpy

# Torch is used for device placement and to hand CUDA tensors to the model later
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """Creates a batch of smooth target heatmaps using a Gaussian kernel."""
    X, Y, Z = grid_dims
    device = grid_indices.device
    
    zz, yy, xx = torch.meshgrid(
        torch.arange(Z, device=device),
        torch.arange(Y, device=device),
        torch.arange(X, device=device),
        indexing='ij'
    )
    grid_coords = torch.stack((xx, yy, zz), dim=-1).float()
    
    target_indices = grid_indices.float().view(-1, 1, 1, 1, 3)
    distance_sq = torch.sum((grid_coords.unsqueeze(0) - target_indices)**2, dim=-1)
    heatmap = torch.exp(-distance_sq / (2 * sigma**2))
    
    return heatmap # Unnormalized for BCEWithLogitsLoss is fine

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
    """Writes ten aligned DALI filelists (five for video, five for audio)."""
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def write(self, records: List[SampleRecord]) -> Tuple[List[str], List[str]]:
        vid_paths = [os.path.join(self.out_dir, f"pov{k}_video.txt") for k in range(5)]
        aud_paths = [os.path.join(self.out_dir, f"pov{k}_audio.txt") for k in range(5)]
        
        with contextlib.ExitStack() as stack:
            files = [stack.enter_context(open(p, "w", encoding="utf-8")) for p in vid_paths + aud_paths]
            vid_files, aud_files = files[:5], files[5:]
            for rec in records:
                for k in range(5):
                    pov_path = rec.pov_videos[k]
                    ticks = ticks_from_filename(pov_path)
                    if not ticks: raise ValueError(f"Could not parse ticks from: {pov_path}")
                    pov_start_tick, pov_end_tick = ticks
                    start, end_exclusive = clamp_window_to_pov(rec.start_f, rec.T_frames, pov_start_tick, pov_end_tick)
                    vid_files[k].write(f"{pov_path} {rec.sample_id} {start} {end_exclusive}\n")
                    packed = rec.sample_id * LABEL_SCALE + start
                    aud_files[k].write(f"{rec.pov_audio[k]} {packed}\n")
        logging.info("Wrote DALI filelists to %s (N=%d)", self.out_dir, len(records))
        return vid_paths, aud_paths

# -------------------------------
# Step 5: DALI pipeline (video + audio)
# -------------------------------

@dataclass
class DaliConfig:
    height: int = 224; width: int = 224; sequence_length: int = 512
    mean: Tuple[float,...] = (0.0, 0.0, 0.0); std: Tuple[float,...] = (1.0, 1.0, 1.0)
    fps: float = 32.0; sample_rate: float = 24000.0; n_mels: int = 128
    n_fft: int = 1024; win_length: int = 1024; hop_length: int = 750
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
    """Builds a single DALI pipeline with 5 video and 5 audio branches."""
    def __init__(self, video_filelists: List[str], audio_filelists: List[str], cfg: DaliConfig):
        if not DALI_AVAILABLE: raise ImportError(f"NVIDIA DALI not available: {_DALI_IMPORT_ERROR}")
        self.pipeline = self._build_pipeline(video_filelists, audio_filelists, cfg)
        out_map = []
        for k in range(5):
            out_map.extend([f"pov{k}", f"labels{k}", f"mel{k}"])
        self.iterator = DALIGenericIterator([self.pipeline], out_map, auto_reset=True, last_batch_policy=LastBatchPolicy.DROP)

    def _build_pipeline(self, vlists, alists, cfg):
        @pipeline_def(batch_size=cfg.batch_size, num_threads=cfg.num_threads, device_id=cfg.device_id, seed=cfg.seed)
        def pipe():
            outputs = []
            for k in range(5):
                video, _ = fn.readers.video(
                    name=f"V{k}",
                    device="gpu",
                    file_list=vlists[k],
                    sequence_length=cfg.sequence_length,
                    pad_sequences=True,
                    shard_id=cfg.shard_id,
                    num_shards=cfg.num_shards,
                    random_shuffle=cfg.shuffle,
                    dtype=types.UINT8,
                    file_list_frame_num=True, 
                    file_list_include_preceding_frame=True,
                )
                frames = fn.crop_mirror_normalize(fn.resize(video, resize_x=cfg.width, resize_y=cfg.height), device="gpu", dtype=types.FLOAT16, output_layout="FCHW", mean=cfg.mean, std=cfg.std)
                
                audio_raw, label_cpu = fn.readers.file(name=f"A{k}", file_list=alists[k], shard_id=cfg.shard_id, num_shards=cfg.num_shards, random_shuffle=cfg.shuffle)
                packed_i64 = fn.cast(label_cpu, dtype=types.INT64)
                sample_id_i64 = packed_i64 // LABEL_SCALE
                start_f_i64 = packed_i64 - sample_id_i64 * LABEL_SCALE

                # Cast to the final desired dtypes
                sample_id_i32 = fn.cast(sample_id_i64, dtype=types.INT32)
                start_f_f32 = fn.cast(start_f_i64, dtype=types.FLOAT)

                # 1. Decode audio to stereo [length, 2]
                decoded, _ = fn.decoders.audio(audio_raw, sample_rate=cfg.sample_rate, downmix=False)
                start_s = start_f_f32 / cfg.fps
                shape_samples = (cfg.sequence_length - 1) * cfg.hop_length + cfg.window_length
                sliced = fn.slice(decoded.gpu(), start=fn.cast(start_s * cfg.sample_rate, dtype=types.INT32), shape=int(shape_samples), axes=[0], out_of_bounds_policy="pad")
                
                # 2. Use slicing syntax (which calls fn.slice) to separate the channels.
                # This is the correct DALI idiom.
                left = sliced[:, 0]
                right = sliced[:, 1]

                # 3. Process each 1D channel separately
                def to_mel_db(channel_1d):
                    # The channel is already 1D, so the squeeze is no longer needed.
                    spec = fn.spectrogram(channel_1d, nfft=cfg.nfft, window_length=cfg.window_length, window_step=cfg.hop_length, center_windows=False)
                    mel = fn.mel_filter_bank(spec, sample_rate=cfg.sample_rate, nfilter=cfg.mel_bins, freq_high=cfg.mel_fmax)
                    db = fn.to_decibels(mel, cutoff_db=cfg.db_cutoff)
                    return fn.transpose(db, perm=[1, 0]) # Transpose to [time, n_mels]

                mel_db_left = to_mel_db(left)
                mel_db_right = to_mel_db(right)

                # 4. Stack the results back together on a new channel axis
                mel_db = fn.stack(mel_db_left, mel_db_right, axis=0) # [2, time, n_mels]

                outputs.extend([frames, sample_id_i32, mel_db])
            return tuple(outputs)
        p = pipe(); p.build(); return p
    
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
    eco_mask: np.ndarray               # [T, 5, 6] uint64
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
        eco_mask = np.zeros((T, 5, 6), dtype=np.uint64)
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

                # --- START MODIFICATION ---
                # Reconstruct correct player indices from the alive bitmask
                # and map them to the sparse player_data list.
                pdl = payload.get("player_data")
                if pdl:
                    alive_indices = [i for i in range(5) if (mask_bits >> i) & 1]
                    
                    # Sanity check: the number of alive players must match the data list length
                    if len(alive_indices) != len(pdl):
                        # This can happen if data is corrupt, log and skip frame.
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
                # --- END MODIFICATION ---

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


    def assemble(self, dali_batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Assembles the final batch dictionary, including ground truth targets."""
        # 1) Stack DALI outputs
        images = torch.stack([dali_batch[f"pov{k}"] for k in range(5)], dim=2)

        # The mel spectrograms now have a stereo channel dimension. We must stack
        # and permute them correctly to match the model's expected input shape.
        mels_list = [dali_batch[f"mel{k}"] for k in range(5)]

        # mel_k shape: [B, 2, T, Mels] (Batch, Channels, Time, Mel Bins)
        # Stack along the player dimension (P) -> [B, 2, 5, T, Mels]
        mel_stacked = torch.stack(mels_list, dim=2)

        # Permute to the model's expected order: [B, T, P, C, Mels]
        mel_permuted = mel_stacked.permute(0, 3, 2, 1, 4)
        
        # Unsqueeze to add the final singleton dimension for the CNN filter.
        # Final shape: [B, T, P, C, Mels, 1]
        mel = mel_permuted.unsqueeze(-1)
        
        # 2) Fetch metadata and ground truth labels from LMDB
        sample_ids = dali_batch["labels0"].view(-1).cpu().tolist()
        
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

        # 3) Stack all ground truth lists into batch tensors
        gt_tensors = {k: torch.stack(v, dim=0).to(self.device) for k, v in gt_lists.items()}

        # 4) Assemble final batch dictionary
        batch = {
            "images": images.to(self.device),
            "mel_spectrogram": mel.to(self.device, dtype=images.dtype),
            "alive_mask": gt_tensors.pop("alive_mask").bool(),
            "meta": meta,
        }

        # 5) Assemble the 'targets' dictionary, converting masks to multi-hot
        targets = {"player": [{} for _ in range(5)], "game_strategy": {}}
        for i in range(5):
            targets["player"][i]["stats"] = gt_tensors["stats"][:, :, i]
            targets["player"][i]["mouse_delta_deg"] = gt_tensors["mouse_delta"][:, :, i]
            targets["player"][i]["pos_coords"] = gt_tensors["position"][:, :, i] # Raw coords for loss fn
            targets["player"][i]["keyboard_logits"] = self._masks_to_multi_hot(gt_tensors["keyboard_mask"][:, :, i], 31)

            # Eco Logits (6x uint64 -> 384 bits, truncated to 224 classes)
            eco_mask_player = gt_tensors["eco_mask"][:, :, i]  # Shape: [B, T, 6]
            eco_parts = [self._masks_to_multi_hot(eco_mask_player[:, :, k], 64) for k in range(6)]
            full_eco_logits = torch.cat(eco_parts, dim=-1)  # Shape: [B, T, 384]
            targets["player"][i]["eco_logits"] = full_eco_logits[..., :224]

            # Inventory Logits (2x uint64 -> 128 bits)
            inv_mask_player = gt_tensors["inventory_mask"][:, :, i]  # Shape: [B, T, 2]
            inv_parts = [self._masks_to_multi_hot(inv_mask_player[:, :, k], 64) for k in range(2)]
            targets["player"][i]["inventory_logits"] = torch.cat(inv_parts, dim=-1) # Shape: [B, T, 128]
            # --- END FIX ---

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
    """Calculates the composite, masked loss across all model prediction heads."""
    def __init__(self, weights: dict):
        super().__init__()
        self.weights = weights
        self.coord_mapper = CoordinateMapper(grid_dims=(64, 64, 8))
        
        # Initialize loss functions with no reduction to allow for manual masking
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)

    def _scalar_loss(self, unmasked_loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Applies a mask and reduces the loss to a scalar."""
        masked_loss = unmasked_loss * mask
        return masked_loss.sum() / mask.sum().clamp(min=1.0)

    def forward(self, predictions: Dict, targets: Dict, alive_mask: torch.Tensor) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=alive_mask.device)
        B, T, _ = alive_mask.shape

        # --- Player Losses (No changes here) ---
        for i in range(5):
            p_pred = predictions["player"][i]
            p_targ = targets["player"][i]
            player_alive_mask = alive_mask[:, :, i].float()

            loss_stats = self.mse_loss(p_pred["stats"], p_targ["stats"]).mean(dim=-1)
            total_loss += self.weights['stats'] * self._scalar_loss(loss_stats, player_alive_mask)
            loss_mouse = self.mse_loss(p_pred["mouse_delta_deg"], p_targ["mouse_delta_deg"]).mean(dim=-1)
            total_loss += self.weights['mouse'] * self._scalar_loss(loss_mouse, player_alive_mask)
            for key in ["keyboard_logits", "eco_logits", "inventory_logits"]:
                loss_bce = self.bce_loss(p_pred[key], p_targ[key]).mean(dim=-1)
                total_loss += self.weights[key.replace('_logits','')] * self._scalar_loss(loss_bce, player_alive_mask)
            pred_flat = p_pred["active_weapon_logits"].view(B * T, -1)
            targ_flat = p_targ["active_weapon_logits"].view(B * T)
            loss_weapon_unmasked = self.ce_loss(pred_flat, targ_flat)
            total_loss += self.weights['weapon'] * self._scalar_loss(loss_weapon_unmasked, player_alive_mask.view(B * T))

        # --- Game Strategy Losses ---
        gs_pred = predictions["game_strategy"]
        gs_targ = targets["game_strategy"]
        frame_mask = alive_mask.any(dim=-1).float()
        loss_round_num = self.mse_loss(gs_pred["round_number"], gs_targ["round_number"].view(B, T, 1)).squeeze(-1)
        total_loss += self.weights['round_number'] * self._scalar_loss(loss_round_num, frame_mask)
        
        # Round State (BCE)
        loss_round_state = self.bce_loss(gs_pred["round_state_logits"], gs_targ["round_state_logits"]).mean(dim=-1)
        total_loss += self.weights['round_state'] * self._scalar_loss(loss_round_state, frame_mask)

        # --- Heatmap Losses (BCE with smooth targets) ---
        # Player Position
        # Stack predictions and targets from the list of players into single tensors
        pred_pos_heatmaps = torch.stack([p["pos_heatmap_logits"] for p in predictions["player"]], dim=2) # Shape: [B, T, 5, Z, Y, X]
        targ_pos_coords = torch.stack([p["pos_coords"] for p in targets["player"]], dim=2)             # Shape: [B, T, 5, 3]

        # Use the boolean alive_mask to select only the valid entries.
        if alive_mask.any():
            # Flatten the batch, time, and player dimensions to get a simple list
            alive_mask_flat = alive_mask.view(-1)
            
            # Select the predictions and targets for alive players, preserving the heatmap shape
            pred_pos_logits_alive = pred_pos_heatmaps.view(-1, *pred_pos_heatmaps.shape[3:])[alive_mask_flat]
            targ_pos_coords_alive = targ_pos_coords.view(-1, 3)[alive_mask_flat]

            # Generate the target heatmap, which will have shape [num_alive, Z, Y, X]
            target_grid_indices = self.coord_mapper.discretize_world_to_grid(targ_pos_coords_alive)
            target_heatmap = create_gaussian_heatmap_target(target_grid_indices, grid_dims=(64, 64, 8), sigma=1.5)
            
            loss_pos = self.bce_loss(pred_pos_logits_alive, target_heatmap).mean()
            total_loss += self.weights['pos_heatmap'] * loss_pos

        # Enemy Position
        targ_enemy_coords = gs_targ["enemy_pos_coords"] # Shape: [B, T, 5, 3]
        pred_enemy_heatmaps = gs_pred["enemy_pos_heatmap_logits"] # Shape: [B, T, Z, Y, X]
        
        # Filter out invalid target positions (often all zeros for dead/unseen enemies)
        valid_enemy_mask = targ_enemy_coords.abs().sum(dim=-1) > 0 # Shape: [B, T, 5]
        
        if valid_enemy_mask.any():
            # Select the valid ground truth coordinates
            valid_targ_coords = targ_enemy_coords[valid_enemy_mask] # Shape: [num_valid_enemies, 3]

            # Expand the single predicted heatmap to match the 5 potential enemy targets
            # then select only the predictions corresponding to valid targets
            expanded_preds = pred_enemy_heatmaps.unsqueeze(2).expand(-1, -1, 5, -1, -1, -1) # Shape: [B, T, 5, Z, Y, X]
            valid_pred_logits = expanded_preds[valid_enemy_mask] # Shape: [num_valid_enemies, Z, Y, X]
            
            # Generate target heatmaps
            target_grid_indices = self.coord_mapper.discretize_world_to_grid(valid_targ_coords)
            target_heatmap = create_gaussian_heatmap_target(target_grid_indices, grid_dims=(64, 64, 8), sigma=1.5)
            
            # The shapes now match: both are [num_valid_enemies, 8, 64, 64]
            loss_enemy_pos = self.bce_loss(valid_pred_logits, target_heatmap).mean()
            total_loss += self.weights['enemy_heatmap'] * loss_enemy_pos
            
        return total_loss
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

@dataclass
class DataArgs:
    data_root: str; manifest: str; split: str = "train"; run_dir: str = "runs/exp1"
    T_frames: int = 64; height: int = 224; width: int = 224
    batch_size: int = 1; seed: int = 42; dali_threads: int = 4

def build_data_iter(args: DataArgs):
    """Convenience function that wires Steps 1→8 and returns (dali_iter, assembler)."""
    # Steps 1-4: Discover rounds, sample windows, write filelists
    manifest = Manifest(args.data_root, args.manifest)
    store = LmdbStore()
    team_rounds = build_team_rounds(args.data_root, manifest.get_games(args.split), store)
    index = EpochIndex(T_frames=args.T_frames, seed=args.seed)
    records, id_map = index.build(team_rounds, epoch=0)
    fl_dir = os.path.join(args.run_dir, "epoch_0")
    video_lists, audio_lists = FilelistWriter(fl_dir).write(records)
    
    # Step 5: DALI
    shard_id, num_shards = get_ddp_info()
    dali_cfg = DaliConfig(
        height=args.height, width=args.width, sequence_length=args.T_frames,
        batch_size=args.batch_size, num_threads=args.dali_threads,
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else 0,
        shard_id=shard_id, num_shards=num_shards,
    )
    dali_iter = DaliInputPipeline(video_lists, audio_lists, dali_cfg)

    # Step 7-8: Metadata Fetcher and Assembler
    fetcher = LmdbMetaFetcher(store)
    assembler = BatchAssembler(id_map, fetcher, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return dali_iter, assembler

if __name__ == "__main__":
    import contextlib
    from model import CS2Transformer, CS2Config
    
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    data_root = os.environ.get("DATA_ROOT", "data")
    manifest_path = os.path.join(data_root, "manifest.json")
    args = DataArgs(data_root=data_root, manifest=manifest_path, batch_size=2, T_frames=32)

    if not DALI_AVAILABLE:
        logging.error("DALI is not available: %s", _DALI_IMPORT_ERROR)
    else:
        try:
            dali_iter, assembler = build_data_iter(args)
            
            # --- SMOKE TEST SETUP ---
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_cfg = CS2Config(context_frames=args.T_frames)
            model = CS2Transformer(model_cfg, use_dummy_vision=True).to(device) # Use dummy vision for speed
            
            loss_weights = {
                'stats': 1.0, 'mouse': 5.0, 'keyboard': 0.5, 'eco': 0.5,
                'inventory': 0.5, 'weapon': 1.0, 'round_number': 0.1,
                'round_state': 1.0, 'pos_heatmap': 2.0, 'enemy_heatmap': 2.0
            }
            loss_fn = CompositeLoss(weights=loss_weights).to(device)
            # --- END SMOKE TEST SETUP ---

            for i in range(5):
                try:
                    with Timer("dali_fetch") as t:
                        batch_raw = next(iter(dali_iter))
                    logging.info("DALI fetched video+audio in %.3fs", t.dt)
                    
                    with Timer("assemble") as t2:
                        batch = assembler.assemble(batch_raw)
                    logging.info("Assembled batch in %.3fs", t2.dt)
                    
                    # --- SMOKE TEST FORWARD PASS AND LOSS CALCULATION ---
                    with Timer("forward_pass") as t3, torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                        predictions = model(batch)
                    logging.info("Forward pass in %.3fs", t3.dt)
                    
                    with Timer("loss_calc") as t4:
                        total_loss = loss_fn(predictions, batch['targets'], batch['alive_mask'])
                    logging.info("Calculated loss in %.3fs -> Total Loss: %.4f", t4.dt, total_loss.item())
                    
                    # Smoke assertion
                    assert total_loss.requires_grad, "Loss must require gradients"
                    logging.info("✅ Smoke assertion passed: Loss requires grad.")

                except StopIteration:
                    logging.info("Finished iterator.")
                    break
        except Exception as e:
            logging.exception("Data loader smoke test failed.")
            logging.error("Failed with error: %s", e)