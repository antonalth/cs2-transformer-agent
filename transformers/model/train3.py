#!/usr/bin/env python3
"""
train3.py — A complete training script for the CS2Transformer model.

This script orchestrates the entire training process, including:
  - Distributed training with DDP.
  - A dual-path data pipeline (on-the-fly or precomputed) using DALI.
  - Per-epoch window resampling for data augmentation.
  - AdamW optimizer with a warmup-cosine learning rate schedule.
  - Support for FP16/BF16 mixed-precision training with gradient scaling.
  - Gradient accumulation and clipping.
  - Checkpointing for model, optimizer, scheduler, and RNG state to enable resumes.
  - Periodic validation and logging to TensorBoard.
  - Optional Exponential Moving Average (EMA) of model weights for stabler evaluation.
"""
from __future__ import annotations

import os
import re
import gc
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

import lmdb
import msgpack
import numpy as np
import msgpack_numpy as mpnp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist


# --- Try to import DALI ---
try:
    from nvidia.dali import fn, types, pipeline_def
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from nvidia.dali.plugin.base_iterator import LastBatchPolicy
    DALI_AVAILABLE = True
except Exception as _e:
    DALI_AVAILABLE = False
    _DALI_IMPORT_ERROR = _e

# --- Import from project files ---
from model import CS2Transformer, CS2Config

# =============================================================================
# SECTION 1: DATA LOADING CORE (Unchanged from original smoke test)
# =============================================================================

TICK_RATE = 64
FPS = 32
TICKS_PER_FRAME = TICK_RATE // FPS
LABEL_SCALE = 1_000_000
TICK_RE = re.compile(r"_([0-9]+)_([0-9]+)\.(mp4|wav)$")

def ticks_from_filename(path: str) -> tuple[int, int] | None:
    m = TICK_RE.search(os.path.basename(path))
    return (int(m.group(1)), int(m.group(2))) if m else None

def ticks_to_frames(start_tick: int, end_tick: int) -> int:
    if end_tick < start_tick: return 0
    return ((end_tick - start_tick) // TICKS_PER_FRAME) + 1

def clamp_window_to_pov(req_start_f: int, T_frames: int, pov_start_tick: int, pov_end_tick: int) -> tuple[int, int]:
    pov_frame_count = ticks_to_frames(pov_start_tick, pov_end_tick)
    if pov_frame_count <= 0: return -1, -1
    start_f = min(req_start_f, pov_frame_count - 1)
    end_f_exclusive = min(start_f + T_frames, pov_frame_count)
    return start_f, end_f_exclusive

@dataclass
class MapBoundaries:
    WORLD_MIN: tuple[float, float, float] = (-3000.0, -3500.0, -500.0)
    WORLD_MAX: tuple[float, float, float] = (3500.0, 2500.0, 1000.0)

class CoordinateMapper:
    def __init__(self, grid_dims: tuple[int, int, int] = (64, 64, 8)):
        self.grid_dims = torch.tensor(grid_dims, dtype=torch.long)
        self.world_min = torch.tensor(MapBoundaries.WORLD_MIN, dtype=torch.float32)
        self.world_max = torch.tensor(MapBoundaries.WORLD_MAX, dtype=torch.float32)
        self.world_range = self.world_max - self.world_min

    def discretize_world_to_grid(self, world_coords: torch.Tensor) -> torch.Tensor:
        device = world_coords.device
        self.world_min = self.world_min.to(device)
        self.world_max = self.world_max.to(device)
        self.world_range = self.world_range.to(device)
        self.grid_dims = self.grid_dims.to(device)
        coords = torch.max(torch.min(world_coords, self.world_max), self.world_min)
        normalized_coords = (coords - self.world_min) / self.world_range
        grid_indices = (normalized_coords * (self.grid_dims - 1e-6)).long()
        return grid_indices

@dataclass
class TeamRound:
    demoname: str; lmdb_path: str; round_num: int; team: str
    start_tick: int; end_tick: int; pov_videos: List[str]; pov_audio: List[str]
    fps: int = FPS; tick_rate: int = TICK_RATE
    @property
    def frame_count(self) -> int: return ticks_to_frames(self.start_tick, self.end_tick)

@dataclass
class SampleRecord:
    sample_id: int; demoname: str; lmdb_path: str; round_num: int; team: str
    pov_videos: List[str]; pov_audio: List[str]; start_f: int
    start_tick_win: int; T_frames: int

class Manifest:
    def __init__(self, data_root: str, manifest_path: str):
        self.data_root = os.path.abspath(data_root)
        with open(manifest_path, "r", encoding="utf-8") as f: self._data = json.load(f)

    def get_games(self, split: str) -> List[Tuple[str, str]]:
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
    def __init__(self, max_readers: int = 512):
        self._envs: Dict[str, lmdb.Environment] = {}
        self._info_cache: Dict[str, Dict[str, Any]] = {}
        self._max_readers = max_readers
    def open(self, lmdb_path: str) -> lmdb.Environment:
        if lmdb_path not in self._envs:
            self._envs[lmdb_path] = lmdb.open(lmdb_path, readonly=True, lock=False, max_readers=self._max_readers, readahead=True)
        return self._envs[lmdb_path]
    def read_info(self, demoname: str, lmdb_path: str) -> Dict[str, Any]:
        cache_key = (demoname, lmdb_path)
        if cache_key in self._info_cache: return self._info_cache[cache_key]
        with self.open(lmdb_path).begin(write=False) as txn:
            blob = txn.get(f"{demoname}_INFO".encode("utf-8"))
            if blob is None: raise FileNotFoundError(f"Missing _INFO for {demoname}")
            info = json.loads(blob.decode("utf-8"))
        self._info_cache[cache_key] = info
        return info

def build_team_rounds(data_root: str, games: List[Tuple[str, str]], store: LmdbStore) -> List[TeamRound]:
    team_rounds: List[TeamRound] = []
    for demoname, lmdb_path in games:
        info = store.read_info(demoname, lmdb_path)
        for r in info["rounds"]:
            if len(r.get("pov_videos", [])) != 5 or len(r.get("pov_audio", [])) != 5: continue
            def _resolve(p): return os.path.abspath(os.path.join(data_root, "recordings", demoname, p))
            videos, audio = [_resolve(pv) for pv in r["pov_videos"]], [_resolve(pa) for pa in r["pov_audio"]]
            if not all(os.path.exists(p) for p in videos + audio):
                logging.warning(f"Skipping {demoname}/{r['round_num']}: missing media file.")
                continue
            team_rounds.append(TeamRound(demoname=demoname, lmdb_path=lmdb_path, round_num=int(r["round_num"]),
                                       team=str(r["team"]).upper(), start_tick=int(r["start_tick"]), end_tick=int(r["end_tick"]),
                                       pov_videos=videos, pov_audio=audio))
    logging.info("Discovered %d team-rounds for split.", len(team_rounds))
    return team_rounds

class EpochIndex:
    def __init__(self, T_frames: int, seed: int, windows_per_round: int = 1):
        self.T_frames = T_frames
        self.seed = seed
        self.windows_per_round = max(1, int(windows_per_round))

    def build(self, team_rounds: List[TeamRound], epoch: int) -> Tuple[List[SampleRecord], Dict[int, SampleRecord]]:
        rnd = random.Random(self.seed + epoch)
        records, id_map = [], {}
        sid = 0
        for tr in team_rounds:
            for _ in range(self.windows_per_round):
                # random start (clamped if shorter than T)
                if tr.frame_count >= self.T_frames:
                    start_f = rnd.randint(0, tr.frame_count - self.T_frames)
                else:
                    start_f = 0
                rec = SampleRecord(
                    sample_id=sid,
                    demoname=tr.demoname, lmdb_path=tr.lmdb_path,
                    round_num=tr.round_num, team=tr.team,
                    pov_videos=tr.pov_videos, pov_audio=tr.pov_audio,
                    start_f=start_f,
                    start_tick_win=tr.start_tick + TICKS_PER_FRAME * start_f,
                    T_frames=self.T_frames
                )
                records.append(rec)
                id_map[sid] = rec
                sid += 1
        return records, id_map

class FilelistWriter:
    def __init__(self, out_dir: str, use_precomputed: bool = False, data_root: str = ""):
        self.out_dir, self.use_precomputed = out_dir, use_precomputed
        self.data_root = os.path.abspath(data_root) if data_root else ""
        os.makedirs(out_dir, exist_ok=True)
        if use_precomputed and not self.data_root:
            raise ValueError("data_root must be provided for precomputed mode.")

    def write(self, records: List[SampleRecord]) -> Tuple[List[str], List[str]]:
        """
        For precomputed mode:
        - video filelists: label = sequential index (0..N-1)
        - audio filelists: label = start_f (int frames)
        - saves sidecar: {seq_id -> (sample_id, start_f)} as label_map.npz

        For on-the-fly (raw media) mode:
        - video filelists: path sample_id start frame_num  (unchanged)
        - audio filelists: label = start_f  (was packed; now plain)
        """
        vid_paths = [os.path.join(self.out_dir, f"pov{k}_video.txt") for k in range(5)]
        aud_paths = [os.path.join(self.out_dir, f"pov{k}_audio.txt") for k in range(5)]

        def _embed_path(kind, demo, base):
            p = Path(self.data_root) / ("vit_embed" if kind == "vit" else "aud_embed") / demo / f"{base}.npy"
            if not p.is_file():
                raise FileNotFoundError(f"Missing precomputed embedding: {p}")
            return p

        # sidecar arrays (one row per record, in the same order)
        sidecar_seq   = []
        sidecar_sid   = []
        sidecar_start = []

        import contextlib
        import numpy as np

        with contextlib.ExitStack() as stack:
            files = [stack.enter_context(open(p, "w")) for p in vid_paths + aud_paths]
            vid_fs, aud_fs = files[:5], files[5:]

            for seq_id, rec in enumerate(records):
                # keep sidecar in the same order
                sidecar_seq.append(seq_id)
                sidecar_sid.append(rec.sample_id)
                sidecar_start.append(rec.start_f)

                if self.use_precomputed:
                    # Labels:
                    #   - video label  = seq_id (0..N-1)
                    #   - audio label  = start_f (frames)
                    for k in range(5):
                        vid_base = os.path.splitext(os.path.basename(rec.pov_videos[k]))[0]
                        aud_base = os.path.splitext(os.path.basename(rec.pov_audio[k]))[0]
                        vid_fs[k].write(f"{_embed_path('vit', rec.demoname, vid_base)} {seq_id}\n")
                        aud_fs[k].write(f"{_embed_path('aud', rec.demoname, aud_base)} {rec.start_f}\n")
                else:
                    # Video filelist line format stays: path sample_id start frame_num
                    for k in range(5):
                        path = rec.pov_videos[k]
                        ticks = ticks_from_filename(path)
                        start, end_exc = clamp_window_to_pov(rec.start_f, rec.T_frames, ticks[0], ticks[1])
                        frame_num = max(0, end_exc - start)
                        vid_fs[k].write(f"{path} {rec.sample_id} {start} {frame_num}\n")

                        # Audio label becomes plain start_f (no packing)
                        aud_fs[k].write(f"{rec.pov_audio[k]} {start}\n")

        # Save sidecar map for debugging / audits: seq_id -> (sample_id, start_f)
        sidecar = {
            "seq_id":   np.asarray(sidecar_seq,   dtype=np.int64),
            "sample_id":np.asarray(sidecar_sid,   dtype=np.int64),
            "start_f":  np.asarray(sidecar_start, dtype=np.int64),
        }
        np.savez(os.path.join(self.out_dir, "label_map.npz"), **sidecar)

        return vid_paths, aud_paths


@dataclass
class DaliConfig:
    sequence_length: int = 128; fps: float = 32.0; sample_rate: float = 24000.0; n_mels: int = 128
    n_fft: int = 1024; win_length: int = 750; hop_length: int = 750; batch_size: int = 1; num_threads: int = 4
    device_id: int = 0; shard_id: int = 0; num_shards: int = 1; seed: int = 42
    mel_bins: int = field(init=False); nfft: int = field(init=False); window_length: int = field(init=False)
    mel_fmax: float = field(init=False); db_cutoff: float = 80.0
    def __post_init__(self):
        self.mel_bins, self.nfft, self.window_length = self.n_mels, self.n_fft, self.win_length
        self.mel_fmax = self.sample_rate / 2.0

class DaliInputPipeline:
    def __init__(self, video_filelists, audio_filelists, cfg, use_precomputed, video_pathlists=None, audio_pathlists=None, last_batch_policy="drop"):
        if not DALI_AVAILABLE: raise ImportError(f"NVIDIA DALI not available: {_DALI_IMPORT_ERROR}")

        if use_precomputed:
            self.pipeline = self._build_npy_pipeline(video_filelists, audio_filelists, video_pathlists, audio_pathlists, cfg)
            out_map = [f"video_embed{k}" for k in range(5)] + [f"audio_embed{k}" for k in range(5)] + ["labels0"]
            reader_name = "P0_VidLblReader"
        else:
            self.pipeline = self._build_media_pipeline(video_filelists, audio_filelists, cfg)
            out_map = [item for k in range(5) for item in (f"pov{k}", f"labels{k}", f"mel{k}")]
            reader_name = "V0"

        self.iterator = DALIGenericIterator(
            [self.pipeline], out_map, reader_name=reader_name, auto_reset=True,
            last_batch_policy=LastBatchPolicy.DROP if last_batch_policy == "drop" else LastBatchPolicy.PARTIAL
        )

    def _build_npy_pipeline(self, vlists, alists, vpaths, apaths, cfg):
        @pipeline_def(batch_size=cfg.batch_size, num_threads=cfg.num_threads, device_id=cfg.device_id, seed=cfg.seed)
        def pipe():
            def read_slice(vlist_lbl, vpath_only, alist_lbl, apath_only, reader_prefix):
                # Read the sequential ID from the video label list
                _, seq_label = fn.readers.file(
                    name=f"{reader_prefix}_VidLblReader",
                    file_list=vlist_lbl,
                    shard_id=cfg.shard_id, num_shards=cfg.num_shards,
                    stick_to_shard=True, random_shuffle=False, shuffle_after_epoch=False
                )

                # Read start_f (frames) from the audio label list
                _, start_label = fn.readers.file(
                    name=f"{reader_prefix}_AudLblReader",
                    file_list=alist_lbl,
                    shard_id=cfg.shard_id, num_shards=cfg.num_shards,
                    stick_to_shard=True, random_shuffle=False, shuffle_after_epoch=False
                )

                # Load npy embeddings from path-only lists (keeps label IO separate)
                v_raw = fn.readers.numpy(
                    name=f"{reader_prefix}_VidEmbedReader",
                    file_list=vpath_only,
                    shard_id=cfg.shard_id, num_shards=cfg.num_shards,
                    stick_to_shard=True, random_shuffle=False, shuffle_after_epoch=False
                )
                a_raw = fn.readers.numpy(
                    name=f"{reader_prefix}_AudEmbedReader",
                    file_list=apath_only,
                    shard_id=cfg.shard_id, num_shards=cfg.num_shards,
                    stick_to_shard=True, random_shuffle=False, shuffle_after_epoch=False
                )

                # Cast labels
                seq_id  = fn.cast(seq_label,   dtype=types.INT32)
                start_f = fn.cast(start_label, dtype=types.INT32)

                # Slice the per-frame embeddings by window start
                v_slice = fn.slice(v_raw, start_f, cfg.sequence_length, axes=[0], out_of_bounds_policy="pad", fill_values=0.0)
                a_slice = fn.slice(a_raw, start_f, cfg.sequence_length, axes=[0], out_of_bounds_policy="pad", fill_values=0.0)

                # Return slices + the seq_id (this is what lands in `labels0`)
                return v_slice.gpu(), a_slice.gpu(), seq_id


            v_embeds, a_embeds = [], []
            v0, a0, label0 = read_slice(vlists[0], vpaths[0], alists[0], apaths[0], "P0")
            v_embeds.append(v0); a_embeds.append(a0)
            for k in range(1, 5):
                vk, ak, _ = read_slice(vlists[k], vpaths[k], alists[k], apaths[k], f"P{k}")
                v_embeds.append(vk); a_embeds.append(ak)
            return (*v_embeds, *a_embeds, label0)
        p = pipe(); p.build(); return p

    def _build_media_pipeline(self, vlists, alists, cfg):
        @pipeline_def(batch_size=cfg.batch_size, num_threads=cfg.num_threads, device_id=cfg.device_id, seed=cfg.seed)
        def pipe():
            outputs = []
            for k in range(5):
                # existing readers
                video, label_vid = fn.readers.video(
                    name=f"V{k}", file_list=vlists[k], sequence_length=cfg.sequence_length, pad_sequences=True,
                    shard_id=cfg.shard_id, num_shards=cfg.num_shards,
                    stick_to_shard=True, random_shuffle=False, shuffle_after_epoch=False,
                    dtype=types.UINT8, file_list_frame_num=True, file_list_include_preceding_frame=True
                )
                audio_raw, label_cpu = fn.readers.file(
                    name=f"A{k}", file_list=alists[k],
                    shard_id=cfg.shard_id, num_shards=cfg.num_shards,
                    stick_to_shard=True, random_shuffle=False, shuffle_after_epoch=False
                )

                # NEW: start_f comes straight from the audio label (no packing)
                start_f = fn.cast(label_cpu, dtype=types.INT32)

                decoded, _ = fn.decoders.audio(audio_raw, sample_rate=cfg.sample_rate, downmix=False)

                # Convert frame start to seconds for audio slicing
                start_s = fn.cast(start_f, dtype=types.FLOAT) / cfg.fps
                shape_samples = (cfg.sequence_length - 1) * cfg.hop_length + cfg.window_length

                sliced = fn.slice(
                    decoded.gpu(),
                    start=fn.cast(start_s * cfg.sample_rate, dtype=types.INT32),
                    shape=[int(shape_samples)],
                    axes=[0],
                    out_of_bounds_policy="pad"
                )

                # ... mel construction unchanged ...

                def to_mel_db(ch):
                    spec = fn.spectrogram(ch, nfft=cfg.nfft, window_length=cfg.window_length, window_step=cfg.hop_length, center_windows=False)
                    mel = fn.mel_filter_bank(spec, sample_rate=cfg.sample_rate, nfilter=cfg.mel_bins, freq_high=cfg.mel_fmax)
                    return fn.transpose(fn.to_decibels(mel, cutoff_db=cfg.db_cutoff), perm=[1, 0])
                mel = fn.stack(to_mel_db(sliced[:, 0]), to_mel_db(sliced[:, 1]), axis=0)
                outputs.extend([fn.transpose(video, perm=[0, 3, 1, 2]), label_vid, mel])
            return tuple(outputs)
        p = pipe(); p.build(); return p
    def __iter__(self): return self
    def __next__(self): return next(self.iterator)

@dataclass
class MetaFetchResult:
    alive_mask: np.ndarray; stats: np.ndarray; mouse_delta: np.ndarray; position: np.ndarray; keyboard_mask: np.ndarray
    eco_mask: np.ndarray; inventory_mask: np.ndarray; active_weapon_idx: np.ndarray; round_number: np.ndarray
    round_state_mask: np.ndarray; enemy_positions: np.ndarray

class LmdbMetaFetcher:
    def __init__(self, store: LmdbStore): self.store = store
    @staticmethod
    def _key(d, r, t, tick): return f"{d}_round_{r:03d}_team_{t}_tick_{tick:08d}".encode("utf-8")
    @staticmethod
    def _bitmask_to_weapon_index(mask: np.ndarray) -> int:
        """Converts a [2] uint64 weapon bitmask to a single item index."""
        if mask.sum() == 0: return -1
        for i in range(128):
            if (mask[i // 64] >> np.uint64(i % 64)) & np.uint64(1):
                return i
        return -1
    def fetch(self, rec: SampleRecord) -> MetaFetchResult:
        env, T = self.store.open(rec.lmdb_path), rec.T_frames
        alive = np.zeros((T, 5), dtype=np.bool_); stats = np.zeros((T, 5, 3), np.float32)
        mouse = np.zeros((T, 5, 2), np.float32); pos = np.zeros((T, 5, 3), np.float32)
        kbd = np.zeros((T, 5), np.uint32); eco = np.zeros((T, 5, 4), np.uint64)
        inv = np.zeros((T, 5, 2), np.uint64); wep = np.full((T, 5), -1, np.int32)
        rnd_num = np.full((T,), rec.round_num, np.int32); rnd_state = np.zeros((T,), np.uint8)
        enemy_pos = np.zeros((T, 5, 3), np.float32)
        ticks = rec.start_tick_win + (np.arange(T, dtype=np.int32) * TICKS_PER_FRAME)
        with env.begin(write=False) as txn:
            for f, tick in enumerate(ticks):
                blob = txn.get(self._key(rec.demoname, rec.round_num, rec.team, int(tick)))
                if not blob: continue
                payload = msgpack.unpackb(blob, raw=False, object_hook=mpnp.decode)
                if not payload.get("game_state"): continue
                gs = payload["game_state"][0]
                alive_slots = [i for i in range(5) if (int(gs['team_alive']) >> i) & 1]
                for slot in alive_slots: alive[f, slot] = True
                rnd_state[f] = gs['round_state']; enemy_pos[f] = gs['enemy_pos']
                pdl = payload.get("player_data")
                if pdl and len(alive_slots) == len(pdl):
                    for p_idx, p_data_arr in zip(alive_slots, pdl):
                        p = p_data_arr[0]
                        stats[f, p_idx] = [p['health'], p['armor'], p['money']]; mouse[f, p_idx] = p['mouse']
                        pos[f, p_idx] = p['pos']; kbd[f, p_idx] = p['keyboard_bitmask']; eco[f, p_idx] = p['eco_bitmask']
                        inv[f, p_idx] = p['inventory_bitmask']; wep[f, p_idx] = self._bitmask_to_weapon_index(p['active_weapon_bitmask'])
        return MetaFetchResult(alive, stats, mouse, pos, kbd, eco, inv, wep, rnd_num, rnd_state, enemy_pos)

class BatchAssembler:
    def __init__(self, id_to_sample, fetcher, device):
        self.id_to_sample, self.fetcher, self.device = id_to_sample, fetcher, device
        self.MOUSE_DELTA_MEAN = torch.tensor([0.009522, -0.000312], device=device)
        self.MOUSE_DELTA_STD = torch.tensor([3.305156, 0.649809], device=device)
    @staticmethod
    def _masks_to_multi_hot(masks, num_classes):
        powers = torch.arange(num_classes, device=masks.device, dtype=torch.long).view(1, -1)
        return ((masks.long().unsqueeze(-1) >> powers) & 1).float()
    def assemble(self, dali_batch, use_precomputed):
        if use_precomputed:
            video_embeddings = torch.stack([dali_batch[f"video_embed{k}"] for k in range(5)], dim=2)
            audio = torch.stack([dali_batch[f"audio_embed{k}"] for k in range(5)], dim=2).permute(0, 1, 2, 4, 3).unsqueeze(-1)
            sample_ids = dali_batch["labels0"].view(-1).cpu().tolist()
            batch = {"video_embeddings": video_embeddings, "mel_spectrogram": audio}
        else:
            images = torch.stack([dali_batch[f"pov{k}"] for k in range(5)], dim=2)
            mel_stacked = torch.stack([dali_batch[f"mel{k}"] for k in range(5)], dim=2)
            mel_permuted = mel_stacked.permute(0, 3, 2, 1, 4)
            mel = mel_permuted.unsqueeze(-1)
            sample_ids = dali_batch["labels0"].view(-1).cpu().tolist()
            batch = {"images": images, "mel_spectrogram": mel}
        gt_lists = defaultdict(list)
        for sid in sample_ids:
            gt_result = self.fetcher.fetch(self.id_to_sample[int(sid)])
            for key, value in gt_result.__dict__.items(): gt_lists[key].append(torch.from_numpy(value))
        gt = {k: torch.stack(v).to(self.device, non_blocking=True) for k,v in gt_lists.items()}
        # ---- robust alive_mask construction ----
        alive = gt.pop("alive_mask", None)

        if alive is None:
            # Try to reconstruct from a per-frame team_alive bitmask if present.
            team_alive_bits = None
            if "team_alive" in gt:
                team_alive_bits = gt["team_alive"]            # expected shape [B, T], int bitmask
            elif "game_state" in gt and isinstance(gt["game_state"], dict) and "team_alive" in gt["game_state"]:
                team_alive_bits = gt["game_state"]["team_alive"]

            if team_alive_bits is not None:
                tab = team_alive_bits.to(dtype=torch.long)
                # Convert [B, T] bitmask into [B, T, 5] boolean mask (one slot per player)
                ar = torch.arange(5, device=tab.device, dtype=torch.long)
                alive = ((tab.unsqueeze(-1) >> ar) & 1).bool()
            elif "stats" in gt:
                # Fallback: infer alive from health in stats [B, T, 5, 3] (health, armor, money)
                alive = (gt["stats"][..., 0] > 0)
            else:
                # Last-resort fallback: mark all frames alive (padding will be handled elsewhere)
                some = next(v for v in gt.values() if torch.is_tensor(v))
                B, T = some.shape[:2]
                alive = torch.ones((B, T, 5), dtype=torch.bool, device=some.device)

        batch["alive_mask"] = alive
        # ---- end robust alive_mask construction ----

        targets = {"player": [{} for _ in range(5)], "game_strategy": {}}
        for i in range(5):
            targets["player"][i]["stats"] = gt["stats"][:, :, i]
            targets["player"][i]["mouse_delta_deg"] = (gt["mouse_delta"][:, :, i] - self.MOUSE_DELTA_MEAN) / self.MOUSE_DELTA_STD
            targets["player"][i]["pos_coords"] = gt["position"][:, :, i]
            targets["player"][i]["keyboard_logits"] = self._masks_to_multi_hot(gt["keyboard_mask"][:, :, i], 31)
            eco_parts = [self._masks_to_multi_hot(gt["eco_mask"][:, :, i, k], 64) for k in range(4)]
            targets["player"][i]["eco_logits"] = torch.cat(eco_parts, dim=-1)[..., :224]
            inv_parts = [self._masks_to_multi_hot(gt["inventory_mask"][:, :, i, k], 64) for k in range(2)]
            targets["player"][i]["inventory_logits"] = torch.cat(inv_parts, dim=-1)
            targets["player"][i]["active_weapon_idx"] = gt["active_weapon_idx"][:, :, i].long()
        targets["game_strategy"]["enemy_pos_coords"] = gt["enemy_positions"]
        targets["game_strategy"]["round_state_logits"] = self._masks_to_multi_hot(gt["round_state_mask"], 5)
        targets["game_strategy"]["round_number"] = gt["round_number"].float()
        batch["targets"] = targets
        return batch

class CompositeLoss(nn.Module):
    def __init__(self, weights, grid_dims=(64,64,8), sigma=1.5):
        super().__init__()
        self.weights = weights
        self.coord_mapper = CoordinateMapper(grid_dims=grid_dims)
        self.mse_loss = nn.MSELoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=-1, label_smoothing=0.05)
        X, Y, Z = grid_dims
        self.register_buffer('xs', torch.arange(X, dtype=torch.float32), persistent=False)
        self.register_buffer('ys', torch.arange(Y, dtype=torch.float32), persistent=False)
        self.register_buffer('zs', torch.arange(Z, dtype=torch.float32), persistent=False)
        self.register_buffer("stats_scale", torch.tensor([1/100.0, 1/100.0, 1/16000.0]), persistent=False)
        self.grid_dims, self.sigma = grid_dims, float(sigma)
    @staticmethod
    def _scalar_loss(unmasked_loss, mask): return (unmasked_loss * mask).sum() / mask.sum().clamp(min=1.0)
    def _gaussian_heatmaps_from_indices(self, centers_xyz_idx):
        if centers_xyz_idx.numel() == 0:
            return torch.empty((0, self.grid_dims[2], self.grid_dims[1], self.grid_dims[0]), device=self.xs.device)
        centers = centers_xyz_idx.to(dtype=torch.float32, device=self.xs.device)
        gx, gy, gz = centers[:, 0], centers[:, 1], centers[:, 2]
        s2 = 2.0 * (self.sigma ** 2)
        hx = torch.exp(-((self.xs.unsqueeze(0) - gx.unsqueeze(1)) ** 2) / s2)
        hy = torch.exp(-((self.ys.unsqueeze(0) - gy.unsqueeze(1)) ** 2) / s2)
        hz = torch.exp(-((self.zs.unsqueeze(0) - gz.unsqueeze(1)) ** 2) / s2)
        return hz[:, :, None, None] * hy[:, None, :, None] * hx[:, None, None, :]
    def _build_targets_heatmaps(self, world_xyz):
        with torch.no_grad():
            grid_idx = self.coord_mapper.discretize_world_to_grid(world_xyz)
            return self._gaussian_heatmaps_from_indices(grid_idx)
    def forward(self, predictions, targets, alive_mask):
        device = self.xs.device; total_loss = torch.tensor(0.0, device=device); losses = {}
        B, T = predictions['player'][0]['stats'].shape[:2]; alive_mask = alive_mask.to(device)
        p_loss_keys = ['stats', 'mouse', 'keyboard', 'eco', 'inventory', 'weapon']
        for key in p_loss_keys: losses[key] = torch.tensor(0.0, device=device)
        for i in range(5):
            p_pred, p_targ = predictions["player"][i], targets["player"][i]
            mask = alive_mask[:, :, i].float()
            losses['stats'] += self.weights['stats'] * self._scalar_loss(self.mse_loss(p_pred["stats"] * self.stats_scale, p_targ["stats"] * self.stats_scale).mean(-1), mask)
            losses['mouse'] += self.weights['mouse'] * self._scalar_loss(self.mse_loss(p_pred["mouse_delta_deg"], p_targ["mouse_delta_deg"]).mean(-1), mask)
            
            # FIX #2: Use dynamically weighted BCE for imbalanced multi-label heads
            for key in ["keyboard_logits", "eco_logits", "inventory_logits"]:
                wkey = key.replace('_logits','')
                logits = p_pred[key]  # [B, T, C]
                labels = p_targ[key]  # [B, T, C]
                with torch.no_grad():
                    pos = labels.float().mean(dim=(0, 1))
                    pos_weight = ((1 - pos) / (pos + 1e-6)).clamp_(1.0, 10.0)
                bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction='none').mean(-1)
                losses[wkey] += self.weights[wkey] * self._scalar_loss(bce, mask)

            # FIX #6: Use dynamically weighted CE for imbalanced weapon classification
            targ_idx = p_targ["active_weapon_idx"].view(-1)
            num_classes = p_pred["active_weapon_idx"].shape[-1]
            with torch.no_grad():
                valid_indices = targ_idx[targ_idx != -1]
                if valid_indices.numel() > 0:
                    counts = torch.bincount(valid_indices, minlength=num_classes)
                    class_weights = 1.0 / (counts.float() + 1e-6)
                    class_weights = (class_weights / class_weights.sum()) * num_classes
                else:
                    class_weights = torch.ones(num_classes, device=device)
            ce_loss_weighted = nn.CrossEntropyLoss(reduction='none', ignore_index=-1, label_smoothing=0.05, weight=class_weights)
            losses['weapon'] += self.weights['weapon'] * self._scalar_loss(ce_loss_weighted(p_pred["active_weapon_idx"].view(B*T, -1), targ_idx), mask.view(B*T))

        for key in p_loss_keys: total_loss += losses[key]
        gs_pred, gs_targ = predictions["game_strategy"], targets["game_strategy"]
        frame_mask = alive_mask.any(dim=-1).float()
        
        # FIX #4: Use Cross-Entropy loss for round number classification
        round_logits  = gs_pred["round_number_logits"].view(B * T, -1)
        round_targets = gs_targ["round_number"].long().view(B*T) - 1  # 1-indexed -> 0-indexed
        C = round_logits.size(-1)
        # Ignore overtime rounds (labels >= C) so they contribute zero loss
        overtime = (round_targets >= C)
        if overtime.any():
            round_targets[overtime] = -1  # -1 matches ignore_index in self.ce_loss
        round_targets = round_targets.clamp(min=-1)  # keep padding as -1

        ce = self.ce_loss(round_logits, round_targets)  # reduction='none', ignore_index=-1
        valid_mask = frame_mask.view(B*T) * (round_targets != -1).float()
        if valid_mask.sum() > 0:
            losses['round_number'] = self.weights['round_number'] * self._scalar_loss(ce, valid_mask)
        else:
            # no valid round labels (all overtime/padding) -> zero loss
            losses['round_number'] = round_logits.sum() * 0.0

        losses['round_state'] = self.weights['round_state'] * self._scalar_loss(
            F.binary_cross_entropy_with_logits(gs_pred["round_state_logits"], gs_targ["round_state_logits"], reduction='none').mean(-1),
            frame_mask
        )
        total_loss += losses['round_number'] + losses['round_state']
        
        # POS heatmap loss (balanced BCE to avoid all-zero collapse)
        pred_pos = torch.stack([p["pos_heatmap_logits"] for p in predictions["player"]], 2)
        targ_pos = torch.stack([p["pos_coords"] for p in targets["player"]], 2)
        alive_flat = alive_mask.view(-1)
        if alive_flat.any():
            pred_alive = pred_pos.view(-1, *pred_pos.shape[3:])[alive_flat]     # [N_alive, Z, Y, X]
            coord_alive = targ_pos.view(-1, 3)[alive_flat]                      # [N_alive, 3]
            target_heatmap = self._build_targets_heatmaps(coord_alive).to(dtype=pred_alive.dtype)

            # Compute a scalar pos_weight per batch to counter extreme sparsity
            pos_sum = target_heatmap.sum()
            total_el = torch.tensor(target_heatmap.numel(), device=target_heatmap.device, dtype=pred_alive.dtype)
            pos_weight = ((total_el - pos_sum) / (pos_sum + 1e-6)).clamp_(1.0, 1000.0)  # clamp for stability

            bce = F.binary_cross_entropy_with_logits(
                pred_alive, target_heatmap, pos_weight=pos_weight, reduction='mean'
            )
            losses['pos_heatmap'] = self.weights['pos_heatmap'] * bce
        else:
            losses['pos_heatmap'] = 0.0 * pred_pos.mean()
        total_loss += losses['pos_heatmap']


        # ENEMY heatmap loss (balanced BCE)
        pred_enemy, targ_enemy = gs_pred["enemy_pos_heatmap_logits"], gs_targ["enemy_pos_coords"]
        valid_enemy = (targ_enemy[..., 0] >= 0)
        if valid_enemy.any():
            b_idx, t_idx, p_idx = valid_enemy.nonzero(as_tuple=True)
            pred_sel = pred_enemy[b_idx, t_idx]                    # [N_valid, Z, Y, X]
            coord_sel = targ_enemy[b_idx, t_idx, p_idx]            # [N_valid, 3]
            target_heatmap = self._build_targets_heatmaps(coord_sel).to(dtype=pred_sel.dtype)

            pos_sum = target_heatmap.sum()
            total_el = torch.tensor(target_heatmap.numel(), device=target_heatmap.device, dtype=pred_sel.dtype)
            pos_weight = ((total_el - pos_sum) / (pos_sum + 1e-6)).clamp_(1.0, 1000.0)

            bce = F.binary_cross_entropy_with_logits(
                pred_sel, target_heatmap, pos_weight=pos_weight, reduction='mean'
            )
            losses['enemy_heatmap'] = self.weights['enemy_heatmap'] * bce
        else:
            losses['enemy_heatmap'] = 0.0 * pred_enemy.mean()
        total_loss += losses['enemy_heatmap']


        return total_loss, {k: v.item() for k, v in losses.items() if v is not None}

# =============================================================================
# SECTION 2: NEW TRAINING COMPONENTS
# =============================================================================

class EMA:
    """Exponential Moving Average of model weights."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.params = {k: v.detach().clone() for k,v in model.state_dict().items() if v.dtype.is_floating_point}
    
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.params:
                # Move the EMA parameter to the model's device if it's not already there
                ema_param = self.params[k].to(v.device)
                
                # Perform the update on the correct device
                ema_param.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
                
                # Store the updated parameter back into the dictionary
                self.params[k] = ema_param

    def state_dict(self): return self.params
    def load_state_dict(self, d): self.params = d
    @torch.no_grad()
    def apply_to(self, model):
        msd = model.state_dict()
        for k, v in self.params.items():
            if k in msd: msd[k].copy_(v)

def setup_distributed(args):
    """Initializes the DDP process group."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(backend=args.dist_backend, init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, torch.distributed.get_rank(), torch.distributed.get_world_size(), local_rank
    return False, 0, 1, 0

def set_seed_all(seed: int, rank: int):
    """Sets random seeds for reproducibility."""
    seed_with_rank = seed + rank
    random.seed(seed_with_rank); np.random.seed(seed_with_rank)
    torch.manual_seed(seed_with_rank); torch.cuda.manual_seed_all(seed_with_rank)

def build_optimizer_scheduler(model, args, total_updates):
    """Build optimizer + LR scheduler where 'total_updates' = number of optimizer updates.
       Supports: AdamW / bnb 8-bit, and schedules: cosine / cosine_restarts / onecycle.
    """
    import math
    import torch
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR

    # ---- param groups with/without weight decay ----
    def split_decay(params):
        decay, no_decay = [], []
        for p in params:
            if not p.requires_grad:
                continue
            (no_decay if p.ndim == 1 else decay).append(p)
        return decay, no_decay

    base_lr = args.lr
    model_params = model.module.parameters() if hasattr(model, 'module') else model.parameters()
    decay, no_decay = split_decay(model_params)
    opt_groups = [
        {"params": decay, "lr": base_lr, "weight_decay": args.weight_decay},
        {"params": no_decay, "lr": base_lr, "weight_decay": 0.0},
    ]

    # ---- optimizer: torch AdamW or bitsandbytes ----
    opt_name = getattr(args, "optim", "adamw")
    optimizer = None
    try:
        import bitsandbytes as bnb  # optional
    except Exception:
        bnb = None

    if opt_name == "adamw8bit":
        if bnb is None:
            raise ImportError("bitsandbytes not installed. `pip install bitsandbytes`")
        optimizer = bnb.optim.AdamW8bit(opt_groups, betas=(0.9, 0.95), eps=1e-8)
    elif opt_name == "paged_adamw8bit":
        if bnb is None:
            raise ImportError("bitsandbytes not installed. `pip install bitsandbytes`")
        optimizer = bnb.optim.PagedAdamW8bit(opt_groups, betas=(0.9, 0.95), eps=1e-8)
    else:
        optimizer = torch.optim.AdamW(opt_groups, betas=(0.9, 0.95), eps=1e-8)

    # ---- schedule params (all in *updates*) ----
    # warmup: prefer --warmup-updates; else convert from --warmup-steps (micro-steps)->updates
    warmup_updates = getattr(args, "warmup_updates", None)
    if warmup_updates is None:
        warmup_updates = math.ceil(getattr(args, "warmup_steps", 0) / max(1, getattr(args, "accum_steps", 1)))
    warmup_updates = max(0, int(warmup_updates))

    # derive updates_per_epoch if not provided (we often have total_updates ≈ updates_per_epoch * epochs)
    epochs = max(1, int(getattr(args, "epochs", 1)))
    updates_per_epoch = max(1, total_updates // epochs)

    # schedule choice
    schedule = getattr(args, "lr_schedule", "cosine")  # "cosine", "cosine_restarts", "onecycle"

    # optional restart params
    cycle_updates = int(getattr(args, "cycle_updates", 0) or (2 * updates_per_epoch))  # default: ~2 epochs per cycle
    cycle_mult = float(getattr(args, "cycle_mult", 2.0))

    # optional onecycle params
    onecycle_div = float(getattr(args, "onecycle_div_factor", 100.0))
    onecycle_final_div = float(getattr(args, "onecycle_final_div_factor",
                                       max(1.0, base_lr / max(getattr(args, "min_lr", 1e-8), 1e-8))))

    # ---- scheduler construction ----
    if schedule == "cosine_restarts":
        # Warmup (linear), then cosine restarts
        warm = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=max(1, warmup_updates)) \
               if warmup_updates > 0 else None
        cosine = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, cycle_updates),
                                             T_mult=int(max(1.0, cycle_mult)),
                                             eta_min=getattr(args, "min_lr", 0.0))
        if warm is not None:
            scheduler = SequentialLR(optimizer, schedulers=[warm, cosine], milestones=[warmup_updates])
        else:
            scheduler = cosine

    elif schedule == "onecycle":
        # OneCycle over total_updates: warm, then decay to near min_lr at the very end
        pct_start = (warmup_updates / max(1, total_updates)) if warmup_updates > 0 else 0.3
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=base_lr,
            total_steps=max(1, total_updates),
            pct_start=min(0.9, max(0.0, pct_start)),
            anneal_strategy="cos",
            div_factor=onecycle_div,
            final_div_factor=onecycle_final_div,
        )

    else:
        # single cosine to min_lr with linear warmup
        def lr_lambda(u):
            # u = completed optimizer updates
            if u < warmup_updates:
                return max(1e-8, u / max(1, warmup_updates))
            progress = (u - warmup_updates) / max(1, total_updates - warmup_updates)
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(getattr(args, "min_lr", 0.0) / base_lr, cosine)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- apply step-0 LR without calling scheduler.step() early ----
    with torch.no_grad():
        if schedule == "onecycle":
            # initial lr = base_lr / div_factor
            init_lr = base_lr / max(1e-8, onecycle_div)
            for pg in optimizer.param_groups:
                pg["lr"] = init_lr
        elif warmup_updates > 0:
            s0 = 1.0 / max(1, warmup_updates)
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr * s0
        else:
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr

    # ---- AMP scaler ----
    use_fp16 = getattr(getattr(model, 'module', model), "cfg", None)
    use_fp16 = (getattr(use_fp16, "compute_dtype", None) == "fp16")
    scaler = torch.amp.GradScaler("cuda", enabled=bool(use_fp16))

    return optimizer, scheduler, scaler


def save_checkpoint(run_dir, filename, epoch, step, best_val, model, optimizer, scheduler, scaler, ema, args):
    """Saves a complete training checkpoint."""
    is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    ckpt = { "epoch": epoch, "global_step": step, "best_val": best_val,
             "model": model.module.state_dict() if is_ddp else model.state_dict(),
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "scaler": scaler.state_dict(), "ema": ema.state_dict() if ema else None,
             "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
             "args": vars(args) }
    path = os.path.join(run_dir, "checkpoints")
    os.makedirs(path, exist_ok=True)
    torch.save(ckpt, os.path.join(path, filename))
    logging.info(f"Saved checkpoint to {os.path.join(path, filename)}")

def load_checkpoint(path, model, optimizer, scheduler, scaler, ema):
    """Loads a training checkpoint."""
    ckpt = torch.load(path, map_location="cpu")
    target = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    target.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    if ema and ckpt.get("ema"): ema.load_state_dict(ckpt["ema"])
    torch.set_rng_state(ckpt["rng"]["torch"])
    for i, s in enumerate(ckpt["rng"]["cuda"]): torch.cuda.set_rng_state(s, device=i)
    logging.info(f"Resumed checkpoint from {path} at epoch {ckpt['epoch']}, step {ckpt['global_step']}")
    return ckpt["epoch"], ckpt["global_step"], ckpt.get("best_val", float("inf"))

def build_epoch_loader(
    args, epoch: int, store: LmdbStore, team_rounds: List[TeamRound], *,
    last_batch_policy="drop", rank=0, world_size=1
):
    """Builds the full data loading pipeline for a given epoch and split."""
    index = EpochIndex(T_frames=args.T_frames, seed=args.seed, windows_per_round=args.windows_per_round)
    records, id_map = index.build(team_rounds, epoch=epoch)
    device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0

    split_name_for_dir = "val" if last_batch_policy == "partial" else "train"
    fl_dir = os.path.join(args.run_dir, f"filelists_{split_name_for_dir}_e{epoch:04d}")

    if rank == 0:
        writer = FilelistWriter(fl_dir, use_precomputed=args.use_precomputed_embeddings, data_root=args.data_root)
        video_lists, audio_lists = writer.write(records)

    if world_size > 1: torch.distributed.barrier(device_ids=[device_id])

    video_lists = [os.path.join(fl_dir, f"pov{k}_video.txt") for k in range(5)]
    audio_lists = [os.path.join(fl_dir, f"pov{k}_audio.txt") for k in range(5)]

    video_path_lists, audio_path_lists = None, None
    if args.use_precomputed_embeddings:
        def make_paths_only_list(src_lst):
            dst = str(Path(src_lst).with_suffix(".paths"))
            if rank == 0:
                with open(src_lst, "r") as s, open(dst, "w") as d:
                    for line in s: d.write(line.rstrip("\n").split()[0] + "\n")
            return dst
        video_path_lists = [make_paths_only_list(p) for p in video_lists]
        audio_path_lists = [make_paths_only_list(p) for p in audio_lists]
    if world_size > 1: torch.distributed.barrier(device_ids=[device_id])

    dali_cfg = DaliConfig(sequence_length=args.T_frames, batch_size=args.batch_size, num_threads=args.dali_threads,
                          device_id=device_id, shard_id=rank, num_shards=world_size, seed=args.seed + epoch)
    dali_pipe = DaliInputPipeline(video_filelists=video_lists, audio_filelists=audio_lists, cfg=dali_cfg,
                                  use_precomputed=args.use_precomputed_embeddings, video_pathlists=video_path_lists,
                                  audio_pathlists=audio_path_lists, last_batch_policy=last_batch_policy)

    fetcher = LmdbMetaFetcher(store)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    assembler = BatchAssembler(id_map, fetcher, device=device)
    steps_per_epoch = (len(records) // (world_size * args.batch_size)) if len(records) > 0 else 0
    return dali_pipe.iterator, assembler, steps_per_epoch

# ==== Fixed-weight evaluation helpers ====

def _to_ddp_module(model):
    # Get the real module if wrapped in DDP
    return model.module if hasattr(model, "module") else model

def build_eval_loss(eval_weights, grid_dims=(64, 64, 8), sigma=1.5, device=None):
    # Create a CompositeLoss with frozen weights ONLY for evaluation
    loss = CompositeLoss(weights=eval_weights, grid_dims=grid_dims, sigma=sigma)
    if device is not None:
        loss = loss.to(device)
    return loss

# ==== Weight-free metric computation ====

class MetricsAccumulator:
    def __init__(self):
        self.sums = defaultdict(float)
        self.counts = defaultdict(float)
    def add(self, key, value, n=1.0):
        self.sums[key] += float(value)
        self.counts[key] += float(n)
    def mean(self, key, default=float("inf")):
        c = self.counts.get(key, 0.0)
        return (self.sums.get(key, 0.0) / max(c, 1e-9)) if c > 0 else default
    def to_dict(self, prefix=None):
        out = {}
        for k in self.sums.keys():
            m = self.mean(k)
            out[(prefix + "/" + k) if prefix else k] = m
        return out

def _sigmoid(x): return torch.sigmoid(x)
def _softmax(x, dim=-1): return torch.softmax(x, dim=dim)

@torch.no_grad()
def compute_weight_free_metrics(predictions, targets, alive_mask, coord_mapper=None):
    """
    Returns raw, weight-free metrics per head.
    Structure expectations (based on your CompositeLoss):
      predictions['player'][i] has:
        'stats', 'mouse_delta_deg', 'keyboard_logits', 'eco_logits',
        'inventory_logits', 'active_weapon_idx', 'pos_heatmap_logits'
      predictions['game_strategy'] has:
        'round_number_logits', 'round_state_logits'
      targets mirror these (keyboard/eco/inventory likely {0,1} multi-labels;
      weapon is class index; pos target is 'pos_coords' (xyz floats); round_number regression).
    """
    acc = MetricsAccumulator()
    device = alive_mask.device
    B, T, P = alive_mask.shape  # players P=5

    # ---- Player heads ----
    # Regression: stats, mouse
    for i in range(P):
        mask = alive_mask[:, :, i]  # [B,T]
        mcount = mask.sum().clamp(min=1)

        p_pred, p_targ = predictions["player"][i], targets["player"][i]
        # stats: L1 / L2
        if "stats" in p_pred and "stats" in p_targ:
            diff = (p_pred["stats"] - p_targ["stats"])[mask]  # [..., D]
            if diff.numel() > 0:
                mae = diff.abs().mean()
                rmse = (diff.pow(2).mean()).sqrt()
                acc.add("player/stats_MAE", mae, 1)
                acc.add("player/stats_RMSE", rmse, 1)

        # mouse: L1 / L2
        if "mouse_delta_deg" in p_pred and "mouse_delta_deg" in p_targ:
            diff = (p_pred["mouse_delta_deg"] - p_targ["mouse_delta_deg"])[mask]
            if diff.numel() > 0:
                mae = diff.abs().mean()
                rmse = (diff.pow(2).mean()).sqrt()
                acc.add("player/mouse_MAE", mae, 1)
                acc.add("player/mouse_RMSE", rmse, 1)

        # Multi-label BCE heads: keyboard, eco, inventory
        # FIX #2: Report precision and recall in addition to F1
        for key in ["keyboard_logits", "eco_logits", "inventory_logits"]:
            if key in p_pred and key in p_targ:
                logits = p_pred[key][mask]     # [N, C]
                labels = p_targ[key][mask]     # [N, C], {0,1} or soft
                if logits.numel() > 0:
                    probs = _sigmoid(logits)
                    preds = (probs >= 0.5).float()
                    # micro accuracy across classes
                    micro_acc = (preds.eq(labels.round())).float().mean()
                    # precision/recall/F1 (micro)
                    tp = (preds * labels).sum()
                    fp = (preds * (1 - labels)).sum()
                    fn = ((1 - preds) * labels).sum()
                    precision = tp / (tp + fp + 1e-9)
                    recall    = tp / (tp + fn + 1e-9)
                    f1        = 2 * precision * recall / (precision + recall + 1e-9)
                    base = key.replace("_logits", "")
                    acc.add(f"player/{base}_micro_acc", micro_acc, 1)
                    acc.add(f"player/{base}_micro_precision", precision, 1)
                    acc.add(f"player/{base}_micro_recall", recall, 1)
                    acc.add(f"player/{base}_micro_F1", f1, 1)

        # Weapon classification
        # FIX #6: Add top-k accuracy metrics for weapon classification
        if "active_weapon_idx" in p_pred and "active_weapon_idx" in p_targ:
            pred_logits = p_pred["active_weapon_idx"][mask]
            targ_idx = p_targ["active_weapon_idx"][mask].long()
            if pred_logits.numel() > 0 and targ_idx.numel() > 0:
                # Top-1
                pred_idx = pred_logits.argmax(-1)
                acc.add("player/weapon_acc", (pred_idx == targ_idx).float().mean(), 1)
                # Top-K
                _, topk_preds = torch.topk(pred_logits, k=5, dim=-1)
                targ_expanded = targ_idx.unsqueeze(-1)
                acc.add("player/weapon_top3_acc", (topk_preds[:, :3] == targ_expanded).any(dim=-1).float().mean(), 1)
                acc.add("player/weapon_top5_acc", (topk_preds == targ_expanded).any(dim=-1).float().mean(), 1)

        # Position heatmap: voxel accuracy & within-1-voxel hit rate
        # FIX #1: Correct voxel index decoding and flattening logic for [Z,Y,X] layout
        if "pos_heatmap_logits" in p_pred and "pos_coords" in p_targ:
            logits = p_pred["pos_heatmap_logits"] # [B,T,Z,Y,X]
            if coord_mapper is not None:
                flat_idx = logits.view(B*T, -1).argmax(-1).view(B, T)
                world = p_targ["pos_coords"]
                grid_idx = coord_mapper.discretize_world_to_grid(world)
                X, Y, Z = coord_mapper.grid_dims.cpu().numpy()

                # Correct Z-major flattening to match tensor layout
                targ_flat = (grid_idx[..., 2] * (Y * X) + grid_idx[..., 1] * X + grid_idx[..., 0]).long()
                eq = (flat_idx[mask] == targ_flat[mask]).float()
                if eq.numel() > 0:
                    acc.add("player/pos_voxel_acc", eq.mean(), 1)

                # Correct Z-major decoding of predicted index
                pred_z = flat_idx // (Y * X)
                pred_y = (flat_idx // X) % Y
                pred_x = flat_idx % X
                
                targ_x = grid_idx[...,0]; targ_y = grid_idx[...,1]; targ_z = grid_idx[...,2]
                dx = (pred_x - targ_x).abs(); dy = (pred_y - targ_y).abs(); dz = (pred_z - targ_z).abs()
                near1 = (((dx <= 1) & (dy <= 1) & (dz <= 1)).float())[mask]
                if near1.numel() > 0:
                    acc.add("player/pos_within1voxel_rate", near1.mean(), 1)

    # ---- Game strategy heads ----
    gs_pred, gs_targ = predictions["game_strategy"], targets["game_strategy"]
    frame_mask = alive_mask.any(dim=-1)  # [B,T]
    if "round_state_logits" in gs_pred and "round_state_logits" in gs_targ:
        pred_idx = _softmax(gs_pred["round_state_logits"], dim=-1).argmax(-1)
        targ_idx = gs_targ["round_state_logits"].argmax(-1) if gs_targ["round_state_logits"].dim() == gs_pred["round_state_logits"].dim() else gs_targ["round_state"]
        correct = (pred_idx[frame_mask] == targ_idx[frame_mask]).float()
        if correct.numel() > 0:
            acc.add("game/round_state_acc", correct.mean(), 1)

    # FIX #4: Change round number metric from MAE/RMSE to classification accuracy
    if "round_number_logits" in gs_pred and "round_number" in gs_targ:
        pred_logits = gs_pred["round_number_logits"][frame_mask]
        targ_idx = (gs_targ["round_number"][frame_mask].round().long() - 1) # 1-indexed to 0-indexed
        if pred_logits.numel() > 0 and targ_idx.numel() > 0:
            # Top-1
            pred_idx = pred_logits.argmax(-1)
            acc.add("game/round_number_acc", (pred_idx == targ_idx).float().mean(), 1)
            # Top-3
            _, top3_preds = torch.topk(pred_logits, k=3, dim=-1)
            acc.add("game/round_number_top3_acc", (top3_preds == targ_idx.unsqueeze(-1)).any(dim=-1).float().mean(), 1)

    return acc.to_dict()

# =============================================================================
# SECTION 2: NEW TRAINING COMPONENTS (CONTINUED)
# =============================================================================

# ======= Helper: compute calibrated metrics from per-class histograms =======

@torch.no_grad()
def _calibrated_metrics_from_hist(pos_hist: torch.Tensor,
                                  neg_hist: torch.Tensor,
                                  thresholds: torch.Tensor) -> Dict[str, float]:
    """
    pos_hist, neg_hist: int64 tensors on the SAME device, shape [C, T]
      where T == thresholds.numel() bins over [0, 1].
    thresholds: float32 tensor [T] monotonically increasing in [0, 1].

    Returns calibrated micro/macro F1 and threshold summaries.
    """
    device = pos_hist.device
    pos_hist = pos_hist.to(device=device, dtype=torch.long)
    neg_hist = neg_hist.to(device=device, dtype=torch.long)
    thresholds = thresholds.to(device=device, dtype=torch.float32)

    # Totals per class
    total_pos = pos_hist.sum(dim=-1)  # [C]
    total_neg = neg_hist.sum(dim=-1)  # [C]

    # For threshold index t, prediction is (prob >= thresholds[t]):
    # TP/FP at t are sums of hist bins with index >= t (right-cumulative)
    tp_bins = torch.flip(torch.flip(pos_hist, dims=[-1]).cumsum(dim=-1), dims=[-1])  # [C, T]
    fp_bins = torch.flip(torch.flip(neg_hist, dims=[-1]).cumsum(dim=-1), dims=[-1])  # [C, T]
    fn_bins = total_pos.unsqueeze(-1) - tp_bins                                       # [C, T]

    # Per-class F1 across thresholds
    prec = tp_bins.to(torch.float32) / (tp_bins + fp_bins + 1e-9)
    rec  = tp_bins.to(torch.float32) / (tp_bins + fn_bins + 1e-9)
    f1   = 2.0 * prec * rec / (prec + rec + 1e-9)                                     # [C, T]

    # Valid classes: at least one positive AND one negative globally
    valid = (total_pos > 0) & (total_neg > 0)
    C = pos_hist.size(0)
    best_idx = torch.zeros(C, dtype=torch.long, device=device)
    if valid.any():
        best_idx[valid] = torch.argmax(f1[valid], dim=-1)
    best_thr = thresholds[best_idx]                                                   # [C]

    # Read class-wise counts at the chosen threshold for each class
    tp_best = tp_bins.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)                  # [C]
    fp_best = fp_bins.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)                  # [C]
    fn_best = total_pos - tp_best                                                     # [C]

    # Micro metrics
    tp_micro = tp_best.sum().to(torch.float32)
    fp_micro = fp_best.sum().to(torch.float32)
    fn_micro = fn_best.sum().to(torch.float32)
    prec_micro = (tp_micro / (tp_micro + fp_micro + 1e-9)).item()
    rec_micro  = (tp_micro / (tp_micro + fn_micro + 1e-9)).item()
    f1_micro   = (2.0 * prec_micro * rec_micro / (prec_micro + rec_micro + 1e-9))

    # Macro-F1 over valid classes only
    macro_f1 = 0.0
    if valid.any():
        prec_c = (tp_best[valid].to(torch.float32) /
                  (tp_best[valid] + fp_best[valid] + 1e-9))
        rec_c  = (tp_best[valid].to(torch.float32) /
                  (tp_best[valid] + fn_best[valid] + 1e-9))
        f1_c   = 2.0 * prec_c * rec_c / (prec_c + rec_c + 1e-9)
        macro_f1 = float(f1_c.mean().item())

    # Threshold summaries
    if valid.any():
        mean_thr = float(best_thr[valid].mean().item())
        median_thr = float(best_thr[valid].median().item())
    else:
        mean_thr = 0.5
        median_thr = 0.5

    return {
        "calibrated_micro_precision": prec_micro,
        "calibrated_micro_recall":    rec_micro,
        "calibrated_micro_F1":        f1_micro,
        "calibrated_macro_F1":        macro_f1,
        "mean_optimal_threshold":     mean_thr,
        "median_optimal_threshold":   median_thr,
    }


# ============================ VALIDATE (DDP-safe) ============================

@torch.no_grad()
def validate(dali_iter, assembler, model, eval_loss_fn, args, *,
             coord_mapper=None,
             also_weighted_view: bool = True,
             train_like_loss_fn: Optional[nn.Module] = None,
             # --- calibration sampling controls ---
             calibrate_mode: str = "count",    # "count" or "fraction"
             calibrate_target: float = 5.0,    # if "count": max examples per head; if "fraction": e.g., 0.05 for 5%
             rng: Optional[torch.Generator] = None) -> Dict[str, float]:
    """
    Runs validation with:
      - fixed/* losses (frozen weights view)
      - optional weighted_view/total_like_train
      - metrics/* (0.5-thresh, weight-free)
      - metrics_calibrated/* from a tiny random subset (histogram-based)

    Sampling:
      - calibrate_mode="count": uses up to `calibrate_target` masked examples per head per epoch (default 5).
      - calibrate_mode="fraction": Bernoulli sample each masked example with prob=`calibrate_target` (e.g., 0.05 for 5%).
    """
    model.eval()
    mdl = _to_ddp_module(model)
    device = next(mdl.parameters()).device
    if rng is None:
        rng = torch.Generator(device=device)
        # Leave unseeded for stochasticity across epochs; set manual_seed() if you want determinism.

    totals = defaultdict(float)
    counts = 0

    # Multi-label heads to calibrate
    heads = ["keyboard_logits", "eco_logits", "inventory_logits"]

    # Threshold grid (shared across ranks)
    T = 101
    thr = torch.linspace(0.0, 1.0, T, device=device, dtype=torch.float32)

    # Per-head histogram state (allocated lazily once we know C)
    hist = {
        h: {
            "pos": None,   # [C, T] int64
            "neg": None,   # [C, T] int64
        } for h in heads
    }
    # Per-head cap state for "count" mode
    samples_used = {h: 0 for h in heads}

    # AMP dtype if configured
    amp_dtype = None
    if getattr(mdl, "cfg", None) is not None and getattr(mdl.cfg, "compute_dtype", None) in ("bf16", "fp16"):
        amp_dtype = torch.bfloat16 if mdl.cfg.compute_dtype == "bf16" else torch.float16

    try:
        while True:
            batch_raw = next(dali_iter)[0]
            batch = assembler.assemble(batch_raw, args.use_precomputed_embeddings)

            with torch.autocast(device_type="cuda", enabled=(amp_dtype is not None), dtype=amp_dtype):
                outputs = model(batch)

                # ---- Fixed-weight loss view ----
                fixed_total, fixed_losses = eval_loss_fn(outputs, batch["targets"], batch["alive_mask"])

                # ---- Optional train-like weighted view ----
                if also_weighted_view and (train_like_loss_fn is not None):
                    weighted_total, _ = train_like_loss_fn(outputs, batch["targets"], batch["alive_mask"])
                else:
                    weighted_total = None

            # Aggregate losses
            totals["fixed/total"] += float(fixed_total)
            for k, v in fixed_losses.items():
                totals[f"fixed/{k}"] += float(v)
            if weighted_total is not None:
                totals["weighted_view/total_like_train"] += float(weighted_total)

            # Weight-free metrics at 0.5
            wf = compute_weight_free_metrics(outputs, batch["targets"], batch["alive_mask"], coord_mapper=coord_mapper)
            for k, v in wf.items():
                totals[f"metrics/{k}"] += float(v)

            # ---- Histogram accumulation for calibration (tiny random subset) ----
            # Determine number of players robustly
            num_players = min(len(outputs.get("player", [])), batch["alive_mask"].shape[-1])
            for i in range(num_players):
                alive = batch["alive_mask"][:, :, i].to(torch.bool)  # [B, T]
                if not alive.any():
                    continue

                for key in heads:
                    if (key not in outputs["player"][i]) or (key not in batch["targets"]["player"][i]):
                        continue

                    # Early skip if "count" mode is already satisfied for this head
                    if calibrate_mode == "count" and samples_used[key] >= int(calibrate_target):
                        continue

                    # logits/labels: [B, T, C]
                    logits_bt = outputs["player"][i][key].to(device=device)
                    labels_bt = batch["targets"]["player"][i][key].to(device=device, dtype=torch.float32)

                    # ----- construct sampling mask -----
                    if calibrate_mode == "fraction":
                        p = float(max(0.0, min(1.0, calibrate_target)))
                        # Bernoulli sampling on alive positions
                        rand = torch.rand_like(alive, dtype=torch.float32, generator=rng, device=device)
                        sample_mask = alive & (rand < p)  # [B, T] bool
                    else:  # "count"
                        remaining = int(max(0, int(calibrate_target) - samples_used[key]))
                        if remaining <= 0:
                            continue
                        # Flatten alive coords, sample up to 'remaining'
                        idx = torch.nonzero(alive, as_tuple=False)  # [M, 2] (b, t)
                        M = idx.shape[0]
                        if M == 0:
                            continue
                        take = min(remaining, M)
                        perm = torch.randperm(M, device=device, generator=rng)[:take]
                        sel = idx[perm]  # [take, 2]
                        sample_mask = torch.zeros_like(alive, dtype=torch.bool, device=device)
                        sample_mask[sel[:, 0], sel[:, 1]] = True
                        samples_used[key] += take  # update cap

                    if not sample_mask.any():
                        continue

                    # Masked -> [N, C]
                    logits = logits_bt[sample_mask]                  # [N, C]
                    labels = labels_bt[sample_mask].to(torch.float32)  # [N, C]
                    if logits.numel() == 0:
                        continue

                    # Lazily allocate hist tensors once we know C
                    C = logits.shape[-1]
                    if hist[key]["pos"] is None:
                        hist[key]["pos"] = torch.zeros((C, T), dtype=torch.long, device=device)
                        hist[key]["neg"] = torch.zeros((C, T), dtype=torch.long, device=device)

                    # Probabilities -> bin indices in [0, T-1]
                    probs = torch.sigmoid(logits).to(torch.float32)            # [N, C]
                    bins = torch.clamp((probs * (T - 1)).long(), 0, T - 1)     # [N, C]

                    # Update per-class pos/neg hist (loop over C keeps memory tiny; C is small)
                    for c in range(C):
                        b_c = bins[:, c]                   # [N]
                        y_c = (labels[:, c] > 0.5)         # [N] bool

                        if y_c.any():
                            hist[key]["pos"][c].scatter_add_(
                                0, b_c[y_c], torch.ones_like(b_c[y_c], dtype=torch.long, device=device)
                            )
                        if (~y_c).any():
                            hist[key]["neg"][c].scatter_add_(
                                0, b_c[~y_c], torch.ones_like(b_c[~y_c], dtype=torch.long, device=device)
                            )

            counts += 1
    except StopIteration:
        pass

    # ---- DDP: sum tiny histograms across ranks (CUDA ints; NCCL-friendly) ----
    ddp = dist.is_available() and dist.is_initialized()
    if ddp:
        for key in heads:
            if hist[key]["pos"] is not None:
                dist.all_reduce(hist[key]["pos"], op=dist.ReduceOp.SUM)
                dist.all_reduce(hist[key]["neg"], op=dist.ReduceOp.SUM)

    # ---- Compute calibrated metrics per head from histograms ----
    for key in heads:
        base = key.replace("_logits", "")
        if hist[key]["pos"] is None:
            # No data sampled for this head in this eval
            for n, v in {
                "micro_precision": 0.0,
                "micro_recall": 0.0,
                "micro_F1": 0.0,
                "macro_F1": 0.0,
                "mean_optimal_threshold": 0.5,
                "median_optimal_threshold": 0.5,
            }.items():
                totals[f"metrics_calibrated/player/{base}_{n}"] = float(v)
            continue

        cal = _calibrated_metrics_from_hist(hist[key]["pos"], hist[key]["neg"], thr)
        totals[f"metrics_calibrated/player/{base}_micro_precision"] = float(cal["calibrated_micro_precision"])
        totals[f"metrics_calibrated/player/{base}_micro_recall"]    = float(cal["calibrated_micro_recall"])
        totals[f"metrics_calibrated/player/{base}_micro_F1"]        = float(cal["calibrated_micro_F1"])
        totals[f"metrics_calibrated/player/{base}_macro_F1"]        = float(cal["calibrated_macro_F1"])
        totals[f"metrics_calibrated/player/{base}_mean_optimal_threshold"]   = float(cal["mean_optimal_threshold"])
        totals[f"metrics_calibrated/player/{base}_median_optimal_threshold"] = float(cal["median_optimal_threshold"])

    # ---- Average batch-wise totals (calibrated metrics are already final) ----
    out = {}
    denom = max(1, counts)
    for k, v in totals.items():
        if k.startswith("metrics_calibrated/"):
            out[k] = float(v)
        else:
            out[k] = float(v) / denom

    return out



# =============================================================================
# SECTION 3: MAIN TRAINING ORCHESTRATOR
# =============================================================================

def train(args, model_cfg):
    """Main training function."""
    is_ddp, rank, world_size, local_rank = setup_distributed(args)
    set_seed_all(args.seed, rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if rank == 0:
        os.makedirs(args.run_dir, exist_ok=True)
    writer = SummaryWriter(args.run_dir) if rank == 0 else None

    # --- Build Model, Loss, Optimizer ---
    model = CS2Transformer(model_cfg).to(device)
    if args.compile: model = torch.compile(model)
    if is_ddp: model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
    
    loss_weights = { 'stats': 1.0, 'mouse': 1.0, 'keyboard': 1.0, 'eco': 0.5, 'inventory': 0.5, 'weapon': 1.0,
                     'round_number': 0.2, 'round_state': 0.5, 'pos_heatmap': 1.0, 'enemy_heatmap': 1.0 }
    loss_fn = CompositeLoss(weights=loss_weights).to(device)
    
    # ---- Balanced-loss setup ----
    base_loss_weights = loss_weights.copy()           # keep your chosen priors
    ema_loss = {k: 1.0 for k in base_loss_weights}    # start neutral
    ema_beta = args.balance_momentum
    sum_base_w = sum(base_loss_weights.values()) or 1.0

    update_count = 0          # counts optimizer updates (not micro-steps)
    eval_count = 0            # how many validations have run
    # ---- /balanced-loss setup ----


    manifest = Manifest(args.data_root, args.manifest)
    store = LmdbStore()
    logging.info("Building training set index...")
    train_team_rounds = build_team_rounds(args.data_root, manifest.get_games("train"), store)
    logging.info("Building validation set index...")
    val_team_rounds = build_team_rounds(args.data_root, manifest.get_games("val"), store)

    effective_samples = len(train_team_rounds) * max(1, args.windows_per_round)
    micro_steps_per_epoch = (effective_samples // (world_size * args.batch_size)) if effective_samples else 0
    updates_per_epoch = math.ceil(micro_steps_per_epoch / max(1, args.accum_steps))
    total_updates = updates_per_epoch * args.epochs

    optimizer, scheduler, scaler = build_optimizer_scheduler(model, args, total_updates)


    # If your scheduler’s warmup is specified in *micro-steps*, convert it to *updates*
    # so that build_optimizer_scheduler gets consistent units.
    if hasattr(args, "warmup_steps"):
        args._orig_warmup_steps = args.warmup_steps  # optional: keep a copy
        args.warmup_steps = math.ceil(args.warmup_steps / max(1, args.accum_steps))

    optimizer, scheduler, scaler = build_optimizer_scheduler(model, args, total_updates)

    ema = EMA(model, decay=args.ema_decay) if args.ema_decay and args.ema_decay > 0 else None

    start_epoch, global_step, best_val = 0, 0, float("inf")
    if args.resume:
        start_epoch, global_step, best_val = load_checkpoint(args.resume, model, optimizer, scheduler, scaler, ema)

    # --- Main Training Loop ---
    for epoch in range(start_epoch, args.epochs):
        train_iter, train_asm, _ = build_epoch_loader(
            args, epoch, store, train_team_rounds, last_batch_policy="drop", rank=rank, world_size=world_size
        )
        model.train()
        
        try:
            while True:
                # FIX: Unpack the list returned by the DALI iterator to get the dictionary
                batch_raw = next(train_iter)[0]
                batch = train_asm.assemble(batch_raw, args.use_precomputed_embeddings)
                
                with torch.autocast("cuda", enabled=(model_cfg.compute_dtype in ["bf16", "fp16"]),
                                    dtype=torch.bfloat16 if model_cfg.compute_dtype == "bf16" else torch.float16):
                    preds = model(batch)
                    total_loss, loss_dict = loss_fn(preds, batch["targets"], batch["alive_mask"])
                    total_loss = total_loss / args.accum_steps
                
                scaler.scale(total_loss).backward()

                if (global_step + 1) % args.accum_steps == 0:
                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if ema: ema.update(model)
                    scheduler.step()

                    update_count += 1

                if rank == 0 and (global_step % args.log_every == 0):
                    writer.add_scalar("train/total_loss", total_loss.item() * args.accum_steps, global_step)
                    for k, v in loss_dict.items(): writer.add_scalar(f"train_loss/{k}", v, global_step)
                    writer.add_scalar("opt/lr", optimizer.param_groups[0]['lr'], global_step)
                    logging.info(f"Step: {global_step}, Loss: {total_loss.item() * args.accum_steps:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

                if args.eval_every > 0 and (global_step > 0 and global_step % args.eval_every == 0) and val_team_rounds:
                    # pre-clean to reduce peak memory
                    torch.cuda.synchronize(); torch.cuda.empty_cache(); gc.collect()

                    val_iter = val_asm = None
                    try:
                        val_iter, val_asm, _ = build_epoch_loader(
                            args, 0, store, val_team_rounds, last_batch_policy="partial", rank=rank, world_size=world_size
                        )
                        # --- START VALIDATION PATCH ---
                        mdl = _to_ddp_module(model)
                        device = next(mdl.parameters()).device

                        # Use base config weights for a stable evaluation metric
                        frozen_eval_weights = {k: float(v) for k, v in base_loss_weights.items()}

                        eval_loss_fn = build_eval_loss(frozen_eval_weights,
                                                       grid_dims=getattr(mdl, "grid_dims", (64,64,8)),
                                                       sigma=getattr(mdl, "sigma", 1.5),
                                                       device=device)
                        
                        # The diagnostic weighted view uses the current training loss_fn
                        train_like_loss_fn = loss_fn
                        
                        # If a CoordinateMapper is available, pass it for metrics
                        coord_mapper = getattr(eval_loss_fn, "coord_mapper", None)

                        val_out = validate(val_iter, val_asm, model, eval_loss_fn, args,
                                           coord_mapper=coord_mapper,
                                           also_weighted_view=True,
                                           train_like_loss_fn=train_like_loss_fn)
                        # --- END VALIDATION PATCH ---

                    finally:
                        # aggressively tear down validation pipeline
                        try:
                            if hasattr(val_iter, "iterator") and hasattr(val_iter.iterator, "_pipes"):
                                for p in val_iter.iterator._pipes:
                                    try: p._pipe.release()
                                    except Exception: pass
                        except Exception:
                            pass
                        del val_iter, val_asm
                        torch.cuda.synchronize(); torch.cuda.empty_cache(); gc.collect()

                    # --- Log raw validation losses ---
                    if writer:
                        # Log the fixed, comparable validation loss
                        writer.add_scalar("val_fixed/total_loss", val_out["fixed/total"], global_step)
                        for k, v in val_out.items():
                            if k.startswith("fixed/") and k != "fixed/total":
                                writer.add_scalar(f"val_loss/{k.split('/', 1)[1]}", v, global_step)
                        
                        # Log the diagnostic weighted view
                        if "weighted_view/total_like_train" in val_out:
                            writer.add_scalar("val_weighted_view/total_like_train", val_out["weighted_view/total_like_train"], global_step)
                        
                        # Log all weight-free metrics
                        for k, v in val_out.items():
                            if k.startswith("metrics/"):
                                writer.add_scalar(k, v, global_step)

                    # --- Update EMA of raw per-head losses (skip "total") ---
                    # Use the fixed-weight per-head losses for rebalancing
                    for k in base_loss_weights.keys():
                        if f"fixed/{k}" in val_out:
                            ema_loss[k] = ema_beta * ema_loss[k] + (1.0 - ema_beta) * float(val_out[f"fixed/{k}"])

                    # --- Rebalance after warmup and every N evals ---
                    eval_count += 1
                    if (update_count >= args.balance_losses_after_updates) and (eval_count % max(1, args.balance_every_evals) == 0):
                        # scale ∝ 1 / EMA(loss); then renormalize to keep total weight sum unchanged
                        scales = {k: 1.0 / max(ema_loss[k], 1e-8) for k in base_loss_weights}
                        unnorm = {k: base_loss_weights[k] * scales[k] for k in base_loss_weights}
                        Z = sum_base_w / max(sum(unnorm.values()), 1e-12)
                        balanced = {k: Z * unnorm[k] for k in base_loss_weights}

                        # apply to the running loss function
                        if hasattr(loss_fn, "weights"):
                            loss_fn.weights.update(balanced)
                        else:
                            loss_fn.weights = balanced  # fallback

                        if writer:
                            for k in balanced:
                                writer.add_scalar(f"weights/{k}", balanced[k], global_step)

                    # --- Use the FIXED metric for model selection ---
                    primary_early_stop_metric = val_out["fixed/total"]
                    if rank == 0 and primary_early_stop_metric < best_val:
                        best_val = primary_early_stop_metric
                        save_checkpoint(args.run_dir, "best.ckpt", epoch, global_step, best_val, model, optimizer, scheduler, scaler, ema, args)
                
                if rank == 0 and args.save_every > 0 and (global_step > 0 and global_step % args.save_every == 0):
                    save_checkpoint(args.run_dir, "last.ckpt", epoch, global_step, best_val, model, optimizer, scheduler, scaler, ema, args)
                
                global_step += 1
        except StopIteration: pass
        
        if rank == 0:
            save_checkpoint(args.run_dir, "last.ckpt", epoch, global_step, best_val, model, optimizer, scheduler, scaler, ema, args)

    if writer: writer.close()
    if world_size > 1: torch.distributed.destroy_process_group()

def smoke_test(args, model_cfg):
    """Original smoke test logic."""
    manifest = Manifest(args.data_root, args.manifest)
    store = LmdbStore()
    team_rounds = build_team_rounds(args.data_root, manifest.get_games(args.split), store)
    dali_iter, assembler, _ = build_epoch_loader(args, 0, store, team_rounds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CS2Transformer(model_cfg).to(device)
    loss_weights = { 'stats': 1.0, 'mouse': 1.0, 'keyboard': 1.0, 'eco': 1.0, 'inventory': 1.0, 'weapon': 0.25,
                     'round_number': 1.0, 'round_state': 1.0, 'pos_heatmap': 1.0, 'enemy_heatmap': 1.0 }
    loss_fn = CompositeLoss(weights=loss_weights).to(device)
    for i in range(args.num_steps):
        try:
            # FIX: Unpack the list returned by the DALI iterator
            batch_raw = next(dali_iter)[0]
            batch = assembler.assemble(batch_raw, args.use_precomputed_embeddings)
            predictions = model(batch)
            total_loss, detailed_losses_dict = loss_fn(predictions, batch['targets'], batch['alive_mask'])
            logging.info(f"Step {i+1} OK. Total Loss: {total_loss.item():.4f}")
            if args.detailed_loss: logging.info(f"  Losses: {detailed_losses_dict}")
        except StopIteration:
            logging.info("Finished iterator.")
            break

def main():
    parser = argparse.ArgumentParser(description="CS2Transformer Training Script")
    # Core args
    parser.add_argument("--data-root", type=str, default=os.environ.get("DATA_ROOT", "data"))
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--run-dir", type=str, default="runs/exp1")
    parser.add_argument("--mode", type=str, choices=["train", "smoke"], default="smoke")
    # Data args
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--T-frames", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--use-precomputed-embeddings", action='store_true')
    parser.add_argument("--dali-threads", type=int, default=4)
    parser.add_argument("--windows-per-round", type=int, default=1,
                        help="How many temporal windows to sample per team-round each epoch")

    # Training args
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adamw8bit",
                    choices=["adamw", "adamw8bit", "paged_adamw8bit"],
                    help="Use torch AdamW or bitsandbytes 8-bit optimizers")
    # near other training args
    parser.add_argument("--lr-schedule", type=str, default="cosine",
        choices=["cosine", "cosine_restarts", "onecycle"])
    parser.add_argument("--warmup-updates", type=int, default=1500)  # in optimizer updates
    parser.add_argument("--cycle-updates", type=int, default=0)      # restarts: 0 -> derive from epoch length
    parser.add_argument("--cycle-mult", type=float, default=2.0)     # multiplicative cycle growth
    parser.add_argument("--onecycle-div-factor", type=float, default=100.0)
    parser.add_argument("--onecycle-final-div-factor", type=float, default=10000.0)


    # Logistics
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--resume", type=str, default="")
    # Advanced
    parser.add_argument("--compile", action='store_true')
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dist-backend", type=str, default="nccl")
    # Smoke test args
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--detailed-loss", action='store_true')

    parser.add_argument("--balance-losses-after-updates", type=int, default=1500,
                    help="Start balancing after this many optimizer updates (not micro-steps)")
    parser.add_argument("--balance-momentum", type=float, default=0.99,
                        help="EMA momentum for per-head raw losses (0.9–0.99 is typical)")
    parser.add_argument("--balance-every-evals", type=int, default=1,
                        help="Recompute balanced weights every N validation runs")


    args = parser.parse_args()
    if args.manifest is None: args.manifest = os.path.join(args.data_root, "manifest.json")

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")

    model_cfg = CS2Config(context_frames=args.T_frames)

    if args.mode == "train":
        train(args, model_cfg)
    elif args.mode == "smoke":
        smoke_test(args, model_cfg)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()