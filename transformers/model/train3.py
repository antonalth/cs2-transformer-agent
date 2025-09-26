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
from torch.utils.tensorboard import SummaryWriter

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
        # FIX: Restored the more robust manifest parser from train2.py.
        # This version handles both simple strings and dictionary entries.
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
    def __init__(self, T_frames: int, seed: int):
        self.T_frames, self.seed = T_frames, seed
    def build(self, team_rounds: List[TeamRound], epoch: int) -> Tuple[List[SampleRecord], Dict[int, SampleRecord]]:
        rnd = random.Random(self.seed + epoch)
        records, id_map = [], {}
        for sid, tr in enumerate(team_rounds):
            start_f = rnd.randint(0, tr.frame_count - self.T_frames) if tr.frame_count >= self.T_frames else 0
            rec = SampleRecord(sample_id=sid, demoname=tr.demoname, lmdb_path=tr.lmdb_path, round_num=tr.round_num,
                               team=tr.team, pov_videos=tr.pov_videos, pov_audio=tr.pov_audio, start_f=start_f,
                               start_tick_win=tr.start_tick + TICKS_PER_FRAME * start_f, T_frames=self.T_frames)
            records.append(rec); id_map[sid] = rec
        return records, id_map

class FilelistWriter:
    def __init__(self, out_dir: str, use_precomputed: bool = False, data_root: str = ""):
        self.out_dir, self.use_precomputed, self.data_root = out_dir, use_precomputed, data_root
        os.makedirs(out_dir, exist_ok=True)
        if use_precomputed and not data_root: raise ValueError("data_root required for precomputed mode.")
    def write(self, records: List[SampleRecord]) -> Tuple[List[str], List[str]]:
        vid_paths = [os.path.join(self.out_dir, f"pov{k}_video.txt") for k in range(5)]
        aud_paths = [os.path.join(self.out_dir, f"pov{k}_audio.txt") for k in range(5)]
        def _embed_path(kind, demo, base):
            p = Path(self.data_root) / ("vit_embed" if kind == "vit" else "aud_embed") / demo / f"{base}.npy"
            if not p.is_file(): raise FileNotFoundError(f"Missing precomputed embedding: {p}")
            return p
        with contextlib.ExitStack() as stack:
            files = [stack.enter_context(open(p, "w")) for p in vid_paths + aud_paths]
            vid_fs, aud_fs = files[:5], files[5:]
            for rec in records:
                if self.use_precomputed:
                    packed_label = rec.sample_id * LABEL_SCALE + rec.start_f
                    for k in range(5):
                        vid_base = os.path.splitext(os.path.basename(rec.pov_videos[k]))[0]
                        aud_base = os.path.splitext(os.path.basename(rec.pov_audio[k]))[0]
                        vid_fs[k].write(f"{_embed_path('vit', rec.demoname, vid_base)} {packed_label}\n")
                        aud_fs[k].write(f"{_embed_path('aud', rec.demoname, aud_base)} {packed_label}\n")
                else:
                    for k in range(5):
                        path, ticks = rec.pov_videos[k], ticks_from_filename(rec.pov_videos[k])
                        start, end_exc = clamp_window_to_pov(rec.start_f, rec.T_frames, ticks[0], ticks[1])
                        frame_num = max(0, end_exc - start)
                        vid_fs[k].write(f"{path} {rec.sample_id} {start} {frame_num}\n")
                        packed_audio_label = rec.sample_id * LABEL_SCALE + start
                        aud_fs[k].write(f"{rec.pov_audio[k]} {packed_audio_label}\n")
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
            # FIX: Use a reader name that actually exists in the DALI graph. The label reader for the
            # first POV is a safe choice for determining the dataset size.
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
                _, packed_label = fn.readers.file(name=f"{reader_prefix}_VidLblReader", file_list=vlist_lbl, shard_id=cfg.shard_id, num_shards=cfg.num_shards, stick_to_shard=True)
                v_raw = fn.readers.numpy(name=f"{reader_prefix}_VidNpyReader", file_list=vpath_only, shard_id=cfg.shard_id, num_shards=cfg.num_shards, stick_to_shard=True)
                a_raw = fn.readers.numpy(name=f"{reader_prefix}_AudNpyReader", file_list=apath_only, shard_id=cfg.shard_id, num_shards=cfg.num_shards, stick_to_shard=True)
                packed_i64 = fn.cast(packed_label, dtype=types.INT64)
                sample_id = fn.cast(packed_i64 // LABEL_SCALE, dtype=types.INT32)
                start_f = fn.cast(packed_i64 % LABEL_SCALE, dtype=types.INT32)
                v_slice = fn.slice(v_raw, start_f, cfg.sequence_length, axes=[0], out_of_bounds_policy="pad", fill_values=0.0)
                a_slice = fn.slice(a_raw, start_f, cfg.sequence_length, axes=[0], out_of_bounds_policy="pad", fill_values=0.0)
                return v_slice.gpu(), a_slice.gpu(), sample_id
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
                video, label_vid = fn.readers.video(name=f"V{k}", device="gpu", file_list=vlists[k], sequence_length=cfg.sequence_length, pad_sequences=True,
                                                    shard_id=cfg.shard_id, num_shards=cfg.num_shards, stick_to_shard=True, random_shuffle=False, shuffle_after_epoch=False,
                                                    dtype=types.UINT8, file_list_frame_num=True, file_list_include_preceding_frame=True)
                audio_raw, label_cpu = fn.readers.file(name=f"A{k}", file_list=alists[k], shard_id=cfg.shard_id, num_shards=cfg.num_shards,
                                                       stick_to_shard=True, random_shuffle=False, shuffle_after_epoch=False)
                packed_i64 = fn.cast(label_cpu, dtype=types.INT64)
                start_f = fn.cast(packed_i64 % LABEL_SCALE, dtype=types.FLOAT)
                decoded, _ = fn.decoders.audio(audio_raw, sample_rate=cfg.sample_rate, downmix=False)
                start_s = start_f / cfg.fps
                shape_samples = (cfg.sequence_length - 1) * cfg.hop_length + cfg.window_length
                sliced = fn.slice(decoded.gpu(), start=fn.cast(start_s * cfg.sample_rate, dtype=types.INT32), shape=[int(shape_samples)], axes=[0], out_of_bounds_policy="pad")
                def to_mel_db(ch):
                    spec = fn.spectrogram(ch, nfft=cfg.nfft, window_length=cfg.window_length, window_step=cfg.hop_length, center_windows=False)
                    mel = fn.mel_filter_bank(spec, sample_rate=cfg.sample_rate, nfilter=cfg.mel_bins, freq_high=cfg.mel_fmax)
                    return fn.transpose(fn.to_decibels(mel, cutoff_db=cfg.db_cutoff), perm=[1, 0])
                mel = fn.stack(to_mel_db(sliced[:, 0]), to_mel_db(sliced[:, 1]), axis=0)
                outputs.extend([fn.transpose(video, perm=[0, 3, 1, 2]), label_vid, mel])
            return tuple(outputs)
        p = pipe(); p.build(); return p
    def __iter__(self): return self
    def __next__(self): return next(self.iterator)[0]

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
    def _bitmask_to_weapon_index(mask): return next((i for i in range(128) if (mask[i//64] >> np.uint64(i%64)) & 1), -1)
    def fetch(self, rec: SampleRecord) -> MetaFetchResult:
        env, T = self.store.open(rec.lmdb_path), rec.T_frames
        alive = np.zeros((T, 5), np.uint8); stats = np.zeros((T, 5, 3), np.float32)
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
                for slot in alive_slots: alive[f, slot] = 1
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
            # DALI mel is [B, C, T, Mels] -> stack(dim=2) -> [B, C, P, T, Mels]
            # Permute to [B, T, P, C, Mels]
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
        batch["alive_mask"] = gt.pop("alive_mask").bool()
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
        self.mse_loss = nn.MSELoss(reduction='none'); self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
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
            for key in ["keyboard_logits", "eco_logits", "inventory_logits"]:
                wkey = key.replace('_logits','')
                losses[wkey] += self.weights[wkey] * self._scalar_loss(self.bce_loss(p_pred[key], p_targ[key]).mean(-1), mask)
            losses['weapon'] += self.weights['weapon'] * self._scalar_loss(self.ce_loss(p_pred["active_weapon_idx"].view(B*T, -1), p_targ["active_weapon_idx"].view(B*T)), mask.view(B*T))
        for key in p_loss_keys: total_loss += losses[key]
        gs_pred, gs_targ = predictions["game_strategy"], targets["game_strategy"]
        frame_mask = alive_mask.any(dim=-1).float()
        losses['round_number'] = self.weights['round_number'] * self._scalar_loss(self.mse_loss(gs_pred["round_number"], gs_targ["round_number"].view(B, T, 1)).squeeze(-1), frame_mask)
        losses['round_state'] = self.weights['round_state'] * self._scalar_loss(self.bce_loss(gs_pred["round_state_logits"], gs_targ["round_state_logits"]).mean(-1), frame_mask)
        total_loss += losses['round_number'] + losses['round_state']
        pred_pos = torch.stack([p["pos_heatmap_logits"] for p in predictions["player"]], 2)
        targ_pos = torch.stack([p["pos_coords"] for p in targets["player"]], 2)
        if alive_mask.any():
            alive_flat = alive_mask.view(-1)
            pred_alive = pred_pos.view(-1, *pred_pos.shape[3:])[alive_flat]
            coord_alive = targ_pos.view(-1, 3)[alive_flat]
            target_heatmap = self._build_targets_heatmaps(coord_alive).to(dtype=pred_alive.dtype)
            losses['pos_heatmap'] = self.weights['pos_heatmap'] * self.bce_loss(pred_alive, target_heatmap).mean()
            total_loss += losses['pos_heatmap']
        pred_enemy, targ_enemy = gs_pred["enemy_pos_heatmap_logits"], gs_targ["enemy_pos_coords"]
        valid_enemy = (targ_enemy[..., 0] >= 0)
        if valid_enemy.any():
            b_idx, t_idx, p_idx = valid_enemy.nonzero(as_tuple=True)
            pred_sel = pred_enemy[b_idx, t_idx]; coord_sel = targ_enemy[b_idx, t_idx, p_idx]
            target_heatmap = self._build_targets_heatmaps(coord_sel).to(dtype=pred_sel.dtype)
            losses['enemy_heatmap'] = self.weights['enemy_heatmap'] * self.bce_loss(pred_sel, target_heatmap).mean()
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
            if k in self.params: self.params[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
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

def build_optimizer_scheduler(model, args, total_steps):
    """Builds the AdamW optimizer, parameter groups, and learning rate scheduler."""
    def split_decay(params):
        decay, no_decay = [], []
        for p in params:
            if not p.requires_grad: continue
            if p.ndim == 1: no_decay.append(p)
            else: decay.append(p)
        return decay, no_decay
    
    base_lr, opt_groups = args.lr, []
    for g in model.parameter_groups():
        lr = base_lr * float(g.get("lr_scale", 1.0))
        decay, no_decay = split_decay(g["params"])
        if decay: opt_groups.append({"params": decay, "lr": lr, "weight_decay": args.weight_decay})
        if no_decay: opt_groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(opt_groups, betas=(0.9, 0.95), eps=1e-8)
    def lr_lambda(step):
        if step < args.warmup_steps: return max(1e-8, step / max(1, args.warmup_steps))
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        return max(args.min_lr / args.lr, cosine)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    use_fp16 = getattr(model.module.cfg if hasattr(model, 'module') else model.cfg, "compute_dtype") == "fp16"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
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
    # FIX: This function no longer creates the store or builds team_rounds.
    # It receives them as arguments for efficiency.
    index = EpochIndex(T_frames=args.T_frames, seed=args.seed)
    records, id_map = index.build(team_rounds, epoch=epoch)
    
    # Use split name from team_rounds for directory, assuming all rounds are from same split.
    # A more robust solution might pass the split name explicitly.
    split_name_for_dir = "val" if last_batch_policy == "partial" else "train"
    fl_dir = os.path.join(args.run_dir, f"filelists_{split_name_for_dir}_e{epoch:04d}")

    if rank == 0:
        writer = FilelistWriter(fl_dir, use_precomputed=args.use_precomputed_embeddings, data_root=args.data_root)
        video_lists, audio_lists = writer.write(records)
    
    if world_size > 1: torch.distributed.barrier() # Ensure rank 0 writes lists first
    
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
    if world_size > 1: torch.distributed.barrier()

    device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
    dali_cfg = DaliConfig(sequence_length=args.T_frames, batch_size=args.batch_size, num_threads=args.dali_threads,
                          device_id=device_id, shard_id=rank, num_shards=world_size, seed=args.seed + epoch)
    dali_pipe = DaliInputPipeline(video_filelists=video_lists, audio_filelists=audio_lists, cfg=dali_cfg,
                                  use_precomputed=args.use_precomputed_embeddings, video_pathlists=video_path_lists,
                                  audio_pathlists=audio_path_lists, last_batch_policy=last_batch_policy)
    
    fetcher = LmdbMetaFetcher(store)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    assembler = BatchAssembler(id_map, fetcher, device=device)
    steps_per_epoch = (len(records) // world_size) // args.batch_size
    return dali_pipe.iterator, assembler, steps_per_epoch

@torch.no_grad()
def validate(dali_iter, assembler, model, loss_fn, args):
    """Runs the validation loop and returns averaged metrics."""
    model.eval()
    totals = defaultdict(float); count = 0
    try:
        while True:
            batch_raw = next(dali_iter)
            batch = assembler.assemble(batch_raw, args.use_precomputed_embeddings)
            preds = model(batch)
            loss, loss_dict = loss_fn(preds, batch["targets"], batch["alive_mask"])
            totals["total"] += loss.item()
            for k, v in loss_dict.items(): totals[k] += v
            count += 1
    except StopIteration: pass
    for k in totals: totals[k] /= max(1, count)
    return dict(totals)

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
    
    # FIX: Perform expensive data indexing ONCE before the training loop
    manifest = Manifest(args.data_root, args.manifest)
    store = LmdbStore()
    logging.info("Building training set index...")
    train_team_rounds = build_team_rounds(args.data_root, manifest.get_games("train"), store)
    logging.info("Building validation set index...")
    val_team_rounds = build_team_rounds(args.data_root, manifest.get_games("val"), store)

    # Estimate total steps based on the initial train set size
    steps_per_epoch = (len(train_team_rounds) // world_size) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    optimizer, scheduler, scaler = build_optimizer_scheduler(model, args, total_steps)
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
                batch_raw = next(train_iter)
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
                    scheduler.step()
                    if ema: ema.update(model)

                if rank == 0 and (global_step % args.log_every == 0):
                    writer.add_scalar("train/total_loss", total_loss.item() * args.accum_steps, global_step)
                    for k, v in loss_dict.items(): writer.add_scalar(f"train_loss/{k}", v, global_step)
                    writer.add_scalar("opt/lr", optimizer.param_groups[0]['lr'], global_step)

                if args.eval_every > 0 and (global_step > 0 and global_step % args.eval_every == 0):
                    # For validation, we typically use the same windowing strategy across epochs (epoch 0)
                    val_iter, val_asm, _ = build_epoch_loader(
                        args, 0, store, val_team_rounds, last_batch_policy="partial", rank=rank, world_size=world_size
                    )
                    val_metrics = validate(val_iter, val_asm, model, loss_fn, args)
                    if writer:
                        writer.add_scalar("val/total_loss", val_metrics["total"], global_step)
                        for k, v in val_metrics.items():
                            if k != "total": writer.add_scalar(f"val_loss/{k}", v, global_step)
                    if rank == 0 and val_metrics["total"] < best_val:
                        best_val = val_metrics["total"]
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
    # FIX: Perform one-time data indexing for the smoke test as well.
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
            batch_raw = next(dali_iter)
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
    # Training args
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--accum-steps", type=int, default=1)
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