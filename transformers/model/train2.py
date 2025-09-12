"""
train.py — Data loading layer (Steps 1–9)

This file implements the end-to-end data input subsystem for training, focusing on:
  1) Round discovery from each game's LMDB `_INFO` entry
  2) LMDB environment manager
  3) Epoch index / window sampling (fixed-length frame windows)
  4) Writing 5 aligned DALI filelists (one per POV)
  5) Building a single DALI video pipeline with 5 branches
  6) Tick-vector generator (64-tick demos → 32 fps, 2 ticks per frame)
  7) LMDB feature fetch & reinflation (alive-only → 5 fixed slots), mel + alive_mask
  8) Batch assembler (fuse DALI outputs with LMDB features)
  9) Determinism, DDP sharding, and instrumentation

Notes:
- This module does not define the training loop or losses; it only prepares data into a CS2Batch-like dict.
- We assume each per-game LMDB contains an `<demoname>_INFO` entry with the schema provided.
- Videos are recorded until player death, all starting at the round start tick. DALI zero-pads tails for short POVs.
- Demos are 64 ticks/s; videos are 32 fps ⇒ 2 ticks == 1 frame. LMDB keys exist only on the parity of round start.

If you only want to smoke-test the loader, see the __main__ section at the bottom for a minimal sanity run (DALI required).
"""
from __future__ import annotations

import os
import json
import math
import time
import random
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

import lmdb  # pip install lmdb
import msgpack  # pip install msgpack
import numpy as np
import msgpack_numpy as mpnp  # pip install msgpack-numpy

# Torch is used for device placement and to hand CUDA tensors to the model later
import torch

# --- Try to import DALI; provide a friendly message if missing ---
try:
    from nvidia.dali import fn, types, pipeline_def
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
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


@dataclass
class TeamRound:
    """Canonical record for a single team-round as discovered from `<demoname>_INFO`.
    The `pov_videos` list is ordered by roster slot index [0..4].
    """
    demoname: str
    lmdb_path: str
    round_num: int
    team: str  # "T" or "CT"
    start_tick: int
    end_tick: int
    pov_videos: List[str]
    fps: int = FPS
    tick_rate: int = TICK_RATE

    @property
    def parity(self) -> int:
        return self.start_tick % 2

    @property
    def frame_count(self) -> int:
        # Inclusive of start/end ticks; with 2 ticks per frame
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
    start_f: int               # start frame index from round start
    start_tick_win: int        # starting tick of the sampled window (parity-locked)
    T_frames: int
    parity: int


# -------------------------------
# Step 1: Manifest + `_INFO` reader
# -------------------------------

class Manifest:
    """Loads `manifest.json` and enumerates games per split.

    Expected flexible schema (examples):
      {
        "train": ["gameA", {"demoname": "gameB"}],
        "val":   ["gameC"],
        "test":  []
      }
    or
      {
        "train": [{"demoname": "gameA", "lmdb_path": ".../gameA.lmdb"}],
        ...
      }
    If `lmdb_path` is omitted, we resolve it as `<data_root>/lmdb/<demoname>.lmdb`.
    """
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
    """Opens and caches LMDB environments. Also reads `<demoname>_INFO` once.

    Use one `LmdbStore` instance per process/rank.
    """
    def __init__(self, max_readers: int = 512, map_size: int = 0, readahead: bool = False):
        self._envs: Dict[str, lmdb.Environment] = {}
        self._info_cache: Dict[str, Dict[str, Any]] = {}
        self._max_readers = max_readers
        self._map_size = map_size
        self._readahead = readahead

    def open(self, lmdb_path: str) -> lmdb.Environment:
        env = self._envs.get(lmdb_path)
        if env is None:
            env = lmdb.open(
                lmdb_path,
                readonly=True,
                lock=False,
                max_readers=self._max_readers,
                map_size=self._map_size or 1 << 30,  # 1GB virtual map is fine for read-only
                readahead=self._readahead,
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
        # Basic validation
        if info.get("demoname") and info["demoname"] != demoname:
            logging.warning("INFO.demoname (%s) != expected (%s)", info["demoname"], demoname)
        rounds = info.get("rounds")
        if not isinstance(rounds, list) or not rounds:
            raise ValueError(f"INFO.rounds malformed for {demoname}")
        self._info_cache[cache_key] = info
        return info


def build_team_rounds(data_root: str, games: List[Tuple[str, str]], store: LmdbStore) -> List[TeamRound]:
    """Enumerate all TeamRounds for the split by reading each game's `<demoname>_INFO`.

    Ensures absolute mp4 paths and basic validity.
    """
    team_rounds: List[TeamRound] = []
    for demoname, lmdb_path in games:
        info = store.read_info(demoname, lmdb_path)
        for r in info["rounds"]:
            pov_videos = r.get("pov_videos", [])
            if len(pov_videos) != 5:
                raise ValueError(f"{demoname} round {r.get('round_num')} missing POV videos (got {len(pov_videos)})")
            
            def _resolve_pov_path(data_root: str, demoname: str, pv: str) -> str:
                if os.path.isabs(pv):
                    return pv
                # Try common layouts, prefer the canonical recordings/<demoname> first
                candidates = [
                    os.path.join(data_root, "recordings", demoname, pv),
                    #os.path.join(data_root, pv),
                ]
                for c in candidates:
                    if os.path.exists(c):
                        return os.path.abspath(c)
                raise FileNotFoundError(
                    f"POV video not found for '{pv}'. Tried: {', '.join(map(os.path.abspath, candidates))}"
                )

            # inside build_team_rounds(...)
            pov_abs = [_resolve_pov_path(data_root, demoname, pv) for pv in pov_videos]

            # Validate files exist (fail fast). You can relax to warnings if needed.
            for p in pov_abs:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"POV video not found: {p}")
            tr = TeamRound(
                demoname=demoname,
                lmdb_path=os.path.abspath(lmdb_path),
                round_num=int(r["round_num"]),
                team=str(r["team"]).upper(),
                start_tick=int(r["start_tick"]),
                end_tick=int(r["end_tick"]),
                pov_videos=pov_abs,
            )
            if tr.start_tick >= tr.end_tick:
                raise ValueError(f"Invalid ticks for {demoname} round {tr.round_num} {tr.team}: {tr.start_tick} >= {tr.end_tick}")
            if tr.frame_count < 1:
                logging.warning("Zero-length frame window for %s r%03d %s", demoname, tr.round_num, tr.team)
                continue
            team_rounds.append(tr)
    logging.info("Discovered %d team-rounds across %d games.", len(team_rounds), len(games))
    return team_rounds


# -------------------------------
# Step 3: EpochIndex (window sampler)
# -------------------------------

class EpochIndex:
    """Samples a fixed-length frame window inside each TeamRound for a given epoch.

    The sampling is deterministic given (seed, epoch).
    """
    def __init__(self, T_frames: int, seed: int):
        self.T_frames = T_frames
        self.seed = seed
        self.records: List[SampleRecord] = []
        self.id_to_sample: Dict[int, SampleRecord] = {}

    def build(self, team_rounds: List[TeamRound], epoch: int, allow_padding: bool = True) -> Tuple[List[SampleRecord], Dict[int, SampleRecord]]:
        rnd = random.Random(self.seed + epoch)
        self.records.clear()
        self.id_to_sample.clear()

        sid = 0
        for tr in team_rounds:
            if tr.frame_count >= self.T_frames:
                start_f = rnd.randint(0, tr.frame_count - self.T_frames)
            else:
                if not allow_padding:
                    continue
                start_f = 0
            start_tick_win = tr.start_tick + TICKS_PER_FRAME * start_f
            rec = SampleRecord(
                sample_id=sid,
                demoname=tr.demoname,
                lmdb_path=tr.lmdb_path,
                round_num=tr.round_num,
                team=tr.team,
                pov_videos=tr.pov_videos,
                start_f=start_f,
                start_tick_win=start_tick_win,
                T_frames=self.T_frames,
                parity=tr.parity,
            )
            self.records.append(rec)
            self.id_to_sample[sid] = rec
            sid += 1

        logging.info("EpochIndex built: %d samples (T=%d)", len(self.records), self.T_frames)
        return self.records, self.id_to_sample


# -------------------------------
# Step 4: Filelist writer (5 aligned lists)
# -------------------------------

class FilelistWriter:
    """Writes five aligned DALI filelists (one per POV).

    Each line: "<abs/path.mp4>  <sample_id>  <start_f>  <end_f>".
    """
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def write(self, records: List[SampleRecord]) -> List[str]:
        paths = [os.path.join(self.out_dir, f"pov{k}.txt") for k in range(5)]
        files = [open(p, "w", encoding="utf-8") for p in paths]
        try:
            for rec in records:
                end_f = rec.start_f + rec.T_frames
                for k in range(5):
                    files[k].write(f"{rec.pov_videos[k]} {rec.sample_id} {rec.start_f} {end_f}\n")
        finally:
            for f in files:
                f.close()
        # Basic parity: line counts equal
        counts = [sum(1 for _ in open(p, "r", encoding="utf-8")) for p in paths]
        if len(set(counts)) != 1:
            raise RuntimeError(f"Filelist line count mismatch: {counts}")
        logging.info("Wrote DALI filelists to %s (N=%d)", self.out_dir, counts[0])
        return paths


# -------------------------------
# Step 5: DALI video pipeline (5 branches)
# -------------------------------

@dataclass
class DaliConfig:
    height: int = 224
    width: int = 224
    sequence_length: int = 512
    batch_size: int = 1
    num_threads: int = 4
    device_id: int = 0
    prefetch_queue_depth: int = 2
    additional_decode_surfaces: int = 8
    read_ahead: bool = True
    mean: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    shard_id: int = 0
    num_shards: int = 1


class DaliVideoInput:
    """Builds a single DALI pipeline with 5 video branches and yields CUDA tensors.

    Iterator output per batch:
      {
        "pov0": [B, T, C, H, W] (FP16 CUDA),
        "pov1": ...,
        ...,
        "labels0": [B],  # sample_ids from branch 0 (others must match)
        ...
      }
    """
    def __init__(self, filelists: List[str], cfg: DaliConfig):
        if not DALI_AVAILABLE:
            raise ImportError(
                f"NVIDIA DALI not available: {_DALI_IMPORT_ERROR}.\n"
                "Install DALI to use GPU video decode: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html"
            )
        assert len(filelists) == 5, "Expected 5 filelists (one per POV)"
        self.filelists = filelists
        self.cfg = cfg
        self.pipeline = self._build_pipeline()
        # Output map must match what the pipeline returns
        out_map = [
            "pov0", "labels0",
            "pov1", "labels1",
            "pov2", "labels2",
            "pov3", "labels3",
            "pov4", "labels4",
        ]
        self.iterator = DALIGenericIterator(
            [self.pipeline], out_map, auto_reset=True, dynamic_shape=False, fill_last_batch=False
        )

    def _build_pipeline(self):
        cfg = self.cfg

        @pipeline_def(batch_size=cfg.batch_size, num_threads=cfg.num_threads, device_id=cfg.device_id, prefetch_queue_depth=cfg.prefetch_queue_depth)
        def pipe():
            seqs = []
            labels = []
            for k in range(5):
                v, l = fn.readers.video_resize(
                    file_list=self.filelists[k],
                    file_list_frame_num=True,
                    sequence_length=cfg.sequence_length,
                    step=cfg.sequence_length,
                    random_shuffle=False,
                    pad_sequences=True,
                    read_ahead=cfg.read_ahead,
                    additional_decode_surfaces=cfg.additional_decode_surfaces,
                    resize_x=cfg.width,
                    resize_y=cfg.height,
                    dtype=types.UINT8,
                    shard_id=cfg.shard_id,
                    num_shards=cfg.num_shards,
                    stick_to_shard=True,
                )
                x = fn.crop_mirror_normalize(
                    v.gpu(),
                    output_layout="FCHW",
                    dtype=types.FLOAT16,
                    mean=cfg.mean,
                    std=cfg.std,
                )
                seqs.append(x)
                labels.append(l)
            # Return interleaved (pov0, lab0, pov1, lab1, ...)
            return tuple([seqs[0], labels[0], seqs[1], labels[1], seqs[2], labels[2], seqs[3], labels[3], seqs[4], labels[4]])

        return pipe()

    def __iter__(self):
        return self

    def __next__(self):
        out = next(self.iterator)  # list of dicts (one per GPU). We have one pipeline → [dict]
        batch = out[0]
        # Sanity: labels across POV branches must match
        l0 = batch["labels0"].cpu().numpy().tolist()
        for k in range(1, 5):
            lk = batch[f"labels{k}"].cpu().numpy().tolist()
            if lk != l0:
                raise RuntimeError(f"Label mismatch between branches: {l0} vs {lk}")
        return batch

    def reset(self):
        self.iterator.reset()


# -------------------------------
# Step 6: Tick vector generator
# -------------------------------

def ticks_for_window(start_tick_win: int, T_frames: int) -> np.ndarray:
    """Compute tick indices for a fixed-length window, parity-locked to round start.
    tick[f] = start_tick_win + 2*f
    """
    # Using int32 is enough and compact
    return start_tick_win + (np.arange(T_frames, dtype=np.int32) * TICKS_PER_FRAME)


# -------------------------------
# Step 7: LMDB fetch & reinflation
# -------------------------------

@dataclass
class FeatureFetchResult:
    mel_spectrogram: np.ndarray  # [T, 5, n_mels, mel_t]
    alive_mask: np.ndarray       # [T, 5] (uint8/bool)
    # Optional raw fields for future target extraction / debugging
    game_state_list: List[Dict[str, Any]]
    # per frame, per alive player; not inflated (debugging aid)
    player_data_alive_list: List[List[Any]]


class FeatureFetcher:
    """Fetches per-frame features from LMDB for a sampled window and reinflates to 5 slots.

    - Keys: "{demoname}_round_{rrr}_team_{team}_tick_{tick:08d}"
    - Values: msgpack with {"game_state": gs, "player_data": pdl}, where pdl is alive-only, roster-ordered.
    - Reinflation uses `gs['team_alive']` 5-bit mask to restore fixed 5-slot layout.
    - Mel spectrogram is extracted per alive player if present; dead/missing → zeros.
    """
    def __init__(self, store: LmdbStore):
        self.store = store
        # Cache mel shape per LMDB (n_mels, mel_t). Discovered lazily.
        self._mel_shape_hint: Dict[str, Tuple[int, int]] = {}

    @staticmethod
    def _key(demoname: str, round_num: int, team: str, tick: int) -> bytes:
        return f"{demoname}_round_{round_num:03d}_team_{team}_tick_{tick:08d}".encode("utf-8")

    @staticmethod
    def _parse_player_entry(entry: Any) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
        """Extract a (pi_dict, mel_array_or_None) from a `player_data` entry.
        Supports either tuple-like (pi, maybe_inventory, mel) or dict-like.
        """
        if isinstance(entry, dict):
            pi = entry.get("pi") or entry
            mel = entry.get("mel")
            if isinstance(mel, list):
                mel = np.asarray(mel, dtype=np.float32)
            return pi, mel
        elif isinstance(entry, (list, tuple)):
            # Heuristic: last element may be mel, first is pi
            pi = entry[0] if len(entry) > 0 else {}
            mel = entry[-1] if len(entry) > 0 else None
            if isinstance(mel, list):
                mel = np.asarray(mel, dtype=np.float32)
            return pi, mel
        # Fallback: unknown structure
        return {}, None

    def _ensure_mel_shape(self, lmdb_path: str, env: lmdb.Environment, demoname: str, round_num: int, team: str, start_tick_win: int, T_frames: int) -> Tuple[int, int]:
        """Determine (n_mels, mel_t) if unknown by peeking at the first non-None mel in the window.
        If no mel is found, default to (128, 6) but log a warning.
        """
        if lmdb_path in self._mel_shape_hint:
            return self._mel_shape_hint[lmdb_path]
        with env.begin(write=False) as txn:
            for f in range(T_frames):
                tick = int(start_tick_win + TICKS_PER_FRAME * f)
                blob = txn.get(self._key(demoname, round_num, team, tick))
                if not blob:
                    continue
                payload = msgpack.unpackb(blob, raw=False, object_hook=mpnp.decode)
                pdl = payload.get("player_data") or []
                for ent in pdl:
                    _, mel = self._parse_player_entry(ent)
                    if isinstance(mel, np.ndarray) and mel.ndim == 2:
                        self._mel_shape_hint[lmdb_path] = (mel.shape[0], mel.shape[1])
                        return self._mel_shape_hint[lmdb_path]
        logging.warning("Could not infer mel shape in %s; defaulting to (128, 6)", lmdb_path)
        self._mel_shape_hint[lmdb_path] = (128, 6)
        return self._mel_shape_hint[lmdb_path]

    def fetch(self, rec: SampleRecord) -> FeatureFetchResult:
        env = self.store.open(rec.lmdb_path)
        n_mels, mel_t = self._ensure_mel_shape(rec.lmdb_path, env, rec.demoname, rec.round_num, rec.team, rec.start_tick_win, rec.T_frames)

        T = rec.T_frames
        alive_mask = np.zeros((T, 5), dtype=np.uint8)
        mel = np.zeros((T, 5, n_mels, mel_t), dtype=np.float32)
        gs_list: List[Dict[str, Any]] = []
        pdl_alive_list: List[List[Any]] = []

        ticks = ticks_for_window(rec.start_tick_win, T)
        with env.begin(write=False) as txn:
            for f, tick in enumerate(ticks.tolist()):
                blob = txn.get(self._key(rec.demoname, rec.round_num, rec.team, int(tick)))
                if not blob:
                    gs_list.append({})
                    pdl_alive_list.append([])
                    continue
                payload = msgpack.unpackb(blob, raw=False, object_hook=mpnp.decode)
                gs = payload.get("game_state") or {}
                pdl = payload.get("player_data") or []
                gs_list.append(gs)
                pdl_alive_list.append(pdl)

                # Reinflation: map alive-only list back to fixed 5 slots using team_alive bitmask
                mask_bits = int(gs.get("team_alive", 0))
                j = 0
                for slot in range(5):
                    if (mask_bits >> slot) & 1:
                        alive_mask[f, slot] = 1
                        if j < len(pdl):
                            pi, mel_arr = self._parse_player_entry(pdl[j])
                            j += 1
                            if isinstance(mel_arr, np.ndarray) and mel_arr.ndim == 2:
                                # Clip or pad mel to (n_mels, mel_t) if needed
                                mh, mw = mel_arr.shape
                                h = min(mh, n_mels)
                                w = min(mw, mel_t)
                                mel[f, slot, :h, :w] = mel_arr[:h, :w]
                        # else: no entry despite alive bit → leave zeros (robustness)
                    else:
                        # dead slot: leave zeros
                        pass
        return FeatureFetchResult(mel_spectrogram=mel, alive_mask=alive_mask, game_state_list=gs_list, player_data_alive_list=pdl_alive_list)


# -------------------------------
# Step 8: Batch assembler (fuse DALI + LMDB)
# -------------------------------

class BatchAssembler:
    """Converts a raw DALI batch + FeatureFetcher outputs into model-ready tensors.

    Output dict keys:
      - images: [B, T, 5, C, H, W]  (CUDA, fp16)
      - mel_spectrogram: [B, T, 5, n_mels, mel_t] (CUDA, float32/16)
      - alive_mask: [B, T, 5] (CUDA, uint8)
      - meta: {sample_ids, demonames, round_nums, teams}

    Target extraction for heads can be added later using `game_state_list` and `player_data_alive_list`.
    """
    def __init__(self, id_to_sample: Dict[int, SampleRecord], fetcher: FeatureFetcher, device: Optional[torch.device] = None, amp_prefer_bf16: bool = False):
        self.id_to_sample = id_to_sample
        self.fetcher = fetcher
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_prefer_bf16 = amp_prefer_bf16

    def assemble(self, dali_batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        # 1) Stack the five POV tensors → [B, T, 5, C, H, W]
        povs = []
        labels = None
        for k in range(5):
            x = dali_batch[f"pov{k}"]  # torch.cuda.FloatTensor (fp16) with layout FCHW → [B, T, C, H, W]
            povs.append(x)
            lb = dali_batch[f"labels{k}"]
            if labels is None:
                labels = lb
        # Shape: list of 5 tensors [B,T,C,H,W] → [B,T,5,C,H,W]
        images = torch.stack(povs, dim=2)
        if self.amp_prefer_bf16 and images.dtype == torch.float16:
            images = images.to(torch.bfloat16)

        # 2) For each sample in the batch, fetch LMDB features by sample_id
        sample_ids = labels.cpu().numpy().tolist()
        mel_list = []
        mask_list = []
        meta = {"sample_ids": sample_ids, "demonames": [], "round_nums": [], "teams": []}
        for sid in sample_ids:
            rec = self.id_to_sample[int(sid)]
            meta["demonames"].append(rec.demoname)
            meta["round_nums"].append(rec.round_num)
            meta["teams"].append(rec.team)
            feats = self.fetcher.fetch(rec)
            mel_list.append(torch.from_numpy(feats.mel_spectrogram))
            mask_list.append(torch.from_numpy(feats.alive_mask))
        mel = torch.stack(mel_list, dim=0).to(self.device, non_blocking=True)  # [B,T,5,n_mels,mel_t]
        mel = mel.unsqueeze(3)  # [B,T,5,1,n_mels,mel_t]
        alive_mask = torch.stack(mask_list, dim=0).to(self.device, non_blocking=True)  # [B,T,5]

        # 3) Ensure images are on the target device
        images = images.to(self.device, non_blocking=True)

        batch = {
            "images": images,
            "mel_spectrogram": mel,
            "alive_mask": alive_mask,
            "meta": meta,
        }
        return batch


# -------------------------------
# Step 9: Determinism, DDP helpers, instrumentation
# -------------------------------

class Timer:
    """Simple timer context for instrumentation."""
    def __init__(self, name: str):
        self.name = name
        self.t0 = 0.0
        self.dt = 0.0

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.dt = time.time() - self.t0


def get_ddp_info() -> Tuple[int, int]:
    """Return (shard_id, num_shards) using torch.distributed if available; else (0,1)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    # torchrun sets LOCAL_RANK; use it for device id, but sharding remains 0/1
    return 0, 1


# -------------------------------
# Optional: minimal driver for sanity
# -------------------------------

@dataclass
class DataArgs:
    data_root: str
    manifest: str
    split: str = "train"
    run_dir: str = "runs/exp1"
    T_frames: int = 512
    height: int = 224
    width: int = 224
    batch_size: int = 1
    seed: int = 42
    dali_threads: int = 4
    decode_surfaces: int = 8


def build_data_iter(args: DataArgs):
    """Convenience function that wires Steps 1→8 and returns (dali_iter, assembler).
    This is useful for plugging straight into a training loop later.
    """
    os.makedirs(args.run_dir, exist_ok=True)
    logging.info("=== Building data loader for split=%s ===", args.split)

    # Step 1: Manifest + INFO
    manifest = Manifest(args.data_root, args.manifest)
    games = manifest.get_games(args.split)
    store = LmdbStore()
    team_rounds = build_team_rounds(args.data_root, games, store)

    # Step 3: EpochIndex
    index = EpochIndex(T_frames=args.T_frames, seed=args.seed)
    records, id_map = index.build(team_rounds, epoch=0, allow_padding=True)

    # Step 4: Filelists
    fl_dir = os.path.join(args.run_dir, "epoch_0")
    filelists = FilelistWriter(fl_dir).write(records)

    # Step 5: DALI input
    shard_id, num_shards = get_ddp_info()
    # Resolve normalization & input size from the ViT model (via timm) if available
    dali_mean = (0.0, 0.0, 0.0)
    dali_std = (1.0, 1.0, 1.0)
    height = args.height
    width = args.width
    try:
        from model import CS2Config, resolve_vit_preprocess  # resolve from the actual model config
        vit_name = getattr(CS2Config, "vit_name_timm", "vit_base_patch14_dinov2.lvd142m")
        pp = resolve_vit_preprocess(vit_name)
        # timm returns mean/std in 0..1; DALI expects them in 0..255 when the input is uint8.
        mean_01 = tuple(pp["mean"])
        std_01 = tuple(pp["std"])
        dali_mean = tuple(float(m) * 255.0 for m in mean_01)
        dali_std  = tuple(float(s) * 255.0 for s in std_01)
        height = int(pp["height"]) or height
        width  = int(pp["width"])  or width
        logging.info("Using model-driven preprocessing: vit=%s, HxW=%dx%d, mean=%s, std=%s",
                    vit_name, height, width, mean_01, std_01)
    except Exception as e:
        logging.warning("Falling back to ImageNet defaults for preprocessing (%s).", e)
        mean_01 = (0.485, 0.456, 0.406)
        std_01  = (0.229, 0.224, 0.225)
        dali_mean = tuple(float(m) * 255.0 for m in mean_01)
        dali_std  = tuple(float(s) * 255.0 for s in std_01)

    dali_cfg = DaliConfig(
        height=height,
        width=width,
        sequence_length=args.T_frames,
        batch_size=args.batch_size,
        num_threads=args.dali_threads,
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else 0,
        additional_decode_surfaces=args.decode_surfaces,
        mean=dali_mean,                # <-- added
        std=dali_std,                  # <-- added
        shard_id=shard_id,
        num_shards=num_shards,
    )
    dali_iter = DaliVideoInput(filelists, dali_cfg)

    # Step 7: Feature fetcher
    fetcher = FeatureFetcher(store)

    # Step 8: Assembler
    assembler = BatchAssembler(id_map, fetcher, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    return dali_iter, assembler


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    # Example CLI-less smoke test (will fail if DALI missing or paths are placeholders)
    data_root = os.environ.get("DATA_ROOT", "data")
    manifest_path = os.path.join(data_root, "manifest.json")
    args = DataArgs(data_root=data_root, manifest=manifest_path, batch_size=2)

    if not DALI_AVAILABLE:
        logging.error("DALI is not available: %s", _DALI_IMPORT_ERROR)
    else:
        try:
            dali_iter, assembler = build_data_iter(args)
            with Timer("one batch") as t:
                batch_raw = next(iter(dali_iter))  # dict with pov0..pov4, labels0..labels4
            logging.info("DALI decode fetched in %.3fs", t.dt)
            with Timer("assemble") as t2:
                batch = assembler.assemble(batch_raw)
            logging.info("Assembled batch in %.3fs; images=%s, mel=%s, alive=%s", t2.dt, tuple(batch["images"].shape), tuple(batch["mel_spectrogram"].shape), tuple(batch["alive_mask"].shape))
        except Exception as e:
            logging.exception("Data loader smoke test failed: %s", e)
