"""
train2.py — Data loading layer (Steps 1–9)

This file implements the end-to-end data input subsystem for training, focusing on:
  1) Round discovery from each game's LMDB `_INFO` entry (including video and audio paths)
  2) LMDB environment manager for metadata
  3) Epoch index / window sampling (fixed-length frame windows)
  4) Writing 10 aligned DALI filelists (one per POV for video and audio)
  5) Building a single DALI pipeline with 10 branches (5 video, 5 audio)
  6) Tick-vector generator (64-tick demos → 32 fps, 2 ticks per frame)
  7) LMDB metadata fetch (alive_mask)
  8) Batch assembler (fuse DALI outputs with LMDB features)
  9) Determinism, DDP sharding, and instrumentation

Notes:
- This module defines the data preparation pipeline into a CS2Batch-like dict.
- Mel spectrograms are now generated ON-THE-FLY by DALI from .wav files.
- LMDB is now only used for per-frame metadata (e.g., alive_mask, game state).
- We assume each per-game LMDB contains an `<demoname>_INFO` entry with pov_videos and pov_audio.
- Videos are recorded until player death, all starting at the round start tick. DALI zero-pads tails for short POVs.
- Demos are 64 ticks/s; videos are 32 fps ⇒ 2 ticks == 1 frame. LMDB keys exist only on the parity of round start.

If you only want to smoke-test the loader, see the __main__ section at the bottom for a minimal sanity run (DALI required).
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
        # This POV has no frames, signal an error.
        return -1, -1

    # The last valid frame index in this specific POV video (0-indexed).
    pov_last_f = pov_frame_count - 1

    # Clamp the requested start frame to be a valid index within the POV.
    # If req_start_f is past the end, we start at the very last frame.
    start_f = min(req_start_f, pov_last_f)

    # The end frame for the filelist is exclusive and clamped to the total frame count.
    end_f_exclusive = min(start_f + T_frames, pov_frame_count)

    return start_f, end_f_exclusive


@dataclass
class TeamRound:
    """Canonical record for a single team-round as discovered from `<demoname>_INFO`.
    The `pov_videos` and `pov_audio` lists are ordered by roster slot index [0..4].
    """
    demoname: str
    lmdb_path: str
    round_num: int
    team: str  # "T" or "CT"
    start_tick: int
    end_tick: int
    pov_videos: List[str]
    pov_audio: List[str]  # <-- ADDED
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
    pov_audio: List[str]  # <-- ADDED
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

    Ensures absolute mp4/wav paths and basic validity.
    """
    team_rounds: List[TeamRound] = []
    for demoname, lmdb_path in games:
        info = store.read_info(demoname, lmdb_path)
        for r in info["rounds"]:
            pov_videos = r.get("pov_videos", [])
            pov_audio = r.get("pov_audio", [])  # <-- ADDED
            if len(pov_videos) != 5:
                raise ValueError(f"{demoname} round {r.get('round_num')} missing POV videos (got {len(pov_videos)})")
            if len(pov_audio) != 5:
                raise ValueError(f"{demoname} round {r.get('round_num')} missing POV audio (got {len(pov_audio)})")

            def _resolve_media_path(data_root: str, demoname: str, p: str) -> str:
                if os.path.isabs(p):
                    return p
                # Try common layouts, prefer the canonical recordings/<demoname> first
                candidate = os.path.join(data_root, "recordings", demoname, p)
                if os.path.exists(candidate):
                    return os.path.abspath(candidate)
                raise FileNotFoundError(f"Media file not found: {p}. Tried: {os.path.abspath(candidate)}")

            pov_videos_abs = [_resolve_media_path(data_root, demoname, pv) for pv in pov_videos]
            pov_audio_abs = [_resolve_media_path(data_root, demoname, pa) for pa in pov_audio]

            # Validate files exist (fail fast).
            for p in pov_videos_abs + pov_audio_abs:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Media file not found: {p}")

            tr = TeamRound(
                demoname=demoname,
                lmdb_path=os.path.abspath(lmdb_path),
                round_num=int(r["round_num"]),
                team=str(r["team"]).upper(),
                start_tick=int(r["start_tick"]),
                end_tick=int(r["end_tick"]),
                pov_videos=pov_videos_abs,
                pov_audio=pov_audio_abs,
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
                pov_audio=tr.pov_audio, # <-- ADDED
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
# Step 4: Filelist writer (10 aligned lists)
# -------------------------------

class FilelistWriter:
    """Writes ten aligned DALI filelists (five for video, five for audio).

    Video line: "<abs/path.mp4>  <sample_id>  <start_f>  <end_f_exclusive>".
    Audio line: "<abs/path.wav>   <sample_id>".
    """
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def write(self, records: List[SampleRecord]) -> Tuple[List[str], List[str]]:
        vid_paths = [os.path.join(self.out_dir, f"pov{k}_video.txt") for k in range(5)]
        aud_paths = [os.path.join(self.out_dir, f"pov{k}_audio.txt") for k in range(5)]
        
        vid_files = [open(p, "w", encoding="utf-8") for p in vid_paths]
        aud_files = [open(p, "w", encoding="utf-8") for p in aud_paths]
        try:
            for rec in records:
                for k in range(5):
                    # Video line
                    pov_path = rec.pov_videos[k]
                    ticks = ticks_from_filename(pov_path)
                    if not ticks:
                        raise ValueError(f"Could not parse start/end ticks from filename: {pov_path}")

                    pov_start_tick, pov_end_tick = ticks
                    start, end_exclusive = clamp_window_to_pov(rec.start_f, rec.T_frames, pov_start_tick, pov_end_tick)

                    if start < 0:
                        raise ValueError(f"POV segment {os.path.basename(pov_path)} for sample {rec.sample_id} has no frames.")

                    vid_files[k].write(f"{pov_path} {rec.sample_id} {start} {end_exclusive}\n")
                    
                    # Audio line
                    aud_files[k].write(f"{rec.pov_audio[k]} {rec.sample_id}\n")
        finally:
            for f in vid_files + aud_files:
                f.close()

        counts = [sum(1 for _ in open(p, "r", encoding="utf-8")) for p in vid_paths + aud_paths]
        if len(set(counts)) != 1:
            raise RuntimeError(f"Filelist line count mismatch: {counts}")
        logging.info("Wrote DALI filelists to %s (N=%d)", self.out_dir, counts[0])
        return vid_paths, aud_paths


# -------------------------------
# Step 5: DALI pipeline (video + audio)
# -------------------------------

@dataclass
class DaliConfig:
    # -------- Video --------
    height: int = 224
    width: int = 224
    sequence_length: int = 512
    mean: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    fps: float = 30.0  # used to convert label (frame index) -> seconds

    # Optional explicit resize targets (if None, mirror width/height)
    resize_h: Optional[int] = None
    resize_w: Optional[int] = None

    # Reader behavior
    shuffle: bool = False
    initial_fill: int = 16  # prefill for random shuffling

    # -------- Audio --------
    sample_rate: float = 24000.0

    # Keep your original names, but the pipeline uses these aliases:
    # (n_mels <-> mel_bins), (n_fft <-> nfft), (win_length <-> window_length)
    n_mels: Optional[int] = 128
    mel_bins: Optional[int] = None

    n_fft: Optional[int] = 1024
    nfft: Optional[int] = None

    win_length: Optional[int] = 1024
    window_length: Optional[int] = None

    hop_length: int = 312  # results in ~T_frames spectrograms

    # Mel / dB params used by the pipeline
    mel_fmin: float = 0.0
    mel_fmax: Optional[float] = None  # if None, set to sample_rate / 2 in __post_init__
    db_cutoff: float = 80.0

    # -------- Common / DALI runtime --------
    batch_size: int = 1
    num_threads: int = 4
    device_id: int = 0
    prefetch_queue_depth: int = 2
    additional_decode_surfaces: int = 8
    read_ahead: bool = True
    shard_id: int = 0
    num_shards: int = 1
    seed: int = 42

    def __post_init__(self):
        # Mirror resize_{h,w} from {height,width} if not explicitly set
        if self.resize_h is None:
            self.resize_h = self.height
        if self.resize_w is None:
            self.resize_w = self.width

        # Sync alias pairs (prefer explicitly provided alias if set)
        if self.mel_bins is None and self.n_mels is not None:
            self.mel_bins = self.n_mels
        if self.n_mels is None and self.mel_bins is not None:
            self.n_mels = self.mel_bins

        if self.nfft is None and self.n_fft is not None:
            self.nfft = self.n_fft
        if self.n_fft is None and self.nfft is not None:
            self.n_fft = self.nfft

        if self.window_length is None and self.win_length is not None:
            self.window_length = self.win_length
        if self.win_length is None and self.window_length is not None:
            self.win_length = self.window_length

        # Default mel_fmax to Nyquist if not set
        if self.mel_fmax is None:
            self.mel_fmax = float(self.sample_rate) / 2.0

        # Basic sanity checks
        assert self.sequence_length > 0, "sequence_length must be > 0"
        assert self.batch_size > 0, "batch_size must be > 0"
        assert self.num_threads > 0, "num_threads must be > 0"
        assert self.nfft > 0, "nfft must be > 0"
        assert self.window_length > 0, "window_length must be > 0"
        assert self.hop_length > 0, "hop_length must be > 0"
        assert len(self.mean) == 3 and len(self.std) == 3, "mean/std must be RGB triplets"



class DaliInputPipeline:
    """Builds a single DALI pipeline with 5 video and 5 audio branches.

    Iterator output per batch:
      {
        "pov0": [B, T, C, H, W] (FP16 CUDA), "labels0": [B], "mel0": [B, T_aud, n_mels],
        "pov1": ..., "labels1": ..., "mel1": ...,
        ...
      }
    """
    def __init__(self, video_filelists: List[str], audio_filelists: List[str], cfg: DaliConfig):
        if not DALI_AVAILABLE:
            raise ImportError(
                f"NVIDIA DALI not available: {_DALI_IMPORT_ERROR}.\n"
                "Install DALI to use GPU video/audio processing: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html"
            )
        assert len(video_filelists) == 5, "Expected 5 video filelists"
        assert len(audio_filelists) == 5, "Expected 5 audio filelists"
        self.video_filelists = video_filelists
        self.audio_filelists = audio_filelists
        self.cfg = cfg
        self.pipeline = self._build_pipeline()

        out_map = []
        for k in range(5):
            out_map.extend([f"pov{k}", f"labels{k}", f"mel{k}"])

        self.iterator = DALIGenericIterator(
            [self.pipeline], out_map, auto_reset=True, dynamic_shape=False, fill_last_batch=False
        )


    def _build_pipeline(self):
        cfg = self.cfg
        video_filelists = self.video_filelists
        audio_filelists = self.audio_filelists
        FPS = cfg.fps  # label is start frame; convert to seconds via FPS

        @pipeline_def(
            batch_size=cfg.batch_size,
            num_threads=cfg.num_threads,
            device_id=cfg.device_id,
            prefetch_queue_depth=cfg.prefetch_queue_depth,
            seed=getattr(cfg, "seed", 42),
        )
        def pipe():
            outputs = []

            for k, (vlist, alist) in enumerate(zip(video_filelists, audio_filelists)):
                # -------------------------
                # VIDEO branch (GPU-only)
                # -------------------------
                # The video reader is on the GPU. We only need the video output from it.
                # We will ignore its label output because it would be on the GPU.
                video, _ = fn.readers.video(
                    name=f"VideoReader{k}",
                    device="gpu",
                    file_list=vlist,
                    sequence_length=cfg.sequence_length,
                    file_list_frame_num=True,
                    pad_sequences=True,
                    shard_id=cfg.shard_id,
                    num_shards=cfg.num_shards,
                    random_shuffle=getattr(cfg, "shuffle", False),
                    pad_last_batch=True,
                    initial_fill=getattr(cfg, "initial_fill", 16),
                    additional_decode_surfaces=getattr(cfg, "additional_decode_surfaces", 8),
                    read_ahead=getattr(cfg, "read_ahead", True),
                    dtype=types.UINT8,
                )

                frames = fn.resize(
                    video,
                    resize_x=cfg.resize_w,
                    resize_y=cfg.resize_h,
                    device="gpu",
                )
                frames = fn.crop_mirror_normalize(
                    frames,
                    device="gpu",
                    dtype=types.FLOAT16,
                    output_layout="FCHW",
                    mean=cfg.mean,
                    std=cfg.std,
                )

                # -------------------------
                # AUDIO branch (CPU -> GPU)
                # -------------------------
                # The audio file reader is a CPU operator. It can give us the label
                # (the sample_id) directly on the CPU, which is what we need.
                audio_raw, label_cpu = fn.readers.file(
                    name=f"AudioReader{k}",
                    file_list=alist,
                    shard_id=cfg.shard_id,
                    num_shards=cfg.num_shards,
                    random_shuffle=getattr(cfg, "shuffle", False),
                )

                decoded_audio, _ = fn.decoders.audio(
                    audio_raw,
                    sample_rate=cfg.sample_rate,
                    downmix=True,
                )

                # Compute slice arguments on the CPU using the `label_cpu` tensor.
                start_in_seconds = label_cpu / FPS
                start_in_samples_f = start_in_seconds * cfg.sample_rate
                start_in_samples = fn.cast(start_in_samples_f, dtype=types.INT32)

                # The slice shape can be a constant Python integer.
                shape_in_seconds = cfg.sequence_length / FPS
                shape_in_samples = int(shape_in_seconds * cfg.sample_rate)

                # Move audio to GPU *then* slice on GPU, using CPU arguments.
                audio_gpu = decoded_audio.gpu()
                sliced_audio_gpu = fn.slice(
                    audio_gpu,
                    start=start_in_samples,
                    shape=shape_in_samples,
                    axes=[0],
                    normalized_anchor=False,
                    normalized_shape=False,
                )

                # Spectrogram + Mel on GPU
                spec = fn.spectrogram(
                    sliced_audio_gpu,
                    nfft=cfg.nfft,
                    window_length=cfg.window_length,
                    window_step=cfg.hop_length,
                    center_windows=False,
                )
                mel = fn.mel_filter_bank(
                    spec,
                    sample_rate=cfg.sample_rate,
                    nfilter=cfg.mel_bins,
                    freq_low=cfg.mel_fmin,
                    freq_high=cfg.mel_fmax,
                )
                mel_db = fn.to_decibels(
                    mel,
                    multiplier=10.0,
                    reference=1.0,
                    cutoff_db=cfg.db_cutoff,
                )

                # IMPORTANT: flatten outputs. Use the CPU label for the output.
                outputs += [frames, label_cpu, mel_db]

            return tuple(outputs)

        pipe_inst = pipe()
        pipe_inst.build()
        return pipe_inst


    def __iter__(self):
        return self

    def __next__(self):
        out = next(self.iterator)
        batch = out[0]
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
    """Compute tick indices for a fixed-length window, parity-locked to round start."""
    return start_tick_win + (np.arange(T_frames, dtype=np.int32) * TICKS_PER_FRAME)


# -------------------------------
# Step 7: LMDB metadata fetch
# -------------------------------

@dataclass
class MetaFetchResult:
    alive_mask: np.ndarray       # [T, 5] (uint8/bool)
    game_state_list: List[Dict[str, Any]]


class LmdbMetaFetcher:
    """Fetches per-frame metadata (alive mask) from LMDB for a sampled window.
    The mel spectrogram is now generated by DALI.
    """
    def __init__(self, store: LmdbStore):
        self.store = store

    @staticmethod
    def _key(demoname: str, round_num: int, team: str, tick: int) -> bytes:
        return f"{demoname}_round_{round_num:03d}_team_{team}_tick_{tick:08d}".encode("utf-8")

    def fetch(self, rec: SampleRecord) -> MetaFetchResult:
        env = self.store.open(rec.lmdb_path)
        T = rec.T_frames
        alive_mask = np.zeros((T, 5), dtype=np.uint8)
        gs_list: List[Dict[str, Any]] = []

        ticks = ticks_for_window(rec.start_tick_win, T)
        with env.begin(write=False) as txn:
            for f, tick in enumerate(ticks.tolist()):
                blob = txn.get(self._key(rec.demoname, rec.round_num, rec.team, int(tick)))
                if not blob:
                    gs_list.append({})
                    continue
                
                payload = msgpack.unpackb(blob, raw=False, object_hook=mpnp.decode)
                gs = payload.get("game_state") or {}
                gs_list.append(gs)

                mask_bits = int(gs.get("team_alive", [0])[0])
                for slot in range(5):
                    if (mask_bits >> slot) & 1:
                        alive_mask[f, slot] = 1

        return MetaFetchResult(alive_mask=alive_mask, game_state_list=gs_list)


# -------------------------------
# Step 8: Batch assembler (fuse DALI + LMDB)
# -------------------------------

class BatchAssembler:
    """Converts a raw DALI batch + LmdbMetaFetcher outputs into model-ready tensors."""
    def __init__(self, id_to_sample: Dict[int, SampleRecord], fetcher: LmdbMetaFetcher, device: Optional[torch.device] = None, amp_prefer_bf16: bool = False):
        self.id_to_sample = id_to_sample
        self.fetcher = fetcher
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_prefer_bf16 = amp_prefer_bf16

    def assemble(self, dali_batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        # 1) Stack POV video and mel spectrogram tensors from DALI
        povs = []
        mels = []
        labels = dali_batch["labels0"]
        for k in range(5):
            povs.append(dali_batch[f"pov{k}"])
            mel_k = dali_batch[f"mel{k}"] # [B, T_aud, n_mels]
            mels.append(mel_k)

        images = torch.stack(povs, dim=2)  # [B, T, 5, C, H, W]
        if self.amp_prefer_bf16 and images.dtype == torch.float16:
            images = images.to(torch.bfloat16)

        # [B, 5, T_aud, n_mels] -> [B, T_aud, 5, n_mels] -> [B, T_aud, 5, 1, n_mels]
        mel = torch.stack(mels, dim=1).permute(0, 2, 1, 3).unsqueeze(3)

        # 2) For each sample, fetch alive_mask from LMDB
        sample_ids = labels.cpu().numpy().tolist()
        mask_list = []
        meta = {"sample_ids": sample_ids, "demonames": [], "round_nums": [], "teams": []}
        for sid in sample_ids:
            rec = self.id_to_sample[int(sid)]
            meta["demonames"].append(rec.demoname)
            meta["round_nums"].append(rec.round_num)
            meta["teams"].append(rec.team)
            feats = self.fetcher.fetch(rec)
            mask_list.append(torch.from_numpy(feats.alive_mask))
        
        alive_mask = torch.stack(mask_list, dim=0).to(self.device, non_blocking=True)  # [B,T,5]

        # 3) Move all tensors to the target device
        images = images.to(self.device, non_blocking=True)
        mel = mel.to(self.device, non_blocking=True, dtype=images.dtype)

        # 4) Pad mel spectrogram if its time dimension doesn't match images
        if images.shape[1] != mel.shape[1]:
            pad_len = images.shape[1] - mel.shape[1]
            if pad_len > 0:
                mel = torch.nn.functional.pad(mel, (0, 0, 0, 0, 0, 0, 0, pad_len), "constant", 0)
            mel = mel[:, :images.shape[1], ...]

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
    """Convenience function that wires Steps 1→8 and returns (dali_iter, assembler)."""
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
    video_filelists, audio_filelists = FilelistWriter(fl_dir).write(records)

    # Step 5: DALI input
    shard_id, num_shards = get_ddp_info()
    dali_mean, dali_std, height, width = (0.0,)*3, (1.0,)*3, args.height, args.width
    try:
        from model import CS2Config, resolve_vit_preprocess
        vit_name = getattr(CS2Config, "vit_name_timm", "vit_base_patch14_dinov2.lvd142m")
        pp = resolve_vit_preprocess(vit_name)
        mean_01, std_01 = tuple(pp["mean"]), tuple(pp["std"])
        dali_mean = tuple(float(m) * 255.0 for m in mean_01)
        dali_std = tuple(float(s) * 255.0 for s in std_01)
        height, width = int(pp["height"]) or height, int(pp["width"]) or width
        logging.info("Using model-driven preprocessing: vit=%s, HxW=%dx%d, mean=%s, std=%s",
                     vit_name, height, width, mean_01, std_01)
    except Exception as e:
        logging.warning("Falling back to default preprocessing (%s).", e)

    dali_cfg = DaliConfig(
        height=height, width=width, sequence_length=args.T_frames,
        batch_size=args.batch_size, num_threads=args.dali_threads,
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else 0,
        additional_decode_surfaces=args.decode_surfaces,
        mean=dali_mean, std=dali_std,
        shard_id=shard_id, num_shards=num_shards,
    )
    dali_iter = DaliInputPipeline(video_filelists, audio_filelists, dali_cfg)

    # Step 7: Metadata fetcher
    fetcher = LmdbMetaFetcher(store)

    # Step 8: Assembler
    assembler = BatchAssembler(id_map, fetcher, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    return dali_iter, assembler


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    data_root = os.environ.get("DATA_ROOT", "data")
    manifest_path = os.path.join(data_root, "manifest.json")
    args = DataArgs(data_root=data_root, manifest=manifest_path, batch_size=1)

    if not DALI_AVAILABLE:
        logging.error("DALI is not available: %s", _DALI_IMPORT_ERROR)
    else:
        try:
            dali_iter, assembler = build_data_iter(args)
            with Timer("one batch") as t:
                batch_raw = next(iter(dali_iter))
            logging.info("DALI fetched video+audio in %.3fs", t.dt)
            with Timer("assemble") as t2:
                batch = assembler.assemble(batch_raw)
            logging.info("Assembled batch in %.3fs; images=%s, mel=%s, alive=%s", t2.dt, tuple(batch["images"].shape), tuple(batch["mel_spectrogram"].shape), tuple(batch["alive_mask"].shape))
        except Exception as e:
            logging.exception("Data loader smoke test failed. Ensure manifest.json is correct, media files exist, and LMDB is populated.")
            logging.error("Failed with error: %s", e)