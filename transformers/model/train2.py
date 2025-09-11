"""
train2.py — barebones scaffold (function definitions + comments)

Goal:
- Provide a clear skeleton for training with GPU video decoding via DALI and LMDB-backed labels/features.
- Encode *key insights* from the planning discussion directly into docstrings and TODOs.
- Keep it minimal: no heavy logic, only signatures and comments describing what to implement.

Assumptions / invariants from your data generation:
- Demos are 64 ticks/s; videos are 32 fps → EXACT mapping: 2 ticks == 1 frame.
- All 5 POV mp4s for a team-round start at the same tick (round start) and end when the player dies.
- LMDB keys are written ONLY every 2 ticks with parity defined by round start (even-only or odd-only).
- A "team-round window" is sampled as a fixed-length clip in FRAMES, e.g., T_frames=512.
- We will:
  1) Enumerate all team-rounds (across all games) from manifest.json and LMDB INFO blocks.
  2) For each epoch, pick a valid random window per team-round.
  3) Write 5 aligned DALI filelists (one per POV) **including sample_id as label**.
  4) DALI decodes on GPU (pad after death); we then map frames → LMDB keys using tick(f) = start_tick_win + 2*f.
  5) From LMDB, reinflate per-player data (alive-only list) into fixed 5 slots using a team_alive bitmask.

Notes:
- Replace "..." with your actual implementation.
"""

from __future__ import annotations
import os
import json
import argparse
import random
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

# import torch
# from torch.utils.data import DataLoader
# from model import CS2Transformer, CS2Config

# DALI imports will be needed in actual implementation:
# from nvidia.dali import fn, types, pipeline_def
# from nvidia.dali.plugin.pytorch import DALIGenericIterator

# LMDB + msgpack (used by injection_mold.py outputs):
# import lmdb
# import msgpack


# -----------------------------
# Config / CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    """
    Define CLI arguments relevant to:
    - Paths: data root, manifest.json, run dir for artifacts (filelists, checkpoints, logs).
    - Video: fps (typically 32), resize HxW, T_frames (clip length).
    - DALI: decode surfaces, prefetch, num threads, shard params (if DDP).
    - LMDB: flags for precomputed mels or on-the-fly, cache size.
    - Training: epochs, batch size, lr, wd, amp mode, grad clip, wandb/tb flags, resume.
    - DDP: world size, local rank, seed, deterministic flags.
    """
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--data-root", type=str, required=True, help="Root folder containing manifest.json, lmdb/, recordings/, demos/")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest.json (defaults to <data-root>/manifest.json)")
    parser.add_argument("--run-dir", type=str, default="runs/exp1", help="Directory for filelists, checkpoints, logs")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Which split to iterate")

    # Video / sampling
    parser.add_argument("--fps", type=int, default=32, help="Recorded video fps (32 per spec)")
    parser.add_argument("--tick-rate", type=int, default=64, help="Demo tick rate (64 per spec)")
    parser.add_argument("--t-frames", type=int, default=512, help="Fixed clip length in frames per sample (e.g., 512)")
    parser.add_argument("--size", type=str, default="224x224", help="HxW resize for video frames, e.g., 224x224")

    # DALI (placeholders; implement in build_dali_pipeline)
    parser.add_argument("--dali-decode-surfaces", type=int, default=8)
    parser.add_argument("--dali-read-ahead", action="store_true")
    parser.add_argument("--dali-threads", type=int, default=4)
    parser.add_argument("--prefetch-queue-depth", type=int, default=2)

    # LMDB / features
    parser.add_argument("--lmdb-readahead", action="store_true", help="Enable LMDB readahead (often False for SSD)")
    parser.add_argument("--n-mels", type=int, default=128, help="Mel bins if needed")
    parser.add_argument("--mel-time", type=int, default=6, help="Mel time bins per frame (example placeholder)")
    parser.add_argument("--mels-from-lmdb", action="store_true", help="Expect mel spectrograms in LMDB payloads")

    # Training knobs (minimal placeholders)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--eval-only", action="store_true")

    # DDP placeholders
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--local-rank", type=int, default=0)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """
    Set Python / NumPy / Torch RNG seeds for reproducibility.
    Key insight: Also seed per-epoch window sampling with (seed + epoch) so windows reshuffle deterministically each epoch.
    """
    random.seed(seed)
    # import numpy as np
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


# -----------------------------
# Data model
# -----------------------------

@dataclass
class TeamRound:
    """
    Canonical description of a single team-round available for sampling windows.

    Fields reflect what's expected from manifest.json + per-LMDB INFO record:
    - demoname: base id of the demo/game (used to build LMDB keys)
    - lmdb_path: absolute path to the game's LMDB
    - round_num: integer round id
    - team: string/enum ('CT' or 'T') used in LMDB key naming
    - start_tick, end_tick: inclusive bounds of round ticks in demo time
    - fps: 32 (recording fps), tick_rate: 64 (demo ticks/s)
    - pov_videos: list[str] length 5; absolute paths to mp4 per roster slot (0..4), ordered consistently with roster index
    """
    demoname: str
    lmdb_path: str
    round_num: int
    team: str
    start_tick: int
    end_tick: int
    fps: int
    tick_rate: int
    pov_videos: List[str]


@dataclass
class SampleRecord:
    """
    A single sampled window for an epoch:
    - sample_id: dense integer 0..N-1 PER EPOCH (used as DALI label to rejoin with LMDB)
    - start_f: starting frame index (relative to round start)
    - start_tick_win: exact demo tick for frame 0 in this window; must satisfy parity with round start
    - T_frames: window length in frames
    """
    sample_id: int
    team_round: TeamRound
    start_f: int
    start_tick_win: int
    T_frames: int


# -----------------------------
# Manifest
# -----------------------------

class Manifest:
    """
    Loads manifest.json and resolves absolute paths plus LMDB INFO metadata.

    Responsibilities:
    - Read manifest.json (from create_split.py) to get the list of games in each split and per-team-round metadata
      OR minimally the path to each game's LMDB so we can open *_INFO to enumerate rounds.
    - Normalize relative paths to absolute (based on data_root).
    - Provide get_split(split) -> List[TeamRound].

    Key insights:
    - Each game has EXACTLY one LMDB.
    - The LMDB contains an *_INFO record with round/teams, start/end ticks, and pov video paths.
    - Use that INFO to build TeamRound entries with stable roster order matching POV mp4 ordering.
    """

    def __init__(self, data_root: str, manifest_path: Optional[str]):
        self.data_root = data_root
        self.manifest_path = manifest_path or os.path.join(data_root, "manifest.json")
        self._raw = None  # store raw manifest dict as needed

    def load(self) -> None:
        """Load manifest.json; validate expected structure. Populate self._raw."""
        # with open(self.manifest_path, "r") as f:
        #     self._raw = json.load(f)
        # TODO: add validation and path normalization
        pass

    def get_split(self, split: str) -> List[TeamRound]:
        """
        Build a flat list of TeamRound objects for the requested split.

        Implementation outline:
        - For each game in manifest[split]:
          - Resolve lmdb_path = <data_root>/lmdb/<game>.lmdb (or from manifest).
          - Open LMDB and read INFO record (e.g., f"{demoname}_INFO").
          - For each round and team present in INFO:
            - Collect start_tick, end_tick, pov_videos (5 absolute mp4 paths in recordings/).
            - Construct TeamRound(...).
        - Return the complete list.
        """
        # TODO: implement reading INFO from LMDB, normalize paths
        return []


# -----------------------------
# Epoch indexing (sampling windows)
# -----------------------------

class EpochIndex:
    """
    Per-epoch sampler and lookup.

    Responsibilities:
    - Given a list of TeamRound and a T_frames, sample ONE window per TeamRound (or multiple, as configured).
    - Respect valid range: start_f in [0, max(0, frame_count - T_frames)] where frame_count = ((end_tick - start_tick)//2) + 1.
    - Support padding option: when frame_count < T_frames, either skip or set start_f=0 and rely on pad_sequences + alive_mask.
    - Assign dense sample_id (0..N-1) and store mapping id -> SampleRecord.
    - Provide methods:
        .build(seed_epoch) -> None          # regenerates windows for a new epoch
        .records() -> List[SampleRecord]    # deterministic order used to write DALI filelists
        .by_id(sample_id) -> SampleRecord   # lookup for LMDB fetch during batching

    Key insight:
    - start_tick_win = TeamRound.start_tick + 2 * start_f (parity guaranteed).
    - Use seed + epoch for deterministic reshuffling.
    """

    def __init__(self, team_rounds: List[TeamRound], T_frames: int, allow_padding: bool = True):
        self.team_rounds = team_rounds
        self.T_frames = T_frames
        self.allow_padding = allow_padding
        self._records: List[SampleRecord] = []
        self._by_id: Dict[int, SampleRecord] = {}

    def build(self, seed_for_epoch: int) -> None:
        """Sample windows and populate self._records and self._by_id."""
        # random.seed(seed_for_epoch)
        # For each TeamRound, compute frame_count and choose start_f (or skip if too short and padding disabled).
        # Compute start_tick_win = start_tick + 2 * start_f.
        # Assign incremental sample_id and store SampleRecord.
        pass

    def records(self) -> List[SampleRecord]:
        """Return records in the exact order to be written to filelists (also defines epoch ordering)."""
        return self._records

    def by_id(self, sample_id: int) -> SampleRecord:
        """Return SampleRecord for a given sample_id."""
        return self._by_id[sample_id]


# -----------------------------
# Filelists (for DALI video readers)
# -----------------------------

def write_epoch_filelists(index: EpochIndex, out_dir: str) -> List[str]:
    """
    Emit five text files (pov0.txt ... pov4.txt), each line aligned across files:

        "<abs/path/to/player_k.mp4>  <sample_id>  <start_frame>  <end_frame>"

    Notes:
    - Use file_list_frame_num=True in DALI so start/end are interpreted as frame indices.
    - The label column is *sample_id*; DALI will return it with the batch, letting us rejoin with LMDB.
    - end_frame should be start_f + T_frames (exclusive depending on DALI variant; align with chosen reader).

    Returns:
    - List of file paths in order [pov0.txt, pov1.txt, pov2.txt, pov3.txt, pov4.txt]
    """
    # Ensure out_dir exists, then open 5 files and write aligned lines for each SampleRecord.
    # Return file paths for DALI to consume.
    return []


# -----------------------------
# DALI pipeline & iterator
# -----------------------------

def build_dali_pipeline(
    pov_filelists: List[str],
    size_hw: Tuple[int, int],
    T_frames: int,
    batch_size: int,
    shard_id: int,
    num_shards: int,
    dali_threads: int,
    decode_surfaces: int,
    read_ahead: bool,
    prefetch_queue_depth: int,
):
    """
    Construct a single DALI pipeline with FIVE branches (one per POV).

    What to implement:
    - For each POV filelist:
        reader = fn.readers.video_resize(
            file_list=..., sequence_length=T_frames, step=T_frames,
            file_list_frame_num=True, random_shuffle=False, pad_sequences=True,
            additional_decode_surfaces=decode_surfaces, read_ahead=read_ahead,
            resize_y=H, resize_x=W, dtype=types.UINT8,
            shard_id=shard_id, num_shards=num_shards, stick_to_shard=True,
            enable_frame_num=True
        )
        images = fn.crop_mirror_normalize(reader, output_layout="FCHW",
                                          mean=..., std=..., output_dtype=types.FLOAT16)
      Name the outputs pov0...pov4 so DALIGenericIterator can return them by name.

    Key insights:
    - pad_sequences=True zero-pads tails after player death; losses must be masked accordingly.
    - We keep the *labels* emitted by DALI (they should equal our sample_id).
    - For AMP=bf16, cast after iterator (fp16→bf16) before forward.

    Return:
    - A compiled DALI pipeline (or a tuple containing the pipeline and the list of output names).
    """
    pass


def make_dali_iterator(pipeline, batch_size: int):
    """
    Wrap the built pipeline in a DALIGenericIterator.

    Implementation details:
    - Provide output_map like ["pov0", "pov1", "pov2", "pov3", "pov4"].
    - Ensure 'reader_name' or 'fill_last_batch' are set correctly for epoch boundaries.
    - Iterator should yield a dict with GPU tensors and 'labels' (sample_id) from at least one branch (e.g., pov0).
    """
    # iter = DALIGenericIterator([...])
    # return iter
    pass


# -----------------------------
# LMDB access
# -----------------------------

class LMDBStore:
    """
    Cache and manage LMDB environments per game.

    Responsibilities:
    - get_env(lmdb_path) -> env (open once per path; cache by path)
    - close_all() on shutdown
    - Set appropriate flags: readonly env, max_readers large enough, map_size irrelevant for read-only.

    Tuning:
    - readahead=False may be better on SSDs; allow CLI flag to toggle.
    """

    def __init__(self, readahead: bool = False):
        self.readahead = readahead
        self._cache: Dict[str, Any] = {}

    def get_env(self, lmdb_path: str):
        """Open (or return cached) LMDB environment for the path."""
        # if lmdb_path not in self._cache:
        #     env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=self.readahead, max_readers=256)
        #     self._cache[lmdb_path] = env
        # return self._cache[lmdb_path]
        pass

    def close_all(self) -> None:
        """Close all cached environments."""
        # for env in self._cache.values():
        #     env.close()
        # self._cache.clear()
        pass


class FeatureFetcher:
    """
    Fetch and assemble per-sample LMDB features aligned to decoded frames.

    Responsibilities:
    - Given SampleRecord (retrieved via sample_id), compute frame ticks:
        tick[f] = start_tick_win + 2*f  (parity locked to round start)
    - Build LMDB keys per tick:
        f"{demoname}_round_{round_num:03d}_team_{team}_tick_{tick:08d}".encode()
    - Read each key under a single read-only txn; msgpack-unpack payload.

    Reinflation logic:
    - payload["game_state"] has a team_alive 5-bit mask (LSB = slot 0).
    - payload["player_data"] is a list for *alive players only*, sorted by roster index.
    - Reinflate to fixed 5 slots:
        j = 0
        for slot in 0..4:
            if (mask_bits >> slot) & 1:
                (pi, _, mel) = pdl[j]; j += 1
                write into slot
            else:
                zeros (dead)
    - Produce:
        - mel_spectrogram: [T, 5, n_mels, mel_t] (zeros where dead or missing)
        - alive_mask: [T, 5] (uint8/bool)
        - targets dict for your heads (keyboard logits targets, eco, inventory, active weapon, positions, etc.)
        - Optional: frame_pad_mask if you want to mask DALI's padded frames beyond true footage.

    Missing keys:
    - If txn.get(key) is None: interpret as "no live data" → alive_mask zeros, features zeros for that frame.

    Performance:
    - Keep one env per game in LMDBStore.
    - Use a single txn per sample window to minimize overhead.
    """

    def __init__(self, store: LMDBStore, n_mels: int, mel_time: int, expect_mels: bool):
        self.store = store
        self.n_mels = n_mels
        self.mel_time = mel_time
        self.expect_mels = expect_mels

    def fetch_window(self, rec: SampleRecord) -> Dict[str, Any]:
        """Return dict with mel_spectrogram, alive_mask, and per-head targets for window rec."""
        # env = self.store.get_env(rec.team_round.lmdb_path)
        # ticks = [rec.start_tick_win + 2*f for f in range(rec.T_frames)]
        # with env.begin(write=False) as txn:
        #     for t in ticks:
        #         key = ...
        #         blob = txn.get(key)
        #         if not blob:
        #             # all zeros; continue
        #         payload = msgpack.unpackb(blob, raw=False)
        #         gs = payload["game_state"]; pdl = payload["player_data"]
        #         # team_alive = int(gs["team_alive"])
        #         # reinflate over slots 0..4
        #         # write mel and targets per slot
        # return {...}
        return {}


# -----------------------------
# Batch assembly
# -----------------------------

class BatchAssembler:
    """
    Fuse DALI outputs (GPU) with LMDB features into the model's expected CS2Batch.

    Inputs per step:
    - From DALIGenericIterator:
        pov0..pov4: each [B, T, C, H, W] (FP16 CUDA)
        labels: [B] (int sample_id) from at least one branch (pov0)
    - From FeatureFetcher.fetch_window for each label:
        mel_spectrogram [T, 5, n_mels, mel_t], alive_mask [T, 5], targets dict

    Outputs (example shape expectations):
    - images: [B, T, 5, 3, H, W]  (stack POVs; cast to bf16 if needed)
    - mel_spectrogram: [B, T, 5, 1, n_mels, mel_t] (unsqueeze channel=1)
    - alive_mask: [B, T, 5] (bool/uint8 on device)
    - targets: dict with per-head tensors on device
    - Optional: frame_pad_mask [B, T] / [B, T, 5] to mask DALI tail padding

    Key insights:
    - Ordering: the 5 POV tensors correspond to roster slots 0..4; the reinflation logic uses the same slot order.
    - AMP: DALI emits FP16; if training in BF16, cast images once (cheap).
    - Device: LMDB-derived arrays must be moved to same CUDA device before forward.
    """

    def __init__(self, feature_fetcher: FeatureFetcher, amp_mode: str = "bf16"):
        self.fetcher = feature_fetcher
        self.amp_mode = amp_mode

    def assemble(self, dali_batch: Dict[str, Any], id_lookup: EpochIndex) -> Dict[str, Any]:
        """
        Build the final CS2Batch dict.

        Implementation steps:
        - Extract labels (sample_id) from one branch (e.g., dali_batch["pov0"]["label"]).
        - For each sample_id in the batch:
            rec = id_lookup.by_id(sample_id)
            feats = self.fetcher.fetch_window(rec)
        - Stack LMDB features across the batch; convert dtypes; move to device.
        - Stack pov0..pov4 along a new axis to [B, T, 5, C, H, W]; permute if needed to [B, T, 5, 3, H, W].
        - Return a dict with keys matching your model's forward signature, e.g.:
            {
              "images": ...,
              "mel_spectrogram": ...,
              "alive_mask": ...,
              "targets": {...}
            }
        """
        return {}


# -----------------------------
# Model / Optim / Loss
# -----------------------------

def build_model(args) -> Any:
    """
    Instantiate CS2Config/CS2Transformer from model.py.

    TODO:
    - Mirror CLI params to CS2Config (context frames, compute dtype, vision config, etc.).
    - Optionally freeze ViT for warmup epochs (model.set_vit_frozen(True/False)).
    - Consider torch.compile() if compatible.
    """
    # cfg = CS2Config(...)
    # model = CS2Transformer(cfg, use_dummy_vision=False)
    # return model
    return None


def build_optim_sched(model: Any, args) -> Tuple[Any, Any]:
    """
    Build optimizer and LR scheduler.

    Key insights:
    - Use model.parameter_groups() to set group-wise LR scaling and weight decay.
    - Common choice: AdamW + cosine decay with warmup.
    - Step per-iteration or per-epoch depending on scheduler.
    """
    # groups = model.parameter_groups()
    # for g in groups:
    #     g["lr"] = args.lr * g.get("lr_scale", 1.0)
    #     g["weight_decay"] = args.wd
    # optim = torch.optim.AdamW(groups, betas=(0.9,0.999), eps=1e-8)
    # sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps)
    # return optim, sched
    return None, None


def compute_losses(pred: Dict[str, Any], targets: Dict[str, Any], alive_mask) -> Dict[str, Any]:
    """
    Compute composite masked loss for multi-head outputs.

    Heads (examples from plan):
    - Heatmaps: BCEWithLogits/CE over spatial grid for player/enemy positions.
    - Classification: keyboard/eco/inventory/active_weapon/round_state via CrossEntropy.
    - Regression: mouse_delta_deg, stats via SmoothL1/L2.
    - Round number: regression or CE if discretized.

    Masking:
    - Use alive_mask [B, T, 5] to exclude dead players' contributions.
    - Optionally also apply frame_pad_mask to ignore DALI tail padding.

    Return:
    - {"loss_total": tensor, "loss_dict": {"heatmap":..., "keyboard":..., ...}}
    """
    # TODO: implement
    return {}


# -----------------------------
# Train / Eval loops
# -----------------------------

def train_one_epoch(
    model: Any,
    dali_iter,
    assembler: BatchAssembler,
    optim: Any,
    sched: Any,
    device: str,
    epoch: int,
    args,
) -> Dict[str, float]:
    """
    Single-epoch training loop.

    Steps:
    - Iterate DALI iterator (no DataLoader needed).
    - For each batch:
        batch = assembler.assemble(dali_batch, id_lookup=index)  # must have access to current EpochIndex
        pred = model(batch_inputs)
        loss_pack = compute_losses(pred, batch["targets"], batch["alive_mask"])
        loss = loss_pack["loss_total"]
        AMP: autocast + GradScaler (if fp16) or bfloat16 autocast (no scaler).
        Backprop, grad clip, optim.step(), sched.step(), zero_grad.
    - Track metrics and timing breakdown (decode vs LMDB fetch vs forward/backward).
    - Return running loss averages for logging.
    """
    return {}


@torch.no_grad()
def evaluate(model: Any, dali_iter, assembler: BatchAssembler, device: str, args) -> Dict[str, float]:
    """
    Validation loop.

    Steps mirror train_one_epoch but under no_grad() and model.eval():
    - Assemble batch
    - Forward
    - Compute losses/metrics (accuracy for classification heads, error for regression, IoU for heatmaps)
    - Aggregate across batches (and across DDP ranks if applicable)
    """
    return {}


# -----------------------------
# Checkpointing
# -----------------------------

def save_checkpoint(path: str, state: Dict[str, Any]) -> None:
    """
    Save a checkpoint including:
    - model.state_dict()
    - optimizer/scheduler state
    - epoch, global_step
    - RNG states if desired
    - config snapshot
    """
    # torch.save(state, path)
    pass


def load_checkpoint(path: str, model: Any, optim: Any, sched: Any) -> int:
    """
    Load checkpoint and restore state.
    Return: start_epoch (or global_step) to resume from.

    Notes:
    - Support strict=False if model has evolved.
    - Handle AMP scaler state if using fp16.
    """
    # ckpt = torch.load(path, map_location="cpu")
    # model.load_state_dict(ckpt["model"], strict=False)
    # optim.load_state_dict(ckpt["optim"])
    # sched.load_state_dict(ckpt["sched"])
    # return ckpt.get("epoch", 0)
    return 0


# -----------------------------
# Orchestration
# -----------------------------

def main() -> None:
    """
    High-level orchestration:
    1) Parse args; set seeds; create run_dir subfolders (filelists/, ckpt/, logs/).
    2) Load Manifest and build TeamRound list for selected split.
    3) Build EpochIndex for the split and sample windows for epoch 0.
    4) Write five aligned DALI filelists (pov0..pov4) for the epoch.
    5) Build DALI pipeline + iterator for this epoch (with sharding if DDP).
    6) Init LMDBStore and FeatureFetcher; create BatchAssembler.
    7) Build model, optimizer, scheduler; resume if requested; amp config.
    8) If eval-only: run evaluate() on a validation iterator and exit.
    9) Training loop over epochs:
        - Rebuild EpochIndex with seed + epoch.
        - Rewrite filelists for the epoch.
        - (Re)build DALI pipeline/iterator (tear down previous to avoid leaks).
        - train_one_epoch(...)
        - evaluate(...) periodically or each epoch.
        - save_checkpoint(last.pt); if best metric, save best.pt.
    10) Cleanup: close LMDB envs, finalize loggers.
    """
    # args = parse_args()
    # set_seed(args.seed)
    # manifest = Manifest(args.data_root, args.manifest); manifest.load()
    # trs = manifest.get_split(args.split)
    # index = EpochIndex(trs, T_frames=args.t_frames, allow_padding=True)
    # index.build(seed_for_epoch=args.seed + 0)

    # filelists_dir = os.path.join(args.run_dir, f"epoch_{0}")
    # os.makedirs(filelists_dir, exist_ok=True)
    # pov_lists = write_epoch_filelists(index, filelists_dir)

    # H, W = map(int, args.size.lower().split("x"))
    # pipe = build_dali_pipeline(
    #     pov_filelists=pov_lists, size_hw=(H, W), T_frames=args.t_frames,
    #     batch_size=args.batch_size, shard_id=args.local_rank, num_shards=args.world_size,
    #     dali_threads=args.dali_threads, decode_surfaces=args.dali_decode_surfaces,
    #     read_ahead=args.dali_read_ahead, prefetch_queue_depth=args.prefetch_queue_depth
    # )
    # dali_iter = make_dali_iterator(pipe, args.batch_size)

    # store = LMDBStore(readahead=args.lmdb_readahead)
    # fetcher = FeatureFetcher(store, n_mels=args.n_mels, mel_time=args.mel_time, expect_mels=args.mels_from_lmdb)
    # assembler = BatchAssembler(fetcher, amp_mode=args.amp)

    # model = build_model(args)
    # optim, sched = build_optim_sched(model, args)

    # if args.eval_only:
    #     evaluate(model, dali_iter, assembler, device="cuda", args=args)
    #     return

    # for epoch in range(args.epochs):
    #     # Regenerate epoch windows + filelists + pipeline
    #     index.build(seed_for_epoch=args.seed + epoch)
    #     filelists_dir = os.path.join(args.run_dir, f"epoch_{epoch}")
    #     os.makedirs(filelists_dir, exist_ok=True)
    #     pov_lists = write_epoch_filelists(index, filelists_dir)
    #     # rebuild DALI pipeline/iterator ...
    #     # train and eval ...
    #     pass
    pass


if __name__ == "__main__":
    main()
