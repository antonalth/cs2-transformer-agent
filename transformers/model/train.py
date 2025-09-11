#!/usr/bin/env python3
"""
train.py

Main training harness for the CS2Transformer model.
This script orchestrates the data loading, model training, validation,
and checkpointing for the project.

VERSION: 5.0 (Major Refactor: DALI video decoding from file paths)
"""

# ===================================================================
# 1. IMPORTS
# ===================================================================
import os
import json
import argparse
import random
import tempfile
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

# Third-party libraries
import lmdb
import yaml
import numpy as np
import msgpack
import msgpack_numpy as mpnp
from tqdm import tqdm

# PyTorch imports
import torch

# DALI Imports
try:
    from nvidia.dali import pipeline_def
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from nvidia.dali.types import SampleInfo
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False


# ===================================================================
# 2. CONSTANTS & CONFIGURATION
# ===================================================================
TICKS_PER_FRAME = 2
NUM_PLAYERS = 5

@dataclass
class DataConfig:
    lmdb_root_path: str
    manifest_path: str
    num_workers: int = 8
    dali_num_threads: int = 6
    dali_prefetch: int = 2
    dali_video_read_ahead: bool = True
    dali_video_add_surfaces: int = 12
    map_extents: Dict = field(default_factory=lambda: {
        "x": (-2000.0, 3000.0), "y": (-3500.0, 2500.0), "z": (-500.0, 500.0)
    })

@dataclass
class TrainConfig:
    context_frames: int = 128
    steps_per_epoch: int = 1000
    batch_size: int = 4
    model_name: str = "vit_base_patch14_dinov2.lvd142m"

def load_config_from_yaml(path: str) -> tuple[DataConfig, TrainConfig]:
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    data_cfg = DataConfig(**cfg_dict['data'])
    train_cfg = TrainConfig(**cfg_dict['train'])
    return data_cfg, train_cfg


# ===================================================================
# 3. DATASET INDEXING & EPOCH PREPARATION
# ===================================================================
class DatasetIndexer:
    """Handles the initial setup and validation of the dataset."""
    def __init__(self, lmdb_root_path: str, manifest_path: str):
        print("--- Initializing DatasetIndexer ---")
        self.lmdb_root = Path(lmdb_root_path)
        self.manifest_path = Path(manifest_path)
        manifest_data = self._load_manifest()
        train_demos = manifest_data.get("train_demos", [])
        val_demos = manifest_data.get("validation_demos", [])
        print(f"Manifest loaded successfully. Found {len(train_demos)} train and {len(val_demos)} val demos.")
        self.train_pool = self._index_demos(train_demos, "Training")
        self.val_pool = self._index_demos(val_demos, "Validation")
        print("\n--- Dataset Indexing Complete ---")
        print(f"  - Total Training Round Perspectives Found:   {len(self.train_pool)}")
        print(f"  - Total Validation Round Perspectives Found: {len(self.val_pool)}")
        print("---------------------------------")
    def _load_manifest(self) -> dict:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"FATAL: Manifest file not found at: {self.manifest_path}")
        return json.loads(self.manifest_path.read_text())
    def _index_demos(self, demo_names: list, set_name: str) -> list:
        round_pool = []
        if not demo_names:
            print(f"No demos found for '{set_name}' set in manifest. Skipping.")
            return round_pool
        print(f"\nIndexing {set_name} set...")
        for demo_name in tqdm(demo_names, desc=f"  -> Indexing {set_name} Demos"):
            demo_path = self.lmdb_root / demo_name
            if not demo_path.is_dir():
                raise FileNotFoundError(f"FATAL: Manifest lists demo '{demo_name}' but not found at: {demo_path}")
            demo_name_base = demo_name.removesuffix('.lmdb')
            with lmdb.open(str(demo_path), readonly=True, lock=False) as env:
                with env.begin(write=False) as txn:
                    info_key = f"{demo_name_base}_INFO".encode('utf-8')
                    info_bytes = txn.get(info_key)
                    if info_bytes is None:
                        raise KeyError(f"FATAL: Metadata key '{info_key.decode()}' not in LMDB for '{demo_name}'.")
                    info_data = json.loads(info_bytes.decode('utf-8'))
                    for round_entry in info_data.get("rounds", []):
                        if len(round_entry.get("pov_videos", [])) == NUM_PLAYERS:
                            round_entry["lmdb_path"] = str(demo_path)
                            round_entry["demo_name"] = demo_name_base
                            round_pool.append(round_entry)
        return round_pool

def build_epoch_files(
    sampling_pool: list,
    epoch_size: int,
    context_frames: int,
    ticks_per_frame: int
) -> tuple[List[str], List[dict]]:
    """
    Creates random windows for one epoch and writes 5 temporary file_lists for DALI.
    """
    lines_per_pov = [[] for _ in range(NUM_PLAYERS)]
    epoch_meta = []
    
    # Create a list of samples for the epoch by randomly choosing from the pool
    epoch_samples = random.choices(sampling_pool, k=epoch_size)

    for round_info in epoch_samples:
        t0, t1 = round_info['start_tick'], round_info['end_tick']
        
        # Ensure there's enough room for a full context window
        min_duration = context_frames * ticks_per_frame
        if (t1 - t0) < min_duration:
            continue # Should be rare if data is well-formed, but good to guard

        max_start_tick = t1 - min_duration
        start_tick = random.randint(t0, max_start_tick)
        
        start_frame = (start_tick - t0) // ticks_per_frame
        end_frame = start_frame + context_frames

        for p in range(NUM_PLAYERS):
            abs_path = os.path.abspath(round_info['pov_videos'][p])
            lines_per_pov[p].append(f"{abs_path} 0 {start_frame} {end_frame}")

        epoch_meta.append({
            "lmdb_path": round_info['lmdb_path'],
            "demo_name": round_info['demo_name'],
            "round_num": round_info['round_num'],
            "team": round_info['team'],
            "start_tick": start_tick,
            "context_frames": context_frames
        })

    file_lists = []
    for p in range(NUM_PLAYERS):
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=f".p{p}.txt", mode="w", encoding="utf-8")
        tf.write("\n".join(lines_per_pov[p]) + "\n")
        tf.close()
        file_lists.append(tf.name)

    return file_lists, epoch_meta


# ===================================================================
# 4. DATA TRANSFORMATION HELPERS
# ===================================================================
HEATMAP_DIMS = (8, 64, 64)
KEYBOARD_DIM = 31
ROUND_STATE_DIM = 5
MEL_SPEC_TIME_FRAMES = 8

def _resolve_timm_config(model_name: str):
    import timm
    from timm.data import resolve_model_data_config
    m = timm.create_model(model_name, pretrained=True, num_classes=0)
    cfg = resolve_model_data_config(m)
    H, W = cfg["input_size"][1], cfg["input_size"][2]
    mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406)), dtype=torch.float32)
    std  = torch.tensor(cfg.get("std",  (0.229, 0.224, 0.225)), dtype=torch.float32)
    print(f"[TIMM Config Resolver] Model: {model_name} -> Input: {H}x{W}")
    return (H, W), mean, std

def find_nth_set_bit_pos(n: int, j: int) -> int:
    count = 0
    for i in range(NUM_PLAYERS):
        if (n >> i) & 1:
            if count == j: return i
            count += 1
    return -1

# ===================================================================
# 5. DALI PIPELINE & DATA SOURCE (FOR TARGETS)
# ===================================================================
class DALITargetsSource:
    """Callable source for DALI's external_source operator to fetch non-visual data."""
    def __init__(self, epoch_meta: list, data_cfg: DataConfig):
        self.epoch_meta = epoch_meta
        self.data_cfg = data_cfg
        self.envs = {}
        self._rng = random.Random()
        
    def __call__(self, sample_info: "SampleInfo"):
        # DALI iterates through the source, sample_info.idx gives the sample index
        meta = self.epoch_meta[sample_info.idx]
        
        env = self._get_env(meta['lmdb_path'])
        raw_frame_data = []
        with env.begin(write=False) as txn:
            for i in range(meta['context_frames']):
                current_tick = meta['start_tick'] + (i * TICKS_PER_FRAME)
                key = (f"{meta['demo_name']}_round_{meta['round_num']:03d}_"
                       f"team_{meta['team']}_tick_{current_tick:08d}").encode('utf-8')
                value_bytes = txn.get(key)
                if value_bytes:
                    raw_frame_data.append(msgpack.unpackb(value_bytes, raw=False, object_hook=mpnp.decode))
                else: # Handle missing ticks by appending the last valid frame
                    raw_frame_data.append(raw_frame_data[-1] if raw_frame_data else None)
        
        # If the very first frame is missing, we must generate a dummy
        if raw_frame_data[0] is None:
            # This is a fatal error in data quality, but we can return zeros to avoid a crash
            print(f"WARNING: First frame missing for sample {sample_info.idx}, returning zeros.")
            return self._get_dummy_data(meta['context_frames'])

        return self._process_frames(raw_frame_data, self.data_cfg)

    def _process_frames(self, raw_frame_data, data_cfg):
        def _bitmask_to_multihot_np(m, nc): 
            return np.array([(m >> i) & 1 for i in range(nc)], dtype=np.float32)
        def _coords_to_heatmap_np(coords, dims, ext):
            Z, Y, X = dims; hm = np.zeros(dims, dtype=np.float32)
            if coords is None: return hm
            coords = np.atleast_2d(coords)
            for x, y, z in coords:
                if not np.all(np.isfinite([x, y, z])): continue
                ix = int(((x - ext['x'][0]) / (ext['x'][1] - ext['x'][0])) * (X - 1))
                iy = int(((y - ext['y'][0]) / (ext['y'][1] - ext['y'][0])) * (Y - 1))
                iz = int(((z - ext['z'][0]) / (ext['z'][1] - ext['z'][0])) * (Z - 1))
                hm[np.clip(iz, 0, Z-1), np.clip(iy, 0, Y-1), np.clip(ix, 0, X-1)] = 1.0
            return hm

        seq_len = len(raw_frame_data)
        
        # Pre-allocate numpy arrays for the entire sequence
        mels = np.full((seq_len, NUM_PLAYERS, 1, 128, MEL_SPEC_TIME_FRAMES), -80.0, dtype=np.float32)
        alive = np.zeros((seq_len, NUM_PLAYERS), dtype=np.bool_)
        stats = np.zeros((seq_len, NUM_PLAYERS, 3), dtype=np.float32)
        pos_hm = np.zeros((seq_len, NUM_PLAYERS, *HEATMAP_DIMS), dtype=np.float32)
        mouse = np.zeros((seq_len, NUM_PLAYERS, 2), dtype=np.float32)
        kbd = np.zeros((seq_len, NUM_PLAYERS, KEYBOARD_DIM), dtype=np.float32)
        enemy_hm = np.zeros((seq_len, *HEATMAP_DIMS), dtype=np.float32)
        state = np.zeros((seq_len, ROUND_STATE_DIM), dtype=np.float32)

        for i, frame_data in enumerate(raw_frame_data):
            if frame_data is None: # Use previous frame's data if a tick is missing
                mels[i], alive[i], stats[i], pos_hm[i], mouse[i], kbd[i] = mels[i-1], alive[i-1], stats[i-1], pos_hm[i-1], mouse[i-1], kbd[i-1]
                enemy_hm[i], state[i] = enemy_hm[i-1], state[i-1]
                continue

            gs = frame_data['game_state'][0]
            pd_list = frame_data['player_data']
            mask_int = gs['team_alive']

            for p_idx, p_tuple in enumerate(pd_list):
                slot = find_nth_set_bit_pos(mask_int, p_idx)
                if slot == -1: continue
                
                p_info, _, mel_spec = p_tuple
                p_info = p_info[0]

                alive[i, slot] = True
                if mel_spec is not None:
                    mel_tensor = torch.from_numpy(mel_spec.copy()).unsqueeze(0)
                    padded = torch.nn.functional.pad(mel_tensor, (0, max(0, MEL_SPEC_TIME_FRAMES - mel_tensor.shape[-1])), 'constant', -80.0)
                    mels[i, slot, 0] = padded[:, :, :MEL_SPEC_TIME_FRAMES].numpy()

                stats[i, slot] = [p_info['health'], p_info['armor'], p_info['money']]
                pos_hm[i, slot] = _coords_to_heatmap_np(p_info['pos'], HEATMAP_DIMS, data_cfg.map_extents)
                mouse[i, slot] = p_info['mouse']
                kbd[i, slot] = _bitmask_to_multihot_np(p_info['keyboard_bitmask'], KEYBOARD_DIM)

            enemy_hm[i] = _coords_to_heatmap_np(gs['enemy_pos'], HEATMAP_DIMS, data_cfg.map_extents)
            state[i] = _bitmask_to_multihot_np(gs['round_state'], ROUND_STATE_DIM)
        
        return mels, alive, stats, pos_hm, mouse, kbd, enemy_hm, state

    def _get_env(self, lmdb_path):
        if lmdb_path not in self.envs:
            self.envs[lmdb_path] = lmdb.open(lmdb_path, readonly=True, lock=False)
        return self.envs[lmdb_path]
    
    def _get_dummy_data(self, seq_len):
        mels = np.full((seq_len, NUM_PLAYERS, 1, 128, MEL_SPEC_TIME_FRAMES), -80.0, dtype=np.float32)
        alive = np.zeros((seq_len, NUM_PLAYERS), dtype=np.bool_)
        stats = np.zeros((seq_len, NUM_PLAYERS, 3), dtype=np.float32)
        pos_hm = np.zeros((seq_len, NUM_PLAYERS, *HEATMAP_DIMS), dtype=np.float32)
        mouse = np.zeros((seq_len, NUM_PLAYERS, 2), dtype=np.float32)
        kbd = np.zeros((seq_len, NUM_PLAYERS, KEYBOARD_DIM), dtype=np.float32)
        enemy_hm = np.zeros((seq_len, *HEATMAP_DIMS), dtype=np.float32)
        state = np.zeros((seq_len, ROUND_STATE_DIM), dtype=np.float32)
        return mels, alive, stats, pos_hm, mouse, kbd, enemy_hm, state

@pipeline_def
def create_dali_pipeline(
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    file_lists: List[str],
    target_source_callable,
    target_hw, mean, std
):
    """Defines the DALI processing graph with video readers and a target source."""
    H, W = target_hw
    mean255, std255 = (mean.numpy() * 255.0).tolist(), (std.numpy() * 255.0).tolist()
    
    # --- 1. Video Decoding and Pre-processing ---
    videos = []
    for i in range(NUM_PLAYERS):
        v, _ = fn.readers.video_resize(
            device="gpu",
            file_list=file_lists[i],
            file_list_frame_num=True,
            sequence_length=train_cfg.context_frames,
            step=1,
            random_shuffle=False,
            pad_sequences=True,
            read_ahead=data_cfg.dali_video_read_ahead,
            additional_decode_surfaces=data_cfg.dali_video_add_surfaces,
            resize_y=H, resize_x=W,
            name=f"vid_reader_{i}",
        )
        videos.append(v)

    # Stack videos along a new 'player' dimension, result shape: [B, F, P, H, W, C]
    video_stack = fn.stack(*videos, axis=2)
    
    # Flatten for per-frame processing: [B*F*P, H, W, C]
    B, F, P, H, W, C = video_stack.shape
    video_flat = fn.reshape(video_stack, layout="HW", shape=(-1, H, W, C))

    # Normalize, cast to FP16, and change layout to CHW
    video_norm = fn.crop_mirror_normalize(
        video_flat.gpu(),
        dtype=types.FLOAT16,
        output_layout="CHW",
        mean=mean255,
        std=std255
    )
    
    # --- 2. Target Data Loading ---
    mels, alive, stats, pos_hm, mouse, kbd, enemy_hm, state = fn.external_source(
        source=target_source_callable,
        num_outputs=8,
        batch=False,
        parallel=True,
        dtype=[types.FLOAT, types.BOOL, types.FLOAT, types.FLOAT, types.FLOAT, types.FLOAT, types.FLOAT, types.FLOAT]
    )

    pos_hm_gpu = fn.cast(pos_hm.gpu(), dtype=types.FLOAT16)
    enemy_hm_gpu = fn.cast(enemy_hm.gpu(), dtype=types.FLOAT16)
    mels_gpu = mels.gpu()

    return video_norm, mels_gpu, alive, stats, pos_hm_gpu, mouse, kbd, enemy_hm_gpu, state

# ===================================================================
# 6. SCRIPT ENTRYPOINT
# ===================================================================
if __name__ == "__main__":
    if not DALI_AVAILABLE:
        raise ImportError("NVIDIA DALI is not installed. Please install it to run this script.")

    parser = argparse.ArgumentParser(description="Test the DALI Video Data Pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    args = parser.parse_args()
    
    file_lists_cleanup = []
    try:
        print(f"Loading configuration from: {args.config}")
        data_cfg, train_cfg = load_config_from_yaml(args.config)

        if not torch.cuda.is_available():
            raise RuntimeError("DALI requires a CUDA-enabled PyTorch environment.")

        target_hw, mean, std = _resolve_timm_config(train_cfg.model_name)
        
        # --- 1. Index dataset ---
        indexer = DatasetIndexer(lmdb_root_path=data_cfg.lmdb_root_path, manifest_path=data_cfg.manifest_path)
        
        # --- 2. Prepare files and metadata for one "epoch" ---
        print("\n--- Preparing epoch data ---")
        epoch_size = train_cfg.steps_per_epoch * train_cfg.batch_size
        file_lists, epoch_meta = build_epoch_files(
            sampling_pool=indexer.train_pool,
            epoch_size=epoch_size,
            context_frames=train_cfg.context_frames,
            ticks_per_frame=TICKS_PER_FRAME
        )
        file_lists_cleanup = file_lists
        print(f"Generated {len(file_lists)} file lists for DALI video readers.")
        print(f"Created epoch metadata for {len(epoch_meta)} samples.")

        # --- 3. Instantiate the target data source ---
        target_source = DALITargetsSource(epoch_meta, data_cfg)
        
        # --- 4. Build and run the DALI pipeline ---
        print("\n--- Using NVIDIA DALI Data Pipeline ---")
        device_id = int(os.getenv("LOCAL_RANK", "0"))
        
        pipeline = create_dali_pipeline(
            batch_size=train_cfg.batch_size,
            num_threads=data_cfg.dali_num_threads,
            device_id=device_id,
            seed=int(os.getenv("DL_SEED", "1337")),
            prefetch_queue_depth=data_cfg.dali_prefetch,
            py_num_workers=data_cfg.num_workers,
            py_start_method='spawn',
            # Custom args
            data_cfg=data_cfg,
            train_cfg=train_cfg,
            file_lists=file_lists,
            target_source_callable=target_source,
            target_hw=target_hw,
            mean=mean,
            std=std
        )
        pipeline.build()
        torch.cuda.set_device(device_id)
        print(f"DALI pipeline built successfully for device {device_id}.")

        output_map = ["images", "mel", "alive", "stats", "pos_hm", "mouse", "kbd", "enemy_hm", "state"]
        
        dali_loader = DALIGenericIterator(
            [pipeline], output_map, 
            reader_name=None, auto_reset=True, last_batch_padded=True
        )
        
        print("\nAttempting to fetch one batch from the DALI loader...")
        batch = next(iter(dali_loader))[0]
        print("Successfully fetched one batch!")

        # --- 5. Reshape and verify the output batch ---
        B, T, P = train_cfg.batch_size, train_cfg.context_frames, NUM_PLAYERS

        # Reshape the flattened video tensor back to its structured form
        C, H, W = batch["images"].shape[1:]
        images_tensor = batch["images"].reshape(B, T, P, C, H, W)
        
        batch_for_inspection = {
            "inputs": {
                "images": images_tensor,
                "mel_spectrogram": batch["mel"],
                "alive_mask": batch["alive"]
            },
            "targets": {
                "player_stats": batch["stats"],
                "player_pos_heatmaps": batch["pos_hm"],
                "player_mouse": batch["mouse"],
                "player_keyboard": batch["kbd"],
                "enemy_pos_heatmap": batch["enemy_hm"],
                "round_state": batch["state"]
            }
        }
        
        print("\n--- Batch Content (Reshaped) ---")
        for key, value in batch_for_inspection.items():
            print(f"\nTop-level key: '{key}'")
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        print(f"  - Sub-key: '{sub_key}', Shape: {sub_value.shape}, DType: {sub_value.dtype}, Device: {sub_value.device}")
        print("--------------------------------\n")
        print("Data Pipeline test successful!")

    except Exception as e:
        print(f"\nAn error occurred during the test:\n{e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up temporary file lists...")
        for fl_path in file_lists_cleanup:
            try:
                os.remove(fl_path)
            except OSError:
                pass