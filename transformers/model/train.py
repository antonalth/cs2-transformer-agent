#!/usr/bin/env python3
"""
train.py

Main training harness for the CS2Transformer model.
This script orchestrates the data loading, model training, validation,
and checkpointing for the project.

VERSION: 4.8 (Refactor: Emit per-sample, per-player labels from DALI source)
"""

# ===================================================================
# 1. IMPORTS
# ===================================================================
import os
import json
import argparse
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Any, Sequence

# Third-party libraries
import lmdb
import yaml
import cv2
import numpy as np
import msgpack
import msgpack_numpy as mpnp
from tqdm import tqdm

# PyTorch imports
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, DataLoader

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
    num_workers: int = 4
    use_dali: bool = True
    dali_num_threads: int = 4
    dali_prefetch: int = 2
    map_extents: Dict = field(default_factory=lambda: {
        "x": (-2000.0, 3000.0), "y": (-3500.0, 2500.0), "z": (-500.0, 500.0)
    })

@dataclass
class TrainConfig:
    context_frames: int = 128
    steps_per_epoch: int = 10000
    batch_size: int = 4
    model_name: str = "vit_base_patch14_dinov2.lvd142m"

def load_config_from_yaml(path: str) -> tuple[DataConfig, TrainConfig]:
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    data_cfg = DataConfig(**cfg_dict['data'])
    train_cfg = TrainConfig(**cfg_dict['train'])
    return data_cfg, train_cfg


# ===================================================================
# 3. DATASET INDEXING & VALIDATION
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
            raise FileNotFoundError(f"FATAL: Manifest file not found at the specified path: {self.manifest_path}")
        try:
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"FATAL: Error decoding JSON from manifest file: {self.manifest_path}\n{e}")
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
            env = lmdb.open(str(demo_path), readonly=True, lock=False, readahead=False, meminit=False)
            with env.begin(write=False) as txn:
                info_key = f"{demo_name_base}_INFO".encode('utf-8')
                info_bytes = txn.get(info_key)
                if info_bytes is None:
                    env.close()
                    raise KeyError(f"FATAL: Could not find metadata key '{info_key.decode()}' in LMDB for '{demo_name}'.")
                info_data = json.loads(info_bytes.decode('utf-8'))
                cursor = txn.cursor()
                for round_entry in info_data.get("rounds", []):
                    round_num, start_tick, end_tick = round_entry
                    for team in ['T', 'CT']:
                        test_key = f"{demo_name_base}_round_{round_num:03d}_team_{team}_tick_{start_tick:08d}".encode('utf-8')
                        if cursor.set_key(test_key):
                            metadata = { "lmdb_path": str(demo_path), "demo_name": demo_name_base, "round_num": round_num, "team": team, "start_tick": start_tick, "end_tick": end_tick }
                            round_pool.append(metadata)
            env.close()
        return round_pool

# ===================================================================
# 4. DATA TRANSFORMATION HELPERS
# ===================================================================
HEATMAP_DIMS = (8, 64, 64)
KEYBOARD_DIM = 31
ROUND_STATE_DIM = 5
MEL_SPEC_TIME_FRAMES = 8

if DALI_AVAILABLE:
    INTERP_MAP = {
        "bicubic":  types.INTERP_CUBIC,
        "bilinear": types.INTERP_LINEAR,
        "nearest":  types.INTERP_NN,
        "lanczos":  types.INTERP_LANCZOS3,
    }

def _resolve_timm_config(model_name: str):
    try:
        import timm
        from timm.data import resolve_model_data_config
    except ImportError:
        raise ImportError("Please install timm: `pip install timm`")
    m = timm.create_model(model_name, pretrained=True, num_classes=0)
    cfg = resolve_model_data_config(m)
    H, W = cfg["input_size"][1], cfg["input_size"][2]
    interp = cfg.get("interpolation", "bicubic")
    mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406)), dtype=torch.float32)
    std  = torch.tensor(cfg.get("std",  (0.229, 0.224, 0.225)), dtype=torch.float32)
    print(f"[TIMM Config Resolver] Model: {model_name} -> Input: {H}x{W}, Interp: {interp}")
    return (H, W), interp, mean, std

def find_nth_set_bit_pos(n: int, j: int) -> int:
    count = 0
    for i in range(NUM_PLAYERS):
        if (n >> i) & 1:
            if count == j: return i
            count += 1
    return -1

# ===================================================================
# 5. DALI PIPELINE & DATA SOURCE
# ===================================================================
class DALIExternalSource:
    """
    Callable, stateful source for DALI's external_source operator.
    By implementing `__call__` instead of `__iter__`, DALI can create
    independent instances of this class in each worker process,
    allowing for true multiprocessing.
    """
    def __init__(self, sampling_pool: list, data_cfg: DataConfig, train_cfg: TrainConfig, mean_rgb: torch.Tensor):
        self.pool = sampling_pool
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg
        self.mean = mean_rgb
        self.chunk_size_frames = self.train_cfg.context_frames
        
        self.envs = {}
        self.current_chunk_data = None
        self.current_idx_in_chunk = 0
        self.items_in_chunk = self.train_cfg.context_frames * NUM_PLAYERS
        
        self.chunk_count = 0
        self.current_chunk_id = -1
        
        self._rng = random.Random()

    def __call__(self, sample_info: "SampleInfo"):
        if self.current_chunk_data is None or self.current_idx_in_chunk >= self.items_in_chunk:
            self._load_and_process_new_chunk(sample_info)
            self.current_idx_in_chunk = 0

        idx = self.current_idx_in_chunk
        t = idx // NUM_PLAYERS

        jpeg  = self.current_chunk_data['jpegs'][idx]
        mel   = self.current_chunk_data['mels'][idx]
        alive = self.current_chunk_data['alive'][idx]
        stats = self.current_chunk_data['stats'][idx]
        pos   = self.current_chunk_data['pos_hm'][idx]
        mouse = self.current_chunk_data['mouse'][idx]
        kbd   = self.current_chunk_data['kbd'][idx]

        enemy = self.current_chunk_data['enemy_hm_seq'][t]
        state = self.current_chunk_data['state_seq'][t]

        chunk_id = self.current_chunk_id
        sample_id = np.int32(idx)

        self.current_idx_in_chunk += 1
        return (chunk_id, sample_id, jpeg, mel, alive, stats, pos, mouse, kbd, enemy, state)

    def _load_and_process_new_chunk(self, sample_info: "SampleInfo"):
        worker_id = getattr(sample_info, "worker_id", 0)
        if not hasattr(self, "_seeded"):
            base_seed = int(os.getenv("DL_SEED", "1337"))
            self._rng.seed((base_seed ^ (worker_id + 1) << 16) & 0xFFFFFFFF)
            self._seeded = True
            
        self.current_chunk_id = np.int64((worker_id << 32) | self.chunk_count)
        self.chunk_count += 1
        
        while True:
            round_info = self._rng.choice(self.pool)
            start_tick, end_tick = round_info['start_tick'], round_info['end_tick']
            total_frames = (end_tick - start_tick) // TICKS_PER_FRAME
            if total_frames < self.chunk_size_frames: continue

            max_start_frame_idx = total_frames - self.chunk_size_frames
            start_frame_idx = self._rng.randint(0, max_start_frame_idx)
            
            raw_chunk_data = []
            env = self._get_env(round_info['lmdb_path'])
            with env.begin(write=False) as txn:
                for i in range(self.chunk_size_frames):
                    current_tick = start_tick + ((start_frame_idx + i) * TICKS_PER_FRAME)
                    key = (f"{round_info['demo_name']}_round_{round_info['round_num']:03d}_"
                           f"team_{round_info['team']}_tick_{current_tick:08d}").encode('utf-8')
                    value_bytes = txn.get(key)
                    raw_chunk_data.append(msgpack.unpackb(value_bytes, raw=False, object_hook=mpnp.decode) if value_bytes else None)
            
            if any(item is None for item in raw_chunk_data): continue
            
            self.current_chunk_data = self._process_chunk(raw_chunk_data, round_info)
            return

    def _process_chunk(self, raw_chunk_data, round_info) -> dict:
        dummy_jpeg = self._get_dummy_mean_jpeg()

        def _bitmask_to_multihot_np(m, nc): 
            return np.array([(m >> i) & 1 for i in range(nc)], dtype=np.float32)

        def _coords_to_heatmap_np(coords, dims, ext):
            Z, Y, X = dims
            hm = np.zeros(dims, dtype=np.float32)
            if coords is None or (isinstance(coords, np.ndarray) and coords.size == 0):
                return hm
            coords = np.atleast_2d(coords)
            for x, y, z in coords:
                if not np.all(np.isfinite([x, y, z])): 
                    continue
                ix = int(((x - ext['x'][0]) / (ext['x'][1] - ext['x'][0])) * (X - 1))
                iy = int(((y - ext['y'][0]) / (ext['y'][1] - ext['y'][0])) * (Y - 1))
                iz = int(((z - ext['z'][0]) / (ext['z'][1] - ext['z'][0])) * (Z - 1))
                hm[np.clip(iz, 0, Z-1), np.clip(iy, 0, Y-1), np.clip(ix, 0, X-1)] = 1.0
            return hm

        all_jpegs, all_mels = [], []
        all_alive, all_stats, all_pos, all_mouse, all_kbd = [], [], [], [], []
        enemy_seq, state_seq = [], []
        
        for i, frame_data in enumerate(raw_chunk_data):
            gs = frame_data['game_state'][0]
            pd_list = frame_data['player_data']
            mask_int = gs['team_alive']

            slot_to_jpeg = {s: dummy_jpeg for s in range(NUM_PLAYERS)}
            slot_to_mel = {s: np.full((1, 128, MEL_SPEC_TIME_FRAMES), -80.0, dtype=np.float32) for s in range(NUM_PLAYERS)}
            slot_alive = np.zeros(NUM_PLAYERS, dtype=np.bool_)
            slot_stats = np.zeros((NUM_PLAYERS, 3), dtype=np.float32)
            slot_pos   = np.zeros((NUM_PLAYERS, *HEATMAP_DIMS), dtype=np.float32)
            slot_mouse = np.zeros((NUM_PLAYERS, 2), dtype=np.float32)
            slot_kbd   = np.zeros((NUM_PLAYERS, KEYBOARD_DIM), dtype=np.float32)

            for p_idx, p_tuple in enumerate(pd_list):
                slot = find_nth_set_bit_pos(mask_int, p_idx)
                if slot == -1: continue
                
                p_info, jpeg_bytes, mel_spec = p_tuple[0][0], p_tuple[1], p_tuple[2]

                slot_alive[slot] = True
                slot_to_jpeg[slot] = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                if mel_spec is not None:
                    mel = torch.from_numpy(mel_spec.copy()).unsqueeze(0)
                    mel = torch.nn.functional.pad(mel, (0, max(0, MEL_SPEC_TIME_FRAMES - mel.shape[-1])), 'constant', -80.0)[:, :, :MEL_SPEC_TIME_FRAMES]
                    slot_to_mel[slot] = mel.numpy()

                slot_stats[slot] = [p_info['health'], p_info['armor'], p_info['money']]
                slot_pos[slot]   = _coords_to_heatmap_np(p_info['pos'], HEATMAP_DIMS, self.data_cfg.map_extents)
                slot_mouse[slot] = p_info['mouse']
                slot_kbd[slot]   = _bitmask_to_multihot_np(p_info['keyboard_bitmask'], KEYBOARD_DIM)

            for s in range(NUM_PLAYERS):
                all_jpegs.append(slot_to_jpeg[s])
                all_mels.append(slot_to_mel[s])
                all_alive.append(slot_alive[s])
                all_stats.append(slot_stats[s])
                all_pos.append(slot_pos[s])
                all_mouse.append(slot_mouse[s])
                all_kbd.append(slot_kbd[s])

            enemy_seq.append(_coords_to_heatmap_np(gs['enemy_pos'], HEATMAP_DIMS, self.data_cfg.map_extents))
            state_seq.append(_bitmask_to_multihot_np(gs['round_state'], ROUND_STATE_DIM))

        return {
            "jpegs": all_jpegs,
            "mels": np.stack(all_mels),
            "alive": np.asarray(all_alive, dtype=np.bool_),
            "stats": np.asarray(all_stats, dtype=np.float32),
            "pos_hm": np.asarray(all_pos, dtype=np.float32),
            "mouse": np.asarray(all_mouse, dtype=np.float32),
            "kbd":   np.asarray(all_kbd, dtype=np.float32),
            "enemy_hm_seq": np.asarray(enemy_seq, dtype=np.float32),
            "state_seq":    np.asarray(state_seq, dtype=np.float32),
        }

    def _get_env(self, lmdb_path):
        if lmdb_path not in self.envs:
            self.envs[lmdb_path] = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        return self.envs[lmdb_path]
    
    def _get_dummy_mean_jpeg(self):
        if hasattr(self, "_cached_dummy_jpeg"): return self._cached_dummy_jpeg
        rgb = (self.mean.numpy() * 255.0).round().astype(np.uint8)
        img = np.full((1, 1, 3), rgb, dtype=np.uint8)
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        assert ok, "Failed to encode dummy JPEG"
        self._cached_dummy_jpeg = np.frombuffer(bytes(enc), dtype=np.uint8)
        return self._cached_dummy_jpeg

# _FIX: Renamed `seed` to `pipeline_seed` to avoid shadowing warning
@pipeline_def
def create_dali_pipeline(external_source_callable, target_hw, interp_str, mean, std, pipeline_seed=1337):
    """Defines the DALI processing graph."""
    target_h, target_w = target_hw
    mean255, std255 = (mean.numpy() * 255.0).tolist(), (std.numpy() * 255.0).tolist()
    interp_dali = INTERP_MAP.get(interp_str, types.INTERP_LINEAR)
    
    chunk_id, sample_id, jpegs, mels, alive, stats, pos_hm, mouse, kbd, enemy_hm, state = fn.external_source(
        source=external_source_callable,
        num_outputs=11,
        batch=False,
        parallel=True,
        dtype=[types.INT64, types.INT32, types.UINT8, types.FLOAT, types.BOOL, types.FLOAT, types.FLOAT,
               types.FLOAT, types.FLOAT, types.FLOAT, types.FLOAT],
        ndim=[0, 0, 1, 3, 0, 1, 3, 1, 1, 3, 1]
    )
    
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    images = fn.resize(images, size=(target_h, target_w), mode="not_larger", interp_type=interp_dali)
    # _FIX: `fill_value` must be a scalar float. Changed from mean255 to 0.0.
    images = fn.paste(images, ratio=1.0, paste_x=0.5, paste_y=0.5,
                      min_canvas_size=(target_h, target_w),
                      fill_value=0.0)
    images = fn.crop_mirror_normalize(images, dtype=types.FLOAT16, output_layout="CHW",
                                      mean=mean255, std=std255)
    
    pos_hm = fn.cast(pos_hm, dtype=types.FLOAT16)
    enemy_hm = fn.cast(enemy_hm, dtype=types.FLOAT16)

    return (
        chunk_id, sample_id, images, mels.gpu(),
        alive,
        stats,
        pos_hm.gpu(),
        mouse,
        kbd,
        enemy_hm.gpu(),
        state
    )

# ===================================================================
# 6. SCRIPT ENTRYPOINT
# ===================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Data Pipeline for the CS2Transformer.")
    
    dummy_config_path = "dummy_config.yaml"
    with open(dummy_config_path, 'w') as f:
        f.write("data:\n  lmdb_root_path: 'data/lmdb'\n  manifest_path: 'data/lmdb/split_manifest.json'\n")
        f.write("  num_workers: 4\n  use_dali: true\n  dali_num_threads: 4\n  dali_prefetch: 2\n")
        f.write("train:\n  context_frames: 128\n  batch_size: 2\n  model_name: 'vit_base_patch14_dinov2.lvd142m'\n")
    
    parser.add_argument("--config", type=str, default=dummy_config_path, help="Path to the YAML config file.")
    args = parser.parse_args()
    
    print(f"Loading configuration from: {args.config}")
    data_cfg, train_cfg = load_config_from_yaml(args.config)

    try:
        if data_cfg.use_dali and not DALI_AVAILABLE: raise ImportError("use_dali=True but NVIDIA DALI is not installed.")
        if data_cfg.use_dali and not torch.cuda.is_available(): raise RuntimeError("use_dali=True requires a CUDA-enabled PyTorch and DALI.")
        
        target_hw, interp_str, mean, std = _resolve_timm_config(train_cfg.model_name)
        indexer = DatasetIndexer(lmdb_root_path=data_cfg.lmdb_root_path, manifest_path=data_cfg.manifest_path)
        dali_source_callable = DALIExternalSource(indexer.train_pool, data_cfg, train_cfg, mean_rgb=mean)
        
        print("\n--- Using NVIDIA DALI Data Pipeline ---")
        
        effective_batch_size = train_cfg.batch_size * train_cfg.context_frames * NUM_PLAYERS
        
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("use_dali=True requires a CUDA-enabled environment (no GPUs found).")
        if local_rank >= num_gpus:
            raise RuntimeError(f"LOCAL_RANK={local_rank} but only {num_gpus} GPU(s) visible.")
        device_id = local_rank

        pipeline = create_dali_pipeline(
            batch_size=effective_batch_size,
            num_threads=data_cfg.dali_num_threads,
            device_id=device_id,
            seed=1337, # This is the main pipeline seed
            prefetch_queue_depth=data_cfg.dali_prefetch,
            py_num_workers=data_cfg.num_workers,
            py_start_method='spawn',
            # _FIX: Pass the seed to the renamed argument
            external_source_callable=dali_source_callable,
            target_hw=target_hw,
            interp_str=interp_str,
            mean=mean,
            std=std,
            pipeline_seed=int(os.getenv("DL_SEED", "1337"))
        )
        
        pipeline.build()
        torch.cuda.set_device(device_id)
        print(f"DALI pipeline built successfully for device {device_id}. Python workers started via 'spawn'.")

        output_map = ["chunk_id", "sample_id", "images", "mel", "alive", "stats", "pos_hm", "mouse", "kbd", "enemy_hm", "state"]
        
        dali_loader = DALIGenericIterator([pipeline], output_map, reader_name=None, auto_reset=True, last_batch_padded=True)
        
        print("Attempting to fetch one batch from the DALI loader...")
        first_batch_from_dali = next(iter(dali_loader))[0]
        print("Successfully fetched one batch!")

        print("Sorting batch to restore sequential order...")
        chunk_id_cpu = first_batch_from_dali["chunk_id"].cpu()
        sample_id_cpu = first_batch_from_dali["sample_id"].cpu()
        
        _, idx1 = torch.sort(sample_id_cpu, stable=True)
        chunk_sorted = chunk_id_cpu[idx1]
        _, idx2 = torch.sort(chunk_sorted, stable=True)
        
        sorted_idx_cpu = idx1[idx2]
        sorted_idx_gpu = sorted_idx_cpu.cuda(device_id)
        
        sorted_batch = {}
        for k, v in first_batch_from_dali.items():
            if v.is_cuda:
                sorted_batch[k] = v.index_select(0, sorted_idx_gpu)
            else:
                sorted_batch[k] = v.index_select(0, sorted_idx_cpu)
                
        print("Batch sorted successfully.")

        B, T, P = train_cfg.batch_size, train_cfg.context_frames, NUM_PLAYERS

        C, H, W = sorted_batch["images"].shape[1:]
        images_tensor = sorted_batch["images"].reshape(B, T, P, C, H, W)
        mels_tensor   = sorted_batch["mel"].reshape(B, T, P, 1, 128, MEL_SPEC_TIME_FRAMES)
        alive_tensor  = sorted_batch["alive"].reshape(B, T, P)
        stats_tensor  = sorted_batch["stats"].reshape(B, T, P, 3)
        pos_hm_tensor = sorted_batch["pos_hm"].reshape(B, T, P, *HEATMAP_DIMS)
        mouse_tensor  = sorted_batch["mouse"].reshape(B, T, P, 2)
        kbd_tensor    = sorted_batch["kbd"].reshape(B, T, P, KEYBOARD_DIM)
        enemy_tensor  = sorted_batch["enemy_hm"].reshape(B, T, P, *HEATMAP_DIMS)
        state_tensor  = sorted_batch["state"].reshape(B, T, P, ROUND_STATE_DIM)

        batch_for_inspection = {
            "inputs": {
                "images": images_tensor,
                "mel_spectrogram": mels_tensor,
                "alive_mask": alive_tensor
            },
            "targets": {
                "player_stats": stats_tensor,
                "player_pos_heatmaps": pos_hm_tensor,
                "player_mouse": mouse_tensor,
                "player_keyboard": kbd_tensor,
                "enemy_pos_heatmap": enemy_tensor,
                "round_state": state_tensor
            }
        }
        
        print(f"DALI is ENABLED. Image tensor device: '{batch_for_inspection['inputs']['images'].device}'.")
        
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
        if os.path.exists(dummy_config_path):
            os.remove(dummy_config_path)