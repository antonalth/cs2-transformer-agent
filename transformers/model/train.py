#!/usr/bin/env python3
"""
train.py

Main training harness for the CS2Transformer model.
This script orchestrates the data loading, model training, validation,
and checkpointing for the project.

This version is updated to use NVIDIA DALI for GPU-accelerated data preprocessing.
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
# Note: DataLoader is replaced by DALI's iterator
# from torch.utils.data import IterableDataset, DataLoader

# --- DALI IMPORTS ---
try:
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
except ImportError:
    raise ImportError("NVIDIA DALI is not installed. Please install it following the official instructions.")


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
    map_extents: Dict = field(default_factory=lambda: {
        "x": (-2000.0, 3000.0), "y": (-3500.0, 2500.0), "z": (-500.0, 500.0)
    })

@dataclass
class TrainConfig:
    context_frames: int = 512
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
    """
    Handles the initial setup and validation of the dataset. This class
    scans the LMDBs listed in the manifest to create a pool of valid
    round perspectives to sample from during training.
    """
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
                raise FileNotFoundError(f"FATAL: Manifest lists demo '{demo_name}' for the '{set_name}' set, but it was not found at the expected path: {demo_path}")
            
            demo_name_base = demo_name.removesuffix('.lmdb')

            env = lmdb.open(str(demo_path), readonly=True, lock=False, readahead=False, meminit=False)
            with env.begin(write=False) as txn:
                info_key = f"{demo_name_base}_INFO".encode('utf-8')
                info_bytes = txn.get(info_key)
                if info_bytes is None:
                    env.close()
                    raise KeyError(f"FATAL: Could not find metadata key '{info_key.decode()}' in the LMDB for demo '{demo_name}'. The LMDB may be corrupt or incomplete.")
                
                info_data = json.loads(info_bytes.decode('utf-8'))
                cursor = txn.cursor()
                for round_entry in info_data.get("rounds", []):
                    round_num, start_tick, end_tick = round_entry
                    for team in ['T', 'CT']:
                        test_key = f"{demo_name_base}_round_{round_num:03d}_team_{team}_tick_{start_tick:08d}".encode('utf-8')
                        if cursor.set_key(test_key):
                            round_metadata = { "lmdb_path": str(demo_path), "demo_name": demo_name_base, "round_num": round_num, "team": team, "start_tick": start_tick, "end_tick": end_tick }
                            round_pool.append(round_metadata)
                        else:
                            print(f"\n[Warning] Missing perspective 'team={team}' for round {round_num} in demo '{demo_name}'. Skipping.")
            env.close()
        return round_pool


# ===================================================================
# 4. DATA TRANSFORMATION HELPERS
# ===================================================================
# These constants should match the dimensions defined in model.py
HEATMAP_DIMS = (8, 64, 64) # Z, Y, X
KEYBOARD_DIM = 31
ECO_DIM = 224
INVENTORY_DIM = 128
WEAPON_DIM = 128
ROUND_STATE_DIM = 5
MEL_SPEC_TIME_FRAMES = 8

def _resolve_timm_config(model_name: str):
    """Creates a dummy TIMM model to resolve its preprocessing config."""
    try:
        import timm
        from timm.data import resolve_model_data_config
    except ImportError:
        raise ImportError("Please install timm to use the ViT model: `pip install timm`")
    
    m = timm.create_model(model_name, pretrained=True, num_classes=0)
    cfg = resolve_model_data_config(m)
    H, W = cfg["input_size"][1], cfg["input_size"][2]
    interp = cfg.get("interpolation", "bicubic")
    mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406)), dtype=torch.float32)
    std  = torch.tensor(cfg.get("std",  (0.229, 0.224, 0.225)), dtype=torch.float32)
    print(f"[TIMM Config Resolver] Model: {model_name} -> Input: {H}x{W}, Interp: {interp}")
    return (H, W), interp, mean, std

# --- Core Data Transformation Helpers (for CPU-processed data) ---
def find_nth_set_bit_pos(n: int, j: int) -> int:
    """Finds the bit position of the j-th 'on' bit in integer n."""
    count = 0
    for i in range(NUM_PLAYERS):
        if (n >> i) & 1:
            if count == j:
                return i
            count += 1
    return -1

def _bitmask_to_multihot(mask: int, num_classes: int) -> torch.Tensor:
    """Converts a single integer bitmask to a multi-hot tensor."""
    return torch.tensor([(mask >> i) & 1 for i in range(num_classes)], dtype=torch.float32)

def _bitmask_array_to_multihot(mask_array: np.ndarray, num_classes: int) -> torch.Tensor:
    """Converts a numpy array of uint64 bitmasks to a single multi-hot tensor."""
    full_mask = 0
    for i, part in enumerate(mask_array):
        full_mask |= int(part) << (i * 64)
    return _bitmask_to_multihot(full_mask, num_classes)

def _coords_to_heatmap(coords: np.ndarray, dims: Tuple[int, int, int], extents: Dict) -> torch.Tensor:
    """Converts a set of world coordinates to a multi-hot 3D heatmap."""
    Z_DIM, Y_DIM, X_DIM = dims
    heatmap = torch.zeros(dims, dtype=torch.float32)
    if coords is None or coords.size == 0:
        return heatmap
        
    min_x, max_x = extents['x']
    min_y, max_y = extents['y']
    min_z, max_z = extents['z']
    
    coords = np.atleast_2d(coords)
    for x, y, z in coords:
        if not np.all(np.isfinite([x, y, z])): continue
        norm_x = (x - min_x) / (max_x - min_x)
        norm_y = (y - min_y) / (max_y - min_y)
        norm_z = (z - min_z) / (max_z - min_z)
        idx_x = int(norm_x * (X_DIM - 1))
        idx_y = int(norm_y * (Y_DIM - 1))
        idx_z = int(norm_z * (Z_DIM - 1))
        idx_x = max(0, min(X_DIM - 1, idx_x))
        idx_y = max(0, min(Y_DIM - 1, idx_y))
        idx_z = max(0, min(Z_DIM - 1, idx_z))
        heatmap[idx_z, idx_y, idx_x] = 1.0
    return heatmap


# ===================================================================
# 5. DALI DATA PIPELINE
# ===================================================================

class DALIDataFeeder:
    """
    An iterable data feeder for the DALI pipeline. It reads raw, compressed
    data from the LMDB database and yields it to the pipeline.
    """
    def __init__(self, sampling_pool: list, data_cfg: DataConfig, train_cfg: TrainConfig):
        self.pool = sampling_pool
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg
        self.chunk_size_frames = self.train_cfg.context_frames + 1
        self.envs = {}

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_pool = self.pool
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            worker_pool = self.pool[worker_id::num_workers]

        random.seed(random.randint(0, 2**32-1) + (worker_info.id if worker_info else 0))

        while True:
            round_info = random.choice(worker_pool)
            start_tick, end_tick = round_info['start_tick'], round_info['end_tick']
            total_frames = (end_tick - start_tick) // TICKS_PER_FRAME
            if total_frames < self.chunk_size_frames:
                continue

            target_frame_idx = random.randint(0, total_frames - 1)
            max_start_frame_idx = total_frames - self.chunk_size_frames
            possible_start_min = max(0, target_frame_idx - self.chunk_size_frames + 1)
            possible_start_max = min(max_start_frame_idx, target_frame_idx)
            start_frame_idx = random.randint(possible_start_min, possible_start_max)
            
            raw_chunk_data = self._fetch_raw_chunk(round_info, start_tick, start_frame_idx)
            
            if raw_chunk_data is None:
                continue
            
            yield self._prepare_dali_input(raw_chunk_data, round_info)

    def _get_env(self, lmdb_path):
        if lmdb_path not in self.envs:
            self.envs[lmdb_path] = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        return self.envs[lmdb_path]

    def _fetch_raw_chunk(self, round_info: dict, start_tick: int, start_frame_idx: int) -> list | None:
        """Fetches a chunk of raw msgpack data from LMDB."""
        raw_chunk_data = []
        env = self._get_env(round_info['lmdb_path'])
        with env.begin(write=False) as txn:
            for i in range(self.chunk_size_frames):
                current_frame_idx = start_frame_idx + i
                current_tick = start_tick + (current_frame_idx * TICKS_PER_FRAME)
                key = (f"{round_info['demo_name']}_round_{round_info['round_num']:03d}_"
                       f"team_{round_info['team']}_tick_{current_tick:08d}").encode('utf-8')
                value_bytes = txn.get(key)
                if value_bytes:
                    unpacked_data = msgpack.unpackb(value_bytes, raw=False, object_hook=mpnp.decode)
                    raw_chunk_data.append(unpacked_data)
                else:
                    return None
        return raw_chunk_data

    def _prepare_dali_input(self, raw_chunk_data: list, round_info: dict) -> tuple:
        """
        Organizes the raw data into a format DALI's ExternalSource can consume.
        """
        all_jpegs = []
        
        for frame_data in raw_chunk_data[:self.train_cfg.context_frames]:
            jpegs_for_frame = [np.empty(0, dtype=np.uint8)] * NUM_PLAYERS
            team_alive_mask = frame_data['game_state'][0]['team_alive']
            
            for i, p_data_tuple in enumerate(frame_data['player_data']):
                slot_index = find_nth_set_bit_pos(team_alive_mask, i)
                if slot_index != -1:
                    _, jpeg_bytes, _ = p_data_tuple
                    jpegs_for_frame[slot_index] = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            all_jpegs.extend(jpegs_for_frame)
        
        cpu_passthrough_data = {
            "raw_chunk_data": raw_chunk_data,
            "round_info": round_info,
        }
        
        return (all_jpegs, cpu_passthrough_data)

class CS2Pipeline(Pipeline):
    """
    NVIDIA DALI Pipeline for processing CS2 data.
    """
    def __init__(self, feeder: DALIDataFeeder, batch_size: int, num_threads: int, device_id: int,
                 target_hw: tuple, interp_str: str, mean: list, std: list):
        super().__init__(batch_size, num_threads, device_id, seed=12345)
        self.feeder = feeder
        self.target_h, self.target_w = target_hw
        
        self.mean_01 = torch.tensor(mean).view(1, 1, 3).tolist()
        self.std_01 = torch.tensor(std).view(1, 1, 3).tolist()

        interp_map = {
            "bicubic": types.INTERP_CUBIC,
            "lanczos": types.INTERP_LANCZOS3,
            "nearest": types.INTERP_NN
        }
        self.interp_type = interp_map.get(interp_str.lower(), types.INTERP_LINEAR)

        # Define the external source operator that consumes data from our feeder
        self.source = fn.external_source(
            source=self.feeder,
            num_outputs=2,
            # --- FIX IS HERE ---
            # The layout argument expects strings. Use "" for data with no layout.
            layout=["", ""],
            dtype=[types.UINT8, types.DALIDataType.ANY_DATA],
            batch=False,
            parallel=True
        )

    def define_graph(self):
        jpeg_list, cpu_data = self.source()
        jpegs = fn.stack(*jpeg_list)
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)

        resized_images = fn.resize(
            images, device="gpu", resize_x=self.target_w, resize_y=self.target_h,
            mode="not_smaller", interp_type=self.interp_type
        )
        padded_images = fn.pad(
            resized_images, device="gpu", axes=(0, 1),
            fill_value=0, shape=(self.target_h, self.target_w)
        )

        normalized_images = fn.crop_mirror_normalize(
            padded_images, device="gpu", dtype=types.FLOAT, output_layout="CHW",
            mean=[m * 255.0 for m in self.mean_01[0][0]],
            std=[s * 255.0 for s in self.std_01[0][0]]
        )

        return (normalized_images, cpu_data)

def collate_and_process_cpu_data(batch: list, data_cfg: DataConfig, train_cfg: TrainConfig) -> dict:
    """
    This function is executed on the CPU after DALI has processed the images.
    It takes the raw "passthrough" data and generates the final target tensors.
    """
    batch_size = len(batch)
    context_frames = train_cfg.context_frames
    
    all_inputs = {"mel_spectrogram": [], "alive_mask": []}
    all_targets = {
        "player_stats": torch.zeros(batch_size, NUM_PLAYERS, 3, dtype=torch.float32),
        "player_pos_heatmaps": torch.zeros(batch_size, NUM_PLAYERS, *HEATMAP_DIMS, dtype=torch.float32),
        "player_mouse": torch.zeros(batch_size, NUM_PLAYERS, 2, dtype=torch.float32),
        "player_keyboard": torch.zeros(batch_size, NUM_PLAYERS, KEYBOARD_DIM, dtype=torch.float32),
        "player_eco": torch.zeros(batch_size, NUM_PLAYERS, ECO_DIM, dtype=torch.float32),
        "player_inventory": torch.zeros(batch_size, NUM_PLAYERS, INVENTORY_DIM, dtype=torch.float32),
        "player_active_weapon": torch.zeros(batch_size, NUM_PLAYERS, WEAPON_DIM, dtype=torch.float32),
        "enemy_pos_heatmap": torch.zeros(batch_size, *HEATMAP_DIMS, dtype=torch.float32),
        "round_state": torch.zeros(batch_size, ROUND_STATE_DIM, dtype=torch.float32),
        "round_number": torch.zeros(batch_size, 1, dtype=torch.float32),
    }

    for i, sample in enumerate(batch):
        raw_chunk_data = sample['raw_chunk_data']
        round_info = sample['round_info']
        
        frame_mels, alive_masks = [], []
        for frame_data in raw_chunk_data[:context_frames]:
            padded_mels = torch.zeros(NUM_PLAYERS, 1, 128, MEL_SPEC_TIME_FRAMES, dtype=torch.float32)
            team_alive_mask = frame_data['game_state'][0]['team_alive']
            alive_mask = torch.tensor([(team_alive_mask >> i) & 1 for i in range(NUM_PLAYERS)], dtype=torch.bool)
            
            for p_idx, p_data_tuple in enumerate(frame_data['player_data']):
                slot_index = find_nth_set_bit_pos(team_alive_mask, p_idx)
                if slot_index == -1: continue
                _, _, mel_spec = p_data_tuple
                if mel_spec is not None:
                    mel_tensor = torch.from_numpy(mel_spec.copy()).unsqueeze(0)
                    if mel_tensor.shape[-1] > MEL_SPEC_TIME_FRAMES: padded_mels[slot_index] = mel_tensor[:, :, :, :MEL_SPEC_TIME_FRAMES]
                    else: pad_width = MEL_SPEC_TIME_FRAMES - mel_tensor.shape[-1]; padded_mels[slot_index] = torch.nn.functional.pad(mel_tensor, (0, pad_width), 'constant', -80.0)
            
            frame_mels.append(padded_mels)
            alive_masks.append(alive_mask)

        all_inputs["mel_spectrogram"].append(torch.stack(frame_mels, dim=0))
        all_inputs["alive_mask"].append(torch.stack(alive_masks, dim=0))

        target_frame_data = raw_chunk_data[-1]
        gs_target = target_frame_data['game_state'][0]
        pd_target_list = target_frame_data['player_data']
        team_alive_target = gs_target['team_alive']
        
        for p_idx, p_data_tuple in enumerate(pd_target_list):
            slot_index = find_nth_set_bit_pos(team_alive_target, p_idx)
            if slot_index == -1: continue
            p_struct, _, _ = p_data_tuple
            p_info = p_struct[0]
            all_targets["player_stats"][i, slot_index] = torch.tensor([p_info['health'], p_info['armor'], p_info['money']], dtype=torch.float32)
            all_targets["player_pos_heatmaps"][i, slot_index] = _coords_to_heatmap(p_info['pos'], HEATMAP_DIMS, data_cfg.map_extents)
            all_targets["player_mouse"][i, slot_index] = torch.from_numpy(p_info['mouse'])
            all_targets["player_keyboard"][i, slot_index] = _bitmask_to_multihot(p_info['keyboard_bitmask'], KEYBOARD_DIM)
            all_targets["player_eco"][i, slot_index] = _bitmask_array_to_multihot(p_info['eco_bitmask'], ECO_DIM)
            all_targets["player_inventory"][i, slot_index] = _bitmask_array_to_multihot(p_info['inventory_bitmask'], INVENTORY_DIM)
            all_targets["player_active_weapon"][i, slot_index] = _bitmask_array_to_multihot(p_info['active_weapon_bitmask'], WEAPON_DIM)
        
        all_targets["enemy_pos_heatmap"][i] = _coords_to_heatmap(gs_target['enemy_pos'], HEATMAP_DIMS, data_cfg.map_extents)
        all_targets["round_state"][i] = _bitmask_to_multihot(gs_target['round_state'], ROUND_STATE_DIM)
        all_targets["round_number"][i] = torch.tensor([round_info['round_num']], dtype=torch.float32)

    final_inputs = {k: torch.stack(v) for k, v in all_inputs.items()}
    
    return {"inputs": final_inputs, "targets": all_targets}


# ===================================================================
# 6. SCRIPT ENTRYPOINT
# ===================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Tier 1 Data Pipeline for the CS2Transformer with DALI.")
    
    dummy_config_path = "dummy_config.yaml"
    with open(dummy_config_path, 'w') as f:
        f.write("data:\n")
        f.write("  lmdb_root_path: 'data/lmdb' # <-- EDIT THIS PATH\n")
        f.write("  manifest_path: 'data/lmdb/split_manifest.json' # <-- AND THIS PATH\n")
        f.write("  num_workers: 4\n")
        f.write("\ntrain:\n")
        f.write("  context_frames: 128\n")
        f.write("  steps_per_epoch: 100\n")
        f.write("  batch_size: 2\n")
        f.write("  model_name: 'vit_base_patch14_dinov2.lvd142m'\n")
    
    parser.add_argument("--config", type=str, default=dummy_config_path, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    
    print(f"Loading configuration from: {args.config}")
    data_cfg, train_cfg = load_config_from_yaml(args.config)
    
    try:
        target_hw, interp, mean, std = _resolve_timm_config(train_cfg.model_name)
        
        indexer = DatasetIndexer(lmdb_root_path=data_cfg.lmdb_root_path, manifest_path=data_cfg.manifest_path)
        feeder = DALIDataFeeder(sampling_pool=indexer.train_pool, data_cfg=data_cfg, train_cfg=train_cfg)

        pipeline = CS2Pipeline(
            feeder=feeder, batch_size=train_cfg.batch_size,
            num_threads=data_cfg.num_workers, device_id=0,
            target_hw=target_hw, interp_str=interp,
            mean=mean.tolist(), std=std.tolist()
        )

        dali_iterator = DALIGenericIterator(
            pipelines=[pipeline], output_map=["images_gpu", "cpu_passthrough_data"],
            reader_name=None, auto_reset=True
        )

        print("\nAttempting to fetch one batch from the DALI iterator...")
        first_dali_output = next(iter(dali_iterator))[0]
        
        gpu_images = first_dali_output["images_gpu"]
        gpu_images = gpu_images.view(
            train_cfg.batch_size, train_cfg.context_frames,
            NUM_PLAYERS, *gpu_images.shape[1:]
        )

        cpu_processed_data = collate_and_process_cpu_data(
            first_dali_output["cpu_passthrough_data"], data_cfg, train_cfg
        )
        
        first_batch = {
            "inputs": { "images": gpu_images, **cpu_processed_data["inputs"] },
            "targets": cpu_processed_data["targets"]
        }
        
        print("Successfully fetched and processed one batch!")
        print("\n--- Batch Content ---")
        for key, value in first_batch.items():
            print(f"\nTop-level key: '{key}'")
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        print(f"  - Sub-key: '{sub_key}', Shape: {sub_value.shape}, DType: {sub_value.dtype}, Device: {sub_value.device}")
        print("---------------------\n")
        print("DALI Data Pipeline test successful!")

    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\nAn error occurred during the test:\n{e}")
        print("\nPlease ensure paths in your config file are correct.")
    except ImportError as e:
        print(f"\nAn import error occurred: {e}")
    finally:
        if os.path.exists(dummy_config_path):
            os.remove(dummy_config_path)