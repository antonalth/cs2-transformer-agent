#!/usr/bin/env python3
"""
train.py

Main training harness for the CS2Transformer model.
This script orchestrates the data loading, model training, validation,
and checkpointing for the project.
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

# (We will add more imports like model, wandb, etc. in future steps)


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
    # TODO: These should be part of the config file
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
    # ... (code from previous step, no changes needed)
    """
    Handles the initial setup and validation of the dataset.
    ...
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
            env = lmdb.open(str(demo_path), readonly=True, lock=False, readahead=False, meminit=False)
            with env.begin(write=False) as txn:
                info_key = f"{demo_name}_INFO".encode('utf-8')
                info_bytes = txn.get(info_key)
                if info_bytes is None:
                    env.close()
                    raise KeyError(f"FATAL: Could not find metadata key '{info_key.decode()}' in the LMDB for demo '{demo_name}'. The LMDB may be corrupt or incomplete.")
                info_data = json.loads(info_bytes.decode('utf-8'))
                cursor = txn.cursor()
                for round_entry in info_data.get("rounds", []):
                    round_num, start_tick, end_tick = round_entry
                    for team in ['T', 'CT']:
                        test_key = f"{demo_name}_round_{round_num:03d}_team_{team}_tick_{start_tick:08d}".encode('utf-8')
                        if cursor.set_key(test_key):
                            round_metadata = { "lmdb_path": str(demo_path), "demo_name": demo_name, "round_num": round_num, "team": team, "start_tick": start_tick, "end_tick": end_tick }
                            round_pool.append(round_metadata)
                        else:
                            print(f"\n[Warning] Missing perspective 'team={team}' for round {round_num} in demo '{demo_name}'. Skipping.")
            env.close()
        return round_pool

# ===================================================================
# 4. DATA TRANSFORMATION HELPERS
# ===================================================================

# --- TIMM-aware Image Preprocessing Functions ---

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

def _cv2_interpolation(interp_name: str) -> int:
    """Maps TIMM interpolation string to an OpenCV enum."""
    interp_name = (interp_name or "").lower()
    if "bicubic" in interp_name: return cv2.INTER_CUBIC
    if "lanczos" in interp_name: return cv2.INTER_LANCZOS4
    if "nearest" in interp_name: return cv2.INTER_NEAREST
    return cv2.INTER_LINEAR

def _letterbox_resize_pad_opencv(img_np: np.ndarray, target_hw: Tuple[int, int], interp: str, pad_value_rgb_01: Sequence[float]) -> torch.Tensor:
    """Performs a pure OpenCV/NumPy letterbox resize and pad."""
    assert img_np.ndim == 3 and img_np.shape[2] == 3, "Input must be a HWC RGB numpy array"
    Ht, Wt = target_hw
    h, w, c = img_np.shape
    scale = min(Wt / w, Ht / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    if (new_w, new_h) != (w, h):
        img_np = cv2.resize(img_np, (new_w, new_h), interpolation=_cv2_interpolation(interp))
    pad_value_uint8 = (np.array(pad_value_rgb_01, dtype=np.float32) * 255.0).astype(np.uint8)
    canvas = np.full((Ht, Wt, c), pad_value_uint8, dtype=np.uint8)
    top, left = (Ht - new_h) // 2, (Wt - new_w) // 2
    canvas[top:top + new_h, left:left + new_w, :] = img_np
    return torch.from_numpy(canvas).permute(2, 0, 1).float().div(255.0)

def _normalize_inplace(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    """Normalizes a CHW tensor in-place."""
    x.sub_(mean[:, None, None]).div_(std[:, None, None])
    return x

# --- Core Data Transformation Helpers ---

def find_nth_set_bit_pos(n: int, j: int) -> int:
    """Finds the bit position of the j-th 'on' bit in integer n."""
    count = 0
    for i in range(NUM_PLAYERS):
        if (n >> i) & 1:
            if count == j:
                return i
            count += 1
    return -1

# ... (Bitmask and heatmap helpers will be added in later steps)

# ===================================================================
# 5. PYTORCH DATASET & DATALOADER
# ===================================================================

class LMDBStreamerDataset(IterableDataset):
    """
    A PyTorch IterableDataset for streaming and processing data from CS2 LMDBs.
    """
    def __init__(self, sampling_pool: list, data_cfg: DataConfig, train_cfg: TrainConfig):
        super().__init__()
        self.pool = sampling_pool
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg
        self.chunk_size_frames = self.train_cfg.context_frames + 1
        self.envs = {}
        
        # Resolve the TIMM config once. This is inherited by worker processes.
        self.target_hw, self.interp, self.mean, self.std = _resolve_timm_config(self.train_cfg.model_name)

    def _get_env(self, lmdb_path):
        if lmdb_path not in self.envs:
            self.envs[lmdb_path] = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        return self.envs[lmdb_path]
        
    def _process_chunk(self, raw_chunk_data: List[Dict[str, Any]], round_info: dict) -> Dict[str, Any]:
        """Transforms a chunk of raw data from LMDB into processed tensors."""
        all_frame_images, all_frame_mels, all_alive_masks = [], [], []

        for frame_data in raw_chunk_data[:self.train_cfg.context_frames]:
            game_state = frame_data['game_state'][0]
            player_data_list = frame_data['player_data']
            
            padded_images = torch.zeros(NUM_PLAYERS, 3, *self.target_hw, dtype=torch.float32)
            padded_mels = torch.zeros(NUM_PLAYERS, 1, 128, 8, dtype=torch.float32) # Assuming fixed mel shape
            
            team_alive_mask = game_state['team_alive']
            alive_mask = torch.tensor([(team_alive_mask >> i) & 1 for i in range(NUM_PLAYERS)], dtype=torch.bool)
            
            for i, p_data_tuple in enumerate(player_data_list):
                slot_index = find_nth_set_bit_pos(team_alive_mask, i)
                if slot_index == -1: continue

                p_info, jpeg_bytes, mel_spec = p_data_tuple
                
                # --- OpenCV-Optimized Preprocessing Pipeline ---
                img_bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                img_rgb_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_tensor_01 = _letterbox_resize_pad_opencv(img_rgb_np, self.target_hw, self.interp, self.mean.tolist())
                img_tensor_norm = _normalize_inplace(img_tensor_01, self.mean, self.std)
                padded_images[slot_index] = img_tensor_norm

                if mel_spec is not None:
                    # TODO: Implement proper mel spectrogram padding/truncating
                    pass

            all_frame_images.append(padded_images)
            all_frame_mels.append(padded_mels)
            all_alive_masks.append(alive_mask)

        # TODO: Implement full target transformation logic
        target_frame_data = raw_chunk_data[-1]
        
        inputs = {
            "images": torch.stack(all_frame_images, dim=0),
            "mel_spectrograms": torch.stack(all_frame_mels, dim=0),
            "alive_mask": torch.stack(all_alive_masks, dim=0),
        }
        
        targets = { "placeholder": torch.tensor(0) } # Placeholder for now

        return {"inputs": inputs, "targets": targets}

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
                        raw_chunk_data.append(None)
            
            if any(item is None for item in raw_chunk_data):
                continue
            
            yield self._process_chunk(raw_chunk_data, round_info)

# ===================================================================
# 6. SCRIPT ENTRYPOINT
# ===================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Tier 1 Data Pipeline for the CS2Transformer.")
    
    dummy_config_path = "dummy_config.yaml"
    with open(dummy_config_path, 'w') as f:
        f.write("data:\n")
        f.write("  lmdb_root_path: '/path/to/your/lmdbs' # <-- EDIT THIS PATH\n")
        f.write("  manifest_path: '/path/to/your/lmdbs/split_manifest.json' # <-- AND THIS PATH\n")
        f.write("  num_workers: 2\n")
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
        indexer = DatasetIndexer(lmdb_root_path=data_cfg.lmdb_root_path, manifest_path=data_cfg.manifest_path)
        dataset = LMDBStreamerDataset(sampling_pool=indexer.train_pool, data_cfg=data_cfg, train_cfg=train_cfg)
        dataloader = DataLoader(dataset, batch_size=train_cfg.batch_size, num_workers=data_cfg.num_workers)

        print("\nAttempting to fetch one batch from the dataloader...")
        first_batch = next(iter(dataloader))
        print("Successfully fetched one batch!")

        print("\n--- Batch Content ---")
        for key, value in first_batch.items():
            print(f"\nTop-level key: '{key}'")
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        print(f"  - Sub-key: '{sub_key}', Shape: {sub_value.shape}, DType: {sub_value.dtype}")
        print("---------------------\n")
        print("Tier 1 Data Pipeline test successful!")

    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\nAn error occurred during the test:\n{e}")
        print("\nPlease ensure paths in your config file ('dummy_config.yaml') are correct.")
    except ImportError as e:
        print(f"\nAn import error occurred: {e}")
    finally:
        if os.path.exists(dummy_config_path):
            os.remove(dummy_config_path)