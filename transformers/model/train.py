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

# This is a fundamental property of the dataset created by injection_mold.py
TICKS_PER_FRAME = 2

@dataclass
class DataConfig:
    lmdb_root_path: str
    manifest_path: str
    num_workers: int = 8

@dataclass
class TrainConfig:
    context_frames: int = 512  # 16 seconds
    steps_per_epoch: int = 10000
    batch_size: int = 4
    # (More training params like learning_rate will be added later)

def load_config_from_yaml(path: str) -> tuple[DataConfig, TrainConfig]:
    """Loads configuration from a YAML file."""
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
    Handles the initial setup and validation of the dataset.

    This class is instantiated once at the beginning of a training run.
    Its responsibilities are:
    1.  Parse the `split_manifest.json` file to get the train/val demo lists.
    2.  Verify that all demos listed in the manifest actually exist on disk.
    3.  For each demo, iterate through its rounds and validate that the 'T' and 'CT'
        perspectives exist in the LMDB.
    4.  Produce two final "sampling pools" (lists of round-perspective
        dictionaries), one for training and one for validation.
    """

    def __init__(self, lmdb_root_path: str, manifest_path: str):
        """
        Initializes the indexer and orchestrates the entire indexing process.

        Args:
            lmdb_root_path (str): The path to the root folder containing all LMDB directories.
            manifest_path (str): The direct path to the `split_manifest.json` file.
        """
        print("--- Initializing DatasetIndexer ---")
        self.lmdb_root = Path(lmdb_root_path)
        self.manifest_path = Path(manifest_path)

        # 1. Load the manifest file
        manifest_data = self._load_manifest()
        train_demos = manifest_data.get("train_demos", [])
        val_demos = manifest_data.get("validation_demos", [])
        print(f"Manifest loaded successfully. Found {len(train_demos)} train and {len(val_demos)} val demos.")

        # 2. Index both the training and validation sets
        self.train_pool = self._index_demos(train_demos, "Training")
        self.val_pool = self._index_demos(val_demos, "Validation")

        # 3. Print a final summary
        print("\n--- Dataset Indexing Complete ---")
        print(f"  - Total Training Round Perspectives Found:   {len(self.train_pool)}")
        print(f"  - Total Validation Round Perspectives Found: {len(self.val_pool)}")
        print("---------------------------------")

    def _load_manifest(self) -> dict:
        """Loads and validates the JSON manifest file."""
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"FATAL: Manifest file not found at the specified path: {self.manifest_path}"
            )
        try:
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"FATAL: Error decoding JSON from manifest file: {self.manifest_path}\n{e}")

    def _index_demos(self, demo_names: list, set_name: str) -> list:
        """
        Iterates through a list of demo names, validates their existence,
        and aggregates all their available round-perspective metadata.

        Args:
            demo_names (list): A list of demo folder names to process.
            set_name (str): The name of the set ("Training" or "Validation") for logging.

        Returns:
            list: A list of dictionaries, where each dict represents a single round-perspective.
        """
        round_pool = []
        
        if not demo_names:
            print(f"No demos found for '{set_name}' set in manifest. Skipping.")
            return round_pool

        print(f"\nIndexing {set_name} set...")
        for demo_name in tqdm(demo_names, desc=f"  -> Indexing {set_name} Demos"):
            demo_path = self.lmdb_root / demo_name

            if not demo_path.is_dir():
                raise FileNotFoundError(
                    f"FATAL: Manifest lists demo '{demo_name}' for the '{set_name}' set, "
                    f"but it was not found at the expected path: {demo_path}"
                )

            env = lmdb.open(str(demo_path), readonly=True, lock=False, readahead=False, meminit=False)

            with env.begin(write=False) as txn:
                info_key = f"{demo_name}_INFO".encode('utf-8')
                info_bytes = txn.get(info_key)

                if info_bytes is None:
                    env.close()
                    raise KeyError(
                        f"FATAL: Could not find metadata key '{info_key.decode()}' in the "
                        f"LMDB for demo '{demo_name}'. The LMDB may be corrupt or incomplete."
                    )
                
                info_data = json.loads(info_bytes.decode('utf-8'))
                
                cursor = txn.cursor()

                for round_entry in info_data.get("rounds", []):
                    round_num, start_tick, end_tick = round_entry
                    
                    for team in ['T', 'CT']:
                        test_key = f"{demo_name}_round_{round_num:03d}_team_{team}_tick_{start_tick:08d}".encode('utf-8')
                        
                        if cursor.set_key(test_key):
                            # ==============================================================
                            # IMPORTANT NOTE ON TICK STRUCTURE (from injection_mold.py)
                            # --------------------------------------------------------------
                            # The data is sampled at 32 FPS from a 64 tick-rate game,
                            # meaning TICKS_PER_FRAME = 2.
                            #
                            # This means that valid tick keys in the LMDB step by 2.
                            # For example, if start_tick is 100, the valid keys for this
                            # round-perspective will be 100, 102, 104, 106, etc., up to
                            # the end_tick.
                            #
                            # The DataLoader/sampler implementation MUST account for this.
                            # It cannot simply pick a random integer between start_tick
                            # and end_tick; it must pick from the valid, stepped range.
                            # ==============================================================
                            round_metadata = {
                                "lmdb_path": str(demo_path),
                                "demo_name": demo_name,
                                "round_num": round_num,
                                "team": team,
                                "start_tick": start_tick,
                                "end_tick": end_tick
                            }
                            round_pool.append(round_metadata)
                        else:
                            print(f"\n[Warning] Missing perspective 'team={team}' for round {round_num} in demo '{demo_name}'. Skipping.")
            
            env.close()

        return round_pool

# ===================================================================
# 4. PYTORCH DATASET & DATALOADER
# ===================================================================

class LMDBStreamerDataset(IterableDataset):
    """
    A PyTorch IterableDataset for streaming data from a collection of LMDBs.
    This dataset is designed for large-scale, distributed training.
    """
    def __init__(self, sampling_pool: list, data_cfg: DataConfig, train_cfg: TrainConfig):
        super().__init__()
        self.pool = sampling_pool
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg
        self.chunk_size_frames = self.train_cfg.context_frames + 1
        
        # In-worker cache for LMDB environments to avoid re-opening files
        self.envs = {}

    def _get_env(self, lmdb_path):
        """Opens and caches LMDB environments on a per-worker basis."""
        if lmdb_path not in self.envs:
            self.envs[lmdb_path] = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        return self.envs[lmdb_path]

    def _process_chunk(self, raw_chunk_data: list, round_info: dict) -> dict:
        """
        Transforms a chunk of raw data from LMDB into processed tensors.
        
        TODO: This is where the core data transformation logic will live.
        - Padding of player data to a fixed size of 5.
        - Transformation of bitmasks to multi-hot tensors.
        - Transformation of coordinates to heatmaps.
        """
        # For now, let's implement the essential parts: deserialization and image decoding.
        
        images = []
        for frame_data in raw_chunk_data:
            # frame_data is a tuple: (player_info_bytes, jpeg_bytes, mel_spectrogram_bytes)
            # The model expects a list of 5 player images per frame.
            
            # TODO: Implement full 5-player padding logic.
            # For now, we just process the first available player for simplicity.
            if frame_data and frame_data[0]:
                jpeg_bytes = frame_data[0][1] # Get the first player's jpeg
                img_bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                images.append(img_rgb)
            else:
                 # Append a dummy black frame if no players are alive
                images.append(np.zeros((480, 640, 3), dtype=np.uint8))

        # Stack into a single numpy array and convert to tensor
        images_tensor = torch.from_numpy(np.stack(images, axis=0)).permute(0, 3, 1, 2) # T, C, H, W
        
        # The final output will be a dictionary of tensors
        return {"images": images_tensor}
        

    def __iter__(self):
        # Handle DDP sharding
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process data loading
            worker_id = 0
            num_workers = 1
            worker_pool = self.pool
        else:
            # Multi-process data loading
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            # Each worker gets a unique, non-overlapping shard of the data
            worker_pool = self.pool[worker_id::num_workers]

        # Set a unique random seed for each worker to ensure different data streams
        random.seed(random.randint(0, 2**32-1) + worker_id)

        # Infinite loop to continuously stream data
        while True:
            # 1. Randomly select a round-perspective from this worker's pool
            round_info = random.choice(worker_pool)
            start_tick, end_tick = round_info['start_tick'], round_info['end_tick']
            
            # Calculate total number of frames available in this round
            total_frames = (end_tick - start_tick) // TICKS_PER_FRAME
            if total_frames < self.chunk_size_frames:
                continue # Skip this round, it's too short for a full chunk

            # --- Two-Stage Uniform Sampling (by Frame Index) ---
            # Stage 1: Uniformly sample a TARGET frame index
            target_frame_idx = random.randint(0, total_frames - 1)

            # Stage 2: Uniformly sample a valid CHUNK START that contains the target
            max_start_frame_idx = total_frames - self.chunk_size_frames
            
            possible_start_min = max(0, target_frame_idx - self.chunk_size_frames + 1)
            possible_start_max = min(max_start_frame_idx, target_frame_idx)
            
            start_frame_idx = random.randint(possible_start_min, possible_start_max)
            
            # --- Read the chunk from LMDB ---
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
                        # Deserialization happens later in _process_chunk
                        # The value is a msgpack'd dict {"game_state": ..., "player_data": [...]}
                        # player_data is a list of tuples: [(pi_dtype, jpg_bytes, mel_spec), ...]
                        unpacked_data = msgpack.unpackb(value_bytes, raw=False, object_hook=mpnp.decode)
                        raw_chunk_data.append(unpacked_data.get('player_data'))
                    else:
                        # Append None if a key is missing, though this shouldn't happen
                        raw_chunk_data.append(None)

            # Process the raw data into tensors and yield
            yield self._process_chunk(raw_chunk_data, round_info)


# ===================================================================
# 5. SCRIPT ENTRYPOINT (for testing Tier 1)
# ===================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the Tier 1 Data Pipeline for the CS2Transformer."
    )
    # Create a dummy yaml file for testing
    with open('dummy_config.yaml', 'w') as f:
        f.write("data:\n")
        f.write("  lmdb_root_path: '/path/to/your/lmdbs'\n")
        f.write("  manifest_path: '/path/to/your/lmdbs/split_manifest.json'\n")
        f.write("  num_workers: 2\n")
        f.write("\n")
        f.write("train:\n")
        f.write("  context_frames: 128\n")
        f.write("  steps_per_epoch: 100\n")
        f.write("  batch_size: 2\n")
    
    parser.add_argument(
        "--config",
        type=str,
        default="dummy_config.yaml",
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()
    
    print(f"Loading configuration from: {args.config}")
    data_cfg, train_cfg = load_config_from_yaml(args.config)
    
    print("\nNOTE: Replace paths in 'dummy_config.yaml' with your actual dataset paths to run this test.")

    try:
        # 1. Instantiate the Indexer
        indexer = DatasetIndexer(
            lmdb_root_path=data_cfg.lmdb_root_path,
            manifest_path=data_cfg.manifest_path
        )

        # 2. Instantiate the Dataset
        dataset = LMDBStreamerDataset(
            sampling_pool=indexer.train_pool,
            data_cfg=data_cfg,
            train_cfg=train_cfg
        )

        # 3. Instantiate the DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=train_cfg.batch_size,
            num_workers=data_cfg.num_workers
        )

        # 4. Fetch one batch to test the entire pipeline
        print("\nAttempting to fetch one batch from the dataloader...")
        first_batch = next(iter(dataloader))
        print("Successfully fetched one batch!")

        # 5. Print the shapes of the tensors in the batch
        print("\n--- Batch Content ---")
        for key, value in first_batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  - Key: '{key}', Shape: {value.shape}, DType: {value.dtype}")
        print("---------------------\n")
        print("Tier 1 Data Pipeline test successful!")

    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\nAn error occurred during the test:\n{e}")
        print("\nPlease ensure paths in your config file are correct and the dataset is valid.")
    finally:
        # Clean up the dummy config file
        if os.path.exists("dummy_config.yaml"):
            os.remove("dummy_config.yaml")