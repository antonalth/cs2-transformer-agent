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
from pathlib import Path

# Third-party libraries
import lmdb
from tqdm import tqdm

# (We will add more imports like torch, wandb, etc. in future steps)


# ===================================================================
# 2. CONFIGURATION DATACLASSES
# ===================================================================
# (To be added in the next step)


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
                
                # Create a cursor for efficient key existence checks
                cursor = txn.cursor()

                for round_entry in info_data.get("rounds", []):
                    round_num, start_tick, end_tick = round_entry
                    
                    # For each round, check for both 'T' and 'CT' perspectives
                    for team in ['T', 'CT']:
                        # To validate existence, we check if the key for the first tick exists.
                        # This is a reliable and fast way to confirm data is present.
                        test_key = f"{demo_name}_round_{round_num:03d}_team_{team}_tick_{start_tick:08d}".encode('utf-8')
                        
                        # cursor.set_key() is faster than txn.get() as it doesn't retrieve the value
                        if cursor.set_key(test_key):
                            # This perspective exists, add it to the pool
                            round_metadata = {
                                "lmdb_path": str(demo_path),
                                "demo_name": demo_name,
                                "round_num": round_num,
                                "team": team,  # <-- The crucial new field
                                "start_tick": start_tick,
                                "end_tick": end_tick
                            }
                            round_pool.append(round_metadata)
                        else:
                            # This perspective is missing, which can happen. Log a warning.
                            print(f"\n[Warning] Missing perspective 'team={team}' for round {round_num} in demo '{demo_name}'. Skipping.")
            
            env.close()

        return round_pool

# ===================================================================
# 4. SCRIPT ENTRYPOINT (for testing the indexer)
# ===================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the DatasetIndexer component of the training script."
    )
    parser.add_argument(
        "--lmdbpath",
        required=True,
        type=str,
        help="Path to the root directory containing the LMDB folders."
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Path to the split_manifest.json file. If not provided, defaults to <lmdbpath>/split_manifest.json"
    )

    args = parser.parse_args()

    manifest_path = args.manifest_path
    if manifest_path is None:
        manifest_path = os.path.join(args.lmdbpath, "split_manifest.json")

    try:
        indexer = DatasetIndexer(
            lmdb_root_path=args.lmdbpath,
            manifest_path=manifest_path
        )
        
        print("\n--- Example items from the generated sampling pools ---")
        if indexer.train_pool:
            print("\nFirst 3 training round-perspectives found:")
            print(json.dumps(indexer.train_pool[:3], indent=2))
        
        if indexer.val_pool:
            print("\nFirst 3 validation round-perspectives found:")
            print(json.dumps(indexer.val_pool[:3], indent=2))

    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\nAn error occurred during indexing:\n{e}")