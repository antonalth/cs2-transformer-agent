#!/usr/bin/env python3
"""
create_split.py

Scans a directory of LMDBs (where each subdirectory is a unique demo)
and creates a deterministic, demo-wise train/validation split.

The output is a 'split_manifest.json' file saved in the root of the
LMDB directory, which serves as the single source of truth for all
training runs to ensure reproducibility.
"""

import json
import random
import argparse
from pathlib import Path
from datetime import datetime, timezone

# The hardcoded seed ensures that the shuffle is always the same,
# making the split reproducible every time the script is run on the
# same dataset.
HARDCODED_SEED = 42

def create_deterministic_split(lmdb_path: str, val_split_ratio: float, seed: int):
    """
    Finds all demo subdirectories, shuffles them deterministically,
    and saves the train/val split to a JSON manifest file.

    Args:
        lmdb_path (str): The path to the root directory containing all LMDB folders.
        val_split_ratio (float): The fraction of demos to reserve for the validation set.
        seed (int): The random seed to use for shuffling.
    """
    print("--- Starting Deterministic Split Creation ---")

    # 1. Validate the input path
    lmdb_root = Path(lmdb_path)
    if not lmdb_root.is_dir():
        raise NotADirectoryError(f"Error: Provided LMDB path is not a valid directory: {lmdb_path}")

    # 2. Get a list of all unique demos by finding all subdirectories
    print(f"Scanning for demo directories in: {lmdb_root.resolve()}")
    demo_names = [d.name for d in lmdb_root.iterdir() if d.is_dir()]

    if not demo_names:
        print(f"Warning: No subdirectories found in {lmdb_path}. No split created.")
        return

    print(f"Found {len(demo_names)} total demos.")

    # 3. CRITICAL: Sort the list to ensure the initial order is always the same
    # before shuffling. This is a key part of ensuring determinism.
    demo_names.sort()

    # 4. CRITICAL: Use the fixed seed to shuffle the list. The shuffle
    # will be identical every time this script is run with the same seed.
    print(f"Shuffling demos with fixed random seed: {seed}")
    random.seed(seed)
    random.shuffle(demo_names)

    # 5. Split the shuffled list into training and validation sets
    split_index = int(len(demo_names) * (1.0 - val_split_ratio))
    train_demos = demo_names[:split_index]
    val_demos = demo_names[split_index:]

    # 6. Prepare the manifest dictionary with useful metadata
    print("Constructing manifest file...")
    split_manifest = {
        "metadata": {
            "description": "Official train/validation split. Each entry is a demo name (LMDB folder).",
            "dataset_source_path": str(lmdb_root.resolve()),
            "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "validation_split_ratio": val_split_ratio
        },
        "split_summary": {
            "num_train_demos": len(train_demos),
            "num_validation_demos": len(val_demos),
            "total_demos": len(demo_names)
        },
        "train_demos": train_demos,
        "validation_demos": val_demos
    }

    # 7. Save the manifest to a JSON file
    manifest_path = lmdb_root / "split_manifest.json"
    try:
        with open(manifest_path, 'w') as f:
            # Use indent=4 for human-readability
            json.dump(split_manifest, f, indent=4)
    except IOError as e:
        print(f"Error: Could not write manifest file to {manifest_path}. Check permissions.")
        print(e)
        return

    print("\n--- Split Creation Successful ---")
    print(f"Manifest file saved to: {manifest_path.resolve()}")
    print(f"  - Training demos:     {len(train_demos)}")
    print(f"  - Validation demos:   {len(val_demos)}")
    print("---------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a deterministic, demo-wise train/validation split for a set of LMDBs.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--lmdbpath",
        required=True,
        type=str,
        help="Path to the root directory containing the LMDB folders (one folder per demo)."
    )

    parser.add_argument(
        "--val_split_ratio",
        type=float,
        default=0.05,
        help="Fraction of the demos to use for the validation set (e.g., 0.05 for 5%%). Default is 0.05."
    )

    args = parser.parse_args()

    # The seed is hardcoded as per the requirements to ensure reproducibility.
    create_deterministic_split(
        lmdb_path=args.lmdbpath,
        val_split_ratio=args.val_split_ratio,
        seed=HARDCODED_SEED
    )