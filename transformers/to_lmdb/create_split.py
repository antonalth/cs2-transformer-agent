#!/usr/bin/env python3
"""
split_manifest.py

Scans a data directory for corresponding LMDB and recordings folders,
then deterministically shuffles and splits them into training and testing sets.
The resulting split is saved to a manifest.json file in the data directory.
"""

import argparse
import datetime
import json
import random
import sys
from pathlib import Path

def create_split_manifest(data_dir: Path, seed: int, train_percent: int):
    """
    Finds all games, verifies data integrity, splits them, and writes a manifest.

    Args:
        data_dir (Path): The root data directory containing 'lmdb' and 'recordings'.
        seed (int): The seed for deterministic shuffling.
        train_percent (int): The percentage of data to allocate to the training set.
    """
    print(f"Scanning data directory: {data_dir}")

    # 1. Define and validate paths
    lmdb_dir = data_dir / "lmdb"
    recordings_dir = data_dir / "recordings"
    manifest_path = data_dir / "manifest.json"

    if not lmdb_dir.is_dir():
        print(f"Error: LMDB directory not found at '{lmdb_dir}'", file=sys.stderr)
        sys.exit(1)

    if not recordings_dir.is_dir():
        print(f"Error: Recordings directory not found at '{recordings_dir}'", file=sys.stderr)
        sys.exit(1)

    # 2. Find all potential games based on LMDB folders
    print(f"Searching for LMDB databases in: {lmdb_dir}")
    # A game name is the directory name without the '.lmdb' suffix
    game_names = sorted([p.stem for p in lmdb_dir.iterdir() if p.is_dir() and p.name.endswith('.lmdb')])

    if not game_names:
        print("Warning: No '.lmdb' directories found. Manifest will be empty.", file=sys.stderr)
        all_valid_games = []
    else:
        print(f"Found {len(game_names)} potential games. Verifying recordings folders...")
        all_valid_games = []
        for name in game_names:
            expected_rec_path = recordings_dir / name
            if not expected_rec_path.is_dir():
                print(
                    f"Error: Found '{lmdb_dir / (name + '.lmdb')}' but the corresponding "
                    f"recordings folder is missing at '{expected_rec_path}'. Failing fast.",
                    file=sys.stderr
                )
                sys.exit(1)
            all_valid_games.append(name)
        print("All LMDB folders have a matching recordings folder.")


    # 3. Deterministically shuffle and split the list of games
    random.seed(seed)
    random.shuffle(all_valid_games)

    split_index = int(len(all_valid_games) * (train_percent / 100.0))
    
    train_set = all_valid_games[:split_index]
    test_set = all_valid_games[split_index:]

    print(f"Splitting {len(all_valid_games)} games into {len(train_set)} train and {len(test_set)} test samples.")

    # 4. Create the manifest dictionary
    manifest_data = {
        "metadata": {
            "seed": seed,
            "creation_date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "split_ratio_train": train_percent,
            "split_ratio_test": 100 - train_percent
        },
        "train": train_set,
        "test": test_set
    }

    # 5. Write to manifest.json
    try:
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f, indent=4)
        print(f"\nSuccessfully created manifest file at: {manifest_path}")
    except IOError as e:
        print(f"Error: Could not write to manifest file at '{manifest_path}'.\n{e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point and argument parsing."""
    parser = argparse.ArgumentParser(
        description="Create a train/test split manifest for LMDB/recordings data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "data_dir",
        type=Path,
        help="Path to the root data directory (e.g., 'data/')."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the deterministic shuffle of the dataset."
    )
    
    parser.add_argument(
        "--split",
        type=int,
        default=80,
        choices=range(0, 101),
        metavar="[0-100]",
        help="Percentage of data to be used for the training set."
    )
    
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        print(f"Error: The specified data directory does not exist: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    create_split_manifest(args.data_dir, args.seed, args.split)


if __name__ == '__main__':
    main()