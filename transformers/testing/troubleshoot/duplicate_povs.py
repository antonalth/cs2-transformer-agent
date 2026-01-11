#!/usr/bin/env python3
"""
debug_video_duplicates.py

Scans a dataset directory for potential duplicate video content within the same round/team groups.
Useful for detecting if different player POVs are erroneously recording the same stream.

Method:
    Samples every 32nd frame (0, 32, 64...) up to a max duration.
    Calculates the percentage of pixels that match within a tolerance across sampled frames.
    Uses Multiprocessing to scan groups in parallel.

Usage:
    Bulk Scan (Parallel):
        python debug_video_duplicates.py --data_root ./data --workers 8

    Scan and Rename Duplicates:
        python debug_video_duplicates.py --data_root ./data --rename

    Revert Renamed Files:
        python debug_video_duplicates.py --data_root ./data --revertrename

    Detailed Comparison (Single pair):
        python debug_video_duplicates.py --compare path/to/video1.mp4 path/to/video2.mp4
"""

import argparse
import os
import cv2
import numpy as np
import logging
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

# --- Configuration defaults ---
CHECK_INTERVAL = 32       # Sample every Nth frame
RESIZE_DIM = (64, 64)     # Downscale for faster comparison
SIMILARITY_THRESHOLD = 0.95 # Require 95% aggregate pixel similarity to flag
PIXEL_TOLERANCE = 20      # Allow small pixel value diffs (0-255) due to compression noise

def get_video_signature(video_path, max_duration=None):
    """
    Scans the video, sampling every CHECK_INTERVAL-th frame.
    Stops if max_duration (seconds) is reached.
    Returns a numpy array of shape (N_samples, H, W).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frames = []
    frame_idx = 0
    
    # Calculate max frames based on FPS if duration is set
    max_frames = float('inf')
    if max_duration is not None and max_duration > 0:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            max_frames = int(fps * max_duration)
    
    try:
        while frame_idx < max_frames:
            if frame_idx % CHECK_INTERVAL == 0:
                # Fully decode the frame we want to check
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert to grayscale and resize
                gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), RESIZE_DIM)
                frames.append(gray)
            else:
                # Skip decoding for intermediate frames to speed up processing
                if not cap.grab():
                    break
            
            frame_idx += 1
    except Exception:
        pass # Handle corrupt videos gracefully
    finally:
        cap.release()
        
    if not frames:
        return None
        
    return np.array(frames, dtype=np.uint8)

def calculate_similarity(sig1, sig2):
    """
    Compares two video signatures.
    Returns a float 0.0-1.0 representing similarity.
    """
    if len(sig1) == 0 or len(sig2) == 0:
        return 0.0
        
    # Truncate to the length of the shorter video (compare overlap)
    min_len = min(len(sig1), len(sig2))
    s1 = sig1[:min_len]
    s2 = sig2[:min_len]
    
    # Calculate absolute difference across the 3D volume (Time, H, W)
    diff = cv2.absdiff(s1, s2)
    
    # Count pixels that are 'close enough'
    close_pixels = np.sum(diff < PIXEL_TOLERANCE)
    total_pixels = diff.size
    
    if total_pixels == 0: return 0.0
    
    return close_pixels / total_pixels

def parse_filename_info(filename):
    """
    Extracts round and team info from filename.
    Format: {round}_{team}_{player}_{start}_{end}.mp4
    """
    # Ignore .dup suffix if present for parsing
    clean_name = filename.replace('.dup', '')
    parts = clean_name.split('_')
    if len(parts) < 4:
        return None
    return {
        'round': parts[0],
        'team': parts[1]
    }

# --- Worker Function for Multiprocessing ---
def process_group_task(args):
    """
    Worker function to process a single group of videos.
    Args:
        args: tuple (key, video_paths, max_duration)
    Returns:
        list of detected duplicates: [(key, path1, path2, similarity), ...]
    """
    key, video_paths, max_duration = args
    duplicates = []
    
    if len(video_paths) < 2:
        return duplicates

    # Generate signatures
    signatures = {}
    for vp in video_paths:
        sig = get_video_signature(vp, max_duration=max_duration)
        if sig is not None:
            signatures[vp] = sig

    # Compare pairs
    video_paths_sorted = sorted(video_paths, key=lambda p: p.name)
    
    for i in range(len(video_paths_sorted)):
        for j in range(i + 1, len(video_paths_sorted)):
            p1 = video_paths_sorted[i]
            p2 = video_paths_sorted[j]
            
            if p1 not in signatures or p2 not in signatures:
                continue
            
            sim = calculate_similarity(signatures[p1], signatures[p2])
            
            if sim >= SIMILARITY_THRESHOLD:
                # Convert paths to strings for cleaner return values across processes
                duplicates.append((key, str(p1), str(p2), sim))
                
    return duplicates

def compare_pair_detailed(path1, path2, max_duration=None):
    """
    Detailed comparison mode for two specific files (runs in main process).
    """
    if not os.path.exists(path1) or not os.path.exists(path2):
        print("Error: One or both files do not exist.")
        return

    cap1 = cv2.VideoCapture(str(path1))
    cap2 = cv2.VideoCapture(str(path2))
    
    fps = cap1.get(cv2.CAP_PROP_FPS)
    max_frames = int(fps * max_duration) if (max_duration and fps > 0) else float('inf')

    frames1_acc = []
    frames2_acc = []
    frame_idx = 0
    
    print(f"Comparing Pair (Max Duration: {max_duration if max_duration else 'Full'}):\n  A: {path1}\n  B: {path2}\n")
    print(f"{'Frame':<8} | {'Sim':<8} | {'Status'}")
    print("-" * 30)

    try:
        while frame_idx < max_frames:
            is_check_frame = (frame_idx % CHECK_INTERVAL == 0)
            
            if is_check_frame:
                ret1, f1 = cap1.read()
                ret2, f2 = cap2.read()
            else:
                ret1 = cap1.grab()
                ret2 = cap2.grab()
                f1, f2 = None, None

            if not ret1 or not ret2:
                break
                
            if is_check_frame:
                g1 = cv2.resize(cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY), RESIZE_DIM)
                g2 = cv2.resize(cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY), RESIZE_DIM)
                
                diff = cv2.absdiff(g1, g2)
                sim = np.sum(diff < PIXEL_TOLERANCE) / diff.size
                status = "MATCH" if sim >= SIMILARITY_THRESHOLD else "DIFF"
                
                print(f"{frame_idx:<8} | {sim:.4f}   | {status}")
                frames1_acc.append(g1)
                frames2_acc.append(g2)

            frame_idx += 1
            
    finally:
        cap1.release()
        cap2.release()
        
    if frames1_acc:
        s1 = np.array(frames1_acc)
        s2 = np.array(frames2_acc)
        agg_sim = calculate_similarity(s1, s2)
        print("-" * 30)
        print(f"AGGREGATE SIMILARITY: {agg_sim:.4f}")

def revert_renames(data_root):
    """
    Recursively scans data_root for files ending in .dup and renames them back.
    """
    root = Path(data_root)
    logging.info(f"Scanning {root} for .dup files to revert...")
    
    count = 0
    for dup_file in root.rglob("*.dup"):
        original_name = dup_file.with_suffix('') # Removes .dup
        try:
            dup_file.rename(original_name)
            logging.info(f"Reverted: {dup_file.name} -> {original_name.name}")
            count += 1
        except OSError as e:
            logging.error(f"Failed to revert {dup_file}: {e}")
            
    logging.info(f"Revert complete. Renamed {count} files.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, help="Path to dataset root")
    parser.add_argument("--compare", nargs=2, help="Compare two specific video files")
    parser.add_argument("--max_duration", type=float, default=20.0, help="Max duration in seconds to scan (0 for full)")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers")
    parser.add_argument("--rename", action="store_true", help="Rename discovered duplicates to {name}.dup")
    parser.add_argument("--revertrename", action="store_true", help="Revert any .dup files in data_root to original names")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # --- Mode 0: Revert Renames ---
    if args.revertrename:
        if not args.data_root:
            parser.error("--data_root is required for --revertrename")
        revert_renames(args.data_root)
        return

    # --- Mode 1: Pair Comparison ---
    if args.compare:
        compare_pair_detailed(args.compare[0], args.compare[1], args.max_duration)
        return

    # --- Mode 2: Bulk Scan ---
    if not args.data_root:
        parser.error("--data_root is required unless --compare is used")

    recordings_dir = Path(args.data_root) / "recordings"
    if not recordings_dir.exists():
        logging.error(f"Recordings directory not found: {recordings_dir}")
        return

    # 1. Discovery
    logging.info("Scanning file structure...")
    groups = defaultdict(list)
    total_files = 0
    
    for demo_dir in recordings_dir.iterdir():
        if not demo_dir.is_dir(): continue
        for video_file in demo_dir.glob("*.mp4"):
            # If we are scanning a folder with already renamed duplicates, ignore them for the group logic
            # or treat them as valid inputs (depends on workflow). 
            # Here we skip .dup files to prevent double processing if re-run.
            if video_file.suffix == '.dup':
                continue
                
            info = parse_filename_info(video_file.name)
            if info:
                key = (demo_dir.name, info['round'], info['team'])
                groups[key].append(video_file)
                total_files += 1

    logging.info(f"Found {total_files} videos across {len(groups)} groups.")
    logging.info(f"Scanning first {args.max_duration}s of videos with {args.workers} workers...")

    # 2. Preparation for Multiprocessing
    tasks = []
    for key, paths in groups.items():
        if len(paths) >= 2:
            tasks.append((key, paths, args.max_duration))

    total_duplicates = 0
    duplicate_groups_count = 0
    
    # Store unique paths to rename after scanning
    paths_to_rename = set()
    
    # 3. Execution
    with mp.Pool(processes=args.workers) as pool:
        with tqdm(total=len(tasks), desc="Processing") as pbar:
            for result in pool.imap_unordered(process_group_task, tasks):
                pbar.update(1)
                
                if result:
                    duplicate_groups_count += 1
                    key = result[0][0] 
                    tqdm.write(f"\n[DUPLICATE DETECTED] Demo: {key[0]} | Round: {key[1]} | Team: {key[2]}")
                    
                    for _, p1, p2, sim in result:
                        tqdm.write(f"  Match ({sim*100:.1f}%):")
                        tqdm.write(f"    A: {os.path.basename(p1)}")
                        tqdm.write(f"    B: {os.path.basename(p2)}")
                        total_duplicates += 1
                        
                        if args.rename:
                            paths_to_rename.add(p1)
                            paths_to_rename.add(p2)

    print("\n" + "="*50)
    print(f"SCAN COMPLETE")
    print(f"Total Videos Scanned: {total_files}")
    print(f"Duplicate Pairs Found: {total_duplicates}")
    print(f"Affected Round-Perspectives: {duplicate_groups_count}")
    
    if args.rename and paths_to_rename:
        print("-" * 50)
        print(f"Renaming {len(paths_to_rename)} files to .dup...")
        count = 0
        for p_str in paths_to_rename:
            p = Path(p_str)
            if p.exists():
                new_p = p.with_name(p.name + ".dup")
                try:
                    p.rename(new_p)
                    count += 1
                except OSError as e:
                    logging.error(f"Error renaming {p}: {e}")
        print(f"Successfully renamed {count} files.")
        
    print("="*50)

if __name__ == "__main__":
    main()