#!/usr/bin/env python3
import sys
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

# Add the model directory to sys.path so we can import dataset and config
sys.path.append(os.path.join(os.path.dirname(__file__), "../transformers/model"))

from dataset import DatasetRoot, DatasetConfig, Epoch, RoundSample, TICKS_PER_FRAME
from config import ModelConfig

def get_mouse_deltas(dataset_root, num_samples=1000, time_limit=280):
    """
    Collects mouse deltas from the dataset without decoding video/audio.
    Stops if num_samples is reached OR time_limit (seconds) is exceeded.
    """
    print(f"Collecting mouse deltas from up to {num_samples} samples (Time limit: {time_limit}s)...")
    
    # We create a dummy epoch to access helper methods like _get_lmdb_env
    dummy_epoch = Epoch(dataset_root.config, 0, [])
    
    all_deltas_x = []
    all_deltas_y = []
    
    # Iterate through games and rounds to get samples
    samples_collected = 0
    start_time = time.time()
    
    # Randomly sample from available rounds to avoid bias
    all_rounds = []
    for game in dataset_root.train:
        for r in game.rounds:
            all_rounds.append(r)
            
    import random
    random.shuffle(all_rounds)
    
    for r in all_rounds:
        if samples_collected >= num_samples:
            break
        
        if time.time() - start_time > time_limit:
            print(f"Time limit reached ({time_limit}s). Stopping collection.")
            break
            
        # Create a sample from the middle of the round
        if r.frame_count < 200:
            continue
            
        # Take a few windows from this round
        for _ in range(5): 
            start_f = random.randint(0, max(0, r.frame_count - dataset_root.config.epoch_round_sample_length))
            sample = RoundSample(
                round=r,
                start_tick=r.start_tick + TICKS_PER_FRAME * start_f,
                start_frame=start_f,
                length_frames=min(dataset_root.config.epoch_round_sample_length, r.frame_count)
            )
            
            # Extract truth only (bypass video decoding)
            try:
                gt = dummy_epoch._get_truth(sample)
                
                # gt.mouse_delta is [T, 5, 2]
                # Filter for alive players only
                alive_mask = gt.alive_mask # [T, 5]
                
                md = gt.mouse_delta[alive_mask] # [N, 2]
                
                if md.shape[0] > 0:
                    all_deltas_x.append(md[:, 0].numpy())
                    all_deltas_y.append(md[:, 1].numpy())
                    
                samples_collected += 1
                
                if samples_collected % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = samples_collected / elapsed
                    print(f"  Collected {samples_collected}/{num_samples} samples... ({elapsed:.1f}s, {rate:.1f} samp/s)")

                if samples_collected >= num_samples:
                    break
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue

    if not all_deltas_x:
        print("No data collected.")
        return np.array([]), np.array([])

    print(f"Collection finished. Total samples: {samples_collected}")
    return np.concatenate(all_deltas_x), np.concatenate(all_deltas_y)

def mu_law_encode(x, mu=255.0, max_val=1.0):
    x_norm = np.clip(x / max_val, -1.0, 1.0)
    return np.sign(x_norm) * np.log(1.0 + mu * np.abs(x_norm)) / np.log(1.0 + mu)

def inverse_mu_law(y, mu=255.0, max_val=1.0):
    return np.sign(y) * (1.0 / mu) * (np.power(1.0 + mu, np.abs(y)) - 1.0) * max_val

def analyze_distribution(data, name="Mouse X"):
    print(f"\n--- Analysis for {name} ---")
    print(f"Count: {len(data)}")
    print(f"Min: {np.min(data):.4f}, Max: {np.max(data):.4f}")
    print(f"Mean: {np.mean(data):.4f}, Std: {np.std(data):.4f}")
    
    # Percentiles
    p_abs = np.percentile(np.abs(data), [50, 90, 95, 99, 99.9])
    print(f"Abs Percentiles: 50%: {p_abs[0]:.4f}, 90%: {p_abs[1]:.4f}, 95%: {p_abs[2]:.4f}, 99%: {p_abs[3]:.4f}, 99.9%: {p_abs[4]:.4f}")
    
    # 95% Coverage (symmetric)
    limit_95 = p_abs[2]
    print(f"95% of movements are within [-{limit_95:.4f}, {limit_95:.4f}]")
    
    return limit_95

def evaluate_bins(data, bins, title):
    data_tensor = torch.from_numpy(data).float().unsqueeze(1) # [N, 1]
    bins_tensor = torch.from_numpy(bins).float().unsqueeze(0) # [1, B]
    
    # distances
    dists = torch.abs(data_tensor - bins_tensor)
    min_dists, _ = torch.min(dists, dim=1)
    msqe = torch.mean(min_dists**2).item()
    print(f"[{title}] MSQE: {msqe:.6f}")
    return msqe

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="dataset0")
    parser.add_argument("--num_samples", type=int, default=10000, help="Target number of windows")
    parser.add_argument("--time_limit", type=int, default=240, help="Time limit in seconds")
    args = parser.parse_args()

    # Setup Dataset
    cfg = DatasetConfig(data_root=args.data_root, run_dir="tmp_run")
    ds_root = DatasetRoot(cfg)
    
    if not ds_root.train:
        print("No training games found in manifest.")
        return

    # 1. Get Data
    dx, dy = get_mouse_deltas(ds_root, num_samples=args.num_samples, time_limit=args.time_limit)
    
    if len(dx) == 0:
        return

    data = np.concatenate([dx, dy])
    
    # 2. Analyze Coverage
    limit_95 = analyze_distribution(data, "Combined Mouse Delta")
    
    # 3. Setup Mu-Law Bins
    p_99_9 = np.percentile(np.abs(data), 99.9)
    print(f"99.9% percentile (Max Range candidate): {p_99_9:.4f}")
    
    n_bins = 32
    print(f"\n--- Evaluating Binning Strategies (N={n_bins}) ---")
    
    # Strategy A: Linear Bins
    lin_bins = np.linspace(-limit_95, limit_95, n_bins)
    evaluate_bins(data[np.abs(data) <= limit_95], lin_bins, "Linear 95%")
    
    # Strategy B: Quantiles
    quantiles = np.percentile(data, np.linspace(0, 100, n_bins))
    evaluate_bins(data, quantiles, "Empirical Quantiles")
    
    # Strategy C: Mu-Law Grid Search
    best_msqe = float('inf')
    best_params = None
    
    # We want max_val to be at least limit_95, but maybe larger to capture tails.
    # Mu controls the non-linearity. Higher mu -> more bins near 0.
    test_max_vals = [limit_95, p_99_9, p_99_9 * 1.5, 30.0, 90.0]
    test_mus = [10.0, 50.0, 100.0, 255.0, 500.0, 1000.0]
    
    print("\nMu-Law Grid Search:")
    for mv in test_max_vals:
        for mu in test_mus:
            t = np.linspace(-1, 1, n_bins)
            centers_norm = np.sign(t) * (1.0 / mu) * (np.power(1.0 + mu, np.abs(t)) - 1.0)
            centers = centers_norm * mv
            
            # Evaluate on ALL data (including tails)
            msqe = evaluate_bins(data, centers, f"Mu-Law (Max={mv:.2f}, Mu={mu})")
            if msqe < best_msqe:
                best_msqe = msqe
                best_params = (mv, mu, centers)

    print(f"\nBest Mu-Law Params: Max={best_params[0]:.4f}, Mu={best_params[1]}")
    
    # 4. Compare
    print("\n--- Comparison ---")
    print("Quantile Bins (Ideal):")
    print(np.array2string(quantiles, precision=4, suppress_small=True))
    print("Best Mu-Law Bins:")
    print(np.array2string(best_params[2], precision=4, suppress_small=True))
    
    print("\n--- Recommendation ---")
    print(f"Use Mu-Law with mu={best_params[1]} and max_val={best_params[0]:.2f}")
    
    bins_in_95 = np.sum(np.abs(best_params[2]) <= limit_95)
    print(f"Bins within 95% range ({limit_95:.4f}): {bins_in_95}/{n_bins}")

if __name__ == "__main__":
    main()