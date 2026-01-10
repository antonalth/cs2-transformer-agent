#!/usr/bin/env python3
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from io import BytesIO

# Import your existing dataset logic
from config import DatasetConfig
from dataset import DatasetRoot, Epoch, TICKS_PER_FRAME

class FastAnalysisEpoch(Epoch):
    """
    A subclass of Epoch that overrides __getitem__ to ONLY load Ground Truth.
    This bypasses the expensive Video and Audio decoding.
    """
    def __getitem__(self, index):
        s = self.samples[index]
        # We ONLY call _get_truth, skipping _decode_video and _decode_audio
        gt = self._get_truth(s)
        return gt

def generate_mu_law_bins(n_bins=256, max_val=500.0, mu=255.0):
    """
    Generates non-linear bin centers using Mu-Law encoding.
    """
    # Create linear space from -1 to 1
    t = np.linspace(-1, 1, n_bins)
    
    # Inverse Mu-Law transformation to get non-linear distribution
    # sgn(x) * (1/mu) * ((1+mu)^|x| - 1)
    val_norm = np.sign(t) * (1.0 / mu) * ((1.0 + mu)**np.abs(t) - 1.0)
    
    # Scale to max value
    bin_centers = val_norm * max_val
    return bin_centers

def analyze_and_visualize(args):
    # 1. Setup Dataset
    print(f"Loading dataset from: {args.data_root}")
    cfg = DatasetConfig(data_root=args.data_root, run_dir="./runs/analysis")
    ds_root = DatasetRoot(cfg)
    
    # Build epoch (using the logic from your dataset.py)
    # We use the base logic to get the list of samples
    base_epoch = ds_root.build_epoch(args.split, 0)
    
    if len(base_epoch.samples) == 0:
        print("No samples found in epoch.")
        return

    # Swap to our Fast Loader
    fast_loader = FastAnalysisEpoch(cfg, 0, base_epoch.samples)
    
    print(f"Analyzing {len(fast_loader)} samples (Frames per sample: {cfg.epoch_round_sample_length})...")
    
    # 2. Iterate and Collect Data
    all_mouse_x = []
    all_mouse_y = []
    
    # Limit samples if requested for speed
    num_to_process = args.num_samples if args.num_samples > 0 else len(fast_loader)
    
    for i in tqdm(range(num_to_process), desc="Scanning LMDB"):
        gt = fast_loader[i]
        
        # gt.mouse_delta is [T, 5, 2]
        # We flatten T and 5, separating X and Y
        mouse = gt.mouse_delta.numpy() # Convert to numpy
        alive = gt.alive_mask.numpy()
        
        # Only take stats from ALIVE players
        valid_mouse = mouse[alive] # Selects where alive is true -> [N, 2]
        
        if valid_mouse.shape[0] > 0:
            all_mouse_x.append(valid_mouse[:, 0])
            all_mouse_y.append(valid_mouse[:, 1])

    if not all_mouse_x:
        print("No valid mouse data found (all players dead?).")
        return

    # Flatten into massive arrays
    data_x = np.concatenate(all_mouse_x)
    data_y = np.concatenate(all_mouse_y)
    
    # 3. Compute Statistics
    def print_stats(name, data):
        print(f"\n--- {name} Statistics ---")
        print(f"Count:    {len(data)}")
        print(f"Min:      {np.min(data):.4f}")
        print(f"Max:      {np.max(data):.4f}")
        print(f"Mean:     {np.mean(data):.4f}")
        print(f"Std Dev:  {np.std(data):.4f}")
        print(f"Variance: {np.var(data):.4f}")
        
        # Percentiles to see "Flicks" vs "Recoil"
        p = np.percentile(data, [1, 5, 25, 50, 75, 95, 99, 99.9])
        print(f"1%ile:   {p[0]:.4f}")
        print(f"5%ile:   {p[1]:.4f}")
        print(f"25%ile:  {p[2]:.4f}")
        print(f"Median:  {p[3]:.4f}")
        print(f"75%ile:  {p[4]:.4f}")
        print(f"95%ile:  {p[5]:.4f}")
        print(f"99%ile:  {p[6]:.4f}")
        print(f"99.9%ile:{p[7]:.4f}")
        return np.max(np.abs(data)) # Return max abs for plotting range

    max_abs_x = print_stats("Mouse X", data_x)
    max_abs_y = print_stats("Mouse Y", data_y)
    
    global_max = max(max_abs_x, max_abs_y)
    
    # 4. Visualize using CV2
    print("\nGenerating visualization...")
    
    # Define Canvas
    W, H = 1200, 600
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 20 # Dark background
    
    # Generate Bins
    mu_val = 255.0
    # Use the 99.9th percentile or the absolute max as the boundary?
    # Usually absolute max is an outlier error, let's use a reasonable clamp like 500 or 1000
    limit_val = 500.0 
    bins = generate_mu_law_bins(n_bins=256, max_val=limit_val, mu=mu_val)
    
    # --- Helper to draw histogram ---
    def draw_hist_on_canvas(data, color, offset_y, height, label):
        # Clip data to visual limit
        clipped_data = np.clip(data, -limit_val, limit_val)
        
        # Calculate histogram
        hist, _ = np.histogram(clipped_data, bins=200, range=(-limit_val, limit_val))
        
        # Log scale for histogram because "Zero" dominates
        hist_log = np.log1p(hist)
        hist_norm = (hist_log / hist_log.max()) * (height - 20)
        
        # Draw base line
        cv2.line(canvas, (50, offset_y + height), (W-50, offset_y + height), (255, 255, 255), 1)
        
        bin_w = (W - 100) / 200
        for i in range(200):
            h = int(hist_norm[i])
            x1 = int(50 + i * bin_w)
            y1 = int(offset_y + height - h)
            y2 = int(offset_y + height)
            cv2.rectangle(canvas, (x1, y1), (x1 + int(bin_w) + 1, y2), color, -1)
            
        cv2.putText(canvas, f"{label} (Log Scale)", (60, offset_y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

    # Draw Data Histograms
    draw_hist_on_canvas(data_x, (100, 200, 255), 50, 200, "Observed Mouse X")
    draw_hist_on_canvas(data_y, (100, 255, 200), 300, 200, "Observed Mouse Y")
    
    # --- Visualize Mu-Law Bins ---
    # Draw vertical lines for where the bins are
    # Map bins from [-limit, limit] to [50, W-50]
    def map_x(val):
        norm = (val + limit_val) / (2 * limit_val) # 0 to 1
        return int(50 + norm * (W - 100))
    
    mid_y = 570
    # Draw axis
    cv2.line(canvas, (50, mid_y), (W-50, mid_y), (255, 255, 255), 1)
    
    # Draw bins
    for b in bins:
        x = map_x(b)
        if 50 <= x <= W-50:
            # Color center bins differently than outer bins
            dist = abs(b) / limit_val
            color_intensity = int(255 * (1.0 - dist))
            # BGR
            col = (0, color_intensity, 255) 
            cv2.line(canvas, (x, mid_y - 10), (x, mid_y + 10), col, 1)

    cv2.putText(canvas, f"Mu-Law Bins (mu={mu_val}, Range=+-{limit_val})", 
                (50, mid_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Show labels for min/max
    cv2.putText(canvas, f"-{limit_val}", (40, mid_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(canvas, f"0", (map_x(0)-5, mid_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(canvas, f"+{limit_val}", (W-90, mid_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # Show Plot
    cv2.imshow("Mouse Distribution Analysis", canvas)
    print("Press any key on the window to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_samples", type=int, default=-1, help="Limit number of samples to scan (-1 for all)")
    args = parser.parse_args()
    
    analyze_and_visualize(args)