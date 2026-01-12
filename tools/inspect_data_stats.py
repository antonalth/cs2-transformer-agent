
import torch
from torch.utils.data import DataLoader
import sys
import os

# Add model dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../transformers/model')))

from dataset import DatasetConfig, DatasetRoot, TrainingSample
from config import TrainConfig
import numpy as np

def inspect_data():
    data_root = "dataset0"
    if not os.path.exists(data_root):
        print(f"Error: {data_root} does not exist.")
        return

    print(f"Loading data from {data_root}...")
    ds_config = DatasetConfig(data_root=data_root, run_dir="./runs")
    ds_root = DatasetRoot(ds_config)
    
    # Try to load epoch 0
    try:
        ds = ds_root.build_epoch("train", 0)
        print(f"Dataset Size: {len(ds)} samples")
    except Exception as e:
        print(f"Failed to build dataset: {e}")
        return

    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=lambda x: x) # Return list of samples

    print("Inspecting first 50 samples...")
    
    eco_ones = 0
    eco_total = 0
    
    inv_ones = 0
    inv_total = 0
    
    kb_ones = 0
    kb_total = 0
    
    round_nums = []
    alive_counts = []
    
    samples_checked = 0
    
    for i, batch in enumerate(loader):
        for sample in batch:
            # sample is TrainingSample
            # sample.truth is GroundTruth
            
            # 1. Eco
            # eco_mask is [T, 5, 4] (chunks of 64 bits) -> 256 bits total
            # We need to unpack or just check non-zero chunks for sparsity estimate, 
            # but exact bit counting requires unpacking. 
            # Let's just check raw chunks for now. If all chunks are 0, then bits are 0.
            
            # Actually, let's look at the raw tensors.
            # truth.eco_mask: [T, 5, 4] int64
            # truth.inventory_mask: [T, 5, 2] int64
            
            # To count bits set in int64, we can do popcount, or just approximate.
            # Since we want to know if it's *empty*, checking == 0 is good enough for now.
            
            # But wait, the loss uses unpack_chunks.
            
            truth = sample.truth
            
            # Eco Bits
            # Just check if any bit is set in the chunks
            eco_nonzero = (truth.eco_mask != 0).sum().item()
            eco_total += truth.eco_mask.numel() * 64 # rough max bits? No, numel is count of int64s. 
            # But we just want to know sparsity.
            
            # Let's count actual bits for a few samples
            # Unpack logic from model_loss
            def unpack(chunks, n):
                bits = torch.arange(64, device=chunks.device)
                unpacked = []
                for i in range(n):
                    expanded = (chunks[..., i].unsqueeze(-1) >> bits) & 1
                    unpacked.append(expanded)
                return torch.cat(unpacked, dim=-1)
    
            eco_bits = unpack(truth.eco_mask, 4)
            eco_ones += eco_bits.sum().item()
            eco_total += eco_bits.numel()
            
            inv_bits = unpack(truth.inventory_mask, 2)
            inv_ones += inv_bits.sum().item()
            inv_total += inv_bits.numel()
            
            # Round Number
            # truth.round_number: [T, 1]
            round_nums.extend(truth.round_number.flatten().tolist())
            
            # Alive Count
            # truth.alive_mask: [T, 5]
            alive_counts.extend(truth.alive_mask.sum(dim=-1).tolist())
            
            
            # Keyboard
            # truth.keyboard_mask is [T, 5] int32 (bitmask)
            kb_bits = unpack(truth.keyboard_mask.unsqueeze(-1), 1) # Treat as 1 chunk
            kb_ones += kb_bits.sum().item()
            kb_total += kb_bits.numel()

            samples_checked += 1
            if samples_checked >= 10:
                break
        
        if samples_checked >= 10:
            break
            
    print("-" * 30)
    print(f"Samples checked: {samples_checked}")
    
    if kb_total > 0:
        print(f"Kb Bit Density: {kb_ones} / {kb_total} ({kb_ones/kb_total:.6f})")
    
    if eco_total > 0:
        print(f"Eco Bit Density: {eco_ones} / {eco_total} ({eco_ones/eco_total:.6f})")
    
    if inv_total > 0:
        print(f"Inv Bit Density: {inv_ones} / {inv_total} ({inv_ones/inv_total:.6f})")
        
    if round_nums:
        print(f"Round Nums: Min={min(round_nums)}, Max={max(round_nums)}, Mean={np.mean(round_nums):.2f}")
        
    if alive_counts:
        print(f"Alive Counts: Mean={np.mean(alive_counts):.2f}")
        unique, counts = np.unique(alive_counts, return_counts=True)
        print(f"Alive Dist: {dict(zip(unique, counts))}")

if __name__ == "__main__":
    inspect_data()
