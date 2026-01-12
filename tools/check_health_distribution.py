import sys
import os
import random
import torch

# Add model dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../transformers/model')))

from dataset import DatasetConfig, DatasetRoot

def check_health_dist():
    data_root = "dataset0"
    if not os.path.exists(data_root):
        print(f"Error: {data_root} does not exist.")
        return

    print(f"Loading data from {data_root}...")
    ds_config = DatasetConfig(data_root=data_root, run_dir="./runs")
    ds_root = DatasetRoot(ds_config)
    
    try:
        ds = ds_root.build_epoch("train", 0)
        print(f"Dataset Size: {len(ds)} samples")
    except Exception as e:
        print(f"Failed to build dataset: {e}")
        return

    print("Analyzing health distribution over 100 samples (skipping video/audio decode)...")
    
    # Randomly select samples
    indices = random.sample(range(len(ds)), min(100, len(ds)))
    
    total_alive_ticks = 0
    full_hp_ticks = 0
    damaged_ticks = 0
    dead_ticks = 0 
    
    armor_full = 0
    armor_damaged = 0
    armor_none = 0
    
    for i, idx in enumerate(indices):
        if i % 20 == 0: print(f"Processing sample {i}...")
        
        # Bypass __getitem__ to avoid video decoding
        sample = ds.samples[idx]
        truth = ds._get_truth(sample)
        
        alive = truth.alive_mask
        hp = truth.stats[..., 0]
        armor = truth.stats[..., 1]
        
        valid_hp = hp[alive]
        valid_armor = armor[alive]
        
        total_alive_ticks += valid_hp.numel()
        
        full_hp_ticks += (valid_hp == 100).sum().item()
        damaged_ticks += ((valid_hp < 100) & (valid_hp > 0)).sum().item()
        dead_ticks += (valid_hp == 0).sum().item() 
        
        armor_full += (valid_armor == 100).sum().item()
        armor_none += (valid_armor == 0).sum().item()
        armor_damaged += ((valid_armor > 0) & (valid_armor < 100)).sum().item()

    if total_alive_ticks == 0:
        print("No alive ticks found.")
        return

    print("\n--- Health Distribution (Alive Players) ---")
    print(f"Total Ticks: {total_alive_ticks}")
    print(f"Full HP (100):  {full_hp_ticks:8d} ({full_hp_ticks/total_alive_ticks*100:.2f}%)")
    print(f"Damaged (<100): {damaged_ticks:8d} ({damaged_ticks/total_alive_ticks*100:.2f}%)")
    print(f"Zero HP (0):    {dead_ticks:8d} ({dead_ticks/total_alive_ticks*100:.2f}%)")
    
    print("\n--- Armor Distribution (Alive Players) ---")
    print(f"Full Armor (100): {armor_full:8d} ({armor_full/total_alive_ticks*100:.2f}%)")
    print(f"No Armor (0):     {armor_none:8d} ({armor_none/total_alive_ticks*100:.2f}%)")
    print(f"Damaged Armor:    {armor_damaged:8d} ({armor_damaged/total_alive_ticks*100:.2f}%)")

if __name__ == "__main__":
    check_health_dist()