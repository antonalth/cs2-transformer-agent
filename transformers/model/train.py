#!/usr/bin/env python3
"""
Copyright 2025 Anton Althoff

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
------------------------------------------------------------------------
"""

import os
import argparse
import logging
import math
import time
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
# Use standard torch.amp for compatibility
from torch.amp import GradScaler, autocast

import wandb

# --- Local Imports ---
from dataset import DatasetConfig, DatasetRoot, TrainingSample, GroundTruth
from model import CS2Config, CS2BehaviorModel
from model_loss import CS2Loss
import debug

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Trainer")

# ==============================================================================
# 1. Configuration
# ==============================================================================

@dataclass
class TrainConfig:
    # Experiment
    project_name: str = "cs2-behavior-cloning"
    run_name: str = "llama-dinov3-baseline"
    output_dir: str = "./checkpoints"
    
    # Data
    data_root: str = "./cs2_dataset"
    num_workers: int = 4
    
    # Optimization
    batch_size: int = 1          # Per GPU (Video memory is tight!)
    grad_accumulation_steps: int = 16 # Effective batch size = 32
    max_epochs: int = 20
    lr: float = 2e-4             
    weight_decay: float = 0.05
    warmup_steps: int = 2000
    clip_grad_norm: float = 1.0
    
    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_every: int = 1
    mixed_precision: str = "bf16" # "bf16" (Ampere+) or "fp16"

# ==============================================================================
# 2. Helpers
# ==============================================================================

def recursive_to_device(obj: Any, device: torch.device, non_blocking: bool = True):
    """
    Recursively moves Tensors inside nested Dataclasses, Dicts, or Lists to the GPU.
    Skips fields starting with '_' to prevent infinite recursion on metadata.
    """
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    elif is_dataclass(obj):
        changes = {}
        for f in fields(obj):
            val = getattr(obj, f.name)
            # CRITICAL FIX: Skip metadata fields like _roundsample to avoid infinite recursion
            if f.name.startswith("_"):
                changes[f.name] = val
            else:
                changes[f.name] = recursive_to_device(val, device, non_blocking)
        return type(obj)(**changes)
    elif isinstance(obj, dict):
        return {k: recursive_to_device(v, device, non_blocking) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_to_device(v, device, non_blocking) for v in obj]
    return obj

def cs2_collate_fn(batch: List[TrainingSample]) -> TrainingSample:
    """
    Custom collate to stack TrainingSample fields and fix dimensions.
    """
    # 1. Images: Dataset gives [T, P, H, W, C] (if from dataset.py logic) or [T, P, C, H, W]
    # Assuming dataset output is [T, P, H, W, C], we reshape to [B, T, P, C, H, W]
    imgs = torch.stack([s.images for s in batch]) # [B, T, P, H, W, C]
    imgs = imgs.permute(0, 1, 2, 5, 3, 4).contiguous() # [B, T, P, C, H, W]
    
    # 2. Audio: Dataset gives [T, P, 2, 128, 1]
    # Model expects [B, T, P, 2, 128, 32]
    audio_raw = torch.stack([s.audio for s in batch]) 
    
    # FIX: Use expand() instead of F.pad() because F.pad 'replicate' mode doesn't support 6D tensors.
    # Since the last dimension is 1, expanding it is mathematically equivalent to replication.
    target_shape = list(audio_raw.shape)
    target_shape[-1] = 32
    audio_padded = audio_raw.expand(*target_shape).contiguous()
    
    # 3. Ground Truth: Recursively stack tensors
    first_gt = batch[0].truth
    gt_fields = {}
    for f in fields(first_gt):
        gt_fields[f.name] = torch.stack([getattr(s.truth, f.name) for s in batch])
    batched_truth = GroundTruth(**gt_fields)

    # 4. Return batched sample
    return TrainingSample(
        _roundsample=batch[0]._roundsample, 
        images=imgs,
        audio=audio_padded,
        truth=batched_truth
    )

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ==============================================================================
# 3. Training Loop
# ==============================================================================

def train_one_epoch(
    model: nn.Module, 
    criterion: nn.Module, 
    loader: DataLoader, 
    optimizer: optim.Optimizer, 
    scheduler: Any, 
    scaler: GradScaler, 
    cfg: TrainConfig, 
    epoch: int
):
    model.train()
    criterion.train() 
    
    total_loss = 0.0
    steps_in_epoch = len(loader)
    
    t0 = time.time()
    
    for batch_idx, sample in enumerate(loader):
        # 1. Move Data to GPU
        sample = recursive_to_device(sample, cfg.device)
        
        # 2. Forward Pass
        dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16
        try: 
            with autocast(device_type="cuda", dtype=dtype):
                preds = model(
                    images=sample.images, 
                    audio=sample.audio
                )
                
                # Loss Calculation
                loss, metrics = criterion(preds, sample.truth)
                loss = loss / cfg.grad_accumulation_steps

            # 3. Backward
            scaler.scale(loss).backward()

            # 4. Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
                
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                
                # Logging
                metrics["train/loss"] = loss.item() * cfg.grad_accumulation_steps
                metrics["train/lr"] = optimizer.param_groups[0]["lr"]
                metrics["train/step"] = (epoch * steps_in_epoch) + batch_idx
                
                wandb.log(metrics)
            
            total_loss += loss.item() * cfg.grad_accumulation_steps
            
            if batch_idx % 50 == 0:
                elapsed = time.time() - t0
                t0 = time.time()
                logger.info(
                    f"Ep {epoch} [{batch_idx}/{steps_in_epoch}] "
                    f"Loss: {loss.item()*cfg.grad_accumulation_steps:.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                    f"Time/50: {elapsed:.1f}s"
                )

        except torch.cuda.OutOfMemoryError:
            logger.error("OOM Detected! Saving snapshot...")
            debug.save_snapshot(f"oom_epoch{epoch}_batch{batch_idx}.pickle")
            raise

    return total_loss / steps_in_epoch

def validate(model, criterion, loader, cfg, epoch):
    model.eval()
    criterion.eval()
    
    total_loss = 0.0
    agg_metrics = {}
    count = 0
    
    logger.info("Starting Validation...")
    
    with torch.no_grad():
        for sample in loader:
            sample = recursive_to_device(sample, cfg.device)
            dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16
            
            with autocast(device_type="cuda", dtype=dtype):
                preds = model(sample.images, sample.audio)
                loss, metrics = criterion(preds, sample.truth)
            
            total_loss += loss.item()
            
            for k, v in metrics.items():
                agg_metrics[k] = agg_metrics.get(k, 0.0) + v
            count += 1
            
    avg_loss = total_loss / count
    avg_metrics = {f"val/{k}": v / count for k, v in agg_metrics.items()}
    avg_metrics["val/loss"] = avg_loss
    
    logger.info(f"Validation Complete. Avg Loss: {avg_loss:.4f}")
    wandb.log(avg_metrics)
    
    return avg_loss

# ==============================================================================
# 4. Main Driver
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="cs2_run")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--debug", action="store_true", help="Enable memory profiling")
    args = parser.parse_args()

    # Enable debug tools
    if args.debug:
        debug.enable()

    # 1. Setup Config
    cfg = TrainConfig(
        data_root=args.data_root,
        run_name=args.run_name,
        batch_size=args.batch_size,
        grad_accumulation_steps=args.grad_accum
    )
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    wandb.init(project=cfg.project_name, name=cfg.run_name, config=cfg.__dict__)

    # 2. Setup Data
    logger.info("Initializing Dataset...")
    ds_config = DatasetConfig(data_root=cfg.data_root, run_dir="./runs")
    ds_root = DatasetRoot(ds_config)
    
    # 3. Setup Model & Loss
    logger.info("Initializing Model & Loss...")
    # NOTE: Set audio_time_steps to 32 to match our padded collate function
    model_cfg = CS2Config(audio_time_steps=32)
    model = CS2BehaviorModel(model_cfg).to(cfg.device)
    criterion = CS2Loss().to(cfg.device)

    # 4. Setup Optimizer
    model_params = [p for p in model.parameters() if p.requires_grad]
    loss_params = [p for p in criterion.parameters() if p.requires_grad]
    
    optimizer = optim.AdamW(
        model_params + loss_params, 
        lr=cfg.lr, 
        weight_decay=cfg.weight_decay
    )
    
    # 5. Scaler & Scheduler
    scaler = GradScaler(enabled=(cfg.mixed_precision == "fp16"))
    
    dummy_epoch = ds_root.build_epoch("train", 0)
    steps_per_epoch = len(dummy_epoch) // cfg.batch_size
    total_steps = (steps_per_epoch // cfg.grad_accumulation_steps) * cfg.max_epochs
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, total_steps)

    logger.info(f"Ready to train for {cfg.max_epochs} epochs (~{total_steps} updates).")

    # 6. Loop
    for epoch in range(cfg.max_epochs):
        train_ds = ds_root.build_epoch("train", epoch)
        val_ds = ds_root.build_epoch("val", epoch)
        
        train_loader = DataLoader(
            train_ds, 
            batch_size=cfg.batch_size, 
            shuffle=True, 
            num_workers=cfg.num_workers,
            collate_fn=cs2_collate_fn, # Use custom collate
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_ds, 
            batch_size=cfg.batch_size, 
            shuffle=False, 
            num_workers=cfg.num_workers,
            collate_fn=cs2_collate_fn
        )
        
        train_loss = train_one_epoch(
            model, criterion, train_loader, optimizer, scheduler, scaler, cfg, epoch
        )
        
        val_loss = validate(model, criterion, val_loader, cfg, epoch)
        
        if (epoch + 1) % cfg.save_every == 0:
            path = os.path.join(cfg.output_dir, f"{cfg.run_name}_ep{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss_state_dict': criterion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            logger.info(f"Saved checkpoint to {path}")

    wandb.finish()

if __name__ == "__main__":
    main()