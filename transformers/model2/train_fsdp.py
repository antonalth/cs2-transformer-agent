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
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb

# --- Accelerate Imports ---
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import ProjectConfiguration

# --- Local Imports ---
from config import GlobalConfig, DatasetConfig, ModelConfig, TrainConfig
from dataset import DatasetRoot, TrainingSample, GroundTruth
from model import GamePredictorBackbone, ModelPrediction
from model_loss import ModelLoss
import debug

# Initialize Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainerFSDP")

def recursive_to_device(obj: Any, device: torch.device, non_blocking: bool = True):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    elif is_dataclass(obj):
        changes = {}
        for f in fields(obj):
            val = getattr(obj, f.name)
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

def recursive_apply_to_floats(obj: Any, op):
    """Recursively apply an operation to all Float/Double tensors."""
    if torch.is_tensor(obj):
        if obj.is_floating_point():
            return op(obj)
        return obj
    elif is_dataclass(obj):
        changes = {}
        for f in fields(obj):
            val = getattr(obj, f.name)
            changes[f.name] = recursive_apply_to_floats(val, op)
        return type(obj)(**changes)
    elif isinstance(obj, dict):
        return {k: recursive_apply_to_floats(v, op) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_apply_to_floats(v, op) for v in obj]
    return obj

def cs2_collate_fn(batch: List[TrainingSample]) -> TrainingSample:
    imgs = torch.stack([s.images for s in batch])
    audio_raw = torch.stack([s.audio for s in batch])
    
    first_gt = batch[0].truth
    gt_fields = {}
    for f in fields(first_gt):
        gt_fields[f.name] = torch.stack([getattr(s.truth, f.name) for s in batch])
    batched_truth = GroundTruth(**gt_fields)

    return TrainingSample(
        _roundsample=batch[0]._roundsample,
        images=imgs,
        audio=audio_raw,
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
# 4. Training Loop
# ==============================================================================

def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader, 
    optimizer: optim.Optimizer, 
    scheduler: Any, 
    accelerator: Accelerator,
    cfg: TrainConfig, 
    epoch: int,
    global_step: int = 0
):
    model.train()
    criterion.train()
    
    total_loss = 0.0
    steps_in_epoch = len(loader)
    t0 = time.time()
    
    # Determine target dtype for loss inputs to match FSDP params
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}
    target_dtype = dtype_map.get(cfg.mixed_precision, torch.float32)

    for batch_idx, sample in enumerate(loader):
        sample = recursive_to_device(sample, accelerator.device)
        
        # Cast Truth to target dtype (BF16) if mixed precision is enabled
        sample.truth = recursive_apply_to_floats(sample.truth, lambda t: t.to(dtype=target_dtype))

        # Accumulate gradients on the single wrapped module
        with accelerator.accumulate(model):
            
            preds_dict = model(sample.images, sample.audio)
            preds = ModelPrediction(**preds_dict)
            
            # Loss now returns a dictionary
            loss_dict = criterion(preds, sample.truth)
            loss = loss_dict["total"]
            
            # Manual normalization
            loss = loss / cfg.grad_accumulation_steps

            # Backward
            accelerator.backward(loss)

            # Optimizer Step
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                if accelerator.is_main_process:
                    metrics = {"train/loss": loss.item() * cfg.grad_accumulation_steps}
                    # Add component losses
                    for k, v in loss_dict.items():
                        if k != "total":
                            metrics[f"train/loss_{k}"] = v.item() if torch.is_tensor(v) else v
                            
                    metrics["train/lr"] = optimizer.param_groups[0]["lr"]
                    metrics["train/epoch"] = epoch + (batch_idx / steps_in_epoch)
                    # Global samples = batch_idx * processes * batch_size
                    metrics["train/global_samples"] = (epoch * steps_in_epoch * accelerator.num_processes * cfg.batch_size) + (batch_idx * accelerator.num_processes * cfg.batch_size)
                    
                    # Log to wandb using the optimizer step as the X-axis
                    wandb.log(metrics, step=global_step)
            
            total_loss += loss.item() * cfg.grad_accumulation_steps
            
            if batch_idx % 50 == 0 and accelerator.is_main_process:
                elapsed = time.time() - t0
                t0 = time.time()
                logger.info(
                    f"Ep {epoch} [{batch_idx}/{steps_in_epoch}] "
                    f"Step: {global_step} | "
                    f"Loss: {loss.item()*cfg.grad_accumulation_steps:.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                    f"Time/50: {elapsed:.1f}s"
                )

    return total_loss / steps_in_epoch, global_step

def validate(model, criterion, loader, accelerator, cfg, epoch):
    model.eval()
    criterion.eval()

    total_loss = 0.0
    agg_metrics = {}
    count = 0
    
    # Determine target dtype
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}
    target_dtype = dtype_map.get(cfg.mixed_precision, torch.float32)

    if accelerator.is_main_process:
        logger.info("Starting Validation...")
    
    with torch.no_grad():
        for sample in loader:
            sample = recursive_to_device(sample, accelerator.device)
            # Ensure consistency in validation too
            sample.truth = recursive_apply_to_floats(sample.truth, lambda t: t.to(dtype=target_dtype))
            
            preds_dict = model(sample.images, sample.audio)
            preds = ModelPrediction(**preds_dict)
            
            loss_dict = criterion(preds, sample.truth)
            loss = loss_dict["total"]
            
            # Gather loss from all ranks
            avg_loss_across_gpus = accelerator.gather(loss).mean()
            total_loss += avg_loss_across_gpus.item()
            
            # Aggregate component losses
            for k, v in loss_dict.items():
                val = v.item() if torch.is_tensor(v) else v
                # We can't easily gather dicts of tensors without overhead, 
                # just logging local rank 0 approximation or simple mean if strictness needed.
                # For validation, let's just track the total loss strictly and maybe components loosely.
                agg_metrics[k] = agg_metrics.get(k, 0.0) + val
                
            count += 1
            
    avg_loss = total_loss / max(1, count)
    
    if accelerator.is_main_process:
        avg_metrics = {f"val/{k}": v / count for k, v in agg_metrics.items()}
        avg_metrics["val/loss"] = avg_loss
        logger.info(f"Validation Complete. Avg Loss: {avg_loss:.4f}")
        wandb.log(avg_metrics)
    
    return avg_loss

# ==============================================================================
# 5. Main Driver
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="unnamed-run")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory to resume from")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_fsdp", help="Directory to save checkpoints")
    parser.add_argument("--debug", action="store_true", help="Enable memory profiling")
    args = parser.parse_args()

    if args.debug:
        debug.enable()

    # 1. Config Construction
    d_cfg = DatasetConfig(data_root=args.data_root, run_dir="./runs")
    t_cfg = TrainConfig(
        data_root=args.data_root,
        run_name=args.run_name,
        batch_size=args.batch_size,
        grad_accumulation_steps=args.grad_accum,
        output_dir=args.output_dir
    )
    m_cfg = ModelConfig() 
    
    # Create GlobalConfig
    global_cfg = GlobalConfig(dataset=d_cfg, model=m_cfg, train=t_cfg)
    
    # Save config for reproducibility (on main process)
    # We'll do it after accelerator init to ensure output dir exists or use project dir

    logger.info("Initializing Model & Loss")
    model = GamePredictorBackbone(m_cfg)
    criterion = ModelLoss(global_cfg) # Pass full global config

    fsdp_plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
    )

    accelerator = Accelerator(
        mixed_precision=t_cfg.mixed_precision,
        gradient_accumulation_steps=t_cfg.grad_accumulation_steps,
        log_with="wandb",
        project_config=ProjectConfiguration(
            project_dir=t_cfg.output_dir, 
            logging_dir="./logs"
        ),
        fsdp_plugin=fsdp_plugin
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=t_cfg.project_name, 
            config=global_cfg.__dict__, # Log full config structure if possible, or convert to dict
            init_kwargs={"wandb": {"name": t_cfg.run_name}}
        )
        # Save local config copy
        global_cfg.to_file(os.path.join(t_cfg.output_dir, "config.json"))

    # 3. Data
    if accelerator.is_main_process:
        logger.info("Initializing Dataset...")
    ds_root = DatasetRoot(d_cfg)
    
    criterion.to(accelerator.device)
    # Loss is small params (weights), maybe no FSDP needed, but good practice if it has learnable params (none here really)
    # accelerator.register_for_checkpointing(criterion) 

    optimizer = optim.AdamW(
        model.parameters(),
        lr=t_cfg.lr,
        weight_decay=t_cfg.weight_decay
    )

    # Correct total_steps calculation using a prepared dummy loader
    dummy_ds = ds_root.build_epoch("train", 0)
    dummy_loader = DataLoader(dummy_ds, batch_size=t_cfg.batch_size)
    dummy_loader = accelerator.prepare(dummy_loader)
    
    steps_per_epoch = len(dummy_loader)
    total_steps = (steps_per_epoch // t_cfg.grad_accumulation_steps) * t_cfg.max_epochs

    real_warmup = min(t_cfg.warmup_steps, int(0.1 * total_steps))
    if accelerator.is_main_process:
        logger.info(f"Total Optimizer Steps: {total_steps}. Adjusted Warmup: {real_warmup}")

    scheduler = get_cosine_schedule_with_warmup(optimizer, real_warmup, total_steps)

    model, optimizer, scheduler = accelerator.prepare(
        model, optimizer, scheduler
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_from_checkpoint:
        if accelerator.is_main_process:
            logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        
        # Infer epoch from checkpoint path name if possible (e.g. checkpoint_ep2)
        try:
            folder_name = os.path.basename(os.path.normpath(args.resume_from_checkpoint))
            if "checkpoint_ep" in folder_name:
                ep_str = folder_name.split("checkpoint_ep")[-1]
                start_epoch = int(ep_str) + 1
                if accelerator.is_main_process:
                    logger.info(f" inferred start_epoch = {start_epoch} from folder name")
        except ValueError:
            pass

    if accelerator.is_main_process:
        logger.info(f"Ready to train for {t_cfg.max_epochs} epochs (~{total_steps} updates). Starting at epoch {start_epoch}")

    global_step = 0

    # 7. Loop
    for epoch in range(start_epoch, t_cfg.max_epochs):
        train_ds = ds_root.build_epoch("train", epoch)
        val_ds = ds_root.build_epoch("val", epoch)
        
        train_loader = DataLoader(
            train_ds, 
            batch_size=t_cfg.batch_size, 
            shuffle=True, 
            num_workers=t_cfg.num_workers,
            collate_fn=cs2_collate_fn,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_ds, 
            batch_size=t_cfg.batch_size, 
            shuffle=False, 
            num_workers=t_cfg.num_workers,
            collate_fn=cs2_collate_fn
        )
        
        train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
        
        train_loss, global_step = train_one_epoch(
            model, criterion, train_loader, optimizer, scheduler, accelerator, t_cfg, epoch, global_step
        )
        
        val_loss = validate(model, criterion, val_loader, accelerator, t_cfg, epoch)
        
        if (epoch + 1) % t_cfg.save_every == 0:
            if accelerator.is_main_process:
                logger.info(f"Saving checkpoint for epoch {epoch}...")
            accelerator.save_state(output_dir=os.path.join(t_cfg.output_dir, f"checkpoint_ep{epoch}"))

    accelerator.end_training()

if __name__ == "__main__":
    main()
