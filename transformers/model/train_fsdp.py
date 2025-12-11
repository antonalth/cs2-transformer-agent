#!/usr/bin/env python3
import os
import argparse
import logging
import math
import time
import functools
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# FSDP Imports
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)

# HuggingFace definitions for wrapping policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

import wandb

# --- Local Imports ---
from dataset import DatasetConfig, DatasetRoot, TrainingSample, GroundTruth, Epoch
from model_novibe import ModelConfig, GamePredictorBackbone
from model_loss import CS2Loss
import debug

# Set up logging only for Rank 0
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Trainer")

def is_main_process():
    return dist.get_rank() == 0

@dataclass
class TrainConfig:
    project_name: str = "cs2-behavior-cloning"
    run_name: str = "llama-dac-fsdp-512"
    output_dir: str = "./checkpoints"
    
    data_root: str = "./cs2_dataset"
    num_workers: int = 4
    
    # Batch size per GPU. 
    # With FSDP on 5090s, you can likely do batch_size=1 with len=512, 
    # or batch_size=2 if you use activation checkpointing aggressively.
    batch_size: int = 1          
    grad_accumulation_steps: int = 4 # Total effective batch = 4 GPUs * 1 Batch * 4 Accum = 16
    
    max_epochs: int = 20
    lr: float = 2e-4             
    weight_decay: float = 0.05
    warmup_steps: int = 1000
    clip_grad_norm: float = 1.0
    
    save_every: int = 1
    
    # Increase sequence length here
    seq_len: int = 512 

# ==============================================================================
# Helpers
# ==============================================================================

def setup_distributed():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    dist.destroy_process_group()

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
# Training Loop
# ==============================================================================

def train_one_epoch(
    model: nn.Module, 
    criterion: nn.Module, 
    loader: DataLoader, 
    optimizer: optim.Optimizer, 
    scheduler: Any, 
    cfg: TrainConfig, 
    epoch: int,
    local_rank: int
):
    model.train()
    # Ensure criterion is in training mode (for AutomaticWeightedLoss)
    criterion.train() 
    
    total_loss = torch.tensor(0.0, device=local_rank)
    steps_in_epoch = len(loader)
    
    if is_main_process():
        logger.info(f"Start Epoch {epoch}")

    for batch_idx, sample in enumerate(loader):
        sample = recursive_to_device(sample, local_rank)
        
        # FSDP automatically handles autocast if mixed_precision is configured
        preds = model(images=sample.images, audio=sample.audio)
        
        loss, metrics = criterion(preds, sample.truth)
        loss = loss / cfg.grad_accumulation_steps
        
        loss.backward()

        if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
            # FSDP handles gradient clipping internally or via utility, 
            # but standard clip_grad_norm_ works on FSDP modules too.
            model.clip_grad_norm_(cfg.clip_grad_norm)
            
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            # Logging (Rank 0 only)
            if is_main_process():
                # Scale loss back up for logging
                current_loss = loss.item() * cfg.grad_accumulation_steps
                metrics["train/loss"] = current_loss
                metrics["train/lr"] = optimizer.param_groups[0]["lr"]
                metrics["train/step"] = (epoch * steps_in_epoch) + batch_idx
                wandb.log(metrics)
        
        total_loss += loss.detach() * cfg.grad_accumulation_steps

    # Aggregate average loss across all ranks for logging
    dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
    return total_loss.item() / steps_in_epoch

def validate(model, criterion, loader, cfg, epoch, local_rank):
    model.eval()
    criterion.eval()
    
    total_loss = torch.tensor(0.0, device=local_rank)
    count = torch.tensor(0.0, device=local_rank)
    
    with torch.no_grad():
        for sample in loader:
            sample = recursive_to_device(sample, local_rank)
            preds = model(sample.images, sample.audio)
            loss, _ = criterion(preds, sample.truth)
            
            total_loss += loss
            count += 1
            
    # Aggregate stats across GPUs
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    
    avg_loss = (total_loss / count).item()
    
    if is_main_process():
        logger.info(f"Validation Ep {epoch} - Avg Loss: {avg_loss:.4f}")
        wandb.log({"val/loss": avg_loss, "epoch": epoch})
    
    return avg_loss

def save_checkpoint(model, optimizer, criterion, epoch, cfg, rank):
    """
    Saves a checkpoint compatible with non-FSDP loading.
    Requires gathering the full state dict to rank 0.
    """
    # Configure FSDP to gather full state dict to CPU
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        model_state = model.state_dict()

    # Criterion might not be FSDP wrapped (it's small), handle normally
    loss_state = criterion.state_dict()
    
    # Optimizer state is complex in FSDP, usually we skip full optimizer saving
    # for simple fine-tuning unless pausing/resuming is critical. 
    # For now, we just save model weights to save disk space and complexity.

    if rank == 0:
        path = os.path.join(cfg.output_dir, f"{cfg.run_name}_ep{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'loss_state_dict': loss_state,
            # 'optimizer_state_dict': ... (omitted for speed/simplicity)
        }, path)
        logger.info(f"Saved FSDP checkpoint to {path}")

# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    local_rank = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 1. Config
    cfg = TrainConfig(
        data_root=args.data_root,
        batch_size=args.batch_size,
        seq_len=512 # Targeted sequence length
    )
    
    if is_main_process():
        os.makedirs(cfg.output_dir, exist_ok=True)
        wandb.init(project=cfg.project_name, name=cfg.run_name, config=cfg.__dict__)
        logger.info(f"World Size: {world_size}")

    # 2. Data
    # DatasetConfig now uses larger sample length
    ds_config = DatasetConfig(
        data_root=cfg.data_root, 
        run_dir="./runs", 
        epoch_round_sample_length=cfg.seq_len
    )
    ds_root = DatasetRoot(ds_config)
    
    # 3. Model Setup
    model_cfg = ModelConfig()
    # Update model config with gradient checkpointing for memory savings
    model_cfg.gradient_checkpointing = True 
    
    # Instantiate on CPU first or meta device, then move to GPU
    # Note: For FSDP with frozen layers, initialization on GPU is usually safer
    model = GamePredictorBackbone(model_cfg).to(local_rank)
    
    # 4. FSDP Wrapping
    # Policy: Wrap LlamaDecoderLayer. This shards the heavy transformer blocks.
    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )

    # Mixed Precision Policy for RTX 5090 (Ampere+ supports bf16)
    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # Wrap model
    # use_orig_params=True is CRITICAL because the model has frozen layers (vision/audio)
    # mixed requires_grad is only supported with use_orig_params=True
    model = FSDP(
        model,
        auto_wrap_policy=llama_auto_wrap_policy,
        mixed_precision=bf16_policy,
        device_id=local_rank,
        sharding_strategy=ShardingStrategy.FULL_SHARD, # Shards params, grads, and optimizer
        use_orig_params=True,
        limit_all_gathers=True,
    )

    # Criterion (Loss) - Small, keeps on local GPU
    criterion = CS2Loss().to(local_rank)

    # 5. Optimizer
    # Pytorch FSDP handles param flattening, pass model.parameters() directly
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=cfg.lr, 
        weight_decay=cfg.weight_decay
    )
    
    # 6. Scheduler
    # We assume a fixed number of samples per epoch for estimation
    # Since we shard the dataset, steps_per_epoch = Total Samples / Global Batch Size
    dummy_epoch = ds_root.build_epoch("train", 0)
    total_samples = len(dummy_epoch.samples)
    global_batch_size = cfg.batch_size * world_size * cfg.grad_accumulation_steps
    total_steps = (total_samples // global_batch_size) * cfg.max_epochs
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, total_steps)

    if is_main_process():
        logger.info(f"Training on {total_samples} samples.")
        logger.info(f"Global Batch Size: {global_batch_size} (Batch {cfg.batch_size} * GPU {world_size} * Accum {cfg.grad_accumulation_steps})")

    # 7. Loop
    for epoch in range(cfg.max_epochs):
        # Build full epoch list on every rank
        full_train_epoch = ds_root.build_epoch("train", epoch)
        full_val_epoch = ds_root.build_epoch("val", epoch)
        
        # --- Manually Shard the Dataset ---
        # Each rank takes a slice: 0, 4, 8... ; 1, 5, 9... etc
        train_samples = full_train_epoch.samples[rank::world_size]
        val_samples = full_val_epoch.samples[rank::world_size]
        
        # Create lightweight Epoch objects with the sliced samples
        train_ds = Epoch(ds_config, epoch, train_samples)
        val_ds = Epoch(ds_config, epoch, val_samples)
        
        train_loader = DataLoader(
            train_ds, 
            batch_size=cfg.batch_size, 
            shuffle=True, # Shuffle local shard
            num_workers=cfg.num_workers,
            collate_fn=cs2_collate_fn,
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
            model, criterion, train_loader, optimizer, scheduler, cfg, epoch, local_rank
        )
        
        val_loss = validate(model, criterion, val_loader, cfg, epoch, local_rank)
        
        if (epoch + 1) % cfg.save_every == 0:
            save_checkpoint(model, optimizer, criterion, epoch, cfg, rank)

    cleanup_distributed()
    if is_main_process():
        wandb.finish()

if __name__ == "__main__":
    main()