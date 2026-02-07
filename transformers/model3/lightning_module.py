import math
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from dataclasses import fields, is_dataclass
from typing import Any, Dict

from config import GlobalConfig
from model import GamePredictorBackbone, ModelPrediction
from model_loss import ModelLoss
from dataset import TrainingSample, GroundTruth

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

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.01):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class CS2PredictorModule(pl.LightningModule):
    def __init__(self, global_cfg: GlobalConfig):
        super().__init__()
        self.save_hyperparameters(ignore=["global_cfg"]) # Log config manually if needed, or rely on PL
        self.global_cfg = global_cfg
        self.model = GamePredictorBackbone(global_cfg.model)
        self.criterion = ModelLoss(global_cfg)
        
        # We need to manually handle dtype casting for Truth if mixed precision is used
        # PL handles model params and inputs mostly, but our GroundTruth structure might need help
        # Map config string to torch dtype
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}
        self.target_dtype = dtype_map.get(global_cfg.train.mixed_precision, torch.float32)

    def forward(self, images, audio):
        return self.model(images, audio)

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        # Override to handle dataclasses
        return recursive_to_device(batch, device)

    def training_step(self, batch: TrainingSample, batch_idx: int):
        # Cast Truth to target dtype (BF16/FP16) explicitly
        # This was done manually in train_fsdp.py
        batch.truth = recursive_apply_to_floats(batch.truth, lambda t: t.to(dtype=self.target_dtype))

        preds_dict = self(batch.images, batch.audio)
        preds = ModelPrediction(**preds_dict)
        
        loss_dict = self.criterion(preds, batch.truth)
        loss = loss_dict["total"]
        
        # Logging
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in loss_dict.items():
            if k != "total":
                self.log(f"train/loss_{k}", v, on_step=True, on_epoch=False, sync_dist=True)

        return loss

    def validation_step(self, batch: TrainingSample, batch_idx: int):
        batch.truth = recursive_apply_to_floats(batch.truth, lambda t: t.to(dtype=self.target_dtype))

        preds_dict = self(batch.images, batch.audio)
        preds = ModelPrediction(**preds_dict)
        
        loss_dict = self.criterion(preds, batch.truth)
        loss = loss_dict["total"]
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in loss_dict.items():
            if k != "total":
                self.log(f"val/loss_{k}", v, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        # Weight decay splitting
        param_dict = {pn: p for pn, p in self.model.named_parameters() if p.requires_grad}
        decay_params = []
        nodecay_params = []
        
        for n, p in param_dict.items():
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.global_cfg.train.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        optimizer = optim.AdamW(optim_groups, lr=self.global_cfg.train.lr)
        
        # LR Scheduler
        # Calculate total steps accurately using PL's estimated_stepping_batches
        total_steps = self.trainer.estimated_stepping_batches
        
        # Warmup
        warmup_steps = self.global_cfg.train.warmup_steps
        # Safety check: if total_steps is somehow smaller than warmup (e.g. debugging), adjust
        if total_steps < warmup_steps:
             warmup_steps = int(0.1 * total_steps)

        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
