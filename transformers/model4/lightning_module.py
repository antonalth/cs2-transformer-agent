import math
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from dataclasses import fields, is_dataclass
from typing import Any, Dict

from config import GlobalConfig
from model import GamePredictorBackbone, ModelPrediction
from model_loss import ModelLoss, mu_law_encode
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

    def _teacher_forced_prev_actions(self, truth: GroundTruth) -> Dict[str, torch.Tensor]:
        B, T = truth.keyboard_mask.shape
        device = truth.keyboard_mask.device
        no_buy_idx = self.global_cfg.model.eco_dim

        prev_keyboard_mask = torch.zeros_like(truth.keyboard_mask)
        prev_keyboard_mask[:, 1:] = truth.keyboard_mask[:, :-1]

        prev_mouse_x_bin = torch.zeros(B, T, dtype=torch.long, device=device)
        prev_mouse_y_bin = torch.zeros(B, T, dtype=torch.long, device=device)
        prev_mouse_x_bin[:, 1:] = mu_law_encode(
            truth.mouse_delta[:, :-1, 0],
            self.global_cfg.model.mouse_mu,
            self.global_cfg.model.mouse_max,
            self.global_cfg.model.mouse_bins_count,
        )
        prev_mouse_y_bin[:, 1:] = mu_law_encode(
            truth.mouse_delta[:, :-1, 1],
            self.global_cfg.model.mouse_mu,
            self.global_cfg.model.mouse_max,
            self.global_cfg.model.mouse_bins_count,
        )

        prev_eco_buy_idx = torch.full((B, T), no_buy_idx, dtype=torch.long, device=device)
        prev_did_buy = (truth.eco_mask[:, :-1] != 0).any(dim=-1)
        prev_buy_idx = truth.eco_buy_idx[:, :-1].long().clamp(0, self.global_cfg.model.eco_dim - 1)
        prev_eco_buy_idx[:, 1:] = torch.where(
            prev_did_buy,
            prev_buy_idx,
            torch.full_like(prev_buy_idx, no_buy_idx),
        )

        return {
            "prev_keyboard_mask": prev_keyboard_mask,
            "prev_mouse_x_bin": prev_mouse_x_bin,
            "prev_mouse_y_bin": prev_mouse_y_bin,
            "prev_eco_buy_idx": prev_eco_buy_idx,
        }

    def _sample_prev_action_sos_mask(self, truth: GroundTruth) -> torch.Tensor:
        B, T = truth.keyboard_mask.shape
        p = float(self.global_cfg.train.teacher_forcing_sos_dropout)
        num_windows = int(getattr(self.global_cfg.train, "teacher_forcing_sos_windows", 0))
        window_frac = float(getattr(self.global_cfg.train, "teacher_forcing_sos_window_frac", 0.0))
        if p <= 0.0:
            mask = torch.zeros(B, T, dtype=torch.bool, device=truth.keyboard_mask.device)
        elif p >= 1.0:
            mask = torch.ones(B, T, dtype=torch.bool, device=truth.keyboard_mask.device)
        else:
            mask = torch.rand(B, T, device=truth.keyboard_mask.device) < p

        if num_windows > 0 and window_frac > 0.0 and T > 0:
            window_len = max(1, int(round(T * window_frac)))
            window_len = min(window_len, T)
            max_start = T - window_len
            for b in range(B):
                for _ in range(num_windows):
                    start = 0
                    if max_start > 0:
                        start = int(torch.randint(0, max_start + 1, (1,), device=truth.keyboard_mask.device).item())
                    mask[b, start : start + window_len] = True

        if p <= 0.0 and not (num_windows > 0 and window_frac > 0.0):
            return mask
        if p >= 1.0:
            mask[:, :] = True
        mask[:, 0] = True
        return mask

    def forward(self, images, audio, **kwargs):
        return self.model(images, audio, **kwargs)

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        # Override to handle dataclasses
        return recursive_to_device(batch, device)

    def training_step(self, batch: TrainingSample, batch_idx: int):
        # Cast Truth to target dtype (BF16/FP16) explicitly
        # This was done manually in train_fsdp.py
        batch.truth = recursive_apply_to_floats(batch.truth, lambda t: t.to(dtype=self.target_dtype))

        prev_actions = self._teacher_forced_prev_actions(batch.truth)
        prev_actions["prev_action_sos_mask"] = self._sample_prev_action_sos_mask(batch.truth)
        preds_dict = self(batch.images, batch.audio, **prev_actions)
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

        prev_actions = self._teacher_forced_prev_actions(batch.truth)
        preds_dict = self(batch.images, batch.audio, **prev_actions)
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

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            warmup_steps,
            total_steps,
            min_lr_ratio=self.global_cfg.train.min_lr_ratio,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
