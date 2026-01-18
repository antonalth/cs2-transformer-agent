#!/usr/bin/env python3
"""
Sanity-check: can the vision frontend (ViT + Q-Former) predict basic HUD stats?
"""
from __future__ import annotations

import argparse
import logging
import os
import random
from dataclasses import fields, is_dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import cv2
import torchvision.models as tv_models

from config import DatasetConfig, ModelConfig
from dataset import DatasetRoot, GroundTruth, TrainingSample, FRAME_RATE
from model_novibe import GameVideoEncoder
import visualize


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("VisionSanity")


def recursive_to_device(obj: Any, device: torch.device, non_blocking: bool = True):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    if is_dataclass(obj):
        changes = {}
        for f in fields(obj):
            val = getattr(obj, f.name)
            if f.name.startswith("_"):
                changes[f.name] = val
            else:
                changes[f.name] = recursive_to_device(val, device, non_blocking)
        return type(obj)(**changes)
    if isinstance(obj, dict):
        return {k: recursive_to_device(v, device, non_blocking) for k, v in obj.items()}
    if isinstance(obj, list):
        return [recursive_to_device(v, device, non_blocking) for v in obj]
    return obj


def cs2_collate_fn(batch: list[TrainingSample]) -> TrainingSample:
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
        truth=batched_truth,
    )


class VisionSanityHeads(nn.Module):
    def __init__(
        self,
        in_dim: int,
        health_bins: int,
        armor_bins: int,
        money_bins: int,
        weapon_bins: int,
        dtype: torch.dtype,
        weapon_only: bool,
    ):
        super().__init__()
        self.weapon_only = weapon_only
        if not weapon_only:
            self.health = nn.Linear(in_dim, health_bins, dtype=dtype)
            self.armor = nn.Linear(in_dim, armor_bins, dtype=dtype)
            self.money = nn.Linear(in_dim, money_bins, dtype=dtype)
        self.weapon = nn.Linear(in_dim, weapon_bins, dtype=dtype)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [B, T, P, D]
        out = {"weapon": self.weapon(x)}
        if not self.weapon_only:
            out["health"] = self.health(x)
            out["armor"] = self.armor(x)
            out["money"] = self.money(x)
        return out


class VisionSanityModel(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        health_bins: int,
        armor_bins: int,
        money_bins: int,
        weapon_bins: int,
        vision_backend: str,
        efficientnet_pretrained: bool,
        cnn_chunk_size: int,
        cnn_image_size: int,
        weapon_only: bool,
        freeze_video: bool,
    ):
        super().__init__()
        self.vision_backend = vision_backend
        self.cnn_chunk_size = cnn_chunk_size
        self.cnn_image_size = cnn_image_size

        if vision_backend == "qformer":
            self.video = GameVideoEncoder(cfg)
            in_dim = cfg.qformer_hidden_size
        elif vision_backend == "efficientnet_b0":
            weights = tv_models.EfficientNet_B0_Weights.DEFAULT if efficientnet_pretrained else None
            self.video = tv_models.efficientnet_b0(weights=weights)
            self.video.classifier = nn.Identity()
            self.video.to(dtype=cfg.dtype)
            in_dim = 1280
        else:
            raise ValueError(f"Unknown vision_backend: {vision_backend}")

        self.heads = VisionSanityHeads(
            in_dim=in_dim,
            health_bins=health_bins,
            armor_bins=armor_bins,
            money_bins=money_bins,
            weapon_bins=weapon_bins,
            dtype=cfg.dtype,
            weapon_only=weapon_only,
        )
        if freeze_video:
            for p in self.video.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        # images: [B, T, P, C, H, W]
        if self.vision_backend == "qformer":
            q_feats = self.video(images)  # [B, T, P, N_q, D_q]
            pooled = q_feats.mean(dim=3)  # [B, T, P, D_q]
            pooled = pooled.to(dtype=self.heads.weapon.weight.dtype)
            return self.heads(pooled)

        # EfficientNet path
        B, T, P, C, H, W = images.shape
        flat = images.reshape(B * T * P, C, H, W)
        if flat.dtype == torch.uint8:
            flat = flat.float().div(255.0)
        else:
            flat = flat.float()
        flat = flat.to(dtype=self.video.features[0][0].weight.dtype)

        if self.cnn_image_size and (flat.shape[-2] != self.cnn_image_size or flat.shape[-1] != self.cnn_image_size):
            flat = F.interpolate(flat, size=(self.cnn_image_size, self.cnn_image_size), mode="bilinear", align_corners=False)

        mean = torch.tensor([0.485, 0.456, 0.406], device=flat.device, dtype=flat.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=flat.device, dtype=flat.dtype).view(1, 3, 1, 1)
        if flat.shape[1] == 3:
            flat = (flat - mean) / std

        feats = []
        for i in range(0, flat.shape[0], self.cnn_chunk_size):
            chunk = flat[i : i + self.cnn_chunk_size]
            feats.append(self.video(chunk))
        feats = torch.cat(feats, dim=0)
        feats = feats.view(B, T, P, -1)
        return self.heads(feats)


def bins_from_stats(stats: torch.Tensor, hp_bin_size: int, armor_bin_size: int, money_bin_size: int,
                    money_bins: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    health = stats[..., 0]
    armor = stats[..., 1]
    money = stats[..., 2]

    hp_bins = torch.clamp((health // hp_bin_size).long(), 0, (100 // hp_bin_size))
    armor_bins = torch.clamp((armor // armor_bin_size).long(), 0, (100 // armor_bin_size))
    money_bins = torch.clamp((money // money_bin_size).long(), 0, money_bins - 1)
    return hp_bins, armor_bins, money_bins


def masked_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # logits: [B, T, P, C], targets/mask: [B, T, P]
    B, T, P, C = logits.shape
    flat_logits = logits.reshape(B * T * P, C)
    flat_targets = targets.reshape(B * T * P)
    flat_mask = mask.reshape(B * T * P)
    if flat_mask.any():
        flat_logits = flat_logits[flat_mask]
        flat_targets = flat_targets[flat_mask]
        return F.cross_entropy(flat_logits.float(), flat_targets.long())
    return torch.zeros((), device=logits.device)


def masked_accuracy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    B, T, P, C = logits.shape
    flat_logits = logits.reshape(B * T * P, C)
    flat_targets = targets.reshape(B * T * P)
    flat_mask = mask.reshape(B * T * P)
    if not flat_mask.any():
        return torch.zeros((), device=logits.device)
    preds = flat_logits.argmax(dim=-1)
    correct = (preds == flat_targets) & flat_mask
    return correct.float().sum() / flat_mask.float().sum()


def run_eval(model: nn.Module, loader: DataLoader, device: torch.device,
             hp_bin_size: int, armor_bin_size: int, money_bin_size: int,
             money_bins: int, weapon_bins: int, max_batches: int,
             weapon_only: bool, is_distributed: bool) -> Dict[str, float]:
    model.eval()
    totals = torch.zeros(4, device=device)  # hp, armor, money, weapon
    count = 0

    with torch.no_grad():
        for i, sample in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            sample = recursive_to_device(sample, device)
            preds = model(sample.images)

            hp_bins, armor_bins, money_bins_t = bins_from_stats(
                sample.truth.stats, hp_bin_size, armor_bin_size, money_bin_size, money_bins
            )
            alive = sample.truth.alive_mask
            weapon_idx = sample.truth.active_weapon_idx
            weapon_mask = alive & (weapon_idx >= 0) & (weapon_idx < weapon_bins)

            if not weapon_only:
                totals[0] += masked_accuracy(preds["health"], hp_bins, alive)
                totals[1] += masked_accuracy(preds["armor"], armor_bins, alive)
                totals[2] += masked_accuracy(preds["money"], money_bins_t, alive)
            totals[3] += masked_accuracy(preds["weapon"], weapon_idx, weapon_mask)
            count += 1

    if is_distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        count_tensor = torch.tensor(count, device=device, dtype=torch.long)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        count = int(count_tensor.item())

    if count == 0:
        return {"hp_acc": 0.0, "armor_acc": 0.0, "money_acc": 0.0, "weapon_acc": 0.0}
    avg = (totals / count).tolist()
    return {"hp_acc": avg[0], "armor_acc": avg[1], "money_acc": avg[2], "weapon_acc": avg[3]}

def _tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().numpy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 3:
        # OpenCV expects BGR; dataset frames are RGB.
        arr = arr[..., ::-1]
    return np.ascontiguousarray(arr)

def _build_pred_dict_for_t(
    hp_bin: int,
    armor_bin: int,
    money_bin: int,
    hp_pred: torch.Tensor | None,
    armor_pred: torch.Tensor | None,
    money_pred: torch.Tensor | None,
    weapon_pred: torch.Tensor,
) -> Dict[str, Any]:
    pred_players = []
    for p in range(5):
        pred_players.append(
            {
                "health": int((hp_pred[p].item() * hp_bin) if hp_pred is not None else 0),
                "armor": int((armor_pred[p].item() * armor_bin) if armor_pred is not None else 0),
                "money": int((money_pred[p].item() * money_bin) if money_pred is not None else 0),
                "pos": [0.0, 0.0, 0.0],
                "mouse": [0.0, 0.0],
                "keyboard_bitmask": 0,
                "eco_bitmask": np.zeros(4, dtype=np.uint64),
                "inventory_bitmask": np.zeros(2, dtype=np.uint64),
                "active_weapon_idx": int(weapon_pred[p].item()),
            }
        )

    return {
        "game_state": {
            "tick": 0,
            "round_state": 0,
            "team_alive": 0,
            "enemy_alive": 0,
            "enemy_pos": [[0.0, 0.0, 0.0] for _ in range(5)],
        },
        "player_data": pred_players,
    }

def generate_val_video(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    hp_bin: int,
    armor_bin: int,
    money_bin: int,
    video_out: str,
    weapon_only: bool,
    video_seconds: int,
    video_seed: int,
):
    model.eval()
    rng = random.Random(video_seed)
    target_frames = int(video_seconds * FRAME_RATE)
    frames_written = 0
    total_samples = max(1, (target_frames + 1) // dataset.config.epoch_round_sample_length)

    indices = [rng.randrange(len(dataset)) for _ in range(total_samples)]

    first_sample = dataset[indices[0]]
    first_frame = _tensor_to_uint8(first_sample.images[0, 0])
    h, w = first_frame.shape[:2]
    grid_w, grid_h = w * 3, h * 2
    os.makedirs(os.path.dirname(video_out) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_out, fourcc, FRAME_RATE, (grid_w, grid_h))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {video_out}")

    for idx in indices:
        if frames_written >= target_frames:
            break
        sample = dataset[idx]
        sample = recursive_to_device(sample, device)
        images = sample.images.unsqueeze(0)
        with torch.no_grad():
            preds = model(images)

        images = sample.images  # [T, P, C, H, W]
        T = images.shape[0]

        hp_pred = preds["health"][0].argmax(dim=-1).cpu() if not weapon_only else None
        armor_pred = preds["armor"][0].argmax(dim=-1).cpu() if not weapon_only else None
        money_pred = preds["money"][0].argmax(dim=-1).cpu() if not weapon_only else None
        weapon_pred = preds["weapon"][0].argmax(dim=-1).cpu()

        gt_fields = {f.name: getattr(sample.truth, f.name).cpu() for f in fields(sample.truth)}
        gt_single = GroundTruth(**gt_fields)

        for t in range(T):
            if frames_written >= target_frames:
                break
            frames = [_tensor_to_uint8(images[t, p]) for p in range(5)]
            gt_dict = visualize.convert_tensor_to_viz_data(gt_single, t)
            pred_dict = _build_pred_dict_for_t(
                hp_bin,
                armor_bin,
                money_bin,
                hp_pred[t] if hp_pred is not None else None,
                armor_pred[t] if armor_pred is not None else None,
                money_pred[t] if money_pred is not None else None,
                weapon_pred[t],
            )
            combined = visualize.visualize_frame(frames, t, gt_dict, pred_dict)
            writer.write(combined)
            frames_written += 1

    writer.release()
    logger.info("Wrote validation video to %s", video_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--run_dir", type=str, default="./runs/vision_sanity")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--frames_per_sample", type=int, default=256)
    parser.add_argument("--epoch_windows_per_round", type=int, default=1)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=50)
    parser.add_argument("--hp_bin", type=int, default=10)
    parser.add_argument("--armor_bin", type=int, default=10)
    parser.add_argument("--money_bin", type=int, default=500)
    parser.add_argument("--max_money", type=int, default=16000)
    parser.add_argument("--weapon_bins", type=int, default=128)
    parser.add_argument("--video_out", type=str, default="./runs/vision_sanity/val_pred.mp4")
    parser.add_argument("--video_seconds", type=int, default=300)
    parser.add_argument("--video_seed", type=int, default=123)
    parser.add_argument("--vision_backend", type=str, default="qformer", choices=["qformer", "efficientnet_b0"])
    parser.add_argument("--efficientnet_pretrained", action="store_true", help="Use ImageNet pretrained EfficientNet")
    parser.add_argument("--cnn_chunk_size", type=int, default=64)
    parser.add_argument("--cnn_image_size", type=int, default=224)
    parser.add_argument("--weapon_only", action="store_true", help="Only train the weapon head")
    parser.add_argument("--freeze_video", action="store_true", help="Freeze the vision encoder (ViT/Q-Former or CNN)")
    parser.add_argument("--ddp", action="store_true", help="Use torch.distributed DDP")
    parser.add_argument("--precision", type=str, choices=["bf16", "fp16", "fp32"], default=None)
    args = parser.parse_args()

    is_distributed = False
    local_rank = 0
    if args.ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA devices")
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        is_distributed = True
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    is_main = (not is_distributed) or dist.get_rank() == 0
    if args.precision is None:
        precision = "bf16" if device.type == "cuda" else "fp32"
    else:
        precision = args.precision

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    model_dtype = dtype_map[precision]

    money_bins = (args.max_money // args.money_bin) + 1
    health_bins = (100 // args.hp_bin) + 1
    armor_bins = (100 // args.armor_bin) + 1

    logger.info("Building dataset...")
    ds_config = DatasetConfig(
        data_root=args.data_root,
        run_dir=args.run_dir,
        epoch_round_sample_length=args.frames_per_sample,
        epoch_windows_per_round=args.epoch_windows_per_round,
    )
    ds_root = DatasetRoot(ds_config)
    train_ds = ds_root.build_epoch("train", 0)
    val_ds = ds_root.build_epoch("val", 0)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=cs2_collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=cs2_collate_fn,
    )

    disable_qformer_ckpt = args.ddp and args.vision_backend == "qformer"
    model_cfg = ModelConfig(dtype=model_dtype, gradient_checkpointing=not disable_qformer_ckpt)
    model = VisionSanityModel(
        model_cfg,
        health_bins=health_bins,
        armor_bins=armor_bins,
        money_bins=money_bins,
        weapon_bins=args.weapon_bins,
        vision_backend=args.vision_backend,
        efficientnet_pretrained=args.efficientnet_pretrained,
        cnn_chunk_size=args.cnn_chunk_size,
        cnn_image_size=args.cnn_image_size,
        weapon_only=args.weapon_only,
        freeze_video=args.freeze_video,
    ).to(device)

    if is_distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    logger.info(
        "Starting training: frames=%d, health_bins=%d, armor_bins=%d, money_bins=%d, weapon_bins=%d",
        args.frames_per_sample, health_bins, armor_bins, money_bins, args.weapon_bins
    )

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running = {"loss": 0.0, "hp": 0.0, "armor": 0.0, "money": 0.0, "weapon": 0.0}
        count = 0

        for step, sample in enumerate(train_loader):
            if args.max_train_batches is not None and step >= args.max_train_batches:
                break

            sample = recursive_to_device(sample, device)
            preds = model(sample.images)

            hp_bins, armor_bins, money_bins_t = bins_from_stats(
                sample.truth.stats, args.hp_bin, args.armor_bin, args.money_bin, money_bins
            )
            alive = sample.truth.alive_mask
            weapon_idx = sample.truth.active_weapon_idx
            weapon_mask = alive & (weapon_idx >= 0) & (weapon_idx < args.weapon_bins)

            loss_weapon = masked_ce(preds["weapon"], weapon_idx, weapon_mask)
            if args.weapon_only:
                loss = loss_weapon
            else:
                loss_hp = masked_ce(preds["health"], hp_bins, alive)
                loss_armor = masked_ce(preds["armor"], armor_bins, alive)
                loss_money = masked_ce(preds["money"], money_bins_t, alive)
                loss = loss_hp + loss_armor + loss_money + loss_weapon

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running["loss"] += loss.item()
            running["weapon"] += masked_accuracy(preds["weapon"], weapon_idx, weapon_mask).item()
            if not args.weapon_only:
                running["hp"] += masked_accuracy(preds["health"], hp_bins, alive).item()
                running["armor"] += masked_accuracy(preds["armor"], armor_bins, alive).item()
                running["money"] += masked_accuracy(preds["money"], money_bins_t, alive).item()
            count += 1

            if step % 20 == 0:
                if args.weapon_only and is_main:
                    logger.info(
                        "ep %d step %d | loss %.4f | weapon %.3f",
                        epoch, step,
                        running["loss"] / max(1, count),
                        running["weapon"] / max(1, count),
                    )
                elif is_main:
                    logger.info(
                        "ep %d step %d | loss %.4f | hp %.3f armor %.3f money %.3f weapon %.3f",
                        epoch, step,
                        running["loss"] / max(1, count),
                        running["hp"] / max(1, count),
                        running["armor"] / max(1, count),
                        running["money"] / max(1, count),
                        running["weapon"] / max(1, count),
                    )

        val_metrics = run_eval(
            model,
            val_loader,
            device,
            args.hp_bin,
            args.armor_bin,
            args.money_bin,
            money_bins,
            args.weapon_bins,
            max_batches=args.max_val_batches,
            weapon_only=args.weapon_only,
            is_distributed=is_distributed,
        )
        if args.weapon_only and is_main:
            logger.info("ep %d val | weapon %.3f", epoch, val_metrics["weapon_acc"])
        elif is_main:
            logger.info(
                "ep %d val | hp %.3f armor %.3f money %.3f weapon %.3f",
                epoch, val_metrics["hp_acc"], val_metrics["armor_acc"], val_metrics["money_acc"], val_metrics["weapon_acc"]
            )

        if is_main:
            base, ext = os.path.splitext(args.video_out)
            per_epoch_out = f"{base}_ep{epoch}{ext or '.mp4'}"
            generate_val_video(
                model,
                val_ds,
                device,
                args.hp_bin,
                args.armor_bin,
                args.money_bin,
                per_epoch_out,
                args.weapon_only,
                args.video_seconds,
                args.video_seed + epoch,
            )

    if is_main:
        generate_val_video(
            model,
            val_ds,
            device,
            args.hp_bin,
            args.armor_bin,
            args.money_bin,
            args.video_out,
            args.weapon_only,
            args.video_seconds,
            args.video_seed,
        )
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
