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
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GlobalConfig
from dataset import GroundTruth
from model import ModelPrediction


def mu_law_encode(x, mu=255.0, max_val=30.0, bins=256):
    x = torch.nan_to_num(x, nan=0.0)
    x = torch.clamp(x, -max_val, max_val)
    x_norm = x / max_val
    sign = torch.sign(x_norm)
    encoded = sign * torch.log1p(mu * torch.abs(x_norm)) / torch.log1p(torch.tensor(mu, device=x.device))
    encoded = (encoded + 1) / 2 * (bins - 1e-5)
    return encoded.long().clamp(0, bins - 1)


def mu_law_decode(y, mu=255.0, max_val=30.0, bins=256):
    y = y.float()
    y_norm = (y / (bins - 1)) * 2 - 1
    x_abs = (1 / mu) * (torch.pow(1 + mu, torch.abs(y_norm)) - 1)
    x_norm = torch.sign(y_norm) * x_abs
    return x_norm * max_val


def bin_value(x, min_val, max_val, num_bins):
    x = torch.nan_to_num(x, nan=min_val)
    norm = (x - min_val) / (max_val - min_val)
    norm = torch.clamp(norm, 0.0, 1.0)
    bins = (norm * (num_bins - 1e-5)).long()
    return bins.clamp(0, num_bins - 1)


def unbin_value(b, min_val, max_val, num_bins):
    b = b.float()
    t = b / (num_bins - 1)
    return min_val + t * (max_val - min_val)


class ModelLoss(nn.Module):
    def __init__(self, cfg: GlobalConfig):
        super().__init__()
        self.cfg = cfg
        self.model_cfg = cfg.model
        self.weights = cfg.model.loss_weights
        self.enabled = cfg.model.loss_enabled
        self.keyboard_dim = cfg.model.keyboard_dim
        self.mouse_bins = cfg.model.mouse_bins_count
        self.eco_dim = cfg.model.eco_dim
        self.mouse_center_bin = int(
            mu_law_encode(
                torch.tensor([0.0]),
                self.model_cfg.mouse_mu,
                self.model_cfg.mouse_max,
                self.model_cfg.mouse_bins_count,
            ).item()
        )

    def _is_enabled(self, name: str) -> bool:
        return bool(self.enabled.get(name, True))

    def _weight(self, name: str) -> float:
        return float(self.weights.get(name, 1.0))

    def _valid_mask(self, gt: GroundTruth, device: torch.device) -> torch.Tensor:
        return gt.alive_mask.reshape(-1).to(device=device, dtype=torch.float32)

    def forward(self, pred: ModelPrediction, gt: GroundTruth) -> dict[str, torch.Tensor]:
        device = pred.keyboard_logits.device
        valid_mask = self._valid_mask(gt, device)
        num_valid = valid_mask.sum().clamp(min=1.0)

        losses: dict[str, torch.Tensor] = {}
        total_loss = pred.keyboard_logits.new_zeros(())

        if self._is_enabled("mouse"):
            mouse_target = gt.mouse_delta.to(device=device)
            mouse_x_target = mu_law_encode(
                mouse_target[..., 0].reshape(-1),
                self.model_cfg.mouse_mu,
                self.model_cfg.mouse_max,
                self.mouse_bins,
            )
            mouse_y_target = mu_law_encode(
                mouse_target[..., 1].reshape(-1),
                self.model_cfg.mouse_mu,
                self.model_cfg.mouse_max,
                self.mouse_bins,
            )

            mouse_x_logits = pred.mouse_x.reshape(-1, self.mouse_bins)
            mouse_y_logits = pred.mouse_y.reshape(-1, self.mouse_bins)

            mouse_bin_weights = torch.ones(
                self.mouse_bins,
                device=device,
                dtype=mouse_x_logits.dtype,
            )
            mouse_bin_weights[self.mouse_center_bin] = self.model_cfg.mouse_center_bin_weight

            mouse_x_loss = F.cross_entropy(
                mouse_x_logits,
                mouse_x_target,
                weight=mouse_bin_weights,
                reduction="none",
            )
            mouse_y_loss = F.cross_entropy(
                mouse_y_logits,
                mouse_y_target,
                weight=mouse_bin_weights,
                reduction="none",
            )
            mouse_loss = ((mouse_x_loss + mouse_y_loss) * valid_mask).sum() / num_valid
            mouse_loss = mouse_loss * self._weight("mouse")
            losses["mouse"] = mouse_loss
            total_loss = total_loss + mouse_loss

        if self._is_enabled("keyboard"):
            keyboard_mask = gt.keyboard_mask.to(device=device).reshape(-1, 1).long()
            bits = torch.arange(self.keyboard_dim, device=device).view(1, self.keyboard_dim)
            keyboard_target = ((keyboard_mask >> bits) & 1).float()
            keyboard_logits = pred.keyboard_logits.reshape(-1, self.keyboard_dim)
            keyboard_loss = F.binary_cross_entropy_with_logits(
                keyboard_logits,
                keyboard_target,
                reduction="none",
            ).mean(dim=-1)
            keyboard_loss = (keyboard_loss * valid_mask).sum() / num_valid
            keyboard_loss = keyboard_loss * self._weight("keyboard")
            losses["keyboard"] = keyboard_loss
            total_loss = total_loss + keyboard_loss

        if self._is_enabled("eco_purchase"):
            did_buy = (gt.eco_mask.to(device=device).reshape(-1, 4) != 0).any(dim=-1).float()
            eco_purchase_logits = pred.eco_purchase_logits.reshape(-1)
            eco_purchase_loss = F.binary_cross_entropy_with_logits(
                eco_purchase_logits,
                did_buy,
                reduction="none",
            )
            eco_purchase_loss = (eco_purchase_loss * valid_mask).sum() / num_valid
            eco_purchase_loss = eco_purchase_loss * self._weight("eco_purchase")
            losses["eco_purchase"] = eco_purchase_loss
            total_loss = total_loss + eco_purchase_loss

        if self._is_enabled("eco_buy"):
            did_buy = (gt.eco_mask.to(device=device).reshape(-1, 4) != 0).any(dim=-1)
            eco_buy_mask = valid_mask * did_buy.float()
            num_buys = eco_buy_mask.sum().clamp(min=1.0)

            eco_buy_target = gt.eco_buy_idx.to(device=device).reshape(-1).long()
            eco_buy_target = eco_buy_target.clamp(min=0, max=self.eco_dim - 1)
            eco_buy_logits = pred.eco_buy_logits.reshape(-1, self.eco_dim)

            eco_buy_loss = F.cross_entropy(
                eco_buy_logits,
                eco_buy_target,
                reduction="none",
            )
            eco_buy_loss = (eco_buy_loss * eco_buy_mask).sum() / num_buys
            eco_buy_loss = eco_buy_loss * self._weight("eco_buy")
            losses["eco_buy"] = eco_buy_loss
            total_loss = total_loss + eco_buy_loss

        losses["total"] = total_loss
        return losses
