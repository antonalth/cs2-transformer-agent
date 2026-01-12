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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
from dataclasses import dataclass
from typing import Dict, Tuple, Any

from model_novibe import ModelPrediction
from config import ModelConfig

class DynamicLossScaler(nn.Module):
    """
    Normalize loss groups to comparable scale using an EMA during warmup,
    then freeze the scale factors so easy tasks can naturally fade out.
    """
    def __init__(self, num_losses: int, momentum: float = 0.01, warmup_steps: int = 2000):
        super().__init__()
        self.momentum = momentum
        self.warmup_steps = warmup_steps

        self.register_buffer("running_magnitudes", torch.ones(num_losses))
        self.register_buffer("base_magnitudes", torch.ones(num_losses))
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    def forward(self, loss_list: list[torch.Tensor]):
        raw = torch.stack([l.detach().float() for l in loss_list])  # float32 for stability

        if self.training:
            self.step += 1
            if self.step <= self.warmup_steps:
                self.running_magnitudes = (1 - self.momentum) * self.running_magnitudes + self.momentum * raw
                self.base_magnitudes = self.running_magnitudes.clone()

        scales = 1.0 / (self.base_magnitudes + 1e-5)  # constant after warmup
        scaled_losses = [loss_list[i] * scales[i].to(loss_list[i].dtype) for i in range(len(loss_list))]
        return scaled_losses, scales

class DynamicWeightAverage(nn.Module):
    """
    Dynamic Weight Averaging (DWA)
    w_i(t) = K * softmax( r_i(t) / T )
    r_i(t) = L_i(t-1) / (L_i(t-2) + eps)

    We approximate epoch losses with windowed EMAs updated every `update_every` steps.
    """
    def __init__(self, num_losses: int, temperature: float = 2.0, momentum: float = 0.01, update_every: int = 200):
        super().__init__()
        self.num_losses = num_losses
        self.temperature = temperature
        self.momentum = momentum
        self.update_every = update_every

        self.register_buffer("ema_window", torch.ones(num_losses))
        self.register_buffer("ema_prev", torch.ones(num_losses))
        self.register_buffer("ema_prev2", torch.ones(num_losses))
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.register_buffer("weights", torch.ones(num_losses))

    def forward(self, loss_list: list[torch.Tensor]) -> torch.Tensor:
        # Use raw detached losses; ratios are scale-invariant (good for mixed loss types).
        raw = torch.stack([l.detach().float() for l in loss_list])

        if not self.training:
            return self.weights

        self.step += 1
        self.ema_window = (1 - self.momentum) * self.ema_window + self.momentum * raw

        # Update DWA weights only every `update_every` steps
        if (self.step % self.update_every) == 0:
            self.ema_prev2 = self.ema_prev.clone()
            self.ema_prev = self.ema_window.clone()

            # Need two history windows before ratios mean anything
            if self.step >= 2 * self.update_every:
                r = self.ema_prev / (self.ema_prev2 + 1e-8)
                logits = r / self.temperature
                w = self.num_losses * torch.softmax(logits, dim=0)
                self.weights = w

        return self.weights


class CS2Metrics:
    def __init__(self):
        self.epsilon = 1e-7

    def calculate(self, preds: ModelPrediction, truth: Any, mouse_bin_centers: torch.Tensor) -> Dict[str, float]:
        m = {}
        device = preds.keyboard_logits.device
        alive_mask = truth.alive_mask
        alive_count = alive_mask.sum().float() + self.epsilon
        
        # --- Keyboard F1 ---
        kb_probs = torch.sigmoid(preds.keyboard_logits)
        kb_pred_bin = (kb_probs > 0.5).long()
        
        gt_kb_bits = (truth.keyboard_mask.unsqueeze(-1).long() >> torch.arange(32, device=device)) & 1
        
        # Mask only alive players
        alive_expanded = alive_mask.unsqueeze(-1)
        tp = (kb_pred_bin * gt_kb_bits * alive_expanded).sum().float()
        fp = (kb_pred_bin * (1 - gt_kb_bits) * alive_expanded).sum().float()
        fn = ((1 - kb_pred_bin) * gt_kb_bits * alive_expanded).sum().float()
        
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        
        m["metric/kb_f1"] = f1.item()
        m["stat/kb_action_rate"] = (kb_pred_bin.float() * alive_expanded).sum().item() / (alive_count * 32)

        # --- Mouse MAE ---
        idx_x = torch.argmax(preds.mouse_x, dim=-1)
        idx_y = torch.argmax(preds.mouse_y, dim=-1)
        
        px_x = mouse_bin_centers[idx_x]
        px_y = mouse_bin_centers[idx_y]
        
        mae_x = (torch.abs(px_x - truth.mouse_delta[..., 0]) * alive_mask).sum() / alive_count
        mae_y = (torch.abs(px_y - truth.mouse_delta[..., 1]) * alive_mask).sum() / alive_count
        
        m["metric/mouse_mae_x"] = mae_x.item()
        m["metric/mouse_mae_y"] = mae_y.item()
        # Movement rate: move > 0.1 deg
        m["stat/mouse_move_rate"] = (( (torch.abs(px_x) > 0.1) | (torch.abs(px_y) > 0.1) ) * alive_mask).sum().item() / alive_count

        # --- Stats MAE ---
        stats_pred = torch.sigmoid(preds.stats_logits)
        
        hp_err = (torch.abs(stats_pred[..., 0] - (truth.stats[..., 0] / 100.0)) * alive_mask).sum() / alive_count
        
        # Money is normalized via log1p(m)/9.68 in CS2Loss.stats
        # Denormalize to get actual money for metric
        money_pred_actual = torch.expm1(stats_pred[..., 2] * 9.68)
        money_err = (torch.abs(money_pred_actual - truth.stats[..., 2]) * alive_mask).sum() / alive_count
        
        m["metric/hp_mae"] = hp_err.item() * 100.0
        m["metric/money_mae"] = money_err.item()
        
        return m

class CS2Loss(nn.Module):
    # Map dimensions/constants
    MAP_MIN = torch.tensor([-4000.0, -4000.0, -500.0])
    MAP_MAX = torch.tensor([ 4000.0,  4000.0,  2000.0])
    MAP_SIZE = MAP_MAX - MAP_MIN
    BINS_X = 256
    BINS_Y = 256
    BINS_Z = 32

    def __init__(self, cfg: ModelConfig = None):
        super().__init__()
        # We group losses into 15 distinct categories for weighting
        # 0: MouseX, 1: MouseY, 2: Keyboard, 3: Eco, 4: Inv, 5: Wep
        # 6: HP, 7: Armor, 8: Money
        # 9: Player Pos, 10: Enemy Pos, 11: Round State, 12: Round Num
        # 13: Team Alive, 14: Enemy Alive
        self.mouse_bins_count = cfg.mouse_bins_count
        self.num_loss_groups = 15

    
        centers = self._generate_mu_law_bins(self.mouse_bins_count, min_val=-90.0, max_val=90.0, mu=255.0)
        self.register_buffer("mouse_bin_centers", centers)
        
        widths = torch.empty_like(centers)
        widths[1:-1] = 0.5 * (centers[2:] - centers[:-2])
        widths[0] = centers[1] - centers[0]
        widths[-1] = centers[-1] - centers[-2]
        self.register_buffer("mouse_bin_widths", widths.abs().clamp_min(1e-6))

        self.scaler = DynamicLossScaler(
            num_losses=self.num_loss_groups,
            momentum=0.01,
            warmup_steps=getattr(cfg, "loss_scale_warmup_steps", 2000),
        )

        self.dwa = DynamicWeightAverage(
            num_losses=self.num_loss_groups,
            temperature=cfg.dwa_temperature,
            momentum=cfg.dwa_momentum,
            update_every=cfg.dwa_update_every,
        )

        self.metrics_calc = CS2Metrics()

    def forward(self, preds: ModelPrediction, truth: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates all losses, weights them, and returns Total Loss + Metrics Dict.
        """
        # --- 1. Compute Raw Losses ---
        
        l_mouse_x, l_mouse_y = self.mouse(preds.mouse_x, preds.mouse_y, truth.mouse_delta, truth.alive_mask)
        l_key = self.keyboard(preds.keyboard_logits, truth.keyboard_mask, truth.alive_mask)
        l_eco = self.eco(preds.eco_logits, truth.eco_mask, truth.alive_mask)
        l_inv = self.inventory(preds.inventory_logits, truth.inventory_mask, truth.alive_mask)
        l_wep = self.weapon(preds.weapon_logits, truth.active_weapon_idx, truth.alive_mask)
        l_hp, l_armor, l_money = self.stats(preds.stats_logits, truth.stats, truth.alive_mask)
        
        l_p_pos = self.position(
            preds.player_pos_x, preds.player_pos_y, preds.player_pos_z, 
            truth.position, truth.alive_mask
        )
        l_e_pos = self.enemy_position(
            preds.enemy_pos_x, preds.enemy_pos_y, preds.enemy_pos_z, 
            truth.enemy_positions, truth.enemy_alive_mask
        )
        
        l_rnd_st = self.round_state(preds.round_state_logits, truth.round_state_mask)
        l_rnd_nm = self.round_number(preds.round_num_logit, truth.round_number)
        
        l_t_alv  = self.alive_count(preds.team_alive_logits, truth.alive_mask)
        l_e_alv  = self.alive_count(preds.enemy_alive_logits, truth.enemy_alive_mask)
        
        # 15 losses
        loss_list = [
            l_mouse_x, l_mouse_y, l_key, l_eco, l_inv, l_wep, 
            l_hp, l_armor, l_money,
            l_p_pos, l_e_pos, l_rnd_st, l_rnd_nm, l_t_alv, l_e_alv
        ]

        loss_list_norm, scalers = self.scaler(loss_list) # norm magnitudes (for warmup phase only)
        dwa_w = self.dwa(loss_list)  # tensor [15], float32

        total_loss = 0.0
        for i in range(self.num_loss_groups):
            total_loss = total_loss + loss_list_norm[i] * dwa_w[i].to(loss_list_norm[i].dtype)
        total_loss = total_loss / self.num_loss_groups

        metrics = self.metrics_calc.calculate(preds, truth, self.mouse_bin_centers)

        info = {
            "loss/total": total_loss.item(),
            
            "raw/mouse_x": l_mouse_x.item(),
            "raw/mouse_y": l_mouse_y.item(),
            "raw/keyboard": l_key.item(),
            "raw/eco": l_eco.item(),
            "raw/inv": l_inv.item(),
            "raw/wep": l_wep.item(),
            "raw/hp": l_hp.item(),
            "raw/armor": l_armor.item(),
            "raw/money": l_money.item(),
            
            "raw/player_pos": l_p_pos.item(),
            "raw/enemy_pos": l_e_pos.item(),
            
            "raw/round_state": l_rnd_st.item(),
            "raw/round_num": l_rnd_nm.item(),
            
            "raw/team_alive": l_t_alv.item(),
            "raw/enemy_alive": l_e_alv.item(),
        }

        group_names = [
            "mouse_x", "mouse_y", "keyboard", "eco", "inv", "wep", 
            "hp", "armor", "money", 
            "player_pos", "enemy_pos", "round_state", "round_num", 
            "team_alive", "enemy_alive"
        ]

        for i, name in enumerate(group_names):
            info[f"scale/{name}"] = float(scalers[i].item())
            info[f"dwa/{name}"] = float(dwa_w[i].item())

        metrics.update(info)

        return total_loss, metrics

    @staticmethod
    def _generate_mu_law_bins(n_bins, min_val, max_val, mu):
        """Generates bin centers concentrated near 0."""
        t = torch.linspace(-1, 1, n_bins)
        # Inverse Mu-Law
        val_norm = torch.sign(t) * (1.0 / mu) * (torch.pow(1.0 + mu, torch.abs(t)) - 1.0)
        return val_norm * max_val
        
    @staticmethod
    def _focal_loss_norm(pred, gt, alpha=0.95, gamma=2.0):
        """
        Computes Focal Loss normalized by the number of positive targets.
        This prevents loss washing out when data is extremely sparse (e.g. 99.9% zeros).
        """
        # Sum of focal loss over all elements
        total_loss = sigmoid_focal_loss(pred, gt, alpha=alpha, gamma=gamma, reduction='sum')
        
        # Count positives
        num_pos = gt.sum().float()
        
        # Normalize
        return total_loss / (num_pos + 1.0)

    def _get_gaussian_targets(self, gt_values, bin_centers, bin_widths, sigma_bins=2.0):
        # Clamp GT
        gt = gt_values.float().clamp(bin_centers.min().float(), bin_centers.max().float())
        centers = bin_centers.float()

        # Find nearest bin index
        idx = torch.argmin(torch.abs(centers.unsqueeze(0) - gt.unsqueeze(1)), dim=1)

        # Look up pre-calculated sigma
        sigma_val = (sigma_bins * bin_widths[idx]).unsqueeze(1)

        dist = torch.exp(-0.5 * ((centers.unsqueeze(0) - gt.unsqueeze(1)) / sigma_val) ** 2)
        return dist / (dist.sum(dim=1, keepdim=True) + 1e-8)


    def mouse(self, pred_x, pred_y, gt_delta, mask):
        """
        Discretized Mouse Loss with Gaussian Smoothing + Imbalance Weighting.
        pred_x, pred_y: [B, T, 5, 256]
        gt_delta: [B, T, 5, 2]
        mask: [B, T, 5]
        """
        # 1. Masking and Flattening
        m_flat = mask.view(-1)
        if m_flat.sum() == 0: 
            z = torch.tensor(0.0, device=pred_x.device, dtype=pred_x.dtype)
            return z, z

        # pred_x: [N, 256] (Only alive players)
        p_x = pred_x.reshape(-1, self.mouse_bins_count)[m_flat]
        p_y = pred_y.reshape(-1, self.mouse_bins_count)[m_flat]

        # gt: [N] (Only alive players)
        g_x = gt_delta[..., 0].reshape(-1)[m_flat]
        g_y = gt_delta[..., 1].reshape(-1)[m_flat]

        # 2. Generate Gaussian Soft Targets
        # Helper uses self.mouse_bin_centers to create bell curve around GT
        tgt_x = self._get_gaussian_targets(g_x, self.mouse_bin_centers, self.mouse_bin_widths, sigma_bins=2.0)
        tgt_y = self._get_gaussian_targets(g_y, self.mouse_bin_centers, self.mouse_bin_widths, sigma_bins=2.0)
        
        # 3. KL Divergence (Requires LogSoftmax input)
        log_x = F.log_softmax(p_x.float(), dim=-1)
        log_y = F.log_softmax(p_y.float(), dim=-1)
        
        # Calculate raw loss per sample
        loss_x_raw = F.kl_div(log_x, tgt_x, reduction='none').sum(dim=1)
        loss_y_raw = F.kl_div(log_y, tgt_y, reduction='none').sum(dim=1)
        
        # 4. Apply Weights for Class Imbalance
        # If GT movement > 0.05, weight it 10x to punish 'camping' predictions
        move_mask_x = torch.abs(g_x) > 0.05
        move_mask_y = torch.abs(g_y) > 0.05
        
        w_x = torch.where(move_mask_x, 10.0, 1.0)
        w_y = torch.where(move_mask_y, 10.0, 1.0)
        
        final_x = (loss_x_raw * w_x).mean()
        final_y = (loss_y_raw * w_y).mean()
        
        return final_x, final_y

    @staticmethod
    def keyboard(pred, gt, mask):
        # Focal Loss (Multi-label, sparse)
        # Expand 32-bit int mask
        gt_long = gt.long() 
        gt_t = (gt_long.unsqueeze(-1) >> torch.arange(32, device=pred.device)) & 1
        m_flat = mask.view(-1)
        p_flat = pred.view(-1, 32)[m_flat]
        
        g_flat = gt_t.to(dtype=pred.dtype).view(-1, 32)[m_flat]
        
        if p_flat.shape[0] == 0: return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        return CS2Loss._focal_loss_norm(p_flat, g_flat, alpha=0.95, gamma=2.0)

    @staticmethod
    def _unpack_chunks(chunks, num_chunks, target_dtype):
        # Helper for Eco/Inv
        bits = torch.arange(64, device=chunks.device)
        unpacked = []
        for i in range(num_chunks):
            expanded = (chunks[..., i].unsqueeze(-1) >> bits) & 1
            unpacked.append(expanded)
        return torch.cat(unpacked, dim=-1).to(dtype=target_dtype)

    @staticmethod
    def eco(pred, gt_chunks, mask):
        # Eco: 4 chunks (256 bits), Focal Loss
        gt_t = CS2Loss._unpack_chunks(gt_chunks, 4, pred.dtype)
        m_flat = mask.view(-1)
        p_flat = pred.view(-1, 256)[m_flat]
        g_flat = gt_t.view(-1, 256)[m_flat]
        if p_flat.shape[0] == 0: return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        return CS2Loss._focal_loss_norm(p_flat, g_flat, alpha=0.95, gamma=2.0)

    @staticmethod
    def inventory(pred, gt_chunks, mask):
        # Inv: 2 chunks (128 bits), Focal Loss
        gt_t = CS2Loss._unpack_chunks(gt_chunks, 2, pred.dtype)
        m_flat = mask.view(-1)
        p_flat = pred.view(-1, 128)[m_flat]
        g_flat = gt_t.view(-1, 128)[m_flat]
        if p_flat.shape[0] == 0: return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        return CS2Loss._focal_loss_norm(p_flat, g_flat, alpha=0.95, gamma=2.0)

    @staticmethod
    def weapon(pred, gt_idx, mask):
        # CrossEntropy (Single Class, Mutually Exclusive)
        m_flat = mask.view(-1)
        p_flat = pred.view(-1, 128)[m_flat]
        g_flat = gt_idx.view(-1)[m_flat].long()
        if p_flat.shape[0] == 0: return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        return F.cross_entropy(p_flat, g_flat, ignore_index=-1)

    @staticmethod
    def stats(pred, gt, mask):
        # Focal Loss for HP/Armor + L1 for Money
        # gt: [Health, Armor, Money]
        m_flat = mask.view(-1)
        if m_flat.sum() == 0: 
            z = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
            return z, z, z

        # Normalize GT
        h_norm = gt[..., 0] / 100.0
        a_norm = gt[..., 1] / 100.0
        # Log Norm for Money: log(m+1) / log(16001)
        mon_norm = torch.log1p(gt[..., 2]) / 9.68  # log(16001) ~ 9.68
        
        # Split predictions
        p_flat = pred.view(-1, 3)[m_flat]
        
        # HP/Armor: Focal Loss (Target is [0,1], pred is logits)
        # Note: _focal_loss_norm expects logits and [0,1] targets
        h_target = h_norm.view(-1)[m_flat].to(dtype=pred.dtype)
        a_target = a_norm.view(-1)[m_flat].to(dtype=pred.dtype)
        
        l_hp = CS2Loss._focal_loss_norm(p_flat[:, 0], h_target, alpha=0.95, gamma=2.0)
        l_armor = CS2Loss._focal_loss_norm(p_flat[:, 1], a_target, alpha=0.95, gamma=2.0)
        
        # Money: L1 Loss (Regression)
        # Apply sigmoid to money pred to bound it [0,1] matching normalization
        mon_pred = torch.sigmoid(p_flat[:, 2])
        mon_target = mon_norm.view(-1)[m_flat].to(dtype=pred.dtype)
        l_money = F.l1_loss(mon_pred, mon_target)
        
        return l_hp, l_armor, l_money

    @staticmethod
    def _positions_to_bins(pos, device):
        norm = (pos - CS2Loss.MAP_MIN.to(device)) / CS2Loss.MAP_SIZE.to(device)
        norm = torch.clamp(norm, 0.0, 0.999)
        bins = torch.tensor([CS2Loss.BINS_X, CS2Loss.BINS_Y, CS2Loss.BINS_Z], device=device)
        return (norm * bins).long()

    @staticmethod
    def position(pred_x, pred_y, pred_z, gt_pos, mask):
        # Core Position Loss
        gt_idx = CS2Loss._positions_to_bins(gt_pos, pred_x.device)
        m_flat = mask.view(-1)
        
        # Flatten Preds
        px = pred_x.contiguous().view(-1, CS2Loss.BINS_X)[m_flat]
        py = pred_y.contiguous().view(-1, CS2Loss.BINS_Y)[m_flat]
        pz = pred_z.contiguous().view(-1, CS2Loss.BINS_Z)[m_flat]
        
        # Flatten Targets
        tx = gt_idx[..., 0].contiguous().view(-1)[m_flat]
        ty = gt_idx[..., 1].contiguous().view(-1)[m_flat]
        tz = gt_idx[..., 2].contiguous().view(-1)[m_flat]

        if px.shape[0] == 0: return torch.tensor(0.0, device=pred_x.device, dtype=pred_x.dtype)
        
        lx = F.cross_entropy(px, tx, label_smoothing=0.1)
        ly = F.cross_entropy(py, ty, label_smoothing=0.1)
        lz = F.cross_entropy(pz, tz, label_smoothing=0.1)
        return lx + ly + lz

    @staticmethod
    def enemy_position(pred_x, pred_y, pred_z, gt_pos, mask):
        # Sort enemies by X coord, push dead to end
        gt_x = gt_pos[..., 0]
        sort_key = gt_x.clone()
        sort_key[~mask] = float('inf')
        
        _, sort_indices = torch.sort(sort_key, dim=-1)
        
        # Reorder GT and Mask
        idx_exp = sort_indices.unsqueeze(-1).expand(-1, -1, -1, 3)
        sorted_gt = torch.gather(gt_pos, dim=2, index=idx_exp)
        sorted_mask = torch.gather(mask, dim=2, index=sort_indices)
        
        return CS2Loss.position(pred_x, pred_y, pred_z, sorted_gt, sorted_mask)

    @staticmethod
    def round_state(pred, gt_byte):
        # Multi-label BCE
        bits = torch.arange(5, device=pred.device)
        gt = (gt_byte.unsqueeze(-1) >> bits) & 1
        return F.binary_cross_entropy_with_logits(pred, gt.to(dtype=pred.dtype))

    @staticmethod
    def round_number(pred, gt_int):
        # MSE
        target = gt_int.float() / 30.0
        target = torch.clamp(target, 0.0, 1.0) 
        return F.mse_loss(torch.sigmoid(pred).squeeze(-1), target.to(dtype=pred.dtype))

    @staticmethod
    def alive_count(pred, mask):
        # CrossEntropy (Class 0-5)
        count = mask.long().sum(dim=-1)
        return F.cross_entropy(pred.view(-1, 6), count.view(-1))