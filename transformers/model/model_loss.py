"""
model_loss.py
------------------------------------------------------------------------
Contains the ModelPrediction dataclass, the learnable AutomaticWeightedLoss,
and the specific mathematical implementations for CS2 stats.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
from dataclasses import dataclass
from typing import Dict, Tuple, Any

from model_novibe import ModelPrediction

class AutomaticWeightedLoss(nn.Module):
    """
    Automatically weighs multiple loss terms using uncertainty weighting 
    (Kendall & Gal, CVPR 2018).
    
    The model learns 'log_variance' parameters. High variance = Low Weight.
    """
    def __init__(self, num_losses: int):
        super().__init__()
        # Initialize to 0.0 (Variance = 1.0, Weight = 0.5)
        self.params = nn.Parameter(torch.zeros(num_losses))

    def forward(self, loss_list: list[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Stack losses into [N]
        losses = torch.stack(loss_list)
        
        # Precision = 1 / (2 * sigma^2)
        precision = 0.5 * torch.exp(-self.params)
        
        # Loss = precision * loss + log(sigma)
        # Note: 0.5 * params is equivalent to log(sigma) since params = log(sigma^2)
        weighted_losses = (precision * losses) + (0.5 * self.params)
        
        # Return total loss and the raw scalar weights for logging
        total_loss = weighted_losses.sum()
        
        # Calculate human-readable weights (1 / 2*sigma^2) for debugging
        current_weights = precision.detach() # No clone needed usually, but detach is crucial
        
        return total_loss, current_weights


class CS2Loss(nn.Module):
    # Map constants
    MAP_MIN = torch.tensor([-4000.0, -4000.0, -500.0])
    MAP_MAX = torch.tensor([ 4000.0,  4000.0,  2000.0])
    MAP_SIZE = MAP_MAX - MAP_MIN
    BINS_X = 256
    BINS_Y = 256
    BINS_Z = 32

    def __init__(self):
        super().__init__()
        # We group losses into 6 distinct categories for weighting
        # 0: Mouse, 1: Keyboard, 2: Eco/Items, 3: Stats, 4: Position, 5: Global
        self.awl = AutomaticWeightedLoss(num_losses=6)

    def forward(self, preds: ModelPrediction, truth: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates all losses, weights them, and returns Total Loss + Metrics Dict.
        """
        # --- 1. Compute Raw Losses ---
        
        # Group 0: Mouse
        l_mouse = self.mouse(preds.mouse_delta, truth.mouse_delta, truth.alive_mask)
        
        # Group 1: Keyboard
        l_key = self.keyboard(preds.keyboard_logits, truth.keyboard_mask, truth.alive_mask)
        
        # Group 2: Economy & Items
        l_eco = self.eco(preds.eco_logits, truth.eco_mask, truth.alive_mask)
        l_inv = self.inventory(preds.inventory_logits, truth.inventory_mask, truth.alive_mask)
        l_wep = self.weapon(preds.weapon_logits, truth.active_weapon_idx, truth.alive_mask)
        l_group_eco = l_eco + l_inv + l_wep

        # Group 3: Stats
        l_stats = self.stats(preds.stats_logits, truth.stats, truth.alive_mask)
        
        # Group 4: Position (Player + Enemy)
        l_p_pos = self.position(
            preds.player_pos_x, preds.player_pos_y, preds.player_pos_z, 
            truth.position, truth.alive_mask
        )
        l_e_pos = self.enemy_position(
            preds.enemy_pos_x, preds.enemy_pos_y, preds.enemy_pos_z, 
            truth.enemy_positions, truth.enemy_alive_mask
        )
        l_group_pos = l_p_pos + l_e_pos

        # Group 5: Global State
        l_rnd_st = self.round_state(preds.round_state_logits, truth.round_state_mask)
        l_rnd_nm = self.round_number(preds.round_num_logit, truth.round_number)
        l_t_alv  = self.alive_count(preds.team_alive_logits, truth.alive_mask)
        l_e_alv  = self.alive_count(preds.enemy_alive_logits, truth.enemy_alive_mask)
        l_group_glob = l_rnd_st + l_rnd_nm + l_t_alv + l_e_alv

        # --- 2. Apply Automatic Weighting ---
        
        # Must be in specific order matching __init__ logic
        loss_list = [l_mouse, l_key, l_group_eco, l_stats, l_group_pos, l_group_glob]
        
        total_loss, weights = self.awl(loss_list)

        # --- 3. Construct Metrics for Logging ---
        metrics = {
            "loss/total": total_loss.item(),
            
            # --- High Level Groups (Used for weighting) ---
            "raw/mouse": l_mouse.item(),
            "raw/keyboard": l_key.item(),
            "raw/eco_group": l_group_eco.item(),
            "raw/stats": l_stats.item(),
            "raw/pos_group": l_group_pos.item(),
            "raw/global_group": l_group_glob.item(),
            
            # --- Granular Debugging (Every single loss term) ---
            "debug/mouse": l_mouse.item(),
            "debug/keyboard": l_key.item(),
            
            "debug/eco_bitmask": l_eco.item(),
            "debug/inventory_bitmask": l_inv.item(),
            "debug/active_weapon": l_wep.item(),
            
            "debug/stats": l_stats.item(),
            
            "debug/player_pos": l_p_pos.item(),
            "debug/enemy_pos": l_e_pos.item(),
            
            "debug/round_state": l_rnd_st.item(),
            "debug/round_num": l_rnd_nm.item(),
            "debug/team_alive_count": l_t_alv.item(),
            "debug/enemy_alive_count": l_e_alv.item(),
            
            # --- Learned Uncertainties ---
            "sigma/mouse": weights[0].item(),
            "sigma/key":   weights[1].item(),
            "sigma/eco":   weights[2].item(),
            "sigma/stats": weights[3].item(),
            "sigma/pos":   weights[4].item(),
            "sigma/glob":  weights[5].item(),
        }

        return total_loss, metrics

    # ==========================================================================
    # STATIC MATH IMPLEMENTATIONS
    # ==========================================================================

    @staticmethod
    def mouse(pred, gt, mask):
        # Huber Loss for robustness against outliers (flicks)
        m_flat = mask.view(-1)
        p_flat = pred.view(-1, 2)[m_flat]
        g_flat = gt.view(-1, 2)[m_flat]
        if p_flat.shape[0] == 0: return torch.tensor(0.0, device=pred.device)
        return F.huber_loss(p_flat, g_flat, delta=1.0)

    @staticmethod
    def keyboard(pred, gt, mask):
        # Focal Loss (Multi-label, sparse)
        # Expand 32-bit int mask
        gt_t = (gt.unsqueeze(-1) >> torch.arange(32, device=pred.device)) & 1
        m_flat = mask.view(-1)
        p_flat = pred.view(-1, 32)[m_flat]
        g_flat = gt_t.float().view(-1, 32)[m_flat]
        if p_flat.shape[0] == 0: return torch.tensor(0.0, device=pred.device)
        return sigmoid_focal_loss(p_flat, g_flat, alpha=0.25, gamma=2.0, reduction='mean')

    @staticmethod
    def _unpack_chunks(chunks, num_chunks):
        # Helper for Eco/Inv
        bits = torch.arange(64, device=chunks.device)
        unpacked = []
        for i in range(num_chunks):
            expanded = (chunks[..., i].unsqueeze(-1) >> bits) & 1
            unpacked.append(expanded)
        return torch.cat(unpacked, dim=-1).float()

    @staticmethod
    def eco(pred, gt_chunks, mask):
        # Eco: 4 chunks (256 bits), Focal Loss
        gt_t = CS2Loss._unpack_chunks(gt_chunks, 4)
        m_flat = mask.view(-1)
        p_flat = pred.view(-1, 256)[m_flat]
        g_flat = gt_t.view(-1, 256)[m_flat]
        if p_flat.shape[0] == 0: return torch.tensor(0.0, device=pred.device)
        return sigmoid_focal_loss(p_flat, g_flat, alpha=0.25, gamma=2.0, reduction='mean')

    @staticmethod
    def inventory(pred, gt_chunks, mask):
        # Inv: 2 chunks (128 bits), Focal Loss
        gt_t = CS2Loss._unpack_chunks(gt_chunks, 2)
        m_flat = mask.view(-1)
        p_flat = pred.view(-1, 128)[m_flat]
        g_flat = gt_t.view(-1, 128)[m_flat]
        if p_flat.shape[0] == 0: return torch.tensor(0.0, device=pred.device)
        return sigmoid_focal_loss(p_flat, g_flat, alpha=0.25, gamma=2.0, reduction='mean')

    @staticmethod
    def weapon(pred, gt_idx, mask):
        # CrossEntropy (Single Class, Mutually Exclusive)
        m_flat = mask.view(-1)
        p_flat = pred.view(-1, 128)[m_flat]
        g_flat = gt_idx.view(-1)[m_flat].long()
        if p_flat.shape[0] == 0: return torch.tensor(0.0, device=pred.device)
        return F.cross_entropy(p_flat, g_flat, ignore_index=-1)

    @staticmethod
    def stats(pred, gt, mask):
        # MSE with Normalization
        # gt: [Health, Armor, Money]
        target = torch.stack([
            gt[..., 0] / 100.0, 
            gt[..., 1] / 100.0, 
            gt[..., 2] / 16000.0
        ], dim=-1)
        
        m_flat = mask.view(-1)
        p_flat = torch.sigmoid(pred).view(-1, 3)[m_flat]
        g_flat = target.view(-1, 3)[m_flat]
        if p_flat.shape[0] == 0: return torch.tensor(0.0, device=pred.device)
        return F.mse_loss(p_flat, g_flat)

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

        if px.shape[0] == 0: return torch.tensor(0.0, device=pred_x.device)
        
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
        return F.binary_cross_entropy_with_logits(pred, gt.float())

    @staticmethod
    def round_number(pred, gt_int):
        # MSE
        target = gt_int.float() / 30.0
        target = torch.clamp(target, 0.0, 1.0) 
        return F.mse_loss(torch.sigmoid(pred).squeeze(-1), target)

    @staticmethod
    def alive_count(pred, mask):
        # CrossEntropy (Class 0-5)
        count = mask.long().sum(dim=-1)
        return F.cross_entropy(pred.view(-1, 6), count.view(-1))