import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss

import dataset

# Maybe it works? Otherwise go back to manual weighting
class AutomaticWeightedLoss(nn.Module):
    """
    Automatically weighs multiple loss terms using uncertainty weighting 
    (Kendall & Gal, CVPR 2018).
    
    Effectively learns the 'noise' of each task. If a task is hard/noisy, 
    the model learns to increase the variance (sigma), which reduces the 
    weight of that loss term.
    """
    def __init__(self, num_losses: int):
        super().__init__()
        # We initialize log_variance to 0.0 (which means variance=1.0, weight=1.0)
        # We use Parameter so the optimizer trains these alongside the model.
        self.params = nn.Parameter(torch.zeros(num_losses))

    def forward(self, loss_list):
        """
        Args:
            loss_list: A list or tuple of scalar tensors [L1, L2, ..., Ln]
        Returns:
            total_loss: Scalar sum of weighted losses
            weights: The calculated weights (for logging)
        """
        # Stack losses into a tensor [N]
        losses = torch.stack(loss_list)
        
        # 1. Calculate Precision (1 / (2 * sigma^2))
        # We use exp(-p) to ensure the precision is always positive.
        # The '0.5' comes from the Gaussian likelihood formulation.
        precision = 0.5 * torch.exp(-self.params)
        
        # 2. Weighted Loss: precision * loss + log_term
        # The log term prevents the model from setting precision to 0 (infinity sigma).
        # We use 0.5 * self.params because self.params is log(sigma^2)
        weighted_losses = (precision * losses) + (0.5 * self.params)
        
        return weighted_losses.sum()

class ModelLoss:
    # Huber loss for mouse
    def mouse(pred_mouse, gt_mouse, alive_mask):
        """
        Args:
            pred_mouse: [B, T, 5, 2] - Model output
            gt_mouse:   [B, T, 5, 2] - Ground Truth from dataset
            alive_mask: [B, T, 5]    - Boolean mask
        """
        mask_flat = alive_mask.view(-1)
        pred_flat = pred_mouse.view(-1, 2)[mask_flat]
        gt_flat   = gt_mouse.view(-1, 2)[mask_flat]

        if pred_flat.shape[0] == 0:
            return torch.tensor(0.0, device=pred_mouse.device)
        
        loss = F.huber_loss(pred_flat, gt_flat, delta=1.0, reduction='mean')
        return loss

    # Focal loss between 0 and 1 for keyboard
    def keyboard(pred_logits, gt_bitmask, alive_mask):
        """
        Args:
            pred_logits: [B, T, 5, 32] - Raw model outputs (no sigmoid applied yet)
            gt_bitmask:  [B, T, 5]     - Int32 bitmask
            alive_mask:  [B, T, 5]     - Boolean mask
        """
        gt_targets = (gt_bitmask.unsqueeze(-1) >> torch.arange(32, device=pred_logits.device)) & 1
        gt_targets = gt_targets.float()

        # 2. Flatten based on alive mask
        mask_flat = alive_mask.view(-1)
        pred_flat = pred_logits.view(-1, 32)[mask_flat]
        gt_flat   = gt_targets.view(-1, 32)[mask_flat]

        if pred_flat.shape[0] == 0:
            return torch.tensor(0.0, device=pred_logits.device)

        loss = sigmoid_focal_loss(
            inputs=pred_flat,
            targets=gt_flat,
            alpha=0.25, 
            gamma=2.0, 
            reduction='mean'
        )
        
        return loss

    # Helper
    def unpack_64bit_n(chunks):
        device = chunks.device
        unpacked_parts = []
        bits = torch.arange(64, device=device)
        num = chunks.shape[-1]
        
        for i in range(num):
            chunk = chunks[..., i]  # Shape: [B, T, 5]
            expanded = (chunk.unsqueeze(-1) >> bits) & 1  # Shape: [B, T, 5, 64]
            unpacked_parts.append(expanded)
        
        full_mask = torch.cat(unpacked_parts, dim=-1)  # Shape: [B, T, 5, 64*i]
        return full_mask.float()

    # Focal loss for buy/sell/drop actions
    def eco(pred_logits, gt_eco_chunks, alive_mask):
        """
        Args:
            pred_logits:   [B, T, 5, 256] - Model output
            gt_eco_chunks: [B, T, 5, 4]   - Compressed Ground Truth
            alive_mask:    [B, T, 5]      - Boolean mask
        """
        gt_targets = unpack_64bit_n(gt_eco_chunks)
        mask_flat = alive_mask.view(-1)
        pred_flat = pred_logits.view(-1, 256)[mask_flat]
        gt_flat   = gt_targets.view(-1, 256)[mask_flat]
        if pred_flat.shape[0] == 0:
            return torch.tensor(0.0, device=pred_logits.device)

        loss = sigmoid_focal_loss(
            inputs=pred_flat, 
            targets=gt_flat, 
            alpha=0.25, 
            gamma=2.0, 
            reduction='mean'
        )
        
        return loss
    
    # Focal loss for inventory items
    def inventory(pred_logits, gt_inventory_chunks, alive_mask):
        """
        Args:
            pred_logits: [B, T, 5, 128] - Model output (128 bits for 2x uint64)
            gt_inventory_chunks: [B, T, 5, 2] - Ground truth chunks
            alive_mask: [B, T, 5]
        """
        gt_targets = unpack_64bit_n(gt_inventory_chunks)
        mask_flat = alive_mask.view(-1)
        pred_flat = pred_logits.view(-1, 128)[mask_flat]
        gt_flat   = gt_targets.view(-1, 128)[mask_flat]

        if pred_flat.shape[0] == 0:
            return torch.tensor(0.0, device=pred_logits.device)

        loss = sigmoid_focal_loss(
            inputs=pred_flat, 
            targets=gt_flat, 
            alpha=0.25, 
            gamma=2.0, 
            reduction='mean'
        )
        
        return loss

    # Cross Entropy for active weapon (only one)
    def weapon(pred_logits, gt_idx, alive_mask):
        """
        Args:
            pred_logits: [B, T, 5, 128] - Model output (one logit per item ID)
            gt_idx:      [B, T, 5]      - Class indices (ints), contains -1
            alive_mask:  [B, T, 5]      - Boolean mask
        """
        mask_flat = alive_mask.view(-1)
        pred_flat = pred_logits.view(-1, 128)[mask_flat]
        gt_flat = gt_idx.view(-1)[mask_flat].long()

        if pred_flat.shape[0] == 0:
            return torch.tensor(0.0, device=pred_logits.device)

        loss = F.cross_entropy(
            pred_flat, 
            gt_flat, 
            ignore_index=-1, 
            reduction='mean'
        )

        return loss
    
    # MSE loss for health, armor and money
    def stats(pred_logits, gt_stats, alive_mask):
        """
        Args:
            pred_logits: [B, T, 5, 3] - Model raw outputs (no activation applied yet)
            gt_stats:    [B, T, 5, 3] - Ground truth floats (Health, Armor, Money)
            alive_mask:  [B, T, 5]    - Boolean mask
        """
        gt_health = gt_stats[..., 0]
        gt_armor  = gt_stats[..., 1]
        gt_money  = gt_stats[..., 2]

        target_health = gt_health / 100.0
        target_armor  = gt_armor  / 100.0
        target_money  = gt_money  / 16000.0

        targets_normalized = torch.stack([target_health, target_armor, target_money], dim=-1)
        preds_normalized = torch.sigmoid(pred_logits)
        mask_flat = alive_mask.view(-1)
        
        pred_flat = preds_normalized.view(-1, 3)[mask_flat]
        target_flat = targets_normalized.view(-1, 3)[mask_flat]

        if pred_flat.shape[0] == 0:
            return torch.tensor(0.0, device=pred_logits.device)

        loss = F.mse_loss(pred_flat, target_flat, reduction='mean')

        return loss
    
    MAP_MIN = torch.tensor([-4000.0, -4000.0, -500.0])
    MAP_MAX = torch.tensor([ 4000.0,  4000.0,  2000.0])
    MAP_SIZE = MAP_MAX - MAP_MIN
    BINS_X = 256
    BINS_Y = 256
    BINS_Z = 32

    def positions_to_bins(pos_tensor):
        device = pos_tensor.device
        min_b = MAP_MIN.to(device)
        size_b = MAP_SIZE.to(device)

        # 1. Normalize to [0, 1]
        norm = (pos_tensor - min_b) / size_b
        norm = torch.clamp(norm, 0.0, 0.999) # Prevent overflow at 1.0

        # 2. Scale to bin counts
        # We construct a tensor of [256, 256, 32] to multiply against
        bins_tensor = torch.tensor([BINS_X, BINS_Y, BINS_Z], device=device)

        indices = (norm * bins_tensor).long()
        return indices

    def position(pred_x, pred_y, pred_z, gt_pos, alive_mask):
        """
        Args:
            pred_x, pred_y: [B, T, 5, 256] (We slice to get just the player)
            pred_z:         [B, T, 5, 32]
            gt_pos:         [B, T, 5, 3]
            alive_mask:     [B, T, 5]
        """
        target_indices = positions_to_bins(gt_pos)
        target_x = target_indices[..., 0]
        target_y = target_indices[..., 1]
        target_z = target_indices[..., 2]

        mask_flat = alive_mask.view(-1)
        
        p_x_flat = pred_x.view(-1, BINS_X)[mask_flat]
        p_y_flat = pred_y.view(-1, BINS_Y)[mask_flat]
        p_z_flat = pred_z.view(-1, BINS_Z)[mask_flat]
        
        t_x_flat = target_x.view(-1)[mask_flat]
        t_y_flat = target_y.view(-1)[mask_flat]
        t_z_flat = target_z.view(-1)[mask_flat]

        if p_x_flat.shape[0] == 0:
            return torch.tensor(0.0, device=pred_x.device)

        loss_x = F.cross_entropy(p_x_flat, t_x_flat, label_smoothing=0.1)
        loss_y = F.cross_entropy(p_y_flat, t_y_flat, label_smoothing=0.1)
        loss_z = F.cross_entropy(p_z_flat, t_z_flat, label_smoothing=0.1)
        
        return loss_x + loss_y + loss_z

    def enemy_position(pred_x, pred_y, pred_z, gt_pos, alive_mask):
        gt_x = gt_pos[..., 0]
        sort_key = gt_x.clone()
        sort_key[~alive_mask] = float('inf') 

        _, sort_indices = torch.sort(sort_key, dim=-1)
        idx_expanded = sort_indices.unsqueeze(-1).expand(-1, -1, -1, 3)
        
        sorted_gt_pos = torch.gather(gt_pos, dim=2, index=idx_expanded)
        sorted_mask = torch.gather(alive_mask, dim=2, index=sort_indices)

        return position(pred_x, pred_y, pred_z, sorted_gt_pos, sorted_mask)

    def round_state(pred_logits, gt_state_byte):
    """
    Args:
        pred_logits: [B, T, 5] - Logits for (Freeze, Active, Plant, T_Win, CT_Win)
        gt_state_byte: [B, T]  - uint8 Tensor
    """
    bits = torch.arange(5, device=pred_logits.device)
    gt_targets = (gt_state_byte.unsqueeze(-1) >> bits) & 1
    return F.binary_cross_entropy_with_logits(
        pred_logits, 
        gt_targets.float()
    )