#!/usr/bin/env python3
"""
visualize_inference.py

Loads a trained model checkpoint, runs inference on validation data,
and produces a side-by-side visualization video of GT vs Prediction.
"""

import os
import cv2
import torch
import numpy as np
import argparse
import logging
from pathlib import Path

from dataclasses import dataclass, fields, is_dataclass
from typing import List, Any

from accelerate import Accelerator, FullyShardedDataParallelPlugin

from config import GlobalConfig, DatasetConfig, ModelConfig, TrainConfig
from dataset import DatasetRoot, GroundTruth, TrainingSample

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

from model import GamePredictorBackbone, ModelPrediction
from model_loss import mu_law_decode, unbin_value, mu_law_encode, bin_value

# =============================================================================
# MAPPINGS (Must match injection_mold.py exactly)
# =============================================================================

ITEM_NAMES = sorted(list(set([
    "Desert Eagle", "Dual Berettas", "Five-SeveN", "Glock-18", "AK-47",
    "AUG", "AWP", "FAMAS", "G3SG1", "Galil AR", "M249", "M4A4", "MAC-10",
    "P90", "MP5-SD", "UMP-45", "XM1014", "PP-Bizon", "MAG-7", "Negev",
    "Sawed-Off", "Tec-9", "Zeus x27", "P2000", "MP7", "MP9", "Nova",
    "P250", "SCAR-20", "SG 553", "SSG 08", "Knife", "knife", "Flashbang",
    "High Explosive Grenade", "Smoke Grenade", "Molotov", "Decoy Grenade",
    "Incendiary Grenade", "C4 Explosive", "Kevlar Vest", "Kevlar & Helmet",
    "Heavy Assault Suit", "item_nvg", "Defuse Kit", "Rescue Kit",
    "Medi-Shot", "knife_t", "M4A1-S", "USP-S", "Trade Up Contract",
    "CZ75-Auto", "R8 Revolver", "Charm Detachments", "Bayonet", "Classic Knife",
    "Flip Knife", "Gut Knife", "Karambit", "M9 Bayonet", "Huntsman Knife",
    "Falchion Knife", "Bowie Knife", "Butterfly Knife", "Shadow Daggers",
    "Paracord Knife", "Survival Knife", "Ursus Knife", "Navaja Knife",
    "Nomad Knife", "Stiletto Knife", "Talon Knife", "Skeleton Knife", "Kukri Knife"
])))

KEYBOARD_ONLY_ACTIONS = [
    "IN_ATTACK", "IN_JUMP", "IN_DUCK", "IN_FORWARD", "IN_BACK", "IN_USE", "IN_CANCEL", "IN_TURNLEFT", 
    "IN_TURNRIGHT", "IN_MOVELEFT", "IN_MOVERIGHT", "IN_ATTACK2", "IN_RELOAD", "IN_ALT1", "IN_ALT2", 
    "IN_SPEED", "IN_WALK", "IN_ZOOM", "IN_WEAPON1", "IN_WEAPON2", "IN_BULLRUSH", "IN_GRENADE1", 
    "IN_GRENADE2", "IN_ATTACK3", "IN_SCORE", "IN_INSPECT", "SWITCH_1", "SWITCH_2", "SWITCH_3", "SWITCH_4", "SWITCH_5"]

# =============================================================================
# DECODING HELPERS (Strings)
# =============================================================================

def decode_keyboard(mask: int) -> str:
    """Decodes uint32 keyboard mask into a short string."""
    active = []
    # Common movement keys for concise display
    move_map = {
        "IN_FORWARD": "W", "IN_BACK": "S", "IN_MOVELEFT": "A", "IN_MOVERIGHT": "D",
        "IN_JUMP": "JUMP", "IN_DUCK": "DUCK", "IN_WALK": "WALK", "IN_ATTACK": "ATK1",
        "IN_ATTACK2": "ATK2", "IN_RELOAD": "R", "IN_USE": "E", "IN_SCORE": "TAB"
    }
    
    for i, action in enumerate(KEYBOARD_ONLY_ACTIONS):
        if (mask >> i) & 1:
            active.append(move_map.get(action, action.replace("IN_", "")))
    return " ".join(active)

def decode_round_state(mask: int) -> str:
    states = []
    if (mask >> 0) & 1: states.append("FREEZE")
    if (mask >> 1) & 1: states.append("LIVE")
    if (mask >> 2) & 1: states.append("PLANT")
    if (mask >> 3) & 1: states.append("T_WIN")
    if (mask >> 4) & 1: states.append("CT_WIN")
    return "|".join(states) if states else "NONE"

def convert_gt_to_viz(gt: GroundTruth, t: int, batch_idx: int = 0, cfg: ModelConfig = None) -> dict:
    """
    Extracts GT data for batch element `batch_idx` at time `t`.
    """
    gs = {
        'tick': t * 2,
        'round_state': int(gt.round_state_mask[batch_idx, t].item()),
        'team_alive': 0,
        'enemy_alive': 0,
        'enemy_pos': gt.enemy_positions[batch_idx, t].tolist()
    }

    pd = []
    team_alive_mask = 0;
    enemy_alive_mask = 0;
    alive_t = gt.alive_mask[batch_idx, t];
    alive_e = gt.enemy_alive_mask[batch_idx, t];

    for p in range(5):
        if alive_t[p].item(): team_alive_mask |= (1 << p);
        if alive_e[p].item(): enemy_alive_mask |= (1 << p);
        
        stats = gt.stats[batch_idx, t, p].tolist();
        pos = gt.position[batch_idx, t, p];
        mouse = gt.mouse_delta[batch_idx, t, p];
        
        # Calculate Bins for GT if cfg is provided
        m_bins = [None, None];
        p_bins = [None, None, None];
        if cfg:
            m_bins = [
                mu_law_encode(mouse[0], cfg.mouse_mu, cfg.mouse_max, cfg.mouse_bins_count).item(),
                mu_law_encode(mouse[1], cfg.mouse_mu, cfg.mouse_max, cfg.mouse_bins_count).item()
            ];
            PMIN, PMAX = -4096.0, 4096.0;
            p_bins = [
                bin_value(pos[0], PMIN, PMAX, cfg.bins_x).item(),
                bin_value(pos[1], PMIN, PMAX, cfg.bins_y).item(),
                bin_value(pos[2], PMIN, PMAX, cfg.bins_z).item()
            ];

        p_dict = {
            'health': int(stats[0]),
            'armor': int(stats[1]),
            'money': int(stats[2]),
            'pos': pos.tolist(),
            'pos_bins': p_bins,
            'mouse': mouse.tolist(),
            'mouse_bins': m_bins,
            'keyboard_bitmask': int(gt.keyboard_mask[batch_idx, t, p].item()),
            'active_weapon_idx': int(gt.active_weapon_idx[batch_idx, t, p].item())
        };
        pd.append(p_dict);
        
    gs['team_alive'] = team_alive_mask;
    gs['enemy_alive'] = enemy_alive_mask;
    return {'game_state': gs, 'player_data': pd}

def convert_pred_to_viz(pred: ModelPrediction, t: int, batch_idx: int = 0, cfg: ModelConfig = None) -> dict:
    """
    Converts raw logits from ModelPrediction to visualized values.
    """
    rs_logits = pred.round_state_logits[batch_idx, t, 0]
    rs_idx = torch.argmax(rs_logits).item()
    
    ta_logits = pred.team_alive_logits[batch_idx, t, 0]
    ea_logits = pred.enemy_alive_logits[batch_idx, t, 0]
    ta_count = torch.argmax(ta_logits).item()
    ea_count = torch.argmax(ea_logits).item()
    
    e_pos = []
    PMIN, PMAX = -4096.0, 4096.0
    for e_idx in range(5):
        bx = torch.argmax(pred.enemy_pos_x[batch_idx, t, e_idx])
        by = torch.argmax(pred.enemy_pos_y[batch_idx, t, e_idx])
        bz = torch.argmax(pred.enemy_pos_z[batch_idx, t, e_idx])
        ex = unbin_value(bx, PMIN, PMAX, cfg.bins_x).item()
        ey = unbin_value(by, PMIN, PMAX, cfg.bins_y).item()
        ez = unbin_value(bz, PMIN, PMAX, cfg.bins_z).item()
        e_pos.append([ex, ey, ez])

    gs = {
        'tick': t * 2,
        'round_state': rs_idx,
        'team_alive_count': ta_count,
        'enemy_alive_count': ea_count,
        'enemy_pos': e_pos
    }

    pd = []
    for p in range(5):
        h_bin = torch.argmax(pred.health_logits[batch_idx, t, p])
        a_bin = torch.argmax(pred.armor_logits[batch_idx, t, p])
        m_bin = torch.argmax(pred.money_logits[batch_idx, t, p])
        
        h = unbin_value(h_bin, 0, 100, cfg.health_bins).item()
        a = unbin_value(a_bin, 0, 100, cfg.armor_bins).item()
        m = unbin_value(m_bin, 0, 16000, cfg.money_bins).item()
        
        pbx = torch.argmax(pred.player_pos_x[batch_idx, t, p])
        pby = torch.argmax(pred.player_pos_y[batch_idx, t, p])
        pbz = torch.argmax(pred.player_pos_z[batch_idx, t, p])
        
        px = unbin_value(pbx, PMIN, PMAX, cfg.bins_x).item()
        py = unbin_value(pby, PMIN, PMAX, cfg.bins_y).item()
        pz = unbin_value(pbz, PMIN, PMAX, cfg.bins_z).item()
        
        mbx = torch.argmax(pred.mouse_x[batch_idx, t, p])
        mby = torch.argmax(pred.mouse_y[batch_idx, t, p])
        
        mx = mu_law_decode(mbx, cfg.mouse_mu, cfg.mouse_max, cfg.mouse_bins_count).item()
        my = mu_law_decode(mby, cfg.mouse_mu, cfg.mouse_max, cfg.mouse_bins_count).item()
        
        w_idx = torch.argmax(pred.active_weapon_logits[batch_idx, t, p]).item()
        
        k_logits = pred.keyboard_logits[batch_idx, t, p]
        k_mask = 0
        for k_i in range(32):
            if torch.sigmoid(k_logits[k_i]) > 0.5:
                k_mask |= (1 << k_i)
                
        p_dict = {
            'health': int(h),
            'armor': int(a),
            'money': int(m),
            'pos': [px, py, pz],
            'pos_bins': [pbx.item(), pby.item(), pbz.item()],
            'mouse': [mx, my],
            'mouse_bins': [mbx.item(), mby.item()],
            'keyboard_bitmask': k_mask,
            'active_weapon_idx': w_idx
        }
        pd.append(p_dict)
        
    return {'game_state': gs, 'player_data': pd}

# =============================================================================
# RENDERING
# =============================================================================

COLORS = {
    'gt': (0, 255, 0),       # Green
    'pred': (0, 0, 255),     # Red
    'text_bg': (0, 0, 0),    # Black outline
    'white': (255, 255, 255)
}

def draw_text_lines(img, lines, x, y, color, scale=0.4, thickness=1):
    line_height = int(20 * (scale / 0.4))
    for i, line in enumerate(lines):
        curr_y = y + (i * line_height)
        cv2.putText(img, line, (x, curr_y), cv2.FONT_HERSHEY_SIMPLEX, scale, COLORS['text_bg'], thickness + 2, cv2.LINE_AA)
        cv2.putText(img, line, (x, curr_y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return y + (len(lines) * line_height)

def render_player_panel(frame, gt_data, pred_data, player_idx=0):
    h, w, _ = frame.shape
    display_frame = frame.copy()
    
    # Helper for lines
    def get_lines(data, prefix=""):
        hp, arm, mon = data['health'], data['armor'], data['money']
        idx = data.get('active_weapon_idx', -1)
        wep_str = ITEM_NAMES[idx] if 0 <= idx < len(ITEM_NAMES) else "Knife/None"
        pos = data['pos']
        mouse = data['mouse']
        keys = decode_keyboard(data['keyboard_bitmask'])
        
        # Get Bins if available (for Pred)
        mouse_bins = data.get('mouse_bins', [None, None])
        pos_bins = data.get('pos_bins', [None, None, None])
        
        m_str = f"{mouse[0]:.1f}, {mouse[1]:.1f}"
        if mouse_bins[0] is not None:
            m_str += f" (bins: {mouse_bins[0]},{mouse_bins[1]})"
            
        p_str = f"{pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}"
        if pos_bins[0] is not None:
            p_str += f" (bins: {pos_bins[0]},{pos_bins[1]},{pos_bins[2]})"

        return [
            f"{prefix} P{player_idx}: HP {hp} | AP {arm} | ${mon}",
            f"   Wep: {wep_str}",
            f"   Pos: {p_str}",
            f"   Aim: {m_str}",
            f"   Key: {keys}"
        ]

    # Draw GT
    gt_lines = get_lines(gt_data, "GT")
    next_y = draw_text_lines(display_frame, gt_lines, 10, 20, COLORS['gt'])
    
    # Draw Pred
    if pred_data:
        pred_lines = get_lines(pred_data, "PR")
        draw_text_lines(display_frame, pred_lines, 10, next_y + 5, COLORS['pred'])
        
    return display_frame

def render_global_panel(h, w, gt_gs, pred_gs):
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(panel, "GAME STATE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['white'], 2)
    
    # GT Lines
    r_state = decode_round_state(gt_gs['round_state'])
    t_alive = gt_gs.get('team_alive', 0)
    e_alive = gt_gs.get('enemy_alive', 0)
    
    gt_lines = [
        f"Tick: {gt_gs.get('tick', '?')}",
        f"State: {r_state}",
        f"T Alive Mask: {bin(t_alive)[2:].zfill(5)}",
        f"E Alive Mask: {bin(e_alive)[2:].zfill(5)}",
        "Enemies (GT):"
    ]
    for i, pos in enumerate(gt_gs['enemy_pos']):
        if (e_alive >> i) & 1:
             gt_lines.append(f" E{i}: {pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}")

    next_y = draw_text_lines(panel, gt_lines, 10, 60, COLORS['gt'], scale=0.5)
    
    # Pred Lines
    if pred_gs:
        pr_state = decode_round_state(pred_gs['round_state'])
        pred_lines = [
            "",
            "Prediction:",
            f"State: {pr_state}",
            f"T Count: {pred_gs['team_alive_count']}",
            f"E Count: {pred_gs['enemy_alive_count']}",
            "Enemies (Pred):"
        ]
        for i, pos in enumerate(pred_gs['enemy_pos']):
            pred_lines.append(f" E{i}: {pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}")
            
        draw_text_lines(panel, pred_lines, 10, next_y + 10, COLORS['pred'], scale=0.5)

    return panel

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--data_root", type=str, default="./dataset0")
    parser.add_argument("--output", type=str, default="inference_vis.mp4")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # 1. Load Config
    # Assuming config is in the checkpoint folder or we construct default
    config_path = os.path.join(args.checkpoint, "config.json")
    if os.path.exists(config_path):
        global_cfg = GlobalConfig.from_file(config_path)
    else:
        print("Warning: config.json not found in checkpoint, using defaults.")
        d_cfg = DatasetConfig(data_root=args.data_root, run_dir="./runs")
        m_cfg = ModelConfig()
        t_cfg = TrainConfig()
        global_cfg = GlobalConfig(d_cfg, m_cfg, t_cfg)
        
    # Override data root if provided
    global_cfg.dataset.data_root = args.data_root
    
    # 2. Model
    model = GamePredictorBackbone(global_cfg.model)
    
    fsdp_plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
    )
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model, optimizer = accelerator.prepare(model, optimizer)
    
    accelerator.load_state(args.checkpoint)
    model.eval()
    
    # 3. Data
    ds_root = DatasetRoot(global_cfg.dataset)
    val_ds = ds_root.build_epoch("val", 0) # epoch 0
    if len(val_ds) == 0:
        print("Validation set empty! Falling back to train.")
        val_ds = ds_root.build_epoch("train", 0)
        
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, collate_fn=cs2_collate_fn, shuffle=True
    )
    
    # 4. Loop
    print(f"Processing {args.num_samples} samples...")
    
    # Setup Video
    writer = None
    
    with torch.no_grad():
        for i, sample in enumerate(loader):
            if i >= args.num_samples: break
            
            # Move to device
            images = sample.images.to(accelerator.device) # [B, T, 5, C, H, W]
            audio = sample.audio.to(accelerator.device)
            
            # Inference
            preds_dict = model(images, audio)
            preds = ModelPrediction(**preds_dict)
            
            # Render Loop
            B, T, P, C, H, W = images.shape
            # Assuming B=1
            
            # Init Writer once we know H, W
            if writer is None:
                grid_w, grid_h = W * 3, H * 2
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.output, fourcc, 32, (grid_w, grid_h)) # 32 FPS hardcoded
            
            # Convert images to uint8 numpy for drawing
            # images is [1, T, 5, C, H, W] (Float or Int?)
            # Dataset returns float normalized? No, it returns what `decoder` gives.
            # `dataset.py` _decode_video returns decoder output. TorchCodec returns uint8 usually?
            # Let's check. If float, scale. If uint8, good.
            # Actually dataset.py pad_or_truncate_to preserves dtype.
            
            imgs_cpu = images.cpu()
            if imgs_cpu.dtype.is_floating_point:
                imgs_cpu = (imgs_cpu * 255).byte()
            else:
                imgs_cpu = imgs_cpu.byte()
                
            for t in range(T):
                # Get Data dicts
                gt_data = convert_gt_to_viz(sample.truth, t, batch_idx=0, cfg=global_cfg.model)
                pred_data = convert_pred_to_viz(preds, t, batch_idx=0, cfg=global_cfg.model)
                
                # Get Frames
                # imgs_cpu: [1, T, 5, C, H, W]
                # Need [5, H, W, C] BGR for opencv
                frames = []
                for p in range(5):
                    # C, H, W -> H, W, C
                    frm = imgs_cpu[0, t, p].permute(1, 2, 0).numpy()
                    frm = cv2.cvtColor(frm, cv2.COLOR_RGB2BGR)
                    frames.append(frm)
                    
                # Render
                # Stitch
                panels = []
                for p in range(5):
                    panels.append(render_player_panel(frames[p], gt_data['player_data'][p], pred_data['player_data'][p], p))
                
                panels.append(render_global_panel(H, W, gt_data['game_state'], pred_data['game_state']))
                
                row1 = np.hstack(panels[0:3])
                row2 = np.hstack(panels[3:6])
                combined = np.vstack([row1, row2])
                
                writer.write(combined)
            
            print(f"Sample {i+1} done.")
            
    if writer:
        writer.release()
    print(f"Video saved to {args.output}")

if __name__ == "__main__":
    main()
