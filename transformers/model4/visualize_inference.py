#!/usr/bin/env python3
"""
visualize_inference.py

Loads a trained PyTorch Lightning model checkpoint, runs inference on validation data,
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

from config import GlobalConfig, DatasetConfig, ModelConfig, TrainConfig
from dataset import DatasetRoot, GroundTruth, TrainingSample

# Import Lightning Module
from lightning_module import CS2PredictorModule
from model import ModelPrediction
from model_loss import mu_law_decode, unbin_value, mu_law_encode, bin_value

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
    mapping = {0: "FREEZE", 1: "LIVE", 2: "PLANT", 3: "T_WIN", 4: "CT_WIN"}
    return mapping.get(mask, f"UNK({mask})")

def convert_gt_to_viz(gt: GroundTruth, t: int, batch_idx: int = 0, cfg: ModelConfig = None) -> dict:
    """
    Extracts GT data for batch element `batch_idx` at time `t`.
    """
    # Global
    r_state = int(gt.round_state_mask[batch_idx, t].item())
    r_num = int(gt.round_number[batch_idx, t].item())
    
    # Alive masks
    alive_t = gt.alive_mask[batch_idx, t]
    alive_e = gt.enemy_alive_mask[batch_idx, t]
    
    team_alive_count = alive_t.sum().item()
    enemy_alive_count = alive_e.sum().item()
    enemy_pos = gt.enemy_positions[batch_idx, t]
    enemy_pos_bins = [encode_position_bins(enemy, cfg) for enemy in enemy_pos] if cfg else [[None, None, None] for _ in range(5)]
    
    gs = {
        'tick': t * 2,
        'round_state': r_state,
        'round_num': r_num,
        'team_alive_count': team_alive_count,
        'enemy_alive_count': enemy_alive_count,
        'team_alive_mask': int(sum((1 << i) for i, alive in enumerate(alive_t.tolist()) if alive)),
        'enemy_alive_mask': int(sum((1 << i) for i, alive in enumerate(alive_e.tolist()) if alive)),
        'enemy_pos': enemy_pos.tolist(),
        'enemy_pos_bins': enemy_pos_bins,
    }

    pd = []

    for p in range(5):
        stats = gt.stats[batch_idx, t, p].tolist()
        pos = gt.position[batch_idx, t, p]
        mouse = gt.mouse_delta[batch_idx, t, p]
        
        m_bins = [None, None]
        p_bins = [None, None, None]
        if cfg:
            m_bins = encode_mouse_bins(mouse, cfg)
            p_bins = encode_position_bins(pos, cfg)

        # Eco
        eco_idx = int(gt.eco_buy_idx[batch_idx, t, p].item())
        # Check purchase bit (logic: if any bit in eco_mask is set, they bought something)
        # But for Viz we usually just want to know if they are pressing buy.
        # Let's rely on eco_buy_idx being valid.
        
        p_dict = {
            'health': int(stats[0]),
            'armor': int(stats[1]),
            'money': int(stats[2]),
            'pos': pos.tolist(),
            'pos_bins': p_bins,
            'mouse': mouse.tolist(),
            'mouse_bins': m_bins,
            'keyboard_bitmask': int(gt.keyboard_mask[batch_idx, t, p].item()),
            'active_weapon_idx': int(gt.active_weapon_idx[batch_idx, t, p].item()),
            'eco_buy_idx': eco_idx,
            'eco_purchase': False # GT doesn't have a direct "buying now" boolean easily accessible without parsing mask, will skip for now or assume idx > -1?
        }
        pd.append(p_dict)
        
    return {'game_state': gs, 'player_data': pd}

def convert_pred_to_viz(pred: ModelPrediction, t: int, batch_idx: int = 0, cfg: ModelConfig = None) -> dict:
    """
    Converts raw logits from ModelPrediction to visualized values.
    """
    # Global
    rs_logits = pred.round_state_logits[batch_idx, t, 0] # [5]
    rs_idx = torch.argmax(rs_logits).item()
    
    rn_logits = pred.round_num_logits[batch_idx, t, 0] # [31]
    rn_idx = torch.argmax(rn_logits).item()
    
    ta_logits = pred.team_alive_logits[batch_idx, t, 0]
    ea_logits = pred.enemy_alive_logits[batch_idx, t, 0]
    ta_count = torch.argmax(ta_logits).item()
    ea_count = torch.argmax(ea_logits).item()
    
    e_pos = []
    enemy_pos_bins = []
    for e_idx in range(5):
        bx = torch.argmax(pred.enemy_pos_x[batch_idx, t, e_idx])
        by = torch.argmax(pred.enemy_pos_y[batch_idx, t, e_idx])
        bz = torch.argmax(pred.enemy_pos_z[batch_idx, t, e_idx])
        ex = unbin_value(bx, POSITION_MIN, POSITION_MAX, cfg.bins_x).item()
        ey = unbin_value(by, POSITION_MIN, POSITION_MAX, cfg.bins_y).item()
        ez = unbin_value(bz, POSITION_MIN, POSITION_MAX, cfg.bins_z).item()
        e_pos.append([ex, ey, ez])
        enemy_pos_bins.append([bx.item(), by.item(), bz.item()])

    gs = {
        'tick': t * 2,
        'round_state': rs_idx,
        'round_num': rn_idx,
        'team_alive_count': ta_count,
        'enemy_alive_count': ea_count,
        'enemy_pos': e_pos,
        'enemy_pos_bins': enemy_pos_bins,
    }

    pd = []
    for p in range(5):
        # Stats
        h_bin = torch.argmax(pred.health_logits[batch_idx, t, p])
        a_bin = torch.argmax(pred.armor_logits[batch_idx, t, p])
        m_bin = torch.argmax(pred.money_logits[batch_idx, t, p])
        
        h = unbin_value(h_bin, 0, 100, cfg.health_bins).item()
        a = unbin_value(a_bin, 0, 100, cfg.armor_bins).item()
        m = unbin_value(m_bin, 0, 16000, cfg.money_bins).item()
        
        # Pos
        pbx = torch.argmax(pred.player_pos_x[batch_idx, t, p])
        pby = torch.argmax(pred.player_pos_y[batch_idx, t, p])
        pbz = torch.argmax(pred.player_pos_z[batch_idx, t, p])
        
        px = unbin_value(pbx, POSITION_MIN, POSITION_MAX, cfg.bins_x).item()
        py = unbin_value(pby, POSITION_MIN, POSITION_MAX, cfg.bins_y).item()
        pz = unbin_value(pbz, POSITION_MIN, POSITION_MAX, cfg.bins_z).item()
        
        # Mouse
        mbx = torch.argmax(pred.mouse_x[batch_idx, t, p])
        mby = torch.argmax(pred.mouse_y[batch_idx, t, p])
        
        mx = mu_law_decode(mbx, cfg.mouse_mu, cfg.mouse_max, cfg.mouse_bins_count).item()
        my = mu_law_decode(mby, cfg.mouse_mu, cfg.mouse_max, cfg.mouse_bins_count).item()
        
        # Weapon
        w_idx = torch.argmax(pred.active_weapon_logits[batch_idx, t, p]).item()
        
        # Keyboard
        k_logits = pred.keyboard_logits[batch_idx, t, p]
        k_mask = 0
        for k_i in range(32):
            if torch.sigmoid(k_logits[k_i]) > 0.5:
                k_mask |= (1 << k_i)
                
        # Eco
        eco_idx = torch.argmax(pred.eco_buy_logits[batch_idx, t, p]).item()
        eco_pur_logit = pred.eco_purchase_logits[batch_idx, t, p].item()
        eco_pur = torch.sigmoid(torch.tensor(eco_pur_logit)).item() > 0.5
                
        p_dict = {
            'health': int(h),
            'armor': int(a),
            'money': int(m),
            'pos': [px, py, pz],
            'pos_bins': [pbx.item(), pby.item(), pbz.item()],
            'mouse': [mx, my],
            'mouse_bins': [mbx.item(), mby.item()],
            'keyboard_bitmask': k_mask,
            'active_weapon_idx': w_idx,
            'eco_buy_idx': eco_idx,
            'eco_purchase': eco_pur,
            'eco_purchase_prob': torch.sigmoid(torch.tensor(eco_pur_logit)).item()
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

POSITION_MIN = -4096.0
POSITION_MAX = 4096.0


def encode_position_bins(pos, cfg: ModelConfig):
    return [
        int(bin_value(pos[0], POSITION_MIN, POSITION_MAX, cfg.bins_x).item()),
        int(bin_value(pos[1], POSITION_MIN, POSITION_MAX, cfg.bins_y).item()),
        int(bin_value(pos[2], POSITION_MIN, POSITION_MAX, cfg.bins_z).item()),
    ]


def encode_mouse_bins(mouse, cfg: ModelConfig):
    return [
        int(mu_law_encode(mouse[0], cfg.mouse_mu, cfg.mouse_max, cfg.mouse_bins_count).item()),
        int(mu_law_encode(mouse[1], cfg.mouse_mu, cfg.mouse_max, cfg.mouse_bins_count).item()),
    ]


def format_bin_triplet(bins):
    return f"[{bins[0]},{bins[1]},{bins[2]}]"


def format_bin_pair(bins):
    return f"[{bins[0]},{bins[1]}]"


def format_pos_continuous(pos):
    return f"({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})"


def format_mouse_continuous(mouse):
    return f"({mouse[0]:.1f}, {mouse[1]:.1f})"

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
        wep_str = ITEM_NAMES[idx] if 0 <= idx < len(ITEM_NAMES) else f"UNK({idx})"
        
        pos = data['pos']
        mouse = data['mouse']
        keys = decode_keyboard(data['keyboard_bitmask'])
        
        # Eco
        eco_idx = data.get('eco_buy_idx', -1)
        eco_str = ITEM_NAMES[eco_idx] if 0 <= eco_idx < len(ITEM_NAMES) else ""
        is_buying = data.get('eco_purchase', False)

        if prefix == "GT":
            pos_str = f"{format_bin_triplet(data['pos_bins'])} {format_pos_continuous(pos)}"
            mouse_str = f"{format_bin_pair(data['mouse_bins'])} {format_mouse_continuous(mouse)}"
        else:
            pos_str = format_bin_triplet(data['pos_bins'])
            mouse_str = f"{format_bin_pair(data['mouse_bins'])} {format_mouse_continuous(mouse)}"
        
        lines = [
            f"{prefix} P{player_idx}: HP {hp} | AP {arm} | ${mon}",
            f"   Wep: {wep_str}",
            f"   Pos: {pos_str}",
            f"   Aim: {mouse_str}",
            f"   Key: {keys}"
        ]
        
        if is_buying and eco_str:
            prob = data.get('eco_purchase_prob', 1.0)
            lines.append(f"   BUY: {eco_str} ({prob:.2f})")
            
        return lines

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
    
    # Split panel into two columns: GT (Left) | Pred (Right)
    col_w = w // 2
    
    cv2.putText(panel, "GAME STATE (GT)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['gt'], 2)
    cv2.putText(panel, "GAME STATE (PRED)", (col_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['pred'], 2)
    
    # --- GT Column ---
    r_state_idx = gt_gs['round_state']
    r_state = decode_round_state(r_state_idx)
    r_num = gt_gs.get('round_num', '?')
    t_alive = gt_gs.get('team_alive_count', 0)
    e_alive = gt_gs.get('enemy_alive_count', 0)
    t_mask = gt_gs.get('team_alive_mask', 0)
    e_mask = gt_gs.get('enemy_alive_mask', 0)
    
    gt_lines = [
        f"Tick: {gt_gs.get('tick', '?')}",
        f"State: {r_state} [{r_state_idx}] | R: {r_num}",
        f"T Alive: {t_alive} | Mask: {bin(t_mask)[2:].zfill(5)}",
        f"E Alive: {e_alive} | Mask: {bin(e_mask)[2:].zfill(5)}",
        "Enemies:"
    ]
    
    for i, pos in enumerate(gt_gs['enemy_pos']):
        line = f" E{i}: {format_bin_triplet(gt_gs['enemy_pos_bins'][i])} {format_pos_continuous(pos)}"
        gt_lines.append(line)

    draw_text_lines(panel, gt_lines, 10, 60, COLORS['gt'], scale=0.5)
    
    # --- Pred Column ---
    if pred_gs:
        pr_state_idx = pred_gs['round_state']
        pr_state = decode_round_state(pr_state_idx)
        pr_num = pred_gs.get('round_num', '?')
        
        pred_lines = [
            f"Tick: {gt_gs.get('tick', '?')}", 
            f"State: {pr_state} [{pr_state_idx}] | R: {pr_num}",
            f"T Alive: {pred_gs['team_alive_count']}",
            f"E Alive: {pred_gs['enemy_alive_count']}",
            "Enemies:"
        ]
        
        for i, _ in enumerate(pred_gs['enemy_pos']):
            line = f" E{i}: {format_bin_triplet(pred_gs['enemy_pos_bins'][i])}"
            pred_lines.append(line)
            
        draw_text_lines(panel, pred_lines, col_w + 10, 60, COLORS['pred'], scale=0.5)

    return panel

def run_inference_and_video(model, loader, output_path, global_cfg, num_samples, device):
    """
    Reusable inference loop that generates a video.
    """
    writer = None
    processed = 0
    
    model.eval()
    
    with torch.no_grad():
        for i, sample in enumerate(loader):
            if processed >= num_samples: break
            
            # Move to device and cast to model's dtype
            sample.images = sample.images.to(device=device, dtype=model.model.cfg.dtype)
            sample.audio = sample.audio.to(device=device, dtype=model.model.cfg.dtype)
            # sample.truth does not need to be on device for viz conversion unless we use it for loss
            
            # Forward
            preds_dict = model(sample.images, sample.audio)
            preds = ModelPrediction(**preds_dict)
            
            B, T, P, C, H, W = sample.images.shape
            
            if writer is None:
                grid_w, grid_h = W * 3, H * 2
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                if os.path.dirname(output_path):
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                writer = cv2.VideoWriter(output_path, fourcc, 32, (grid_w, grid_h))
            
            imgs_cpu = sample.images.float().cpu()
            if imgs_cpu.max() <= 1.0:
                 imgs_cpu = (imgs_cpu * 255).byte()
            else:
                 imgs_cpu = imgs_cpu.clamp(0, 255).byte()
                
            for t in range(T):
                gt_data = convert_gt_to_viz(sample.truth, t, batch_idx=0, cfg=global_cfg.model)
                pred_data = convert_pred_to_viz(preds, t, batch_idx=0, cfg=global_cfg.model)
                
                frames = []
                for p in range(5):
                    frm = imgs_cpu[0, t, p].permute(1, 2, 0).numpy()
                    frm = cv2.cvtColor(frm, cv2.COLOR_RGB2BGR)
                    frames.append(frm)
                    
                panels = []
                for p in range(5):
                    panels.append(render_player_panel(frames[p], gt_data['player_data'][p], pred_data['player_data'][p], p))
                
                panels.append(render_global_panel(H, W, gt_data['game_state'], pred_data['game_state']))
                
                row1 = np.hstack(panels[0:3])
                row2 = np.hstack(panels[3:6])
                combined = np.vstack([row1, row2])
                
                writer.write(combined)
            
            processed += 1
            print(f"Viz sample {processed}/{num_samples} done.")
            
    if writer:
        writer.release()
        print(f"Video saved to {output_path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to lightning .ckpt file")
    parser.add_argument("--data_root", type=str, default="./dataset0")
    parser.add_argument("--output", type=str, default="inference_vis.mp4")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # 1. Load Config
    # Attempt to find config.json in checkpoint directory or parent
    ckpt_path = Path(args.checkpoint)
    config_path = ckpt_path.parent / "config.json"
    
    if config_path.exists():
        global_cfg = GlobalConfig.from_file(config_path)
        print(f"Loaded config from {config_path}")
    else:
        # Check grandparent (if checkpoint is in 'checkpoints/' subdir)
        config_path = ckpt_path.parent.parent / "config.json"
        if config_path.exists():
            global_cfg = GlobalConfig.from_file(config_path)
            print(f"Loaded config from {config_path}")
        else:
            print("Warning: config.json not found near checkpoint, using defaults.")
            d_cfg = DatasetConfig(data_root=args.data_root, run_dir="./runs")
            m_cfg = ModelConfig()
            t_cfg = TrainConfig()
            global_cfg = GlobalConfig(d_cfg, m_cfg, t_cfg)
        
    # Override data root if provided
    global_cfg.dataset.data_root = args.data_root
    
    # 2. Model
    print(f"Loading model from {args.checkpoint}...")
    # Load from PL checkpoint
    # We pass global_cfg to __init__ because it's required and strict_loading might fail if params mismatch,
    # but more importantly, the model needs the config structure.
    model = CS2PredictorModule.load_from_checkpoint(args.checkpoint, global_cfg=global_cfg, strict=False)
    
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    
    # 3. Data
    ds_root = DatasetRoot(global_cfg.dataset)
    val_ds = ds_root.build_dataset("val") # Build validation dataset
    if len(val_ds) == 0:
        print("Validation set empty! Falling back to train.")
        val_ds = ds_root.build_dataset("train")
        
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, collate_fn=cs2_collate_fn, shuffle=True
    )
    
    # 4. Run
    run_inference_and_video(model, loader, args.output, global_cfg, args.num_samples, device)

if __name__ == "__main__":
    main()
