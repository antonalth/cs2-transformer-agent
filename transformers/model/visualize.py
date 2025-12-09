#!/usr/bin/env python3
"""
visualize.py

Helper module to visualize CS2 dataset samples.
Decodes the compact NumPy/LMDB representations OR PyTorch GroundTruth tensors
into human-readable text and overlays them onto a composite video frame.
"""

import cv2
import numpy as np
import torch

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

# Reconstruct ECO_ACTIONS to match injection_mold logic
safe_item_names = [name.replace(' ', '_').replace('&', 'and').replace('-','_') for name in ITEM_NAMES]
item_id_map_names = [
    "deagle", "elite", "fiveseven", "glock", "ak47", "aug", "awp", "famas", "g3sg1", "galilar",
    "m249", "m4a1", "mac10", "p90", "mp5sd", "ump45", "xm1014", "ppbizon", "mag7", "negev",
    "sawedoff", "tec9", "zeus", "p2000", "mp7", "mp9", "nova", "p250", "scar20", "sg556", "ssg08",
    "knife", "flashbang", "hegrenade", "smokegrenade", "molotov", "decoy", "incgrenade", "c4",
    "vest", "vesthelm", "heavyassaultsuit", "nvgs", "defuser", "rescue_kit", "medishot", "knifet",
    "m4a1_silencer", "usp_silencer", "tradeupcontract", "cz75auto", "r8revolver", "charmdetachments",
    "bayonet", "knife_default_ct", "flipknife", "gutknife", "karambit", "knife_m9_bayonet",
    "huntsmanknife", "falchionknife", "bowieknife", "butterflyknife", "shadowdaggers",
    "paracordknife", "survivalknife", "ursusknife", "navajaknife", "nomadknife", "stilettoknife",
    "talonknife", "skeletonknife", "kukriknife"
]

ECO_ACTIONS = ["IN_BUYZONE"]
for name in safe_item_names:
    ECO_ACTIONS.append(f"BUY_{name}")
for name in item_id_map_names:
    ECO_ACTIONS.append(f"SELL_{name}"); ECO_ACTIONS.append(f"DROP_{name}")
ECO_ACTIONS = sorted(list(set(ECO_ACTIONS)))

# =============================================================================
# DECODING HELPERS
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

def decode_multi_array_mask(mask_array: np.ndarray, mapping_list: list) -> list:
    """Decodes an array of uint64s (like eco_bitmask) back to strings."""
    active_items = []
    
    # Ensure input is a uint64 numpy array. 
    # This handles lists (from JSON) and int64 arrays (from PyTorch tensors).
    # Casting int64 -> uint64 in NumPy preserves the bit pattern, which is what we want.
    mask_array = np.array(mask_array, dtype=np.uint64)
        
    for i, action in enumerate(mapping_list):
        arr_idx = i // 64
        bit_idx = i % 64
        if arr_idx < len(mask_array):
            if (mask_array[arr_idx] >> np.uint64(bit_idx)) & np.uint64(1):
                active_items.append(action)
    return active_items

def decode_inventory(mask_array) -> list:
    """Decodes inventory/weapon bitmasks."""
    return decode_multi_array_mask(mask_array, ITEM_NAMES)

def decode_eco(mask_array) -> list:
    """Decodes eco/buy/drop bitmasks."""
    full_list = decode_multi_array_mask(mask_array, ECO_ACTIONS)
    return [x.replace("BUY_", "+").replace("SELL_", "-").replace("DROP_", "v") for x in full_list]

def decode_round_state(mask: int) -> str:
    states = []
    if (mask >> 0) & 1: states.append("FREEZE")
    if (mask >> 1) & 1: states.append("LIVE")
    if (mask >> 2) & 1: states.append("PLANT")
    if (mask >> 3) & 1: states.append("T_WIN")
    if (mask >> 4) & 1: states.append("CT_WIN")
    return "|".join(states) if states else "NONE"

# =============================================================================
# TENSOR ADAPTER
# =============================================================================

def convert_tensor_to_viz_data(gt_object, t: int):
    """
    Extracts data from a GroundTruth object (Batched Tensors) at time t
    and converts it into the dictionary format expected by the renderer.
    """
    
    # Check if this is actually a tensor object or already a dict (support both)
    if isinstance(gt_object, dict):
        return gt_object # Assume it's already in the right format

    # 1. Game State Extraction
    # We initialize alive masks to 0 and reconstruct them from the boolean tensors below
    game_state = {
        'tick': t * 2, # Approximation
        'round_state': int(gt_object.round_state_mask[t].item()),
        'team_alive': 0, 
        'enemy_alive': 0, 
        'enemy_pos': gt_object.enemy_positions[t].tolist()
    }
    
    # 2. Player Data Extraction (Team)
    player_data = []
    team_alive_mask = 0
    
    for p in range(5):
        alive = bool(gt_object.alive_mask[t, p].item())
        if alive:
            team_alive_mask |= (1 << p)
            
        # Stats: [Health, Armor, Money]
        stats = gt_object.stats[t, p].tolist()
        
        # Weapon Handling
        wep_idx = int(gt_object.active_weapon_idx[t, p].item())
        
        p_dict = {
            'health': int(stats[0]),
            'armor': int(stats[1]),
            'money': int(stats[2]),
            'pos': gt_object.position[t, p].tolist(),
            'mouse': gt_object.mouse_delta[t, p].tolist(),
            'keyboard_bitmask': int(gt_object.keyboard_mask[t, p].item()),
            'eco_bitmask': gt_object.eco_mask[t, p].cpu().numpy(),
            'inventory_bitmask': gt_object.inventory_mask[t, p].cpu().numpy(),
            'active_weapon_idx': wep_idx
        }
        player_data.append(p_dict)

    # 3. Enemy Data Extraction
    enemy_alive_mask = 0
    if hasattr(gt_object, 'enemy_alive_mask'):
        for p in range(5):
            # Check availability just in case
            if p < gt_object.enemy_alive_mask.shape[1]: 
                if bool(gt_object.enemy_alive_mask[t, p].item()):
                    enemy_alive_mask |= (1 << p)

    game_state['team_alive'] = team_alive_mask
    game_state['enemy_alive'] = enemy_alive_mask
    
    return {'game_state': game_state, 'player_data': player_data}

# =============================================================================
# DRAWING HELPERS
# =============================================================================

COLORS = {
    'gt': (0, 255, 0),       # Green
    'pred': (0, 0, 255),     # Red
    'text_bg': (0, 0, 0),    # Black outline
    'dead_bg': (10, 10, 10), # Dark gray
    'white': (255, 255, 255)
}

def draw_text_lines(img, lines, x, y, color, scale=0.4, thickness=1):
    """Draws a list of strings vertically with a black outline."""
    line_height = int(20 * (scale / 0.4))
    for i, line in enumerate(lines):
        curr_y = y + (i * line_height)
        # Outline
        cv2.putText(img, line, (x, curr_y), cv2.FONT_HERSHEY_SIMPLEX, scale, COLORS['text_bg'], thickness + 2, cv2.LINE_AA)
        # Text
        cv2.putText(img, line, (x, curr_y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return y + (len(lines) * line_height)

def render_player_panel(frame, gt_data, pred_data=None, player_idx=0):
    """
    Renders a single player's perspective.
    """
    h, w, _ = frame.shape
    
    # Check if dead based on health
    is_dead = gt_data['health'] <= 0
    
    if is_dead:
        display_frame = np.zeros_like(frame) + 20 
        cv2.putText(display_frame, "DEAD", (w//2 - 40, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 2)
    else:
        display_frame = frame.copy()

    # --- Prepare Data Strings ---
    def get_lines(data, prefix=""):
        hp = data['health']
        arm = data['armor']
        mon = data['money']
        
        # Handle Weapon: Could be Index (Tensor source) or Bitmask (NumPy source)
        if 'active_weapon_idx' in data:
            idx = data['active_weapon_idx']
            wep_str = ITEM_NAMES[idx] if 0 <= idx < len(ITEM_NAMES) else "Knife/None"
        else:
            active_wep = decode_inventory(data.get('active_weapon_bitmask', [0,0]))
            wep_str = active_wep[0] if active_wep else "Knife"
        
        pos = data['pos']
        mouse = data['mouse']
        keys = decode_keyboard(data['keyboard_bitmask'])
        eco = decode_eco(data['eco_bitmask'])
        
        lines = [
            f"{prefix} P{player_idx}: HP {hp} | AP {arm} | ${mon}",
            f"   Wep: {wep_str}",
            f"   Pos: {pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}",
            f"   Aim: {mouse[0]:.2f}, {mouse[1]:.2f}",
            f"   Key: {keys}"
        ]
        if eco:
            # Truncate eco if too long
            eco_str = ','.join(eco)
            if len(eco_str) > 40: eco_str = eco_str[:37] + "..."
            lines.append(f"   Eco: {eco_str}")
        return lines

    # Top Left: Ground Truth
    gt_lines = get_lines(gt_data, "GT")
    next_y = draw_text_lines(display_frame, gt_lines, 10, 20, COLORS['gt'])

    # Below GT: Prediction
    if pred_data is not None:
        pred_lines = get_lines(pred_data, "PR")
        draw_text_lines(display_frame, pred_lines, 10, next_y + 5, COLORS['pred'])

    return display_frame

def render_global_panel(h, w, game_state, pred_game_state=None):
    """
    Renders the 6th panel (bottom right).
    """
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    
    cv2.putText(panel, "GAME STATE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['white'], 2)
    
    # Check if 'tick' is real or approximate
    tick_val = game_state.get('tick', '?')
    r_state = decode_round_state(game_state['round_state'])
    
    t_alive = game_state.get('team_alive', 0)
    e_alive = game_state.get('enemy_alive', 0)

    # Helper to format binary string (e.g., "11011")
    def fmt_alive(mask):
        return bin(mask)[2:].zfill(5)[::-1] # Reversed so index 0 is left (or right, depending on preference)

    gt_lines = [
        f"Tick/Frame: {tick_val}",
        f"State: {r_state}",
        f"T Alive: {bin(t_alive)[2:].zfill(5)} (Mask)",
        f"E Alive: {bin(e_alive)[2:].zfill(5)} (Mask)",
        "",
        "Enemy Positions (GT):"
    ]
    
    for i, pos in enumerate(game_state['enemy_pos']):
        # Only show position if the enemy is actually alive according to the bitmask
        is_alive = (e_alive >> i) & 1
        
        # Simple heuristic: hide 0,0,0 or obviously empty positions if not marked alive
        if is_alive or (abs(pos[0]) > 1.0 or abs(pos[1]) > 1.0):
            status = "" if is_alive else "(DEAD?)"
            gt_lines.append(f" E{i}: {pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f} {status}")

    next_y = draw_text_lines(panel, gt_lines, 10, 60, COLORS['gt'], scale=0.5)

    if pred_game_state is not None:
        pred_lines = ["", "Prediction (Global):"]
        # Add global prediction comparisons here if available
        draw_text_lines(panel, pred_lines, 10, next_y + 10, COLORS['pred'], scale=0.5)

    return panel

# =============================================================================
# MAIN EXPORT
# =============================================================================

def visualize_frame(frames: list, t: int, ground_truth, prediction=None):
    """
    Creates a combined visualization frame.
    
    Args:
        frames: List of 5 numpy arrays (H, W, 3) -> The 5 POV images.
        t: Integer index of the current frame (used to slice tensors).
        ground_truth: GroundTruth object (Batched Tensors) OR Dict.
        prediction: (Optional) GroundTruth object OR Dict.
    
    Returns:
        A single numpy array (Image) with 3x2 grid layout.
    """
    if len(frames) != 5:
        raise ValueError(f"Expected 5 frames, got {len(frames)}")

    # Convert Inputs to Dictionary Format
    gt_dict = convert_tensor_to_viz_data(ground_truth, t)
    pred_dict = convert_tensor_to_viz_data(prediction, t) if prediction else None

    # Unpack
    gs = gt_dict['game_state']
    pd = gt_dict['player_data']
    
    pred_gs = pred_dict['game_state'] if pred_dict else None
    pred_pd = pred_dict['player_data'] if pred_dict else [None]*5

    # --- Render Individual Panels ---
    rendered_panels = []
    h, w = frames[0].shape[:2]
    
    for i in range(5):
        panel = render_player_panel(frames[i], pd[i], pred_pd[i], player_idx=i)
        rendered_panels.append(panel)

    # --- Render Global Panel ---
    global_panel = render_global_panel(h, w, gs, pred_gs)
    rendered_panels.append(global_panel)

    # --- Stitch Grid ---
    # [P0] [P1] [P2]
    # [P3] [P4] [Global]
    row1 = np.hstack(rendered_panels[0:3])
    row2 = np.hstack(rendered_panels[3:6])
    combined = np.vstack([row1, row2])

    return combined