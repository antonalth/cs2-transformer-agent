#!/usr/bin/env python3
"""
lmdb_inspect.py - An interactive tool to inspect and verify the contents
of a CS2 data LMDB.

This script allows you to navigate through the dataset frame-by-frame,
switch player perspectives, play audio, and view detailed metadata overlays
in multiple modes (off, bitmasks, strings) to ensure the data was
processed correctly.
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path

import cv2
import lmdb
import msgpack
import msgpack_numpy as m
import numpy as np
import sounddevice as sd

# --- Platform-specific key press detection ---
try:
    import msvcrt
    import ctypes
    class COORD(ctypes.Structure): _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]
    class CONSOLE_SCREEN_BUFFER_INFO(ctypes.Structure): _fields_ = [("dwSize", COORD), ("dwCursorPosition", COORD), ("wAttributes", ctypes.c_ushort), ("srWindow", ctypes.c_short * 4), ("dwMaximumWindowSize", COORD)]
    STD_OUTPUT_HANDLE = -11
    handle = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
    def getch(): return msvcrt.getch()
    def move_cursor(y, x):
        pos = COORD(x, y)
        ctypes.windll.kernel32.SetConsoleCursorPosition(handle, pos)
except ImportError:
    import tty, termios
    def getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch.encode('utf-8')
    def move_cursor(y, x): sys.stdout.write(f"\x1b[{y};{x}H")

# --- Configuration ---
GAME_TICKS_PER_SEC = 64; EXPECTED_VIDEO_FPS = 32
TICKS_PER_FRAME = GAME_TICKS_PER_SEC // EXPECTED_VIDEO_FPS
AUDIO_SAMPLE_RATE = 44100; FONT = cv2.FONT_HERSHEY_SIMPLEX; FONT_SCALE = 0.4
FONT_COLOR = (0, 255, 0); SHADOW_COLOR = (0, 0, 0); LINE_TYPE = 1
TEXT_START_POS = (10, 20); LINE_HEIGHT = 16
m.patch()

# =============================================================================
# DATA ENCODING MAPPINGS (Canonical Source - Synced with injection_mold.py)
# =============================================================================
KEYBOARD_ONLY_ACTIONS = ["IN_ATTACK", "IN_JUMP", "IN_DUCK", "IN_FORWARD", "IN_BACK", "IN_USE", "IN_CANCEL", "IN_TURNLEFT", "IN_TURNRIGHT", "IN_MOVELEFT", "IN_MOVERIGHT", "IN_ATTACK2", "IN_RELOAD", "IN_ALT1", "IN_ALT2", "IN_SPEED", "IN_WALK", "IN_ZOOM", "IN_WEAPON1", "IN_WEAPON2", "IN_BULLRUSH", "IN_GRENADE1", "IN_GRENADE2", "IN_ATTACK3", "IN_SCORE", "IN_INSPECT", "SWITCH_1", "SWITCH_2", "SWITCH_3", "SWITCH_4", "SWITCH_5"]
ITEM_NAMES = sorted(list(set(["AK-47", "M4A4", "M4A1-S", "Galil AR", "FAMAS", "AUG", "SG 553", "AWP", "SSG 08", "G3SG1", "SCAR-20", "Glock-18", "USP-S", "P250", "P2000", "Dual Berettas", "Five-SeveN", "Tec-9", "CZ75-Auto", "R8 Revolver", "Desert Eagle", "MP9", "MAC-10", "MP7", "MP5-SD", "UMP-45", "P90", "PP-Bizon", "Nova", "XM1014", "MAG-7", "Sawed-Off", "M249", "Negev", "Knife", "knife_t", "knife_ct", "Bayonet", "Flip Knife", "Gut Knife", "Karambit", "M9 Bayonet", "Huntsman Knife", "Falchion Knife", "Bowie Knife", "Butterfly Knife", "Shadow Daggers", "Ursus Knife", "Navaja Knife", "Stiletto Knife", "Talon Knife", "Classic Knife", "Paracord Knife", "Survival Knife", "Nomad Knife", "Skeleton Knife", "High Explosive Grenade", "Flashbang", "Smoke Grenade", "Molotov", "Incendiary Grenade", "Decoy Grenade", "C4 Explosive", "Defuse Kit", "Zeus x27", "Kevlar Vest", "Kevlar and Helmet", "Helmet"])))
ECO_ACTIONS = ["IN_BUYZONE"]
safe_item_names = [name.replace(" ", "_").replace("-", "_") for name in ITEM_NAMES]
for name in safe_item_names: ECO_ACTIONS.append(f"BUY_{name}"); ECO_ACTIONS.append(f"SELL_{name}"); ECO_ACTIONS.append(f"DROP_{name}")
item_id_map_names = ["deagle", "elite", "fiveseven", "glock", "ak47", "aug", "awp", "famas", "g3sg1", "galilar", "m249", "m4a1", "mac10", "p90", "mp5sd", "ump45", "xm1014", "bizon", "mag7", "negev", "sawedoff", "tec9", "p2000", "mp7", "mp9", "nova", "p250", "scar20", "sg556", "ssg08", "knife", "flashbang", "hegrenade", "smokegrenade", "molotov", "decoy", "incgrenade", "c4", "knife_t", "m4a1_silencer", "usp_silencer", "cz75a", "revolver", "defuser", "vest", "vesthelm"]
for name in item_id_map_names: ECO_ACTIONS.append(f"BUY_{name}"); ECO_ACTIONS.append(f"SELL_{name}"); ECO_ACTIONS.append(f"DROP_{name}")
ECO_ACTIONS = sorted(list(set(ECO_ACTIONS)))
BIT_TO_KEYBOARD = {i: action for i, action in enumerate(KEYBOARD_ONLY_ACTIONS)}
BIT_TO_ECO = {i: action for i, action in enumerate(ECO_ACTIONS)}
BIT_TO_ITEM = {i: item for i, item in enumerate(ITEM_NAMES)}

# =============================================================================
# HELPER AND DECODER FUNCTIONS
# =============================================================================
def decode_bitmask(mask, reverse_map):
    actions = [reverse_map[i] for i in range(32) if (mask >> i) & 1 and i in reverse_map]
    return ", ".join(actions) if actions else "None"

def decode_bitmask_array(mask_array, reverse_map):
    actions = []
    for i, sub_mask in enumerate(mask_array):
        for j in range(64):
            if (sub_mask >> j) & 1:
                global_bit_pos = i * 64 + j
                if global_bit_pos in reverse_map:
                    actions.append(reverse_map[global_bit_pos])
    return ", ".join(actions) if actions else "None"

def play_audio(audio_chunk, blocking=False):
    if not audio_chunk: print("No audio chunk to play."); return
    try:
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16).reshape(-1, 2)
        sd.play(audio_array, samplerate=AUDIO_SAMPLE_RATE)
        if blocking: sd.wait()
    except Exception as e: print(f"Error playing audio: {e}")

def create_placeholder_frame(width, height, text):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    (text_width, text_height), _ = cv2.getTextSize(text, FONT, 1, 2)
    text_x, text_y = (width - text_width) // 2, (height + text_height) // 2
    cv2.putText(frame, text, (text_x, text_y), FONT, 1, (150, 150, 150), 2, cv2.LINE_AA)
    return frame

def draw_text(frame, text, pos, color=FONT_COLOR, shadow=SHADOW_COLOR):
    shadow_pos = (pos[0] + 1, pos[1] + 1)
    cv2.putText(frame, text, shadow_pos, FONT, FONT_SCALE, shadow, LINE_TYPE, cv2.LINE_AA)
    cv2.putText(frame, text, pos, FONT, FONT_SCALE, color, LINE_TYPE, cv2.LINE_AA)
    return (pos[0], pos[1] + LINE_HEIGHT)

def get_player_data_for_pov(player_data_list, player_idx, team_alive_mask):
    if not (team_alive_mask >> player_idx) & 1: return None
    living_players_before = bin(team_alive_mask & ((1 << player_idx) - 1)).count('1')
    if living_players_before < len(player_data_list):
        return player_data_list[living_players_before]
    return None

def debug_interactive_search(env):
    print("Loading all keys into memory for searching...")
    try:
        with env.begin(write=False) as txn:
            all_keys = [key.decode('utf-8') for key in txn.cursor().iternext(keys=True, values=False)]
    except Exception as e: print(f"Error loading keys: {e}"); sys.exit(1)
    finally: env.close()
    print(f"Loaded {len(all_keys)} keys. Starting interactive search...")
    search_query = ""
    try:
        while True:
            matching_keys = [k for k in all_keys if search_query in k] if search_query else all_keys[:200]
            try: term_width, term_height = os.get_terminal_size()
            except OSError: term_width, term_height = 80, 24
            os.system('cls' if os.name == 'nt' else 'clear')
            print("--- LMDB Interactive Key Search (Press Ctrl+C to exit) ---")
            display_rows = term_height - 4
            if display_rows > 0 and matching_keys:
                max_len = max(len(k) for k in matching_keys[:display_rows*5]) if matching_keys else 0
                num_columns = max(1, term_width // (max_len + 2)); column_width = (term_width // num_columns)
                display_limit = min(len(matching_keys), display_rows * num_columns)
                for i, key in enumerate(matching_keys[:display_limit]):
                    if i > 0 and i % num_columns == 0: print()
                    print(f"{key:<{column_width-1}}", end="")
                print()
            status_line = f"Showing {min(len(matching_keys), display_limit)} of {len(matching_keys)} matches. Total keys: {len(all_keys)}"
            move_cursor(term_height - 2, 0); print(status_line.ljust(term_width))
            prompt_line = f"Search: {search_query}"; print(prompt_line.ljust(term_width), end="")
            move_cursor(term_height - 1, len(prompt_line)); sys.stdout.flush()
            char_bytes = getch()
            if char_bytes in (b'\x03', b'\x04'): break
            elif char_bytes == b'\x08': search_query = search_query[:-1]
            else:
                try:
                    char = char_bytes.decode('utf-8')
                    if char.isprintable(): search_query += char
                except UnicodeDecodeError: pass
    except KeyboardInterrupt: print("\nExiting interactive search.")
    finally: os.system('cls' if os.name == 'nt' else 'clear'); sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Interactive LMDB Inspector for CS2 Data.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("lmdb_path", type=Path); parser.add_argument("round", type=int, nargs='?', default=1); parser.add_argument("team", type=str, nargs='?', default='T', choices=['T', 'CT'])
    parser.add_argument("player_idx", type=int, nargs='?', default=0); parser.add_argument("tick", type=int, nargs='?', default=-1)
    parser.add_argument("--debug", action='store_true'); parser.add_argument("--autoplay", action='store_true'); args = parser.parse_args()
    if not args.lmdb_path.exists(): print(f"Error: LMDB path not found: {args.lmdb_path}"); sys.exit(1)
    if args.debug:
        env = lmdb.open(str(args.lmdb_path), readonly=True, lock=False, map_size=1024**4); debug_interactive_search(env)
    env = lmdb.open(str(args.lmdb_path), readonly=True, lock=False)
    with env.begin() as txn:
        info_key_bytes = next((k for k in txn.cursor().iternext(keys=True, values=False) if k.endswith(b'_INFO')), None)
        if not info_key_bytes: print("Error: INFO key not found."); sys.exit(1)
        demoname = info_key_bytes.removesuffix(b'_INFO').decode('utf-8')
        metadata = json.loads(txn.get(info_key_bytes))
        round_info = {r[0]: {'start': r[1], 'end': r[2]} for r in metadata['rounds']}
    current_round, current_team, current_player_idx = args.round, args.team, args.player_idx
    run_mode_on = False; overlay_state = 1
    if args.tick == -1:
        if current_round not in round_info: print(f"Error: Round {current_round} not found. Valid rounds: {list(round_info.keys())}"); sys.exit(1)
        current_tick = round_info[current_round]['start']
    else: current_tick = args.tick
    cv2.namedWindow("LMDB Inspector")
    while True:
        key_str = f"{demoname}_round_{current_round:03d}_team_{current_team}_tick_{current_tick:08d}"
        key_bytes = key_str.encode('utf-8')
        with env.begin() as txn: value = txn.get(key_bytes)
        if value:
            data = msgpack.unpackb(value, raw=False, object_hook=m.decode)
            game_state_record = data['game_state'][0]
            pov_data = get_player_data_for_pov(data['player_data'], current_player_idx, game_state_record['team_alive'])
            player_input_array, jpeg, audio = (None, None, None)
            if pov_data: player_input_array, jpeg, audio = pov_data
            if not run_mode_on and args.autoplay and audio: play_audio(audio, blocking=False)
            if jpeg:
                frame_np = np.frombuffer(jpeg, dtype=np.uint8); frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
                if pov_data and player_input_array.size > 0:
                    player_input_record = player_input_array[0]
                    if overlay_state > 0:
                        overlay_mode_str = "MASKS" if overlay_state == 1 else "STRINGS"; run_status = "ON" if run_mode_on else "OFF"
                        pos = draw_text(frame, f"KEY: {key_str}", TEXT_START_POS)
                        pos = draw_text(frame, f"POV: Player {current_player_idx} ({current_team}) | RUN: {run_status} | OVERLAY: {overlay_mode_str}", pos)
                        pos = draw_text(frame, "-"*60, pos); pos = draw_text(frame, f"[GAME STATE] Round: {current_round} | Tick: {game_state_record['tick']}", pos)
                        pos = draw_text(frame, f"Team Alive: {game_state_record['team_alive']:05b} | Enemy Alive: {game_state_record['enemy_alive']:05b}", pos)
                        pos = draw_text(frame, "-"*60, pos); pos = draw_text(frame, "[PLAYER INPUT]", pos)
                        pos = draw_text(frame, f"Health: {player_input_record['health']} | Armor: {player_input_record['armor']} | Money: ${player_input_record['money']}", pos)
                        pos = draw_text(frame, f"Pos: ({player_input_record['pos'][0]:.1f}, {player_input_record['pos'][1]:.1f}, {player_input_record['pos'][2]:.1f})", pos)
                        pos = draw_text(frame, f"Mouse: ({player_input_record['mouse'][0]:.3f}, {player_input_record['mouse'][1]:.3f})", pos)
                        if overlay_state == 1:
                            pos = draw_text(frame, f"Keyboard Mask: {int(player_input_record['keyboard_bitmask'])}", pos)
                            # --- FIX: Dynamically create the string for the larger mask ---
                            eco_mask_str = " ".join(map(str, player_input_record['eco_bitmask']))
                            pos = draw_text(frame, f"Eco Mask: {eco_mask_str}", pos)
                            inv_mask_str = " ".join(map(str, player_input_record['inventory_bitmask']))
                            pos = draw_text(frame, f"Inv Mask: {inv_mask_str}", pos)
                            wep_mask_str = " ".join(map(str, player_input_record['active_weapon_bitmask']))
                            pos = draw_text(frame, f"Wep Mask: {wep_mask_str}", pos)
                        elif overlay_state == 2:
                            pos = draw_text(frame, "Keyboard: " + decode_bitmask(player_input_record['keyboard_bitmask'], BIT_TO_KEYBOARD), pos)
                            pos = draw_text(frame, "Eco: " + decode_bitmask_array(player_input_record['eco_bitmask'], BIT_TO_ECO), pos)
                            pos = draw_text(frame, "Weapon: " + decode_bitmask_array(player_input_record['active_weapon_bitmask'], BIT_TO_ITEM), pos)
                            pos = draw_text(frame, "Inventory: " + decode_bitmask_array(player_input_record['inventory_bitmask'], BIT_TO_ITEM), pos)
                else: frame = create_placeholder_frame(1280, 720, "PLAYER DEAD")
                if not run_mode_on: print("\n" + "="*80); print(f"Displaying: {key_str}"); print("\n--- CONTROLS ---"); print("q: quit|j/k: tick|p: player|t: team|a: audio|o: overlay|r: RUN")
            else: frame = create_placeholder_frame(1280, 720, "PLAYER DEAD")
        else:
            frame = create_placeholder_frame(1280, 720, "NO DATA FOR TICK"); audio = None
            if not run_mode_on: print(f"No data for key: {key_str}")
            if run_mode_on: run_mode_on = False; print("Run mode stopped: No data for tick.")
        cv2.imshow("LMDB Inspector", frame)
        if run_mode_on and audio: play_audio(audio, blocking=True)
        wait_time = 1 if run_mode_on else 0; key = cv2.waitKey(wait_time) & 0xFF
        if key == ord('r'): run_mode_on = not run_mode_on; print(f"Run mode toggled {'ON' if run_mode_on else 'OFF'}"); continue
        if run_mode_on:
            current_tick += TICKS_PER_FRAME
            if current_round in round_info and current_tick > round_info[round_info[current_round]['end']]:
                print("Run mode stopped: End of round."); run_mode_on = False
        else:
            if key == ord('q'): break
            elif key == ord('j'): current_tick += TICKS_PER_FRAME
            elif key == ord('k'): current_tick -= TICKS_PER_FRAME
            elif key == ord('p'): current_player_idx = (current_player_idx + 1) % 5
            elif key == ord('t'): current_team = 'CT' if current_team == 'T' else 'T'
            elif key == ord('a') and audio: play_audio(audio, blocking=True)
            elif key == ord('o'): overlay_state = (overlay_state + 1) % 3
    cv2.destroyAllWindows(); env.close()

if __name__ == "__main__":
    main()