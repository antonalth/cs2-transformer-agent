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

# --- Platform-specific key press detection for interactive debug ---
try:
    # Windows
    import msvcrt
    def getch():
        """Gets a single character from standard input on Windows."""
        return msvcrt.getch().decode('utf-8', errors='ignore')
except ImportError:
    # Unix-like systems (Linux, macOS)
    import tty, termios
    def getch():
        """Gets a single character from standard input on Unix."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


# --- Configuration ---
GAME_TICKS_PER_SEC = 64
EXPECTED_VIDEO_FPS = 32
TICKS_PER_FRAME = GAME_TICKS_PER_SEC // EXPECTED_VIDEO_FPS
AUDIO_SAMPLE_RATE = 44100
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
FONT_COLOR = (0, 255, 0)
SHADOW_COLOR = (0, 0, 0)
LINE_TYPE = 1
TEXT_START_POS = (10, 20)
LINE_HEIGHT = 16

# Make msgpack aware of numpy for deserialization
m.patch()

# =============================================================================
# DATA ENCODING MAPPINGS (Copied from injection_mold.py for decoding)
# =============================================================================
KEYBOARD_ONLY_ACTIONS = [
    "IN_ATTACK", "IN_JUMP", "IN_DUCK", "IN_FORWARD", "IN_BACK", "IN_USE", "IN_CANCEL", "IN_TURNLEFT", "IN_TURNRIGHT", "IN_MOVELEFT", "IN_MOVERIGHT", "IN_ATTACK2", "IN_RELOAD", "IN_ALT1", "IN_ALT2", "IN_SPEED", "IN_WALK", "IN_ZOOM", "IN_WEAPON1", "IN_WEAPON2", "IN_BULLRUSH", "IN_GRENADE1", "IN_GRENADE2", "IN_ATTACK3", "IN_SCORE", "IN_INSPECT", "SWITCH_1", "SWITCH_2", "SWITCH_3", "SWITCH_4", "SWITCH_5",
]
ECO_ACTIONS = [
    "IN_BUYZONE", "DROP_deagle", "DROP_elite", "DROP_fiveseven", "DROP_glock", "DROP_ak47", "DROP_aug", "DROP_awp", "DROP_famas", "DROP_g3sg1", "DROP_galilar", "DROP_m249", "DROP_m4a1", "DROP_mac10", "DROP_p90", "DROP_mp5sd", "DROP_ump45", "DROP_xm1014", "DROP_bizon", "DROP_mag7", "DROP_negev", "DROP_sawedoff", "DROP_tec9", "DROP_p2000", "DROP_mp7", "DROP_mp9", "DROP_nova", "DROP_p250", "DROP_scar20", "DROP_sg556", "DROP_ssg08", "DROP_knife", "DROP_flashbang", "DROP_hegrenade", "DROP_smokegrenade", "DROP_molotov", "DROP_decoy", "DROP_incgrenade", "DROP_c4", "DROP_m4a1_silencer", "DROP_usp_silencer", "DROP_cz75a", "DROP_revolver", "DROP_defuser", "BUY_deagle", "BUY_elite", "BUY_fiveseven", "BUY_glock", "BUY_ak47", "BUY_aug", "BUY_awp", "BUY_famas", "BUY_g3sg1", "BUY_galilar", "BUY_m249", "BUY_m4a1", "BUY_mac10", "BUY_p90", "BUY_mp5sd", "BUY_ump45", "BUY_xm1014", "BUY_bizon", "BUY_mag7", "BUY_negev", "BUY_sawedoff", "BUY_tec9", "BUY_p2000", "BUY_mp7", "BUY_mp9", "BUY_nova", "BUY_p250", "BUY_scar20", "BUY_sg556", "BUY_ssg08", "BUY_knife", "BUY_flashbang", "BUY_hegrenade", "BUY_smokegrenade", "BUY_molotov", "BUY_decoy", "BUY_incgrenade", "BUY_c4", "BUY_m4a1_silencer", "BUY_usp_silencer", "BUY_cz75a", "BUY_revolver", "BUY_defuser", "BUY_vest", "BUY_vesthelm", "SELL_deagle", "SELL_fiveseven", "SELL_glock", "SELL_ak47", "SELL_aug", "SELL_awp", "SELL_famas", "SELL_galilar", "SELL_m4a1", "SELL_mac10", "SELL_p90", "SELL_ump45", "SELL_xm1014", "SELL_bizon", "SELL_mag7", "SELL_sawedoff", "SELL_tec9", "SELL_p2000", "SELL_mp7", "SELL_mp9", "SELL_nova", "SELL_p250", "SELL_ssg08", "SELL_flashbang", "SELL_hegrenade", "SELL_smokegrenade", "SELL_molotov", "SELL_decoy", "SELL_incgrenade",
]
ITEM_NAMES = sorted(list(set([
    "AK-47", "M4A4", "M4A1-S", "Galil AR", "FAMAS", "AUG", "SG 553", "AWP", "SSG 08", "G3SG1", "SCAR-20", "Glock-18", "USP-S", "P250", "P2000", "Dual Berettas", "Five-SeveN", "Tec-9", "CZ75-Auto", "R8 Revolver", "Desert Eagle", "MP9", "MAC-10", "MP7", "MP5-SD", "UMP-45", "P90", "PP-Bizon", "Nova", "XM1014", "MAG-7", "Sawed-Off", "M249", "Negev", "Knife", "Bayonet", "Flip Knife", "Gut Knife", "Karambit", "M9 Bayonet", "Huntsman Knife", "Falchion Knife", "Bowie Knife", "Butterfly Knife", "Shadow Daggers", "Ursus Knife", "Navaja Knife", "Stiletto Knife", "Talon Knife", "Classic Knife", "Paracord Knife", "Survival Knife", "Nomad Knife", "Skeleton Knife", "High Explosive Grenade", "Flashbang", "Smoke Grenade", "Molotov", "Incendiary Grenade", "Decoy Grenade", "C4 Explosive", "Defuse Kit", "Zeus x27", "Kevlar Vest", "Helmet", "knife_ct", "knife_t",
])))

# --- NEW: Reverse mappings for decoding ---
BIT_TO_KEYBOARD = {i: action for i, action in enumerate(KEYBOARD_ONLY_ACTIONS)}
BIT_TO_ECO = {i: action for i, action in enumerate(ECO_ACTIONS)}
BIT_TO_ITEM = {i: item for i, item in enumerate(ITEM_NAMES)}

# =============================================================================
# HELPER AND DECODER FUNCTIONS
# =============================================================================

def decode_bitmask(mask, reverse_map):
    """Decodes a single integer bitmask into a list of strings."""
    actions = [reverse_map[i] for i in range(32) if (mask >> i) & 1 and i in reverse_map]
    return ", ".join(actions) if actions else "None"

def decode_bitmask_array(mask_array, reverse_map):
    """Decodes a multi-integer bitmask array into a list of strings."""
    actions = []
    for i, sub_mask in enumerate(mask_array):
        for j in range(64):
            if (sub_mask >> j) & 1:
                global_bit_pos = i * 64 + j
                if global_bit_pos in reverse_map:
                    actions.append(reverse_map[global_bit_pos])
    return ", ".join(actions) if actions else "None"

def play_audio(audio_chunk, blocking=False):
    """Plays a raw PCM audio chunk using sounddevice."""
    if not audio_chunk:
        # This is a normal occurrence, so we don't print anything here.
        return
    try:
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16).reshape(-1, 2)
        sd.play(audio_array, samplerate=AUDIO_SAMPLE_RATE)
        if blocking:
            sd.wait()
    except Exception as e:
        print(f"Error playing audio: {e}")

def create_placeholder_frame(width, height, text):
    """Creates a black frame with centered text."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    (text_width, text_height), _ = cv2.getTextSize(text, FONT, 1, 2)
    text_x = (width - text_width) // 2
    text_y = (height + text_height) // 2
    cv2.putText(frame, text, (text_x, text_y), FONT, 1, (150, 150, 150), 2, cv2.LINE_AA)
    return frame

def draw_text(frame, text, pos, color=FONT_COLOR, shadow=SHADOW_COLOR):
    """Draws text with a shadow for better visibility."""
    shadow_pos = (pos[0] + 1, pos[1] + 1)
    cv2.putText(frame, text, shadow_pos, FONT, FONT_SCALE, shadow, LINE_TYPE, cv2.LINE_AA)
    cv2.putText(frame, text, pos, FONT, FONT_SCALE, color, LINE_TYPE, cv2.LINE_AA)
    return (pos[0], pos[1] + LINE_HEIGHT)

def get_player_data_for_pov(player_data_list, player_idx, team_alive_mask):
    """Finds the correct player's data from the list based on their liveness."""
    if not (team_alive_mask >> player_idx) & 1:
        return None  # Player is dead

    living_players_before = bin(team_alive_mask & ((1 << player_idx) - 1)).count('1')
    
    if living_players_before < len(player_data_list):
        return player_data_list[living_players_before]
    
    return None # Data inconsistent or player just died this tick

def debug_interactive_search(env):
    """
    Provides a memory-efficient, interactive search interface for LMDB keys
    that dynamically fits the terminal window.
    """
    search_query = ""
    total_keys = env.stat()['entries']

    try:
        while True:
            # --- Prepare for redraw ---
            try:
                # Get terminal dimensions
                term_width, term_height = os.get_terminal_size()
            except OSError:
                # Fallback for environments where this fails (e.g., some IDEs)
                term_width, term_height = 80, 24
            
            # Calculate how many results can be displayed
            header_lines = 4
            prompt_lines = 2
            available_lines = term_height - header_lines - prompt_lines

            # --- Search Logic (Memory Efficient) ---
            # This block iterates the DB instead of loading all keys into RAM
            matching_keys = []
            with env.begin() as txn:
                cursor = txn.cursor()
                for key_bytes, _ in cursor:
                    # If we have enough keys to fill the screen, stop searching
                    if len(matching_keys) >= 500: # Safety limit
                         break
                    key_str = key_bytes.decode('utf-8', errors='ignore')
                    if search_query in key_str:
                        matching_keys.append(key_str)

            # --- Render Output ---
            # Clear screen
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Print header
            print("--- LMDB Interactive Key Search ---")
            print(f"Searching {total_keys} keys. Display is limited to fit terminal. (Press Ctrl+C to exit)\n")
            
            # Print results in columns
            if not matching_keys:
                print("No matches found.")
            else:
                max_len = max(len(k) for k in matching_keys) if matching_keys else 0
                num_columns = max(1, term_width // (max_len + 2)) # +2 for padding
                display_limit = min(len(matching_keys), available_lines * num_columns)

                for i, key in enumerate(matching_keys[:display_limit]):
                    if (i > 0) and (i % num_columns == 0):
                        print()
                    print(f"{key:<{max_len}}  ", end="")
                print() # Final newline

                if len(matching_keys) > display_limit:
                    print(f"\n...and {len(matching_keys) - display_limit} more matches found.")

            # Print search prompt at the bottom (so it doesn't move)
            print("\n" + "="*term_width)
            print(f"Search: {search_query}", end="", flush=True)

            # --- Handle Input ---
            char = getch()
            
            if char in ('\x03', '\x04'): # Ctrl+C, Ctrl+D
                break
            elif char == '\r': # Enter key
                break
            elif char in ('\x08', '\x7f'): # Backspace
                search_query = search_query[:-1]
            # Add any printable character to the query
            elif char and char.isprintable():
                search_query += char

    except KeyboardInterrupt:
        print("\nExiting interactive search.")
    finally:
        env.close()
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Interactive LMDB Inspector for CS2 Data.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("lmdb_path", type=Path, help="Path to the LMDB directory.")
    parser.add_argument("round", type=int, nargs='?', default=1, help="Starting round number (default: 1).")
    parser.add_argument("team", type=str, nargs='?', default='T', choices=['T', 'CT'], help="Starting team (default: T).")
    parser.add_argument("player_idx", type=int, nargs='?', default=0, help="Starting player index [0-4] (default: 0).")
    parser.add_argument("tick", type=int, nargs='?', default=-1, help="Starting tick. If -1, starts at the beginning of the round (default: -1).")
    parser.add_argument("--debug", action='store_true', help="Run an interactive search to find keys in the LMDB and exit.")
    parser.add_argument("--autoplay", action='store_true', help="Automatically play non-blocking audio for each frame (overridden by run mode).")
    args = parser.parse_args()

    if not args.lmdb_path.exists():
        print(f"Error: LMDB path not found at: {args.lmdb_path}"); sys.exit(1)

    env = lmdb.open(str(args.lmdb_path), readonly=True, lock=False)

    if args.debug:
        debug_interactive_search(env)
        # The script will exit from within the debug function

    with env.begin() as txn:
        info_key_bytes = next((k for k in txn.cursor().iternext(keys=True, values=False) if k.endswith(b'_INFO')), None)
        if not info_key_bytes: print("Error: Could not find INFO key in the database."); sys.exit(1)
        demoname = info_key_bytes.removesuffix(b'_INFO').decode('utf-8')
        metadata = json.loads(txn.get(info_key_bytes))
        round_info = {r[0]: {'start': r[1], 'end': r[2]} for r in metadata['rounds']}

    current_round, current_team, current_player_idx = args.round, args.team, args.player_idx
    run_mode_on = False
    overlay_state = 1 # 0=OFF, 1=Bitmasks, 2=Strings

    if args.tick == -1:
        if current_round not in round_info:
             print(f"Error: Round {current_round} not found. Valid rounds: {list(round_info.keys())}"); sys.exit(1)
        current_tick = round_info[current_round]['start']
    else:
        current_tick = args.tick
        
    cv2.namedWindow("LMDB Inspector")
    
    while True:
        key_str = f"{demoname}_round_{current_round:03d}_team_{current_team}_tick_{current_tick:08d}"
        key_bytes = key_str.encode('utf-8')
        
        with env.begin() as txn: value = txn.get(key_bytes)
        
        if value:
            data = msgpack.unpackb(value, raw=False, object_hook=m.decode)
            game_state_record = data['game_state'][0]
            player_data_list = data['player_data']
            
            pov_data = get_player_data_for_pov(player_data_list, current_player_idx, game_state_record['team_alive'])
            
            player_input_array, jpeg, audio = (None, None, None)
            if pov_data:
                player_input_array, jpeg, audio = pov_data

            if not run_mode_on and args.autoplay and audio:
                play_audio(audio, blocking=False)

            if jpeg is not None:
                frame_np = np.frombuffer(jpeg, dtype=np.uint8)
                frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
                if pov_data and player_input_array is not None and player_input_array.size > 0:
                    player_input_record = player_input_array[0]
                    
                    if overlay_state > 0:
                        overlay_mode_str = "MASKS" if overlay_state == 1 else "STRINGS"
                        run_status = "ON" if run_mode_on else "OFF"
                        pos = draw_text(frame, f"KEY: {key_str}", TEXT_START_POS)
                        pos = draw_text(frame, f"POV: Player {current_player_idx} ({current_team}) | RUN: {run_status} | OVERLAY: {overlay_mode_str}", pos)
                        pos = draw_text(frame, "-"*60, pos)
                        pos = draw_text(frame, f"[GAME STATE] Round: {current_round} | Tick: {game_state_record['tick']}", pos)
                        pos = draw_text(frame, f"Team Alive: {game_state_record['team_alive']:05b} | Enemy Alive: {game_state_record['enemy_alive']:05b}", pos)
                        pos = draw_text(frame, "-"*60, pos); pos = draw_text(frame, "[PLAYER INPUT]", pos)
                        pos = draw_text(frame, f"Health: {player_input_record['health']} | Armor: {player_input_record['armor']} | Money: ${player_input_record['money']}", pos)
                        pos = draw_text(frame, f"Pos: ({player_input_record['pos'][0]:.1f}, {player_input_record['pos'][1]:.1f}, {player_input_record['pos'][2]:.1f})", pos)
                        pos = draw_text(frame, f"Mouse: ({player_input_record['mouse'][0]:.3f}, {player_input_record['mouse'][1]:.3f})", pos)
                        
                        if overlay_state == 1: # Bitmask Mode
                            pos = draw_text(frame, f"Keyboard Mask: {int(player_input_record['keyboard_bitmask'])}", pos)
                            pos = draw_text(frame, f"Eco Mask: {int(player_input_record['eco_bitmask'][0])} {int(player_input_record['eco_bitmask'][1])}", pos)
                            pos = draw_text(frame, f"Inv Mask: {int(player_input_record['inventory_bitmask'][0])} {int(player_input_record['inventory_bitmask'][1])}", pos)
                            pos = draw_text(frame, f"Wep Mask: {int(player_input_record['active_weapon_bitmask'][0])} {int(player_input_record['active_weapon_bitmask'][1])}", pos)
                        elif overlay_state == 2: # String Mode
                            pos = draw_text(frame, "Keyboard: " + decode_bitmask(player_input_record['keyboard_bitmask'], BIT_TO_KEYBOARD), pos)
                            pos = draw_text(frame, "Eco: " + decode_bitmask_array(player_input_record['eco_bitmask'], BIT_TO_ECO), pos)
                            pos = draw_text(frame, "Weapon: " + decode_bitmask_array(player_input_record['active_weapon_bitmask'], BIT_TO_ITEM), pos)
                            pos = draw_text(frame, "Inventory: " + decode_bitmask_array(player_input_record['inventory_bitmask'], BIT_TO_ITEM), pos)
                else: # Player is dead or has no input record
                    frame = create_placeholder_frame(1280, 720, "PLAYER DEAD")
            else: # No pov_data or no jpeg
                frame = create_placeholder_frame(1280, 720, "PLAYER DEAD")
        else: # No value for key
            frame = create_placeholder_frame(1280, 720, "NO DATA FOR TICK")
            if not run_mode_on: print(f"No data found for key: {key_str}")
            audio = None
            if run_mode_on: run_mode_on = False; print("Run mode stopped: No data for tick.")

        cv2.imshow("LMDB Inspector", frame)
        
        if run_mode_on and audio:
            play_audio(audio, blocking=True)
        
        wait_time = 1 if run_mode_on else 0
        key = cv2.waitKey(wait_time) & 0xFF
        
        if key == ord('r'):
            run_mode_on = not run_mode_on
            status = "ON" if run_mode_on else "OFF"
            print(f"Run mode toggled {status}")
            if not run_mode_on: # Print controls when stopping run mode
                print("\n--- CONTROLS ---")
                print("q: quit | j/k: next/prev tick | p: next player | t: switch team")
                print("a: play audio | o: toggle overlay | r: toggle RUN mode")
            continue

        if run_mode_on:
            current_tick += TICKS_PER_FRAME
            if current_round in round_info and current_tick > round_info[current_round]['end']:
                print("Run mode stopped: End of round reached.")
                run_mode_on = False
        else:
            if key == ord('q'): break
            elif key == ord('j'): current_tick += TICKS_PER_FRAME
            elif key == ord('k'): current_tick -= TICKS_PER_FRAME
            elif key == ord('p'): current_player_idx = (current_player_idx + 1) % 5
            elif key == ord('t'): current_team = 'CT' if current_team == 'T' else 'T'
            elif key == ord('a'): play_audio(audio, blocking=True)
            elif key == ord('o'): overlay_state = (overlay_state + 1) % 3

    cv2.destroyAllWindows()
    env.close()

if __name__ == "__main__":
    main()