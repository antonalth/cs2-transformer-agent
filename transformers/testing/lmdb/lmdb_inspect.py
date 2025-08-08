#!/usr/bin/env python3
"""
lmdb_inspect.py - An interactive tool to inspect and verify the contents
of a CS2 data LMDB.

This script allows you to navigate through the dataset frame-by-frame,
switch player perspectives, play audio, and view detailed metadata overlays
to ensure the data was processed correctly.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import lmdb
import msgpack
import msgpack_numpy as m
import numpy as np
import sounddevice as sd

# --- Configuration ---
GAME_TICKS_PER_SEC = 64
EXPECTED_VIDEO_FPS = 32
TICKS_PER_FRAME = GAME_TICKS_PER_SEC // EXPECTED_VIDEO_FPS
AUDIO_SAMPLE_RATE = 44100
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
FONT_COLOR = (0, 255, 0)  # Green
SHADOW_COLOR = (0, 0, 0) # Black
LINE_TYPE = 1
TEXT_START_POS = (10, 20)
LINE_HEIGHT = 16

# Make msgpack aware of numpy for deserialization
m.patch()

def play_audio(audio_chunk):
    """Plays a raw PCM audio chunk using sounddevice."""
    if not audio_chunk:
        print("No audio chunk to play for this frame.")
        return
    try:
        # The audio is raw PCM 16-bit stereo, so we need to convert it to a NumPy array
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16).reshape(-1, 2)
        sd.play(audio_array, samplerate=AUDIO_SAMPLE_RATE)
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
    """
    Finds the correct player's data from the list based on their liveness.
    The list only contains data for living players, so we must map the
    player_idx (0-4) to the correct index in the shortened list.
    """
    if not (team_alive_mask >> player_idx) & 1:
        return None, None, None # Player is dead

    # Count how many living players come before our target player_idx
    living_players_before = bin(team_alive_mask & ((1 << player_idx) - 1)).count('1')
    
    if living_players_before < len(player_data_list):
        player_input, jpeg, audio = player_data_list[living_players_before]
        return player_input, jpeg, audio
    
    return None, None, None # Should not happen if data is consistent

def main():
    parser = argparse.ArgumentParser(description="Interactive LMDB Inspector for CS2 Data.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("lmdb_path", type=Path, help="Path to the LMDB directory.")
    parser.add_argument("round", type=int, nargs='?', default=1, help="Starting round number (default: 1).")
    parser.add_argument("team", type=str, nargs='?', default='T', choices=['T', 'CT'], help="Starting team (default: T).")
    parser.add_argument("player_idx", type=int, nargs='?', default=0, help="Starting player index [0-4] (default: 0).")
    parser.add_argument("tick", type=int, nargs='?', default=-1, help="Starting tick. If -1, starts at the beginning of the round (default: -1).")
    # --- NEW: Added debug flag ---
    parser.add_argument("--debug", action='store_true', help="List the first 50 keys in the LMDB and exit.")
    args = parser.parse_args()

    if not args.lmdb_path.exists():
        print(f"Error: LMDB path not found at: {args.lmdb_path}")
        sys.exit(1)

    env = lmdb.open(str(args.lmdb_path), readonly=True, lock=False)

    # --- NEW: Debug functionality to dump keys ---
    if args.debug:
        print(f"--- DUMPING FIRST 50 KEYS FROM {args.lmdb_path} ---")
        count = 0
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                print(key.decode('utf-8'))
                count += 1
                if count >= 50:
                    break
        print("--- END OF KEY DUMP ---")
        env.close()
        sys.exit(0)

    with env.begin() as txn:
        # Find the demoname and round start/end ticks
        info_key = [k for k in txn.cursor().iternext(keys=True, values=False) if k.endswith(b'_INFO')]
        if not info_key:
            print("Error: Could not find INFO key in the database.")
            sys.exit(1)
        demoname = info_key[0].removesuffix(b'_INFO').decode('utf-8')
        metadata = json.loads(txn.get(info_key[0]))
        round_info = {r[0]: {'start': r[1], 'end': r[2]} for r in metadata['rounds']}

    # --- Initial State ---
    current_round = args.round
    current_team = args.team
    current_player_idx = args.player_idx
    overlay_on = True

    if args.tick == -1:
        current_tick = round_info[current_round]['start']
    else:
        current_tick = args.tick
        
    cv2.namedWindow("LMDB Inspector")
    
    while True:
        # Construct the key and fetch data
        key_str = f"{demoname}_round_{current_round:03d}_team_{current_team}_tick_{current_tick:08d}"
        key_bytes = key_str.encode('utf-8')
        
        with env.begin() as txn:
            value = txn.get(key_bytes)
        
        if value:
            data = msgpack.unpackb(value)
            game_state = data['game_state'][0] # Unpack from the single-element array
            player_data_list = data['player_data']

            # Get the specific POV we want to display
            player_input, jpeg, audio = get_player_data_for_pov(player_data_list, current_player_idx, game_state['team_alive'])

            if jpeg:
                # Decode the JPEG image
                frame_np = np.frombuffer(jpeg, dtype=np.uint8)
                frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
                
                # Draw overlay if enabled
                if overlay_on:
                    pos = draw_text(frame, f"KEY: {key_str}", TEXT_START_POS)
                    pos = draw_text(frame, f"POV: Player {current_player_idx} ({current_team})", pos)
                    pos = draw_text(frame, "-"*40, pos)
                    # Display game state
                    pos = draw_text(frame, f"[GAME STATE] Round: {current_round} | Tick: {game_state['tick']}", pos)
                    pos = draw_text(frame, f"Team Alive Mask: {game_state['team_alive']:05b} | Enemy Alive Mask: {game_state['enemy_alive']:05b}", pos)
                    # Display player state
                    pos = draw_text(frame, "-"*40, pos)
                    pos = draw_text(frame, "[PLAYER INPUT]", pos)
                    pos = draw_text(frame, f"Health: {player_input['health']} | Armor: {player_input['armor']} | Money: ${player_input['money']}", pos)
                    pos = draw_text(frame, f"Pos: ({player_input['pos'][0]:.1f}, {player_input['pos'][1]:.1f}, {player_input['pos'][2]:.1f})", pos)
                    pos = draw_text(frame, f"Mouse: ({player_input['mouse'][0]:.3f}, {player_input['mouse'][1]:.3f})", pos)
                    pos = draw_text(frame, f"Keyboard Mask: {int(player_input['keyboard_bitmask'])}", pos)
                    pos = draw_text(frame, f"Eco Mask: {int(player_input['eco_bitmask'][0])} {int(player_input['eco_bitmask'][1])}", pos)
                    pos = draw_text(frame, f"Inv Mask: {int(player_input['inventory_bitmask'][0])} {int(player_input['inventory_bitmask'][1])}", pos)
                    pos = draw_text(frame, f"Wep Mask: {int(player_input['active_weapon_bitmask'][0])} {int(player_input['active_weapon_bitmask'][1])}", pos)

                print("\n" + "="*80)
                print(f"Displaying: {key_str}")
                print("\n--- GAME STATE ---")
                print(game_state)
                print("\n--- PLAYER INPUT (POV) ---")
                print(player_input)
                print("\n--- CONTROLS ---")
                print("q: quit | j: next tick | k: prev tick | p: next player | t: switch team | a: play audio | o: toggle overlay")

            else:
                frame = create_placeholder_frame(1280, 720, "PLAYER DEAD")

        else:
            frame = create_placeholder_frame(1280, 720, "NO DATA FOR TICK")
            print(f"No data found for key: {key_str}")
            audio = None

        cv2.imshow("LMDB Inspector", frame)
        
        # --- Handle User Input ---
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('j'):
            current_tick += TICKS_PER_FRAME
            if current_round in round_info and current_tick > round_info[current_round]['end']:
                current_tick = round_info[current_round]['end']
                print("At end of round.")
        elif key == ord('k'):
            current_tick -= TICKS_PER_FRAME
            if current_round in round_info and current_tick < round_info[current_round]['start']:
                current_tick = round_info[current_round]['start']
                print("At start of round.")
        elif key == ord('p'):
            current_player_idx = (current_player_idx + 1) % 5
        elif key == ord('t'):
            current_team = 'CT' if current_team == 'T' else 'T'
        elif key == ord('a'):
            play_audio(audio)
        elif key == ord('o'):
            overlay_on = not overlay_on

    cv2.destroyAllWindows()
    env.close()
    
if __name__ == "__main__":
    main()