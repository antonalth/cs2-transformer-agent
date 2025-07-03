'''
Your task is to write annotate_many.py

parameters of script --sql input.db --data recording_folder --round 1 --team ct --out video.mp4

Information about input.db
    CREATE TABLE player (
        tick INTEGER,
        steamid INTEGER,
        playername TEXT,
        position_x REAL,
        position_y REAL,
        position_z REAL,
        inventory TEXT,
        active_weapon TEXT,
        health INTEGER,
        armor INTEGER,
        money INTEGER,
        keyboard_input TEXT,
        mouse_x REAL,
        mouse_y REAL,
        is_in_buyzone INTEGER,
        buy_sell_input TEXT,
        PRIMARY KEY (tick, steamid)
    )
     CREATE TABLE rounds (
        round INTEGER PRIMARY KEY,
        starttick INTEGER,
        freezetime_endtick INTEGER,
        endtick INTEGER,
        t_team TEXT,
        ct_team TEXT
    )
    CREATE TABLE RECORDING (
        roundnumber        INTEGER,
        starttick          INTEGER,
        stoptick           INTEGER,
        team               TEXT,
        playername         TEXT,
        is_recorded        BOOLEAN,
        recording_filepath TEXT,
        PRIMARY KEY (starttick, stoptick, playername)
    );
Steps:
- Extract from table RECORDING * where roundnumber = (passed in args), team = (passed in args)
    - if no rows, print notification and exit
    - if rows, check for all entries is_recorded = true, otherwise exit
    - rebuild file paths according to format from the db data, check for existance in recording_folder, if not found, exit
            f"{round_num:02d}_{team}_{player_name}_{start_tick}_{stop_tick}" #video name format .mp4 in av1
- query all db information from table player between (and including) start and stop tick given, once for EACH player entry from the earlier RECORDING query. 
- For each row from the RECORDING table and the corresponding tick data from the player table
    Fully Stringify each tick (will later add to each video frame)
    Fill out any missing ticks (check that we have continous data for every number between starttick and stoptick) with the text "NO_TICK_INFO"
    Align video frames and this tick data from the END going back forwards.
    The video is at 32frames/second, while the demo ticks ran at 64ticks/s
    For each frame starting from the back, add both matched tick strings from the db (add into each frame using cv2 or similar)
        In practice it makes sense to align from the back, and then write from front to back (but make sure to align from the back first so it fits perfectly! (also take below considerations into account))
    Do this until we reach the start tick.
    Any frames before this start tick should be labeled "BEFORE_START" (frames left over with no corresponding tick data)
    If we dont manage to use all tick rows from the db (some are left over at the front), print a warning with the amount of ticks left over.

- once we have overlayed all 5 player videos (povs), we need to merge all 5 videos (3 up top, 2 at bottom in a 3x2 structur) in one large higher res video.
- align the videos based on the calculated ticks from the end. Note: some videos are shorter than others since the players die and the recording stops. 
    So the step at the front where we accurately align each video frame with two ticks from the end of the video is essential to align all frames and povs later.
    Write out compiled video to --out outfile.mp4
'''#!/usr/bin/env python3

import argparse
import sqlite3
import cv2
import numpy as np
import os
import sys

# --- Configuration for the text overlay ---
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 0.45
FONT_COLOR = (50, 255, 50)      # High-visibility green (BGR)
SHADOW_COLOR = (0, 0, 0)          # Black shadow
LINE_TYPE = 1
TEXT_POSITION = (10, 20)
LINE_HEIGHT = 18

# --- Configuration for video processing ---
EXPECTED_VIDEO_FPS = 32
GAME_TICKS_PER_SEC = 64
TICKS_PER_FRAME = GAME_TICKS_PER_SEC // EXPECTED_VIDEO_FPS

def get_round_recordings(db_path, data_folder, round_num, team):
    """Queries the RECORDING table and validates that all required video files exist."""
    print("-> Step 1: Fetching and validating recording metadata...")
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM RECORDING WHERE roundnumber = ? AND team = ? ORDER BY playername"
        cursor.execute(query, (round_num, team))
        recordings = cursor.fetchall()

        if not recordings:
            print(f"Error: No recordings found for round {round_num} and team '{team}'. Exiting.", file=sys.stderr)
            sys.exit(1)

        print(f"   - Found {len(recordings)} recordings for team '{team}' in round {round_num}.")

        valid_recordings = []
        for rec in recordings:
            if not rec['is_recorded']:
                print(f"Error: Recording for player '{rec['playername']}' is marked as not recorded. Exiting.", file=sys.stderr)
                sys.exit(1)

            filename = f"{rec['roundnumber']:02d}_{rec['team']}_{rec['playername']}_{rec['starttick']}_{rec['stoptick']}.mp4"
            filepath = os.path.join(data_folder, filename)

            if not os.path.exists(filepath):
                print(f"Error: Video file not found at expected path: {filepath}. Exiting.", file=sys.stderr)
                sys.exit(1)
            
            rec_dict = dict(rec)
            rec_dict['filepath'] = filepath
            valid_recordings.append(rec_dict)

        print("   - All recordings are marked 'is_recorded' and all video files exist.")
        return valid_recordings

    except sqlite3.Error as e:
        print(f"Database error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn:
            conn.close()

def get_round_start_tick(db_path, round_num):
    """Gets the absolute start tick for the entire round."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT starttick FROM rounds WHERE round = ?", (round_num,))
        round_info = cursor.fetchone()
        if not round_info:
            print(f"Error: Could not find round {round_num} in the 'rounds' table. Exiting.", file=sys.stderr)
            sys.exit(1)
        return round_info['starttick']
    finally:
        if conn:
            conn.close()

def fetch_and_prepare_all_player_data(db_path, recordings):
    """For each recording, fetches and prepares all tick data strings."""
    print("\n-> Step 2: Fetching and preparing all player tick data...")
    all_player_data = {}
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        for rec in recordings:
            player_name, start_tick, stop_tick = rec['playername'], rec['starttick'], rec['stoptick']
            print(f"   - Fetching data for '{player_name}' (Ticks: {start_tick}-{stop_tick})...")
            
            query = "SELECT * FROM player WHERE playername = ? AND tick BETWEEN ? AND ? ORDER BY tick"
            cursor.execute(query, (player_name, start_tick, stop_tick))
            rows = cursor.fetchall()
            db_data = {row['tick']: row for row in rows}

            tick_strings = []
            for tick in range(start_tick, stop_tick + 1):
                if tick in db_data:
                    tick_strings.append(format_tick_data(db_data[tick]))
                else:
                    tick_strings.append(f"TICK: {tick}\nSTATUS: NO_TICK_INFO")

            all_player_data[player_name] = {
                'starttick': start_tick, 'stoptick': stop_tick,
                'filepath': rec['filepath'], 'tick_strings': tick_strings
            }
        
        print("   - All player data has been fetched and prepared.")
        return all_player_data

    except sqlite3.Error as e:
        print(f"Database error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn:
            conn.close()

# MODIFIED FUNCTION WITH ALL REQUESTED CHANGES
def format_tick_data(row):
    """
    Turns a database row into a detailed, multi-line string for display.
    Includes combined keyboard/buy inputs and high-precision position/mouse data.
    """
    # Combine keyboard and buy/sell inputs into a single string
    inputs = []
    if row['keyboard_input']:
        inputs.append(row['keyboard_input'])
    if row['buy_sell_input']:
        inputs.append(row['buy_sell_input'])
    
    input_str = ", ".join(inputs) if inputs else "None"

    # Assemble the multi-line display string
    line1 = f"TICK: {row['tick']}"
    line2 = f"HP: {row['health']:<3} | ARMOR: {row['armor']:<3} | ${row['money']}"
    line3 = f"WEAPON: {row['active_weapon']}"
    line4 = f"POS: ({row['position_x']:.3f}, {row['position_y']:.3f}, {row['position_z']:.3f})"
    line5 = f"INPUT: {input_str} | MOUSE: ({row['mouse_x']:.3f}, {row['mouse_y']:.3f})"

    return "\n".join([line1, line2, line3, line4, line5])


def create_placeholder_frame(width, height, text):
    """Creates a black frame with centered text (e.g., 'PLAYER DEAD')."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    (text_width, text_height), _ = cv2.getTextSize(text, FONT, 0.8, 2)
    text_x = (width - text_width) // 2
    text_y = (height + text_height) // 2
    cv2.putText(frame, text, (text_x, text_y), FONT, 0.8, (150, 150, 150), 2, cv2.LINE_AA)
    return frame

def create_compiled_video(output_path, player_data, round_start, effective_end_tick):
    """Composites all videos into a single output file, stopping when all players are dead."""
    print("\n-> Step 3: Compositing videos into final output...")

    captures, (tile_w, tile_h) = {}, (0, 0)
    player_names = sorted(player_data.keys())

    for name in player_names:
        cap = cv2.VideoCapture(player_data[name]['filepath'])
        if not cap.isOpened(): print(f"FATAL: Could not open video for {name}", file=sys.stderr); sys.exit(1)
        captures[name] = cap
        if tile_w == 0:
            tile_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            tile_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_w, output_h = tile_w * 3, tile_h * 2
    layout = {p: (c * tile_w, r * tile_h) for i, p in enumerate(player_names) for r, c in [(i // 3, i % 3)]}

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, EXPECTED_VIDEO_FPS, (output_w, output_h))
    
    if not out.isOpened():
        print(f"Error: Could not open VideoWriter for output file '{output_path}'.", file=sys.stderr)
        sys.exit(1)

    placeholder_dead = create_placeholder_frame(tile_w, tile_h, "PLAYER DEAD")
    placeholder_waiting = create_placeholder_frame(tile_w, tile_h, "WAITING TO START")
    placeholder_empty = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

    total_round_frames = (effective_end_tick - round_start + 1) // TICKS_PER_FRAME
    print(f"   - Assembling video from global timeline: {round_start} -> {effective_end_tick} ({total_round_frames} frames)")

    for frame_idx in range(total_round_frames):
        current_start_tick = round_start + (frame_idx * TICKS_PER_FRAME)
        final_frame = np.zeros((output_h, output_w, 3), dtype=np.uint8)

        for name in player_names:
            pov_data = player_data[name]
            tile_frame = None

            if current_start_tick >= pov_data['starttick'] and current_start_tick < pov_data['stoptick']:
                ret, frame = captures[name].read()
                if ret:
                    tick_offset = current_start_tick - pov_data['starttick']
                    string_idx_1 = tick_offset
                    string_idx_2 = tick_offset + 1
                    text_to_display = ""
                    if string_idx_1 < len(pov_data['tick_strings']): text_to_display += pov_data['tick_strings'][string_idx_1]
                    if string_idx_2 < len(pov_data['tick_strings']): text_to_display += "\n\n" + pov_data['tick_strings'][string_idx_2]
                    
                    y = TEXT_POSITION[1]
                    for line in text_to_display.split('\n'):
                        shadow_pos = (TEXT_POSITION[0] + 1, y + 1)
                        cv2.putText(frame, line, shadow_pos, FONT, FONT_SCALE, SHADOW_COLOR, LINE_TYPE, cv2.LINE_AA)
                        text_pos = (TEXT_POSITION[0], y)
                        cv2.putText(frame, line, text_pos, FONT, FONT_SCALE, FONT_COLOR, LINE_TYPE, cv2.LINE_AA)
                        y += LINE_HEIGHT
                    tile_frame = frame
                else:
                    tile_frame = placeholder_dead
            elif current_start_tick >= pov_data['stoptick']:
                tile_frame = placeholder_dead
            else:
                tile_frame = placeholder_waiting

            if name in layout:
                x, y = layout[name]
                final_frame[y:y+tile_h, x:x+tile_w] = tile_frame
        
        if len(player_names) == 5:
            x, y = tile_w * 2, tile_h
            final_frame[y:y+tile_h, x:x+tile_w] = placeholder_empty
            
        out.write(final_frame)
        
        if (frame_idx + 1) % 100 == 0:
            print(f"     Processed {frame_idx + 1}/{total_round_frames} composite frames...", end='\r')

    print(f"\n   - Successfully processed {total_round_frames} frames.")
    for cap in captures.values(): cap.release()
    out.release()


def main():
    parser = argparse.ArgumentParser(description="Compiles multiple player POV videos for a specific round into a single, synchronized grid video.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--sql", required=True, help="Path to the input.db SQLite database file.")
    parser.add_argument("--data", required=True, help="Path to the folder containing the POV recordings.")
    parser.add_argument("--round", required=True, type=int, help="The round number to process.")
    parser.add_argument("--team", required=True, choices=['ct', 't', 'CT', 'T'], help="The team to process (CT or T).")
    parser.add_argument("--out", required=True, help="Path for the final compiled output video file.")
    args = parser.parse_args()

    team_for_query = args.team.upper()

    recordings = get_round_recordings(args.sql, args.data, args.round, team_for_query)
    
    round_start_tick = get_round_start_tick(args.sql, args.round)
    effective_end_tick = max(rec['stoptick'] for rec in recordings)
    
    all_player_data = fetch_and_prepare_all_player_data(args.sql, recordings)

    create_compiled_video(args.out, all_player_data, round_start_tick, effective_end_tick)

    print(f"\nScript finished successfully! Output saved to: {args.out}")


if __name__ == "__main__":
    main()