#!/usr/bin/env python3

import argparse
import sqlite3
import cv2
import os
import sys

# --- Configuration for the text overlay ---
# Switched to a more "serious" font, smaller scale, and compact line height
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 0.45
FONT_COLOR = (50, 255, 50)  # High-visibility green (in BGR format)
SHADOW_COLOR = (0, 0, 0)     # Black shadow for readability
LINE_TYPE = 1
TEXT_POSITION = (10, 20)  # (X, Y) from top-left corner
LINE_HEIGHT = 18 # Pixel height for each new line of text

# --- Constants based on the problem description ---
EXPECTED_VIDEO_FPS = 32
GAME_TICKS_PER_SEC = 64
TICKS_PER_FRAME = GAME_TICKS_PER_SEC // EXPECTED_VIDEO_FPS


def parse_filename(filepath):
    """
    Extracts metadata from a filename with the format:
    {round_num:02d}_{team}_{player_name}_{start_tick}_{stop_tick}.mp4
    """
    try:
        basename = os.path.basename(filepath)
        filename_no_ext, _ = os.path.splitext(basename)
        parts = filename_no_ext.split('_')
        
        stop_tick = int(parts[-1])
        start_tick = int(parts[-2])
        round_num = int(parts[0])
        player_name = '_'.join(parts[2:-2])

        if not player_name:
             raise ValueError("Could not extract a valid player name.")

        print(f"-> Parsed from filename:")
        print(f"   - Round Number: {round_num}")
        print(f"   - Player Name:  '{player_name}'")
        print(f"   - Start Tick:   {start_tick}")
        print(f"   - Stop Tick:    {stop_tick}")
        
        return player_name, start_tick, stop_tick
        
    except (IndexError, ValueError) as e:
        print(f"Error: Could not parse filename '{filepath}'. It does not match the expected format.", file=sys.stderr)
        print(f"Expected format: 'round_team_player_name_starttick_stoptick.mp4'", file=sys.stderr)
        print(f"Original error: {e}", file=sys.stderr)
        sys.exit(1)


def fetch_player_data(db_path, player_name, start_tick, stop_tick):
    """
    Queries the SQLite database for all player data for a given player and tick range.
    """
    print(f"\n-> Connecting to database '{db_path}'...")
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = """
            SELECT * FROM player
            WHERE playername = ? AND tick BETWEEN ? AND ?
            ORDER BY tick
        """
        cursor.execute(query, (player_name, start_tick, stop_tick))
        rows = cursor.fetchall()
        print(f"   - Found {len(rows)} data rows for player '{player_name}' between ticks {start_tick} and {stop_tick}.")
        
        return {row['tick']: row for row in rows}

    except sqlite3.Error as e:
        print(f"Database error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn:
            conn.close()


def format_tick_data(row):
    """
    Turns a database row into a detailed, multi-line string for display.
    """
    # Helper to avoid long lines for keyboard input
    keys = row['keyboard_input'] if row['keyboard_input'] else "None"

    # Helper for buyzone status
    buyzone_text = "Yes" if row['is_in_buyzone'] else "No"

    # Format all data into separate lines for clarity
    line1 = f"TICK: {row['tick']}"
    line2 = f"HP: {row['health']:<3} | ARMOR: {row['armor']:<3} | MONEY: ${row['money']}"
    line3 = f"POS: ({row['position_x']:.2f}, {row['position_y']:.2f}, {row['position_z']:.2f})"
    line4 = f"WEAPON: {row['active_weapon']}"
    # Inventory can be long, so it gets its own line
    line5 = f"INVENTORY: {row['inventory']}"
    line6 = f"KEYS: {keys} | MOUSE: ({row['mouse_x']:.4f}, {row['mouse_y']:.4f})"
    line7 = f"IN BUYZONE: {buyzone_text}"

    return "\n".join([line1, line2, line3, line4, line5, line6, line7])


def prepare_tick_strings(db_data, start_tick, stop_tick):
    """
    Creates a complete list of strings for every tick in the range,
    filling in missing data with 'NO_TICK_INFO'.
    """
    print("\n-> Preparing tick data strings for video overlay...")
    all_tick_strings = []
    missing_ticks = 0
    for current_tick in range(start_tick, stop_tick + 1):
        if current_tick in db_data:
            formatted_string = format_tick_data(db_data[current_tick])
            all_tick_strings.append(formatted_string)
        else:
            # Create a minimal string for missing ticks
            all_tick_strings.append(f"TICK: {current_tick}\nSTATUS: NO_TICK_INFO")
            missing_ticks += 1
    
    if missing_ticks > 0:
        print(f"   - Warning: Filled in {missing_ticks} missing ticks with 'NO_TICK_INFO'.")
    else:
        print("   - All ticks in range were found in the database.")
        
    return all_tick_strings


def process_video(input_path, output_path, ordered_tick_strings):
    """
    Reads the input video, aligns tick data from the end, overlays the text,
    and writes the new video to the output path.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{input_path}'", file=sys.stderr)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use 'avc1' (H.264) for better .mp4 compatibility, especially on macOS/Linux
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not open VideoWriter for output file '{output_path}'.", file=sys.stderr)
        print("This may be due to an unsupported codec ('fourcc') on your system. Try 'mp4v' or 'XVID' (with .avi).", file=sys.stderr)
        cap.release()
        sys.exit(1)
        
    print("\n-> Processing video...")
    print(f"   - Input video: {total_frames} frames at {video_fps:.2f} FPS.")
    print(f"   - Game data: {len(ordered_tick_strings)} ticks to map ({TICKS_PER_FRAME} ticks/frame).")

    if abs(video_fps - EXPECTED_VIDEO_FPS) > 1:
        print(f"   - Warning: Video FPS ({video_fps:.2f}) is not the expected {EXPECTED_VIDEO_FPS}. Timing may be slightly off.")

    frame_to_text_map = ["BEFORE_START"] * total_frames
    tick_idx = len(ordered_tick_strings) - 1

    for frame_idx in range(total_frames - 1, -1, -1):
        if tick_idx >= (TICKS_PER_FRAME - 1):
            texts_for_frame = []
            for i in range(TICKS_PER_FRAME):
                texts_for_frame.append(ordered_tick_strings[tick_idx - i])
            
            # Combine the two tick strings, separated by a blank line for readability
            frame_to_text_map[frame_idx] = "\n\n".join(reversed(texts_for_frame))
            tick_idx -= TICKS_PER_FRAME
        else:
            break
            
    ticks_left_over = tick_idx + 1
    if ticks_left_over > 0:
        print(f"\n   - WARNING: {ticks_left_over} tick(s) from the start of the data range could not be mapped to any video frame.")

    print(f"   - Writing annotated video to '{output_path}'...")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        text_to_display = frame_to_text_map[frame_count]
        
        # Draw text with a shadow for better readability
        y = TEXT_POSITION[1]
        for line in text_to_display.split('\n'):
            # Draw shadow (black, offset by 1px)
            shadow_pos = (TEXT_POSITION[0] + 1, y + 1)
            cv2.putText(frame, line, shadow_pos, FONT, FONT_SCALE, SHADOW_COLOR, LINE_TYPE, cv2.LINE_AA)
            # Draw main text (green)
            text_pos = (TEXT_POSITION[0], y)
            cv2.putText(frame, line, text_pos, FONT, FONT_SCALE, FONT_COLOR, LINE_TYPE, cv2.LINE_AA)
            y += LINE_HEIGHT
        
        out.write(frame)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"     Processed {frame_count}/{total_frames} frames...", end='\r')

    print(f"\n   - Successfully processed {frame_count} frames.")

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Annotates a CS:GO video with detailed player data from a SQLite database.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--sql", required=True, help="Path to the merged.db SQLite database file.")
    parser.add_argument("--input", required=True, help="Path to the input video file. \nFilename must be in the format: 'round_team_player_start_stop.mp4'")
    parser.add_argument("--out", required=True, help="Path for the output annotated video file.")
    args = parser.parse_args()

    player_name, start_tick, stop_tick = parse_filename(args.input)
    db_data = fetch_player_data(args.sql, player_name, start_tick, stop_tick)
    ordered_tick_strings = prepare_tick_strings(db_data, start_tick, stop_tick)
    process_video(args.input, args.out, ordered_tick_strings)

    print("\nScript finished successfully!")


if __name__ == "__main__":
    main()