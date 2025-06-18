'''
Information:
new_filename_base = f"{round_num:02d}_{team}_{player_name}_{start_tick}_{stop_tick}" #video name format .mp4

Information about 
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
Requirements:
parameters: --sql merged.db --input video_in_format_from_above.mp4 --out outfile.mp4
Steps:
- Extract from video filename: round number, player name, start tick, stop tick
- Open sqlite merged.db
- query all db information from table player between (and including) start and stop tick with playername given.
- Turn each tick into a well formatted string (we will add onto video)
- Fill out any missing ticks in the data from the db (check that we have continous data for every number between starttick and stoptick) with the text "NO_TICK_INFO"
- Align video frames and this tick data from the end
    The video is at 32frames/second, while the demo ticks ran at 64ticks/s
    For each frame starting from the back, add both matched tick strings from the db (add into each frame using cv2 or similar)
        In practice it makes sense to align from the back, and then write from front to back (but make sure to align from the back first so it fits perfectly! (also take below considerations into account))
    Do this until we reach the start tick.
    Any frames before this start tick should be labeled "BEFORE_START" (frames left over with no corresponding tick data)
    If we dont manage to use all tick rows from the db (some are left over at the front), print a warning with the amount of ticks left over.
    Write out modified video to --out outfile.mp4
'''

#!/usr/bin/env python3

import argparse
import sqlite3
import cv2
import os
import sys

# --- Configuration for the text overlay ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_COLOR = (255, 255, 255)  # White
LINE_TYPE = 2
TEXT_POSITION = (15, 30)  # (X, Y) from top-left corner
LINE_HEIGHT = 25 # Pixel height for each new line of text

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
        
        # Player names can contain underscores, so we parse from the ends.
        stop_tick = int(parts[-1])
        start_tick = int(parts[-2])
        round_num = int(parts[0])
        # Team is part[1], everything in between is the player name
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
        # Use Row factory to access columns by name
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
        
        # Return as a dictionary for fast lookups {tick: row_data}
        return {row['tick']: row for row in rows}

    except sqlite3.Error as e:
        print(f"Database error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn:
            conn.close()


def format_tick_data(row):
    """
    Turns a database row into a well-formatted string for display.
    """
    # Example format, can be customized as needed.
    return (
        f"Tick: {row['tick']} | "
        f"HP: {row['health']}/{row['armor']} | "
        f"Weapon: {row['active_weapon']}"
    )


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
            all_tick_strings.append(f"Tick: {current_tick} | NO_TICK_INFO")
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
    # --- 1. Open Video and get properties ---
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{input_path}'", file=sys.stderr)
        sys.exit(1)

    # Get video properties to create the output file correctly
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # --- FIX IS HERE: CHANGING THE VIDEO CODEC ---
    # The 'mp4v' codec can be unreliable. 'avc1' (H.264) is a much more standard
    # and compatible codec for .mp4 files, especially on macOS.
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    
    # --- IF 'avc1' FAILS, TRY THESE OTHER OPTIONS: ---
    # fourcc = cv2.VideoWriter_fourcc(*'h264')  # Alternative for H.264
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # The original, sometimes works
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Very compatible, but use .avi for output
    
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not open VideoWriter for output file '{output_path}'.", file=sys.stderr)
        print("This may be due to an unsupported codec ('fourcc') on your system.", file=sys.stderr)
        print("Try changing the fourcc code in the script (e.g., to 'h264' or 'XVID').", file=sys.stderr)
        cap.release()
        sys.exit(1)

    
    print("\n-> Processing video...")
    print(f"   - Input video: {total_frames} frames at {video_fps:.2f} FPS.")
    print(f"   - Game data: {len(ordered_tick_strings)} ticks to map ({TICKS_PER_FRAME} ticks/frame).")

    if abs(video_fps - EXPECTED_VIDEO_FPS) > 1:
        print(f"   - Warning: Video FPS ({video_fps:.2f}) is not the expected {EXPECTED_VIDEO_FPS}. Timing may be slightly off.")

    # --- 2. Align Tick Data to Frames from the END ---
    frame_to_text_map = ["BEFORE_START"] * total_frames
    tick_idx = len(ordered_tick_strings) - 1

    for frame_idx in range(total_frames - 1, -1, -1):
        if tick_idx >= (TICKS_PER_FRAME - 1):
            # We have enough ticks left for this frame
            texts_for_frame = []
            for i in range(TICKS_PER_FRAME):
                texts_for_frame.append(ordered_tick_strings[tick_idx - i])
            
            # Combine the two tick strings for this frame, reversing to show older tick first
            frame_to_text_map[frame_idx] = "\n".join(reversed(texts_for_frame))
            tick_idx -= TICKS_PER_FRAME
        else:
            # Not enough ticks left, so this frame is before the start of data
            # The loop will naturally stop and the rest will remain "BEFORE_START"
            break
            
    # --- 3. Check for leftover ticks ---
    ticks_left_over = tick_idx + 1
    if ticks_left_over > 0:
        print(f"\n   - WARNING: {ticks_left_over} tick(s) from the start of the data range could not be mapped to any video frame.")
        print(f"     This can happen if the video clip is shorter than the tick data range implies.")

    # --- 4. Read, Annotate, and Write each Frame ---
    print(f"   - Writing annotated video to '{output_path}'...")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get the pre-calculated text for this frame
        text_to_display = frame_to_text_map[frame_count]
        
        # Add text to frame. Handles multi-line strings.
        y = TEXT_POSITION[1]
        for line in text_to_display.split('\n'):
            cv2.putText(frame, line, (TEXT_POSITION[0], y), FONT, FONT_SCALE, FONT_COLOR, LINE_TYPE)
            y += LINE_HEIGHT
        
        out.write(frame)
        frame_count += 1
        # Simple progress indicator
        if frame_count % 100 == 0:
            print(f"     Processed {frame_count}/{total_frames} frames...", end='\r')

    print(f"\n   - Successfully processed {frame_count} frames.")

    # --- 5. Release resources ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Annotates a CS:GO video with player data from a SQLite database.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--sql", required=True, help="Path to the merged.db SQLite database file.")
    parser.add_argument("--input", required=True, help="Path to the input video file. \nFilename must be in the format: 'round_team_player_start_stop.mp4'")
    parser.add_argument("--out", required=True, help="Path for the output annotated video file.")
    args = parser.parse_args()

    # Step 1: Extract info from filename
    player_name, start_tick, stop_tick = parse_filename(args.input)

    # Step 2: Query the database
    db_data = fetch_player_data(args.sql, player_name, start_tick, stop_tick)
    
    # Step 3: Prepare tick strings, filling in any gaps
    ordered_tick_strings = prepare_tick_strings(db_data, start_tick, stop_tick)
    
    # Step 4: Process the video
    process_video(args.input, args.out, ordered_tick_strings)

    print("\nScript finished successfully!")


if __name__ == "__main__":
    main()