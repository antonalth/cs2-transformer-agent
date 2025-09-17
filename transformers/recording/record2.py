import argparse
import logging
import sqlite3
import time
import sys
import shutil
import signal
from pathlib import Path
from urllib.parse import quote_plus

# Third-party libraries that need to be installed:
# pip install requests
try:
    import requests
except ImportError:
    print("Error: 'requests' library not found. Please install it using 'pip install requests'")
    sys.exit(1)

import re

def sanitize_player_name(player_name: str) -> str:
    """
    Sanitizes a player name to make it safe for use in filenames.
    - Replaces spaces with underscores.
    - Removes any character that is NOT an alphanumeric character, underscore, or hyphen.
    - Strips leading/trailing whitespace.
    """
    if not player_name:
        return "unknown_player"
    
    name = player_name.replace(' ', '_')
    sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '', name)
    return sanitized_name.strip()

# --- Globals for Script Instance ---
# These are set in main() after parsing arguments.
ARGS = None
DB_CONN = None
TEMP_RECORD_DIR = None
CLIENT_ID = None
HTTP_SERVER_URL = "http://localhost:8080"

# --- Logging Setup ---
LOG = logging.getLogger("CS2 Recorder")

def setup_logging(debug=False):
    """Configures the logging for the script."""
    level = logging.DEBUG if debug else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    LOG.addHandler(handler)
    LOG.setLevel(level)
    LOG.name = f"CS2 Recorder (ID: {CLIENT_ID})"

def send_command(command, wait_after=0.5):
    """Sends a command to the game client via the HTTP API server."""
    if not CLIENT_ID:
        LOG.error("Cannot send command, client ID is not set.")
        return

    encoded_command = quote_plus(command)
    url = f"{HTTP_SERVER_URL}/run?id={CLIENT_ID}&cmd={encoded_command}"
    
    LOG.debug(f"Sending command: {command}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        LOG.debug(f"Server responded: {response.text}")
        time.sleep(wait_after)
    except requests.exceptions.RequestException as e:
        LOG.error(f"Failed to send command to HTTP API: {e}")
        LOG.error("Is the server.ts script running? Exiting.")
        cleanup()

def wait_for_ffmpeg_to_finish():
    """
    Polls the HTTP API until ffmpeg.exe is no longer reported as running.
    Includes a retry mechanism to handle server timeouts under heavy load.
    """
    LOG.info("Recording in progress... waiting for ffmpeg.exe to complete.")
    url = f"{HTTP_SERVER_URL}/running?id={CLIENT_ID}&name=ffmpeg.exe"
    
    is_recording = True
    max_retries = 5  # Maximum number of consecutive timeouts before giving up
    retry_count = 0

    while is_recording:
        try:
            # Increased timeout for more resilience on a loaded system
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Reset retry counter on a successful connection
            retry_count = 0
            
            is_running = response.json()
            
            if is_running:
                LOG.debug("ffmpeg.exe is still running...")
                time.sleep(2)  # Poll interval
            else:
                is_recording = False
                
        except (requests.exceptions.RequestException, ValueError) as e:
            retry_count += 1
            LOG.warning(f"Error polling for ffmpeg status: {e}")
            if retry_count >= max_retries:
                LOG.error(f"Could not confirm ffmpeg status after {max_retries} retries. Assuming it has finished to avoid getting stuck.")
                is_recording = False  # Break loop on repeated errors
            else:
                LOG.warning(f"Retrying poll... ({retry_count}/{max_retries})")
                time.sleep(5) # Wait longer before retrying if server is struggling

    LOG.info("ffmpeg.exe process has finished or is assumed to have finished.")
    LOG.info("Waiting an additional 5 seconds for file handles to be released.")
    time.sleep(5) # Increased this wait time as a final safety measure

def setup_environment(demo_file: Path):
    """Performs all initial setup steps using the HTTP API."""
    full_path = str(demo_file.resolve())
    LOG.info(f"Loading demo: {full_path}")
    send_command(f'playdemo "{full_path}"')
    
    LOG.info("Waiting 25 seconds for demo to load...")
    time.sleep(25)
    
    send_command("demo_pause")

    LOG.info(f"Setting up temporary directory: {TEMP_RECORD_DIR}")
    if TEMP_RECORD_DIR.exists():
        LOG.warning("Temp directory already exists. Deleting contents.")
        shutil.rmtree(TEMP_RECORD_DIR)
    TEMP_RECORD_DIR.mkdir(parents=True)

    LOG.info("Sending initial recording setup commands...")
    temp_record_path_str = str(TEMP_RECORD_DIR.resolve())
    
    send_command("mirv_streams record startMovieWav 1", wait_after=1)
    send_command(f'mirv_streams record name "{temp_record_path_str}"', wait_after=1)
    send_command("mirv_streams record screen enabled 1", wait_after=1)
    send_command("mirv_streams record screen settings afxDefault", wait_after=1)
    send_command("exec ffmpeg.cfg", wait_after=1)
    send_command("n3", wait_after=1) # A custom bind/alias?
    send_command("mirv_streams record fps 32", wait_after=1)
    send_command("demoui; cl_drawhud_force_radar -1; spec_mode 0", wait_after=1)
    send_command("volume 0.5; spec_show_xray 0; sv_cheats 1; cl_hide_avatar_images 1; r_show_build_info false", wait_after=1)

    LOG.info("Setup complete. Ready to record clips.")

def process_recordings(demo_file: Path, output_dir: Path):
    """Iterates through DB entries and records each clip based on override settings."""
    global DB_CONN
    
    sql_file = Path(ARGS.sql)
    if not sql_file.exists():
        LOG.error(f"SQL database not found at: {sql_file}")
        return
        
    DB_CONN = sqlite3.connect(sql_file)
    DB_CONN.row_factory = sqlite3.Row
    cursor = DB_CONN.cursor()
    
    cursor.execute("SELECT * FROM RECORDING")
    all_db_entries = cursor.fetchall()
    
    entries_to_record_queue = []
    skipped_due_to_exist_count = 0
    skipped_due_to_db_recorded_count = 0
    updated_db_consistency_count = 0

    demo_name = demo_file.stem
    final_output_dir = output_dir / demo_name
    final_output_dir.mkdir(parents=True, exist_ok=True)
    LOG.info(f"Final recordings will be saved to: {final_output_dir}")

    for entry in all_db_entries:
        round_num = entry['roundnumber']
        start_tick = entry['starttick']
        stop_tick = entry['stoptick']
        player_name = entry['playername']
        team = entry['team']
        db_is_recorded = entry['is_recorded']
        db_filepath = entry['recording_filepath']

        # Construct potential output paths
        new_filename_base = f"{round_num:02d}_{team}_{sanitize_player_name(player_name)}_{start_tick}_{stop_tick}"
        dest_mp4_path = final_output_dir / f"{new_filename_base}.mp4"
        dest_wav_path = final_output_dir / f"{new_filename_base}.wav"

        # Check if output files already exist on disk
        file_exists_on_disk = dest_mp4_path.exists() and dest_wav_path.exists()
        
        if ARGS.override == 2:
            # Override 2: Always re-record
            entries_to_record_queue.append(entry)
            LOG.debug(f"Queuing R{round_num}, P:{player_name} (Override 2: always re-record).")
        elif ARGS.override == 1:
            # Override 1: Re-record if DB says recorded but file missing OR not recorded.
            # If file exists on disk, DB says recorded, and the DB's recorded_filepath also points to an existing file, then skip.
            db_recorded_filepath_exists = db_filepath and Path(db_filepath).resolve().exists()
            if db_is_recorded and file_exists_on_disk and db_recorded_filepath_exists:
                LOG.debug(f"Skipping R{round_num}, P:{player_name}. Already recorded and file(s) exist (--override 1).")
                skipped_due_to_db_recorded_count += 1
            else:
                entries_to_record_queue.append(entry)
                LOG.debug(f"Queuing R{round_num}, P:{player_name} (Override 1: file missing or DB not recorded).")
        else: # ARGS.override == 0 (Default behavior)
            # Default (0): Only record if DB says not recorded AND output file(s) don't exist.
            if db_is_recorded:
                LOG.debug(f"Skipping R{round_num}, P:{player_name}. Already marked recorded in DB (--override 0).")
                skipped_due_to_db_recorded_count += 1
            elif file_exists_on_disk:
                LOG.info(f"Skipping R{round_num}, P:{player_name}. Output file(s) already exist (--override 0).")
                skipped_due_to_exist_count += 1
                # If DB says not recorded but file(s) exist on disk, update DB for consistency
                LOG.info(f"  -> Updating DB: marking R{round_num}, P:{player_name} as recorded (file exists).")
                DB_CONN.execute(
                    "UPDATE RECORDING SET is_recorded = ?, recording_filepath = ? WHERE starttick = ? AND stoptick = ? AND playername = ?",
                    (True, str(dest_mp4_path.resolve()), start_tick, stop_tick, player_name)
                )
                updated_db_consistency_count += 1
            else:
                entries_to_record_queue.append(entry)
                LOG.debug(f"Queuing R{round_num}, P:{player_name} (DB not recorded, file(s) missing).")

    # Commit any DB updates made during the filtering process (for consistency)
    if updated_db_consistency_count > 0:
        DB_CONN.commit()
        LOG.info(f"Committed {updated_db_consistency_count} database updates (marking existing files as recorded).")

    if not entries_to_record_queue:
        LOG.info("No entries to record based on current criteria. Check --override options if you want to re-record.")
        return

    LOG.info(f"Found {len(all_db_entries)} total entries in DB.")
    LOG.info(f"Selected {len(entries_to_record_queue)} entries for recording.")
    if skipped_due_to_exist_count > 0:
        LOG.info(f"Skipped {skipped_due_to_exist_count} entries because files already existed and --override 0 was set.")
    if skipped_due_to_db_recorded_count > 0:
        LOG.info(f"Skipped {skipped_due_to_db_recorded_count} entries because they were already recorded according to DB/files and --override 0 or 1 was set.")


    # Now iterate over the queue and perform actual recording
    for entry in entries_to_record_queue:
        round_num = entry['roundnumber']
        start_tick = entry['starttick']
        stop_tick = entry['stoptick']
        player_name = entry['playername']
        team = entry['team']
        
        LOG.info(f"--- Starting recording for Round {round_num}, Player: {player_name} (Team: {team}) ---") # Added team to log
        
        send_command("mirv_cmd clear", wait_after=2)
        send_command(f"demo_gototick {start_tick}", wait_after=10)
        send_command(f"spec_player {player_name}", wait_after=2)
        
        stop_command = f'mirv_cmd addAtTick {stop_tick} "mirv_streams record end; demo_pause"'
        send_command(stop_command, wait_after=2)
        
        send_command("mirv_streams record start; demo_resume", wait_after=10)
        
        wait_for_ffmpeg_to_finish()
        
        LOG.info("Processing recorded files...")
        # Re-calculate filename base and dest paths. This is safe even if `entry` were mutable.
        new_filename_base = f"{round_num:02d}_{team}_{sanitize_player_name(player_name)}_{start_tick}_{stop_tick}"
        dest_mp4_path = final_output_dir / f"{new_filename_base}.mp4"
        dest_wav_path = final_output_dir / f"{new_filename_base}.wav"

        try:
            take_dirs = [d for d in TEMP_RECORD_DIR.iterdir() if d.is_dir() and d.name.startswith("take")]
            if not take_dirs:
                raise FileNotFoundError("No 'takeXXXX' sub-directory found. Recording likely failed.")
            
            take_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            take_dir = take_dirs[0]
            LOG.debug(f"Found recording sub-directory: {take_dir}")

            mp4_files = list(take_dir.glob("*.mp4"))
            wav_files = list(take_dir.glob("*.wav"))
            
            if not mp4_files: raise FileNotFoundError(f"No .mp4 file found in '{take_dir}'")
            if not wav_files: raise FileNotFoundError(f"No .wav file found in '{take_dir}'")
                
            source_mp4 = mp4_files[0]
            source_wav = wav_files[0]
            
            # These checks ensure that if we decide to re-record (e.g., override 1 or 2),
            # existing files are explicitly removed before moving new ones.
            if dest_mp4_path.exists():
                LOG.warning(f"Destination MP4 file exists, will be overwritten: {dest_mp4_path}")
                dest_mp4_path.unlink()
            if dest_wav_path.exists():
                LOG.warning(f"Destination WAV file exists, will be overwritten: {dest_wav_path}")
                dest_wav_path.unlink()
            
            shutil.move(str(source_mp4), str(dest_mp4_path))
            shutil.move(str(source_wav), str(dest_wav_path))
            LOG.info(f"Moved and renamed clip to: {dest_mp4_path}")

            LOG.debug(f"Removing processed take directory: {take_dir}")
            shutil.rmtree(take_dir)

            LOG.info("Updating database entry...")
            DB_CONN.execute(
                "UPDATE RECORDING SET is_recorded = ?, recording_filepath = ? WHERE starttick = ? AND stoptick = ? AND playername = ?",
                (True, str(dest_mp4_path.resolve()), start_tick, stop_tick, player_name)
            )
            DB_CONN.commit()
            LOG.info("Database updated successfully.")
            
        except FileNotFoundError as e:
            LOG.error(f"File handling failed for R{round_num}, P:{player_name}: {e}. Skipping database update for this entry.")
        except Exception as e:
            LOG.error(f"An unexpected error occurred during file processing for R{round_num}, P:{player_name}: {e}", exc_info=ARGS.debug)

        LOG.info(f"--- Finished processing R{round_num}, P:{player_name} ---")

def cleanup(signum=None, frame=None):
    """Cleans up resources. Can be called by 'finally' or signal handler."""
    LOG.info("\n--- Cleaning up resources ---")
    
    send_command("demo_pause; mirv_streams record end;", wait_after=1)

    if TEMP_RECORD_DIR and TEMP_RECORD_DIR.exists():
        LOG.info(f"Removing temporary directory: {TEMP_RECORD_DIR}")
        shutil.rmtree(TEMP_RECORD_DIR, ignore_errors=True)
        
    if DB_CONN:
        LOG.info("Closing SQLite database connection.")
        DB_CONN.close()
    
    LOG.info("Cleanup complete. Exiting.")
    if signum:
        sys.exit(0)

def main():
    """Main execution function."""
    global ARGS, CLIENT_ID, TEMP_RECORD_DIR
    
    parser = argparse.ArgumentParser(
        description="Automate CS2 video recording using an HTTP API. Controls a single game instance.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--id", required=True, help="The client ID to control via the HTTP API.")
    parser.add_argument("--sql", required=True, help="Path to the input.db SQLite database file.")
    parser.add_argument("--demofile", required=True, help="Path to the .dem demo file.")
    parser.add_argument("--out", required=True, help="Base path for the final output folder.")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug logging.")
    parser.add_argument(
        "--override",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Set the re-recording behavior:\n"
             "0: (Default) Record only entries not marked as recorded in the DB AND where output files do not exist.\n"
             "   If a file exists but DB says not recorded, DB is updated and file is SKIPPED.\n"
             "1: Record entries from (0) AND entries marked as recorded but where the video file in DB or target is missing.\n"
             "   If DB says recorded AND target files exist AND DB's filepath exists, it will be SKIPPED.\n"
             "2: Re-record all entries in the DB, overwriting any existing files."
    )
    
    ARGS = parser.parse_args()
    
    CLIENT_ID = ARGS.id
    TEMP_RECORD_DIR = Path(f"temp/temp_recording_id{CLIENT_ID}")
    
    setup_logging(ARGS.debug)
    
    signal.signal(signal.SIGINT, cleanup)

    demo_file = Path(ARGS.demofile)
    if not demo_file.exists():
        LOG.error(f"Demo file not found: {demo_file}")
        sys.exit(1)
        
    output_dir = Path(ARGS.out)

    try:
        setup_environment(demo_file)
        process_recordings(demo_file, output_dir)
        LOG.info("All recording tasks have been processed.")
    except Exception as e:
        LOG.error(f"An unhandled error occurred during execution: {e}", exc_info=ARGS.debug)
    finally:
        cleanup()

if __name__ == "__main__":
    main()