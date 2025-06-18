'''
parameters:
--sql input.db --demofile path/to/file.dem

first, do setup:
    restart cs2 with hlae:

        LOG.info("restarting cs2+hlae")
        subprocess.Popen(["python", "s1a_restart_all.py"], cwd="../../cnn_scripts", creationflags=subprocess.CREATE_NEW_CONSOLE)
        time.sleep(15)
    start console connection with cs2, and load demo to playback:

        import from libs.mirv_client import connect as mirv_connect
        conn = mirv_connect()

        full_path = str(demo_file.resolve())
        conn.sendCommand(f'playdemo "{full_path}"')
        #sleep 25s
        #send command (like above)
            demo_pause

        #create folder with name recordings/TEMP_RECORD if not exists, if exists delete contents

        #send commands:(wait 1s between each one)
            mirv_streams record startMovieWav 1
            mirv_streams record name "C:\..\currentdir\TEMP_RECORD"
            mirv_streams record screen enabled 1
            mirv_streams record screen settings afxDefault
            exec ffmpeg.cfg
            n1
            mirv_streams record fps 32
            cl_drawhud 0; demoui; demoui; cl_drawhud_force_radar -1; spec_mode 0
            spec_show_xray 0; sv_cheats 1; cl_hide_avatar_images 1;

            
for each entry with starttick, stoptick, roundnumber, playername, team in the input.db RECORDING table
    #send commands (wait 2s between each one)
        mirv_cmd clear
        demo_gototick starttick #wait 4s after this one
        spec_player playername
        mirv_cmd addAtTick stoptick "mirv_streams record end; demo_pause"
        mirv_streams record start; demo_resume

    #poll at 1s intervals until there is no process called "ffmpeg.exe" anymore. (+3s wait-time)
    
    #if not exists create folder named recordings/recording_DEMOFILENAME
    #rename the mp4 file in temp_record to roundnumber_team_starttick_stoptick_playername.mp4 (same with the coressponding .wav) and move to the folder created earlier
    #update the corresponding sql entry to set is_recorded to TRUE, recording_filepath to the corresponding mp4 path

#cleanup (or when CTRL+C is pressed)
    #send commands:
        demo_pause; mirv_streams record end;
    
    #wait 2s
    #remove folder TEMP_RECORD
    #close sqlite


    
Additional Info:
    log to stdout most steps you take, add a --debug to print even more.
database format:
 (format:
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


'''

import argparse
import logging
import sqlite3
import subprocess, os
import time
import sys
import shutil
import signal
from pathlib import Path

# Third-party libraries that need to be installed:
# pip install psutil
try:
    import psutil
except ImportError:
    print("Error: psutil library not found. Please install it using 'pip install psutil'")
    sys.exit(1)

# Assuming mirv_client is in a 'libs' folder relative to this script
# or available in the Python path.
try:
    from libs.mirv_client import connect as mirv_connect
except ImportError:
    print("Error: libs.mirv_client not found. Make sure the 'libs' folder is in the correct location.")
    sys.exit(1)

# --- Globals for Cleanup ---
# These are defined globally so the signal handler can access them.
DB_CONN = None
MIRV_CONN = None
TEMP_RECORD_DIR = Path("recordings/TEMP_RECORD")

# --- Logging Setup ---
LOG = logging.getLogger("CS2 Recorder")

def setup_logging(debug=False):
    """Configures the logging for the script."""
    level = logging.DEBUG if debug else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    LOG.addHandler(handler)
    LOG.setLevel(level)

def send_command(command, wait_after=0.5):
    """Sends a command to the MIRV console connection and waits."""
    if MIRV_CONN:
        LOG.debug(f"Sending command: {command}")
        MIRV_CONN.sendCommand(command)
        time.sleep(wait_after)

def setup_environment(demo_file: Path):
    """Performs all initial setup steps."""
    global MIRV_CONN

    # 1. Restart CS2 with HLAE
    LOG.info("Restarting CS2 with HLAE...")
    
    # Calculate the absolute path to the cnn_scripts directory relative to the current script
    # Path(__file__).resolve() gives the absolute path to *this* script file.
    # .parent gets its parent directory.
    # / "../../cnn_scripts" appends the relative path.
    # .resolve() again resolves the ".." components to get the final absolute absolute path.
    cnn_scripts_dir = (Path(__file__).resolve().parent / "../../cnn_scripts").resolve()

    LOG.debug(f"Calculated cnn_scripts_dir: {cnn_scripts_dir}") # Add this for debug output


    script = os.path.join(cnn_scripts_dir, "s1a_restart_all.py")

    subprocess.Popen([
        "powershell", "-NoProfile", "-Command",
        "Start-Process", "python",
        "-ArgumentList", f"'{script}'"
    ], cwd=cnn_scripts_dir)
        
    LOG.info("Waiting 25 seconds for CS2 to launch...")
    time.sleep(25)

    # 2. Start console connection
    LOG.info("Connecting to MIRV console...")
    try:
        MIRV_CONN = mirv_connect()
    except:
        LOG.error(f"Failed to connect to MIRV")
        LOG.error("Is CS2 running with -afxDetour enabled?")
        sys.exit(1)
    
    LOG.info("Connection successful.")

    # 3. Load demo
    full_path = str(demo_file.resolve())
    LOG.info(f"Loading demo: {full_path}")
    send_command(f'playdemo "{full_path}"')
    
    LOG.info("Waiting 25 seconds for demo to load...")
    time.sleep(25)
    
    send_command("demo_pause")

    # 4. Prepare temporary recording directory
    LOG.info(f"Setting up temporary directory: {TEMP_RECORD_DIR}")
    if TEMP_RECORD_DIR.exists():
        LOG.warning("Temp directory already exists. Deleting contents.")
        shutil.rmtree(TEMP_RECORD_DIR)
    TEMP_RECORD_DIR.mkdir(parents=True)

    # 5. Send initial recording commands
    LOG.info("Sending initial recording setup commands...")
    # The 'name' command needs a fully resolved, backslash-escaped path for Windows
    temp_record_path_str = str(TEMP_RECORD_DIR.resolve())
    
    # Wait 1s between each one
    send_command("mirv_streams record startMovieWav 1", wait_after=1)
    send_command(f'mirv_streams record name "{temp_record_path_str}"', wait_after=1)
    send_command("mirv_streams record screen enabled 1", wait_after=1)
    send_command("mirv_streams record screen settings afxDefault", wait_after=1)
    send_command("exec ffmpeg.cfg", wait_after=1) # Assuming this config exists
    send_command("n3", wait_after=1) # A custom bind/alias?
    send_command("mirv_streams record fps 32", wait_after=1)
    send_command("demoui; cl_drawhud_force_radar -1; spec_mode 0", wait_after=1)
    send_command("volume 0.5; spec_show_xray 0; sv_cheats 1; cl_hide_avatar_images 1; r_show_build_info false", wait_after=1)

    LOG.info("Setup complete. Ready to record clips.")

def wait_for_ffmpeg_to_finish():
    """Polls running processes until ffmpeg.exe is no longer found."""
    LOG.info("Recording in progress... waiting for ffmpeg.exe to complete.")
    is_recording = True
    while is_recording:
        found_ffmpeg = False
        for proc in psutil.process_iter(['name']):
            if proc.info['name'].lower() == 'ffmpeg.exe':
                found_ffmpeg = True
                break
        
        if found_ffmpeg:
            LOG.debug("ffmpeg.exe is still running...")
            time.sleep(1) # Poll interval
        else:
            is_recording = False
    
    LOG.info("ffmpeg.exe process has finished.")
    LOG.info("Waiting an additional 3 seconds for file handles to be released.")
    time.sleep(3)


def process_recordings(demo_file: Path):
    """Iterates through DB entries and records each clip."""
    global DB_CONN
    
    # Connect to the SQLite database
    sql_file = Path(ARGS.sql)
    if not sql_file.exists():
        LOG.error(f"SQL database not found at: {sql_file}")
        return
        
    DB_CONN = sqlite3.connect(sql_file)
    # Use Row factory to access columns by name
    DB_CONN.row_factory = sqlite3.Row
    cursor = DB_CONN.cursor()
    
    # Query for unrecorded entries
    cursor.execute("SELECT * FROM RECORDING WHERE is_recorded IS NOT TRUE")
    record_entries = cursor.fetchall()
    
    if not record_entries:
        LOG.info("No unrecorded entries found in the database.")
        return

    LOG.info(f"Found {len(record_entries)} clips to record.")

    # Prepare final output directory
    demo_name = demo_file.stem
    final_output_dir = Path(f"recordings/recording_{demo_name}")
    final_output_dir.mkdir(parents=True, exist_ok=True)
    LOG.info(f"Final recordings will be saved to: {final_output_dir}")

    for entry in record_entries:
        round_num = entry['roundnumber']
        start_tick = entry['starttick']
        stop_tick = entry['stoptick']
        player_name = entry['playername']
        team = entry['team']
        
        LOG.info(f"--- Starting recording for Round {round_num}, Player: {player_name} ---")
        
        # 1. Send per-clip commands
        send_command("mirv_cmd clear", wait_after=2)
        send_command(f"demo_gototick {start_tick}", wait_after=10)
        send_command(f"spec_player {player_name}", wait_after=2)
        
        stop_command = f'mirv_cmd addAtTick {stop_tick} "mirv_streams record end; demo_pause"'
        send_command(stop_command, wait_after=2)
        
        # Start recording and resume
        send_command("mirv_streams record start; demo_resume", wait_after=10)
        
        # 2. Wait for ffmpeg to finish
        wait_for_ffmpeg_to_finish()
        
        # 3. Move and rename the recorded files
        LOG.info("Processing recorded files...")
        
        new_filename_base = f"{round_num:02d}_{team}_{player_name}_{start_tick}_{stop_tick}"
        
        try:
            # --- MODIFICATION START ---

            # First, find the 'takeXXXX' subdirectory created by HLAE.
            # We assume there will be only one since we process one clip at a time.
            take_dirs = [d for d in TEMP_RECORD_DIR.iterdir() if d.is_dir() and d.name.startswith("take")]
            
            if not take_dirs:
                raise FileNotFoundError("No 'takeXXXX' sub-directory found in temp directory. Recording likely failed.")
            
            if len(take_dirs) > 1:
                LOG.warning(f"Found multiple 'take' directories: {take_dirs}. Using the most recently modified one.")
                # Sort by modification time to get the latest one, just in case.
                take_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            take_dir = take_dirs[0]
            LOG.debug(f"Found recording sub-directory: {take_dir}")

            # Now, find the mp4 and wav files inside that specific take directory.
            mp4_files = list(take_dir.glob("*.mp4"))
            wav_files = list(take_dir.glob("*.wav"))
            
            if not mp4_files:
                raise FileNotFoundError(f"No .mp4 file found in '{take_dir}'")
            if not wav_files:
                raise FileNotFoundError(f"No .wav file found in '{take_dir}'")
                
            source_mp4 = mp4_files[0]
            source_wav = wav_files[0]
            
            # --- MODIFICATION END ---
            
            dest_mp4_path = final_output_dir / f"{new_filename_base}.mp4"
            dest_wav_path = final_output_dir / f"{new_filename_base}.wav"
            
            shutil.move(str(source_mp4), str(dest_mp4_path))
            shutil.move(str(source_wav), str(dest_wav_path))
            LOG.info(f"Moved and renamed clip to: {dest_mp4_path}")

            # Clean up the now-empty take directory
            LOG.debug(f"Removing processed take directory: {take_dir}")
            shutil.rmtree(take_dir)

            # 4. Update the database
            LOG.info("Updating database entry...")
            update_cursor = DB_CONN.cursor()
            update_cursor.execute(
                """
                UPDATE RECORDING
                SET is_recorded = ?, recording_filepath = ?
                WHERE starttick = ? AND stoptick = ? AND playername = ?
                """,
                (True, str(dest_mp4_path.resolve()), start_tick, stop_tick, player_name)
            )
            DB_CONN.commit()
            LOG.info("Database updated successfully.")
            
        except FileNotFoundError as e:
            LOG.error(f"File handling failed: {e}. Skipping database update for this entry.")
        except Exception as e:
            LOG.error(f"An unexpected error occurred during file processing: {e}")

        LOG.info(f"--- Finished recording for Round {round_num}, Player: {player_name} ---")


def cleanup(signum=None, frame=None):
    """Cleans up resources. Can be called by 'finally' or signal handler."""
    LOG.info("\n--- Cleaning up resources ---")
    
    if MIRV_CONN:
        try:
            LOG.info("Sending final commands to stop recording...")
            send_command("demo_pause; mirv_streams record end;", wait_after=2)
        except Exception as e:
            LOG.warning(f"Could not send final commands: {e}")

    if TEMP_RECORD_DIR.exists():
        LOG.info(f"Removing temporary directory: {TEMP_RECORD_DIR}")
        shutil.rmtree(TEMP_RECORD_DIR, ignore_errors=True)
        
    if DB_CONN:
        LOG.info("Closing SQLite database connection.")
        DB_CONN.close()
    
    LOG.info("Killing node process..")
    subprocess.run(r'wsl bash -c "pkill node"', check=False)

    LOG.info("Cleanup complete. Exiting.")
    sys.exit(0)
    if signum: # If called by signal handler, exit explicitly
        sys.exit(0)


def main():
    """Main execution function."""
    global ARGS
    
    parser = argparse.ArgumentParser(description="Automate CS2 video recording from a demo file based on an SQLite database.")
    parser.add_argument("--sql", required=True, help="Path to the input.db SQLite database file.")
    parser.add_argument("--demofile", required=True, help="Path to the .dem demo file.")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug logging.")
    
    ARGS = parser.parse_args()
    
    setup_logging(ARGS.debug)
    
    # Register the cleanup function for Ctrl+C
    signal.signal(signal.SIGINT, cleanup)

    demo_file = Path(ARGS.demofile)
    if not demo_file.exists():
        LOG.error(f"Demo file not found: {demo_file}")
        sys.exit(1)

    try:
        # Step 1: Setup
        setup_environment(demo_file)
        
        # Step 2: Process all recordings
        process_recordings(demo_file)
        
        LOG.info("All recording tasks have been processed.")
        
    except Exception as e:
        LOG.error(f"An unhandled error occurred during execution: {e}", exc_info=ARGS.debug)
    finally:
        # Step 3: Cleanup (this will always run)
        cleanup()


if __name__ == "__main__":
    main()