#!/usr/bin/env python3
"""
injection_mold.py - ML Data Preprocessing and Packaging Script

This script serves as the final stage of the data preparation pipeline. It consumes
the structured game data from an SQLite database and the corresponding raw video/audio
recordings, then "molds" them into a single, highly-optimized, and training-ready
LMDB (Lightning Memory-Mapped Database).

The script iterates through the `rounds` table and, for each complete round, verifies
that all 10 player recordings exist. It then processes the full round timeframe,
packaging the global game state and individual player data into the LMDB.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

# --- Standard Library Imports ---
import argparse
import json
import logging
import os
import signal
import shutil
import sys
from pathlib import Path
import sqlite3
from collections import defaultdict

# --- Third-Party Imports ---
# These must be installed via pip:
# pip install lmdb numpy msgpack-python opencv-python soundfile tqdm
try:
    import lmdb
    import numpy as np
    import msgpack
    import cv2
    import soundfile as sf
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: A required library is missing. {e}", file=sys.stderr)
    print("Please run: pip install lmdb numpy msgpack-python opencv-python soundfile tqdm", file=sys.stderr)
    sys.exit(1)


# =============================================================================
# 2. GLOBAL CONSTANTS & CONFIGURATION
# =============================================================================

# --- LMDB Configuration ---
LMDB_INITIAL_MAP_SIZE = 20 * (1024**3)  # 20 GB
LMDB_RESIZE_INCREMENT = 5 * (1024**3)   # 5 GB
LMDB_REMAINING_THRESHOLD = 200 * (1024**2) # 200 MB
INFO_KEY_SUFFIX = "_INFO"

# --- Game & Video Data Configuration ---
GAME_TICKRATE = 64
VIDEO_FPS = 32
TICKS_PER_FRAME = GAME_TICKRATE // VIDEO_FPS

# --- NumPy Data Structure Definitions (DTypes) ---
game_state_dtype = np.dtype([
    ('tick', np.int32),
    ('round_state_flags', np.uint8),
    ('team_alive_mask', np.uint8),
    ('enemy_alive_mask', np.uint8),
    ('enemy_positions', np.float32, (5, 3))
])

player_input_dtype = np.dtype([
    ('position', np.float32, 3),
    ('mouse_delta', np.float32, 2),
    ('health', np.uint8),
    ('armor', np.uint8),
    ('money', np.int32),
    ('input_flags', np.uint64),
    ('active_weapon_id', np.uint8),
    ('inventory_flags', np.uint32),
    ('is_in_buyzone', bool)
])

# --- Data Mapping Constants ---
KEY_INPUT_MAP = {
    'IN_FORWARD': 1 << 0, 'IN_BACK': 1 << 1, 'IN_MOVELEFT': 1 << 2, 'IN_MOVERIGHT': 1 << 3,
    'IN_JUMP': 1 << 4, 'IN_DUCK': 1 << 5, 'IN_WALK': 1 << 6, 'IN_RELOAD': 1 << 7,
    'IN_ATTACK': 1 << 8, 'IN_ATTACK2': 1 << 9, 'IN_USE': 1 << 10,
    'DROP_': 1 << 11, 'BUY_': 1 << 12, 'SELL_': 1 << 13
}
# TODO: Define these mappings based on final data from extract.py
WEAPON_CATEGORY_MAP = {'Rifle': 1, 'Pistol': 2, 'SMG': 3, 'Heavy': 4, 'Knife': 5, 'Grenade': 6}
INVENTORY_MAP = {'defuser': 1 << 0, 'c4': 1 << 1}

# =============================================================================
# 3. GLOBAL STATE
# =============================================================================

LOG = logging.getLogger("InjectionMold")
CLEANUP_REQUIRED = True


# =============================================================================
# 4. CORE LOGIC FUNCTIONS
# =============================================================================

def fetch_and_validate_round_manifest(db_conn, recordings_dir, overridesql_flag):
    LOG.info("Connecting to database and building round manifest...")
    cursor = db_conn.cursor()

    cursor.execute("SELECT round, starttick, endtick, freezetime_endtick, bomb_planted_tick, t_team, ct_team FROM rounds ORDER BY round")
    all_rounds = cursor.fetchall()
    if not all_rounds:
        LOG.error("No entries found in the 'rounds' table. Cannot proceed.")
        sys.exit(1)

    round_manifest = {}
    for r in all_rounds:
        round_num = r['round']
        round_manifest[round_num] = {'round_data': dict(r), 'recordings': {}}
        
        try:
            t_players = [p[0] for p in json.loads(r['t_team'])]
            ct_players = [p[0] for p in json.loads(r['ct_team'])]
        except (json.JSONDecodeError, TypeError, IndexError):
            LOG.warning(f"Skipping Round {round_num}: Invalid team data format in 'rounds' table.")
            continue

        all_players_in_round = t_players + ct_players
        cursor.execute(f"SELECT * FROM RECORDING WHERE roundnumber = ? AND playername IN ({','.join(['?']*len(all_players_in_round))})",
                       (round_num, *all_players_in_round))
        
        recordings = cursor.fetchall()
        for rec in recordings:
            if not rec['is_recorded'] and not overridesql_flag:
                LOG.warning(f"Skipping Round {round_num}: Player '{rec['playername']}' is marked 'is_recorded=False'. Use --overridesql to ignore.")
                round_manifest.pop(round_num)
                break

            base_filename = f"{rec['roundnumber']:02d}_{rec['team']}_{rec['playername']}_{rec['starttick']}_{rec['stoptick']}"
            mp4_path = recordings_dir / f"{base_filename}.mp4"
            wav_path = recordings_dir / f"{base_filename}.wav"

            if not mp4_path.exists() or not wav_path.exists():
                LOG.warning(f"Skipping Round {round_num}: Missing recording file for player '{rec['playername']}'.")
                if round_num in round_manifest: round_manifest.pop(round_num)
                break
            
            rec_dict = dict(rec)
            rec_dict['mp4_path'] = mp4_path
            rec_dict['wav_path'] = wav_path
            round_manifest[round_num]['recordings'][rec['playername']] = rec_dict
        
        # Final check if all players for the round were found and validated
        if round_num in round_manifest and len(round_manifest[round_num]['recordings']) != len(all_players_in_round):
            LOG.warning(f"Skipping Round {round_num}: Mismatch between players in 'rounds' table and validated recordings in 'RECORDING' table.")
            round_manifest.pop(round_num)

    return round_manifest

def construct_game_state(current_tick, round_data, team_players_data, enemy_players_data):
    gs_array = np.zeros(1, dtype=game_state_dtype)
    gs_array['tick'] = current_tick
    
    round_state_flags = 0
    if round_data and current_tick < round_data['freezetime_endtick']: round_state_flags |= 1 << 0
    if round_data and current_tick >= round_data['freezetime_endtick']: round_state_flags |= 1 << 1
    if round_data and round_data['bomb_planted_tick'] != -1 and current_tick >= round_data['bomb_planted_tick']: round_state_flags |= 1 << 2
    gs_array['round_state_flags'] = round_state_flags

    team_alive_mask, enemy_alive_mask = 0, 0
    enemy_pos_array = np.zeros((5, 3), dtype=np.float32)

    for i, player_data in enumerate(team_players_data):
        if player_data and player_data['health'] > 0: team_alive_mask |= (1 << i)
    
    for i, player_data in enumerate(enemy_players_data):
        if player_data and player_data['health'] > 0:
            enemy_alive_mask |= (1 << i)
            enemy_pos_array[i] = [player_data['position_x'], player_data['position_y'], player_data['position_z']]

    gs_array['team_alive_mask'] = team_alive_mask
    gs_array['enemy_alive_mask'] = enemy_alive_mask
    gs_array['enemy_positions'] = enemy_pos_array
    return gs_array

def construct_player_input(player_data):
    pi_array = np.zeros(1, dtype=player_input_dtype)
    pi_array['position'] = (player_data['position_x'], player_data['position_y'], player_data['position_z'])
    pi_array['mouse_delta'] = (player_data['mouse_x'] or 0.0, player_data['mouse_y'] or 0.0)
    pi_array['health'] = player_data['health']
    pi_array['armor'] = player_data['armor']
    pi_array['money'] = player_data['money']
    pi_array['is_in_buyzone'] = bool(player_data['is_in_buyzone'])

    input_flags = 0
    inputs = (player_data['keyboard_input'] or "").split(',') + (player_data['buy_sell_input'] or "").split(',')
    for key, mask in KEY_INPUT_MAP.items():
        if any(inp.strip().startswith(key) for inp in inputs if inp): input_flags |= mask
    pi_array['input_flags'] = input_flags
    
    pi_array['active_weapon_id'] = 0
    pi_array['inventory_flags'] = 0
    return pi_array

def process_round_perspective(round_num, team_name, players_meta, enemies_meta, round_data, data_by_tick, lmdb_txn, demo_name):
    LOG.info(f"Processing Round {round_num}, Team '{team_name}'...")
    round_start_tick = round_data['starttick']
    round_end_tick = round_data['endtick']

    caps = {p['playername']: cv2.VideoCapture(str(p['mp4_path'])) for p in players_meta}
    auds = {p['playername']: sf.SoundFile(str(p['wav_path'])) for p in players_meta}
    total_frames = (round_end_tick - round_start_tick) // TICKS_PER_FRAME

    for frame_idx in tqdm(range(total_frames), desc=f"  R{round_num}-{team_name}", leave=False):
        current_tick = round_start_tick + (frame_idx * TICKS_PER_FRAME)
        if current_tick > round_end_tick: break

        tick_data_for_all = data_by_tick.get(current_tick, {})
        team_data_at_tick = [tick_data_for_all.get(p['playername']) for p in players_meta]
        enemy_data_at_tick = [tick_data_for_all.get(p['playername']) for p in enemies_meta]

        game_state_np = construct_game_state(current_tick, round_data, team_data_at_tick, enemy_data_at_tick)
        packaged_data = [game_state_np]
        team_alive_mask = game_state_np['team_alive_mask'][0]

        for i, player_meta in enumerate(players_meta):
            if (team_alive_mask >> i) & 1:
                player_name = player_meta['playername']
                player_data_at_tick = team_data_at_tick[i]
                if not player_data_at_tick: continue

                ret, frame = caps[player_name].read()
                if not ret: continue
                
                audio_frames_to_read = int(auds[player_name].samplerate / VIDEO_FPS)
                audio_data = auds[player_name].read(audio_frames_to_read)
                
                _, jpeg_bytes = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                player_input_np = construct_player_input(player_data_at_tick)
                packaged_data.append((player_input_np, jpeg_bytes.tobytes(), audio_data.tobytes()))
        
        key = f"{demo_name}_round_{round_num:02d}_team_{team_name}_tick_{current_tick}".encode('utf-8')
        value = msgpack.packb(packaged_data, use_bin_type=True, default=lambda obj: obj.tolist() if isinstance(obj, np.ndarray) else obj)
        lmdb_txn.put(key, value)

    for cap in caps.values(): cap.release()
    for aud in auds.values(): aud.close()

def write_metadata_info(lmdb_txn, processed_rounds, demoname):
    metadata = {
        "demoname": demoname,
        "rounds": sorted(processed_rounds)
    }
    key = f"{demoname}{INFO_KEY_SUFFIX}".encode('utf-8')
    value = json.dumps(metadata, indent=4).encode('utf-8')
    lmdb_txn.put(key, value)

# =============================================================================
# 5. HELPER & UTILITY FUNCTIONS
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="Mold CS2 recordings and data into a unified LMDB.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--recdir", required=True, help="Path to the directory containing .mp4 and .wav recordings.")
    parser.add_argument("--dbfile", required=True, help="Path to the SQLite database file from extract.py.")
    parser.add_argument("--outlmdb", required=True, help="Path where the output LMDB will be created.")
    parser.add_argument("--overwrite", action="store_true", help="If specified, delete and recreate the LMDB if it exists.")
    parser.add_argument("--overridesql", action="store_true", help="Ignore 'is_recorded=False' flags in the DB.")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug logging.")
    return parser.parse_args()

def setup_logging(is_debug):
    level = logging.DEBUG if is_debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)

def prepare_output_lmdb(output_path, overwrite_flag):
    if os.path.exists(output_path):
        if overwrite_flag:
            LOG.warning(f"Output path '{output_path}' exists. Overwriting as requested.")
            shutil.rmtree(output_path)
        else:
            LOG.error(f"Output path '{output_path}' already exists. Use --overwrite to replace it. Exiting.")
            sys.exit(1)

def interrupt_handler(signum, frame):
    global CLEANUP_REQUIRED
    LOG.warning("\nCTRL+C detected. Shutting down gracefully...")
    CLEANUP_REQUIRED = True
    raise KeyboardInterrupt

# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================

def main():
    global CLEANUP_REQUIRED
    args = parse_arguments()
    setup_logging(args.debug)
    signal.signal(signal.SIGINT, interrupt_handler)

    lmdb_env = None
    demo_name = Path(args.dbfile).stem
    recordings_dir = Path(args.recdir)
    outlmdb_path = Path(args.outlmdb)

    try:
        LOG.info("Step 1: Preparing output and validating data manifest...")
        prepare_output_lmdb(outlmdb_path, args.overwrite)
        
        db_conn = sqlite3.connect(args.dbfile)
        db_conn.row_factory = sqlite3.Row
        
        round_manifest = fetch_and_validate_round_manifest(db_conn, recordings_dir, args.overridesql)
        valid_round_nums = sorted(round_manifest.keys())
        LOG.info(f"Manifest validated. Found {len(valid_round_nums)} complete rounds to process: {valid_round_nums}")

        if not valid_round_nums:
            LOG.warning("No complete rounds found to process. Exiting.")
            sys.exit(0)

        LOG.info("Step 2: Caching all player data for performance...")
        cursor = db_conn.cursor()
        cursor.execute("SELECT * FROM player")
        all_player_data = cursor.fetchall()
        data_by_tick = defaultdict(dict)
        for row in all_player_data:
            data_by_tick[row['tick']][row['playername']] = row
        db_conn.close()

        LOG.info("Step 3: Initializing LMDB environment...")
        lmdb_env = lmdb.open(str(outlmdb_path), map_size=LMDB_INITIAL_MAP_SIZE, writemap=True)

        LOG.info("Step 4: Starting data processing loop...")
        processed_rounds_info = []
        with lmdb_env.begin(write=True) as lmdb_txn:
            for round_num in valid_round_nums:
                manifest_entry = round_manifest[round_num]
                round_data = manifest_entry['round_data']
                
                # Split players by team for processing
                all_recordings = manifest_entry['recordings'].values()
                t_players_meta = sorted([r for r in all_recordings if r['team'] == 'T'], key=lambda x: x['playername'])
                ct_players_meta = sorted([r for r in all_recordings if r['team'] == 'CT'], key=lambda x: x['playername'])
                
                # Process T-side perspective
                process_round_perspective(round_num, 'T', t_players_meta, ct_players_meta, round_data, data_by_tick, lmdb_txn, demo_name)
                # Process CT-side perspective
                process_round_perspective(round_num, 'CT', ct_players_meta, t_players_meta, round_data, data_by_tick, lmdb_txn, demo_name)
                
                processed_rounds_info.append([round_num, round_data['starttick'], round_data['endtick']])

            LOG.info("Step 5: Writing final metadata key...")
            write_metadata_info(lmdb_txn, processed_rounds_info, demo_name)

        LOG.info("All data successfully processed and written to LMDB.")
        CLEANUP_REQUIRED = False

    except KeyboardInterrupt:
        LOG.warning("Process interrupted by user.")
    except Exception as e:
        LOG.error(f"An unhandled exception occurred: {e}", exc_info=args.debug)
    finally:
        LOG.info("Step 6: Finalizing script execution...")
        if lmdb_env:
            lmdb_env.close()
        if CLEANUP_REQUIRED and outlmdb_path.exists():
            LOG.error(f"Process did not complete successfully. Deleting incomplete LMDB at: {outlmdb_path}")
            try:
                shutil.rmtree(outlmdb_path)
                LOG.info("Incomplete LMDB deleted.")
            except OSError as e:
                LOG.error(f"Failed to delete incomplete LMDB: {e}")
        LOG.info("Shutdown complete.")


if __name__ == "__main__":
    main()