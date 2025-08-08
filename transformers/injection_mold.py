#!/usr/bin/env python3
"""
injection_mold.py - Compile CS2 recordings and database into a unified LMDB
for model training. (Multithreaded Version)

This script performs the following actions:
1. Validates the existence of all required video/audio recordings against a
   SQLite database.
2. Uses a pool of worker threads to process each round/team perspective in parallel.
3. Each worker thread fetches:
   - The overall game state (round status, player liveness, enemy positions).
   - The POV video frame and audio chunk for each of the 5 players on a team.
   - The detailed inputs (keyboard, mouse, etc.) for each of the 5 players.
4. Each worker aligns and encodes this data into a structured format using NumPy
   and msgpack, returning the result to the main thread.
5. The main thread is the sole writer to the LMDB, committing the packaged data
   as it receives it from the workers.
6. Includes robust error handling and a cleanup mechanism to prevent corrupt
   database states on interruption.
"""

import argparse
import json
import logging
import os
import signal
import shutil
import sqlite3
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import lmdb
import msgpack
import msgpack_numpy as m
import numpy as np
from tqdm import tqdm

# --- Configuration Constants ---
GAME_TICKS_PER_SEC = 64
EXPECTED_VIDEO_FPS = 32
TICKS_PER_FRAME = GAME_TICKS_PER_SEC // EXPECTED_VIDEO_FPS
AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 2
AUDIO_BIT_DEPTH = 2  # 16-bit
AUDIO_BYTES_PER_FRAME = (AUDIO_SAMPLE_RATE // EXPECTED_VIDEO_FPS) * AUDIO_CHANNELS * AUDIO_BIT_DEPTH

# LMDB Configuration
INITIAL_MAP_SIZE = 20 * 1024 * 1024 * 1024  # 20 GB
MAP_RESIZE_INCREMENT = 5 * 1024 * 1024 * 1024  # 5 GB
MAP_RESIZE_THRESHOLD = 200 * 1024 * 1024  # 200 MB

# --- Globals ---
LOG = logging.getLogger("InjectionMold")
LMDB_PATH_FOR_CLEANUP = None


# =============================================================================
# 1. DATA ENCODING MAPPINGS (No changes here)
# =============================================================================

# Actions related to player movement and weapon handling
KEYBOARD_ONLY_ACTIONS = [
    "IN_ATTACK", "IN_JUMP", "IN_DUCK", "IN_FORWARD", "IN_BACK", "IN_USE", "IN_CANCEL", "IN_TURNLEFT", "IN_TURNRIGHT", "IN_MOVELEFT", "IN_MOVERIGHT", "IN_ATTACK2", "IN_RELOAD", "IN_ALT1", "IN_ALT2", "IN_SPEED", "IN_WALK", "IN_ZOOM", "IN_WEAPON1", "IN_WEAPON2", "IN_BULLRUSH", "IN_GRENADE1", "IN_GRENADE2", "IN_ATTACK3", "IN_SCORE", "IN_INSPECT", "SWITCH_1", "SWITCH_2", "SWITCH_3", "SWITCH_4", "SWITCH_5",
]

# Actions related to economy (buying, selling, dropping)
ECO_ACTIONS = [
    "IN_BUYZONE", "DROP_deagle", "DROP_elite", "DROP_fiveseven", "DROP_glock", "DROP_ak47", "DROP_aug", "DROP_awp", "DROP_famas", "DROP_g3sg1", "DROP_galilar", "DROP_m249", "DROP_m4a1", "DROP_mac10", "DROP_p90", "DROP_mp5sd", "DROP_ump45", "DROP_xm1014", "DROP_bizon", "DROP_mag7", "DROP_negev", "DROP_sawedoff", "DROP_tec9", "DROP_p2000", "DROP_mp7", "DROP_mp9", "DROP_nova", "DROP_p250", "DROP_scar20", "DROP_sg556", "DROP_ssg08", "DROP_knife", "DROP_flashbang", "DROP_hegrenade", "DROP_smokegrenade", "DROP_molotov", "DROP_decoy", "DROP_incgrenade", "DROP_c4", "DROP_m4a1_silencer", "DROP_usp_silencer", "DROP_cz75a", "DROP_revolver", "DROP_defuser", "BUY_deagle", "BUY_elite", "BUY_fiveseven", "BUY_glock", "BUY_ak47", "BUY_aug", "BUY_awp", "BUY_famas", "BUY_g3sg1", "BUY_galilar", "BUY_m249", "BUY_m4a1", "BUY_mac10", "BUY_p90", "BUY_mp5sd", "BUY_ump45", "BUY_xm1014", "BUY_bizon", "BUY_mag7", "BUY_negev", "BUY_sawedoff", "BUY_tec9", "BUY_p2000", "BUY_mp7", "BUY_mp9", "BUY_nova", "BUY_p250", "BUY_scar20", "BUY_sg556", "BUY_ssg08", "BUY_knife", "BUY_flashbang", "BUY_hegrenade", "BUY_smokegrenade", "BUY_molotov", "BUY_decoy", "BUY_incgrenade", "BUY_c4", "BUY_m4a1_silencer", "BUY_usp_silencer", "BUY_cz75a", "BUY_revolver", "BUY_defuser", "BUY_vest", "BUY_vesthelm", "SELL_deagle", "SELL_fiveseven", "SELL_glock", "SELL_ak47", "SELL_aug", "SELL_awp", "SELL_famas", "SELL_galilar", "SELL_m4a1", "SELL_mac10", "SELL_p90", "SELL_ump45", "SELL_xm1014", "SELL_bizon", "SELL_mag7", "SELL_sawedoff", "SELL_tec9", "SELL_p2000", "SELL_mp7", "SELL_mp9", "SELL_nova", "SELL_p250", "SELL_ssg08", "SELL_flashbang", "SELL_hegrenade", "SELL_smokegrenade", "SELL_molotov", "SELL_decoy", "SELL_incgrenade",
]

# --- FIX: A comprehensive, canonical list of all official item names ---
ITEM_NAMES = sorted(list(set([
    # Rifles
    "AK-47", "M4A4", "M4A1-S", "Galil AR", "FAMAS", "AUG", "SG 553", "AWP", "SSG 08", "G3SG1", "SCAR-20",
    # Pistols
    "Glock-18", "USP-S", "P250", "P2000", "Dual Berettas", "Five-SeveN", "Tec-9", "CZ75-Auto", "R8 Revolver", "Desert Eagle",
    # SMGs
    "MP9", "MAC-10", "MP7", "MP5-SD", "UMP-45", "P90", "PP-Bizon",
    # Heavy
    "Nova", "XM1014", "MAG-7", "Sawed-Off", "M249", "Negev",
    # Knives (now comprehensive to reduce reliance on fallbacks)
    "Knife", "knife_t", "knife_ct", "Bayonet", "Flip Knife", "Gut Knife", "Karambit", "M9 Bayonet", "Huntsman Knife", "Falchion Knife", "Bowie Knife", "Butterfly Knife", "Shadow Daggers", "Ursus Knife", "Navaja Knife", "Stiletto Knife", "Talon Knife", "Classic Knife", "Paracord Knife", "Survival Knife", "Nomad Knife", "Skeleton Knife",
    # Grenades
    "High Explosive Grenade", "Flashbang", "Smoke Grenade", "Molotov", "Incendiary Grenade", "Decoy Grenade",
    # Gear & Other
    "C4 Explosive", "Defuse Kit", "Zeus x27", "Kevlar Vest", "Helmet"
])))

# Reverse mappings for decoding are generated from the canonical lists
KEYBOARD_TO_BIT = {action: i for i, action in enumerate(KEYBOARD_ONLY_ACTIONS)}
ECO_TO_BIT = {action: i for i, action in enumerate(ECO_ACTIONS)}
ITEM_TO_INDEX = {item: i for i, item in enumerate(ITEM_NAMES)}

BIT_TO_KEYBOARD = {i: action for i, action in enumerate(KEYBOARD_ONLY_ACTIONS)}
BIT_TO_ECO = {i: action for i, action in enumerate(ECO_ACTIONS)}
BIT_TO_ITEM = {i: item for i, item in enumerate(ITEM_NAMES)}

m.patch()

# =============================================================================
# 2. NUMPY DTYPE DEFINITIONS (No changes here)
# =============================================================================

def define_numpy_dtypes():
    game_state_dtype = np.dtype([('tick', np.int32), ('round_state', np.uint8), ('team_alive', np.uint8), ('enemy_alive', np.uint8), ('enemy_pos', np.float32, (5, 3))])
    player_input_dtype = np.dtype([('pos', np.float32, (3,)), ('mouse', np.float32, (2,)), ('health', np.uint8), ('armor', np.uint8), ('money', np.int32), ('keyboard_bitmask', np.uint32), ('eco_bitmask', np.uint64, (2,)), ('inventory_bitmask', np.uint64, (2,)), ('active_weapon_bitmask', np.uint64, (2,))])
    assert len(KEYBOARD_TO_BIT) <= 32, "Too many keyboard actions for uint32 bitmask"
    assert len(ECO_TO_BIT) <= 128, "Too many economic actions for 128-bit bitmask (2x uint64)"
    assert len(ITEM_TO_INDEX) <= 128, "Too many items for inventory/weapon bitmasks (2x uint64)"
    return game_state_dtype, player_input_dtype

# =============================================================================
# 3. HELPER FUNCTIONS AND SETUP (No changes here)
# =============================================================================

class TqdmLoggingHandler(logging.Handler):
    """
    A custom logging handler that redirects all logging output
    to `tqdm.write()`, ensuring that it does not interfere with
    the progress bar.
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)

def setup_logging(debug=False):
    """
    Configures the logging for the script, using the custom
    TqdmLoggingHandler to prevent conflicts with the progress bar.
    """
    level = logging.DEBUG if debug else logging.INFO
    # Get the root logger
    log = logging.getLogger()
    log.setLevel(level)
    
    # Remove any existing handlers to avoid duplicate messages
    if log.hasHandlers():
        log.handlers.clear()
        
    # Add our custom tqdm-aware handler
    handler = TqdmLoggingHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)

def signal_handler(sig, frame):
    LOG.warning("Interruption detected. Cleaning up partial LMDB database...")
    if LMDB_PATH_FOR_CLEANUP and os.path.exists(LMDB_PATH_FOR_CLEANUP):
        try: shutil.rmtree(LMDB_PATH_FOR_CLEANUP); LOG.info(f"Successfully removed incomplete LMDB at: {LMDB_PATH_FOR_CLEANUP}")
        except OSError as e: LOG.error(f"Error removing LMDB directory: {e}")
    os._exit(1) # Use os._exit to bypass normal cleanup that might hang

def load_and_validate_data(db_path, rec_dir, override_sql):
    LOG.info("-> Phase 1: Validating database and recording files...")
    if not db_path.exists(): LOG.critical(f"Database file not found: {db_path}"); sys.exit(1)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True); conn.row_factory = sqlite3.Row
    rounds_info = {r['round']: dict(r) for r in conn.execute("SELECT * FROM rounds").fetchall()}; LOG.info(f"   - Loaded info for {len(rounds_info)} rounds.")
    recordings_map = {}
    db_recordings = conn.execute("SELECT * FROM RECORDING ORDER BY roundnumber, team, playername").fetchall()
    for rec in db_recordings:
        rec = dict(rec); round_num, team = rec['roundnumber'], rec['team']
        if round_num not in rounds_info or rounds_info[round_num].get('starttick') is None or rounds_info[round_num].get('endtick') is None:
            LOG.warning(f"Skipping recording for round {round_num} because it has missing start/end ticks."); continue
        if not rec['is_recorded'] and not override_sql: LOG.critical(f"Rec for player '{rec['playername']}' in round {round_num} not marked as recorded. Use --overridesql."); sys.exit(1)
        filename_base = f"{rec['roundnumber']:02d}_{rec['team']}_{rec['playername']}_{rec['starttick']}_{rec['stoptick']}"
        mp4_path, wav_path = rec_dir / f"{filename_base}.mp4", rec_dir / f"{filename_base}.wav"
        if not mp4_path.exists(): LOG.critical(f"Video file not found: {mp4_path}"); sys.exit(1)
        if not wav_path.exists(): LOG.critical(f"Audio file not found: {wav_path}"); sys.exit(1)
        rec['mp4_path'], rec['wav_path'] = mp4_path, wav_path
        key = (round_num, team)
        if key not in recordings_map: recordings_map[key] = []
        recordings_map[key].append(rec)
    LOG.info(f"   - Validated {len(db_recordings)} recording entries and their corresponding media files.")
    LOG.info("   - Caching all 'player' table data into memory (this may take a moment)...")
    player_data_cache = { (r['playername'], r['tick']): dict(r) for r in conn.execute("SELECT * FROM player").fetchall() }
    LOG.info(f"   - Cached {len(player_data_cache)} player-tick entries.")
    conn.close()
    demoname = rec_dir.name; LOG.info(f"   - Validation complete. Demo name identified as '{demoname}'.")
    return demoname, rounds_info, recordings_map, player_data_cache

def get_bitmask(actions, mapping):
    mask = 0
    if actions:
        for action in actions:
            if action in mapping: mask |= (1 << mapping[action])
    return mask

def get_bitmask_array(actions, mapping):
    mask = np.zeros(2, dtype=np.uint64)
    if actions:
        for action in actions:
            if action in mapping:
                bit_pos, idx, pos_in_idx = mapping[action], mapping[action] // 64, mapping[action] % 64
                if idx < 2: mask[idx] |= (np.uint64(1) << np.uint64(pos_in_idx))
    return mask

def get_inventory_bitmasks(inventory_json, active_weapon, mapping):
    """
    Creates 128-bit bitmasks for the player's inventory and active weapon.
    Includes a fail-loud mechanism for unknown items.
    """
    inventory_mask, active_weapon_mask = np.zeros(2, dtype=np.uint64), np.zeros(2, dtype=np.uint64)

    def set_bit(mask, item_name):
        # Gracefully handle potential None or non-string item names
        if not isinstance(item_name, str):
            return

        bit_pos = mapping.get(item_name)
        # Handle knife variations as a fallback
        if bit_pos is None and ("knife" in item_name.lower() or "bayonet" in item_name.lower()):
            bit_pos = mapping.get("Knife")

        # --- NEW: Fail-Loud Error Check ---
        # If after the lookup and the fallback, the item is still unknown, raise an error.
        if bit_pos is None:
            raise ValueError(
                f"\n\n"
                f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                f"FATAL: Unknown item name encountered during encoding: '{item_name}'\n"
                f"This item is not in the canonical ITEM_NAMES list.\n"
                f"To fix, add '{item_name}' to the ITEM_NAMES list in both:\n"
                f"  - injection_mold.py\n"
                f"  - lmdb_inspect.py\n"
                f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            )

        # If we get here, bit_pos is valid.
        idx = bit_pos // 64
        pos_in_idx = bit_pos % 64
        if idx < 2:
            mask[idx] |= (np.uint64(1) << np.uint64(pos_in_idx))

    # Process the active weapon and inventory
    if active_weapon:
        set_bit(active_weapon_mask, active_weapon)
    try:
        inventory_list = json.loads(inventory_json) if inventory_json else []
        for item in inventory_list:
            set_bit(inventory_mask, item)
    except (json.JSONDecodeError, TypeError):
        pass  # Ignore malformed inventory JSON, but individual items will still be checked

    return inventory_mask, active_weapon_mask

# =============================================================================
# 4. WORKER FUNCTION
# =============================================================================

def process_round_perspective(round_num, team, demoname, gs_dtype, pi_dtype, rounds_info, recordings_map, player_data_cache):
    """
    This function is executed by each worker thread. It processes one
    round/team perspective and returns a list of (key, data) tuples.
    """
    round_data = rounds_info[round_num]
    team_recordings = recordings_map[(round_num, team)]
    if len(team_recordings) != 5: return []

    t_roster = [p[0] for p in json.loads(round_data['t_team'])]
    ct_roster = [p[0] for p in json.loads(round_data['ct_team'])]
    current_roster = t_roster if team == 'T' else ct_roster
    enemy_roster = ct_roster if team == 'T' else t_roster

    caps = {rec['playername']: cv2.VideoCapture(str(rec['mp4_path'])) for rec in team_recordings}
    auds = {rec['playername']: open(rec['wav_path'], 'rb') for rec in team_recordings}
    for aud_file in auds.values(): aud_file.seek(44)

    round_start_tick = round_data['starttick']
    round_end_tick = max(rec['stoptick'] for rec in team_recordings)

    results = []
    for current_tick in range(round_start_tick, round_end_tick + 1, TICKS_PER_FRAME):
        game_state = np.zeros(1, dtype=gs_dtype)
        game_state[0]['tick'] = current_tick
        
        rs_mask = 0
        if round_data.get('freezetime_endtick') is not None and current_tick < round_data['freezetime_endtick']: rs_mask |= (1 << 0)
        if round_data.get('starttick') is not None and round_data.get('endtick') is not None and current_tick >= round_data['starttick'] and current_tick <= round_data['endtick']: rs_mask |= (1 << 1)
        if round_data.get('bomb_planted_tick', -1) != -1 and current_tick >= round_data['bomb_planted_tick']: rs_mask |= (1 << 2)
        game_state[0]['round_state'] = rs_mask
        
        team_deaths = {p[0]: p[1] for p in json.loads(round_data[f"{team.lower()}_team"])}
        enemy_deaths = {p[0]: p[1] for p in json.loads(round_data[f"{'ct' if team == 'T' else 't'}_team"])}
        team_alive_mask = sum(1 << i for i, n in enumerate(current_roster) if team_deaths.get(n, 0) == -1 or current_tick < team_deaths.get(n, 0))
        enemy_alive_mask = sum(1 << i for i, n in enumerate(enemy_roster) if enemy_deaths.get(n, 0) == -1 or current_tick < enemy_deaths.get(n, 0))
        game_state[0]['team_alive'], game_state[0]['enemy_alive'] = team_alive_mask, enemy_alive_mask
        
        for i, name in enumerate(enemy_roster):
            if (enemy_alive_mask >> i) & 1:
                if (name, current_tick) in player_data_cache:
                    e_data = player_data_cache[(name, current_tick)]
                    game_state[0]['enemy_pos'][i] = [e_data['position_x'], e_data['position_y'], e_data['position_z']]

        player_data_list = []
        for i, rec in enumerate(team_recordings):
            if not ((team_alive_mask >> i) & 1) or current_tick >= rec['stoptick']: continue
            
            playername = rec['playername']
            ret, frame = caps[playername].read()
            audio_chunk = auds[playername].read(AUDIO_BYTES_PER_FRAME)
            
            # --- NEW: Enhanced warning logic ---
            if not ret or len(audio_chunk) < AUDIO_BYTES_PER_FRAME:
                player_start_tick = rec['starttick']
                player_stop_tick = rec['stoptick']

                # Calculate expected vs actual frames processed for this player's POV
                expected_total_frames = (player_stop_tick - player_start_tick) // TICKS_PER_FRAME
                # The current frame index within this player's own video timeline
                current_frame_index = (current_tick - player_start_tick) // TICKS_PER_FRAME
                
                # To avoid negative numbers if the stream is longer than expected
                frames_missing = max(0, expected_total_frames - current_frame_index)

                LOG.warning(
                    f"Media stream for {playername} ended unexpectedly at tick {current_tick}. "
                    f"({current_frame_index}/{expected_total_frames} frames processed, "
                    f"approx. {frames_missing} frames missing from recording)."
                )
                continue # Skip processing this player for this tick
            
            jpeg_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
            tick_data = player_data_cache.get((playername, current_tick))
            player_input = np.zeros(1, dtype=pi_dtype)
            
            if tick_data:
                kb_input_str, buy_sell_input_str = tick_data.get('keyboard_input', '') or '', tick_data.get('buy_sell_input', '') or ''
                all_kb_actions = kb_input_str.split(',')
                keyboard_actions = [a for a in all_kb_actions if not a.startswith('DROP_')]
                player_input[0]['keyboard_bitmask'] = get_bitmask(keyboard_actions, KEYBOARD_TO_BIT)
                drop_actions = [a for a in all_kb_actions if a.startswith('DROP_')]
                eco_actions_list = drop_actions + buy_sell_input_str.split(',')
                if tick_data.get('is_in_buyzone') == 1: eco_actions_list.append('IN_BUYZONE')
                player_input[0]['eco_bitmask'] = get_bitmask_array(filter(None, eco_actions_list), ECO_TO_BIT)
                player_input[0]['pos'] = [tick_data.get('position_x', 0), tick_data.get('position_y', 0), tick_data.get('position_z', 0)]
                player_input[0]['mouse'] = [tick_data.get('mouse_x', 0), tick_data.get('mouse_y', 0)]
                player_input[0]['health'], player_input[0]['armor'], player_input[0]['money'] = tick_data.get('health', 0), tick_data.get('armor', 0), tick_data.get('money', 0)
                inv_mask, wep_mask = get_inventory_bitmasks(tick_data.get('inventory'), tick_data.get('active_weapon'), ITEM_TO_INDEX)
                player_input[0]['inventory_bitmask'], player_input[0]['active_weapon_bitmask'] = inv_mask, wep_mask
            
            player_data_list.append((player_input, jpeg_bytes, audio_chunk))
        
        if not player_data_list: continue
        key = f"{demoname}_round_{round_num:03d}_team_{team}_tick_{current_tick:08d}".encode('utf-8')
        tick_payload = msgpack.packb({"game_state": game_state, "player_data": player_data_list}, use_bin_type=True)
        results.append((key, tick_payload))

    for cap in caps.values(): cap.release()
    for aud in auds.values(): aud.close()
    return results

# =============================================================================
# 5. MAIN DRIVER
# =============================================================================

def main():
    global LMDB_PATH_FOR_CLEANUP

    parser = argparse.ArgumentParser(description="Compile CS2 recordings and database into a unified LMDB.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--recdir", required=True, type=Path, help="Path to recordings directory.")
    parser.add_argument("--dbfile", required=True, type=Path, help="Path to SQLite DB file.")
    parser.add_argument("--outlmdb", required=True, type=Path, help="Path for output LMDB.")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing LMDB.")
    parser.add_argument("--overridesql", action='store_true', help="Ignore 'is_recorded=False' flags.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of worker threads to use.")
    args = parser.parse_args()

    setup_logging(args.debug)
    
    LMDB_PATH_FOR_CLEANUP = args.outlmdb
    signal.signal(signal.SIGINT, signal_handler)

    if args.outlmdb.exists():
        if args.overwrite: shutil.rmtree(args.outlmdb); LOG.warning(f"Removed existing LMDB: {args.outlmdb}")
        else: LOG.critical(f"Output path {args.outlmdb} exists. Use --overwrite. Exiting."); sys.exit(1)
    
    demoname, rounds_info, recordings_map, player_data_cache = load_and_validate_data(args.dbfile, args.recdir, args.overridesql)
    gs_dtype, pi_dtype = define_numpy_dtypes()
    env = lmdb.open(str(args.outlmdb), map_size=INITIAL_MAP_SIZE, writemap=True)
    
    LOG.info(f"-> Phase 2: Processing rounds with {args.workers} worker threads...")
    
    tasks = sorted(recordings_map.keys())
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_round_perspective, r, t, demoname, gs_dtype, pi_dtype, rounds_info, recordings_map, player_data_cache): (r, t) for r, t in tasks}
        
        # Use a tqdm instance that we can manually close
        pbar = tqdm(as_completed(futures), total=len(tasks), desc="Processing Round/Team Perspectives")
        
        for future in pbar:
            try:
                round_results = future.result()
                
                if not round_results: continue

                # Proactive resizing logic
                batch_size = sum(len(payload) for key, payload in round_results)
                info, stats = env.info(), env.stat()
                available_space = info['map_size'] - (info['last_pgno'] * stats['psize'])
                while available_space < batch_size:
                    LOG.warning(f"Resizing LMDB: Need {batch_size / (1024*1024):.2f} MB, have {available_space / (1024*1024):.2f} MB.")
                    new_size = info['map_size'] + MAP_RESIZE_INCREMENT
                    env.set_mapsize(new_size)
                    info, available_space = env.info(), info['map_size'] - (info['last_pgno'] * stats['psize'])

                with env.begin(write=True) as txn:
                    for key, payload in round_results:
                        txn.put(key, payload)

            except Exception as exc:
                # --- FIX: Graceful handling of exceptions from workers ---
                # 1. Close the tqdm progress bar to prevent it from garbling the output.
                pbar.close()
                
                # 2. Log the error clearly. The detailed ValueError message will be part of 'exc'.
                task_id = futures[future]
                LOG.error(f"FATAL ERROR in worker task for Round/Team {task_id}.")
                
                # 3. Shutdown the executor immediately, cancelling pending tasks.
                executor.shutdown(wait=False, cancel_futures=True)
                
                # 4. Re-raise the original exception to show the full traceback and stop the script.
                # This will trigger the signal handler for cleanup.
                raise exc

    LOG.info("-> Phase 3: Writing final metadata...")
    metadata = {"demoname": demoname, "rounds": [[r, rounds_info[r]['starttick'], rounds_info[r]['endtick']] for r in sorted(rounds_info.keys())]}
    with env.begin(write=True) as txn:
        key = f"{demoname}_INFO".encode('utf-8')
        txn.put(key, json.dumps(metadata).encode('utf-8'))

    env.close()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    LOG.info("All processing steps completed successfully.")
    LOG.info(f"Final LMDB database is at: {args.outlmdb}")

if __name__ == '__main__':
    main()