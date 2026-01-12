#!/usr/bin/env python3
"""
Copyright 2025 Anton Althoff

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
------------------------------------------------------------------------
injection_mold.py - Compile CS2 recordings and database into a unified LMDB
for model training.

This version DOES NOT embed video frames or process audio. Instead, it processes
all non-visual data (player states) and creates a final metadata key containing the
paths to the corresponding MP4 and WAV files for each round-perspective, to be used
by a video-aware training script. Audio processing (e.g., spectrograms) is
expected to be handled at train time.

This refactored version is single-threaded to eliminate multiprocessing-related
complexity and improve reliability.

Patched:
- Added perf timers for hot sections (db, LMDB writes).
- Cached per-round JSON (team/enemy death ticks) outside the inner tick loop.
- Added explicit check for WAV file existence.
- Added WAV file paths to final metadata output.
- Added --debug flag to print final metadata to stdout.
"""
import argparse
import json
import logging
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from time import perf_counter
from collections import defaultdict

import lmdb
import numpy as np
from tqdm import tqdm

# Import msgpack and numpy extension for serialization
import msgpack
import msgpack_numpy as mpnp
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


# --- Configuration ---
GAME_TICKS_PER_SEC = 64
EXPECTED_VIDEO_FPS = 32
TICKS_PER_FRAME = GAME_TICKS_PER_SEC // EXPECTED_VIDEO_FPS
INITIAL_MAP_SIZE = 500 * 1024**2  # 500MB
MAP_RESIZE_INCREMENT = 100 * 1024**2  # 100MB

# --- Globals ---
LOG = logging.getLogger("InjectionMold")

# --- Lightweight timing helpers ---
TIMERS = defaultdict(float)
def t(label: str):
    class _T:
        def __enter__(self):
            self.t0 = perf_counter()
        def __exit__(self, *exc):
            TIMERS[label] += perf_counter() - self.t0
    return _T()

# =============================================================================
# DATA ENCODING MAPPINGS (Canonical Source)
# =============================================================================
#dont ask why there are two kinds of item names...
ITEM_NAMES = sorted(list(set([
    "Desert Eagle", "Dual Berettas", "Five-SeveN", "Glock-18", "AK-47",
    "AUG", "AWP", "FAMAS", "G3SG1", "Galil AR", "M249", "M4A4", "MAC-10",
    "P90", "MP5-SD", "UMP-45", "XM1014", "PP-Bizon", "MAG-7", "Negev",
    "Sawed-Off", "Tec-9", "Zeus x27", "P2000", "MP7", "MP9", "Nova",
    "P250", "SCAR-20", "SG 553", "SSG 08", "Knife", "knife", "Flashbang",
    "High Explosive Grenade", "Smoke Grenade", "Molotov", "Decoy Grenade",
    "Incendiary Grenade", "C4 Explosive", "Kevlar Vest", "Kevlar & Helmet",
    "Heavy Assault Suit", "item_nvg", "Defuse Kit", "Rescue Kit",
    "Medi-Shot", "knife_t", "M4A1-S", "USP-S", "Trade Up Contract",
    "CZ75-Auto", "R8 Revolver", "Charm Detachments", "Bayonet", "Classic Knife",
    "Flip Knife", "Gut Knife", "Karambit", "M9 Bayonet", "Huntsman Knife",
    "Falchion Knife", "Bowie Knife", "Butterfly Knife", "Shadow Daggers",
    "Paracord Knife", "Survival Knife", "Ursus Knife", "Navaja Knife",
    "Nomad Knife", "Stiletto Knife", "Talon Knife", "Skeleton Knife", "Kukri Knife"
])))
safe_item_names = [name.replace(' ', '_').replace('&', 'and').replace('-','_') for name in ITEM_NAMES]

KEYBOARD_ONLY_ACTIONS = [
    "IN_ATTACK", "IN_JUMP", "IN_DUCK", "IN_FORWARD", "IN_BACK", "IN_USE", "IN_CANCEL", "IN_TURNLEFT", 
    "IN_TURNRIGHT", "IN_MOVELEFT", "IN_MOVERIGHT", "IN_ATTACK2", "IN_RELOAD", "IN_ALT1", "IN_ALT2", 
    "IN_SPEED", "IN_WALK", "IN_ZOOM", "IN_WEAPON1", "IN_WEAPON2", "IN_BULLRUSH", "IN_GRENADE1", 
    "IN_GRENADE2", "IN_ATTACK3", "IN_SCORE", "IN_INSPECT", "SWITCH_1", "SWITCH_2", "SWITCH_3", "SWITCH_4", "SWITCH_5"]
ECO_ACTIONS = ["IN_BUYZONE"]
for name in safe_item_names:
    ECO_ACTIONS.append(f"BUY_{name}") #; ECO_ACTIONS.append(f"SELL_{name}"); ECO_ACTIONS.append(f"DROP_{name}") #no needs since bullshit in extract.py
item_id_map_names = [
    "deagle", "elite", "fiveseven", "glock", "ak47", "aug", "awp", "famas", "g3sg1", "galilar",
    "m249", "m4a1", "mac10", "p90", "mp5sd", "ump45", "xm1014", "ppbizon", "mag7", "negev",
    "sawedoff", "tec9", "zeus", "p2000", "mp7", "mp9", "nova", "p250", "scar20", "sg556", "ssg08",
    "knife", "flashbang", "hegrenade", "smokegrenade", "molotov", "decoy", "incgrenade", "c4",
    "vest", "vesthelm", "heavyassaultsuit", "nvgs", "defuser", "rescue_kit", "medishot", "knifet",
    "m4a1_silencer", "usp_silencer", "tradeupcontract", "cz75auto", "r8revolver", "charmdetachments",
    "bayonet", "knife_default_ct", "flipknife", "gutknife", "karambit", "knife_m9_bayonet",
    "huntsmanknife", "falchionknife", "bowieknife", "butterflyknife", "shadowdaggers",
    "paracordknife", "survivalknife", "ursusknife", "navajaknife", "nomadknife", "stilettoknife",
    "talonknife", "skeletonknife", "kukriknife"
]
for name in item_id_map_names:
    ECO_ACTIONS.append(f"SELL_{name}"); ECO_ACTIONS.append(f"DROP_{name}") #ECO_ACTIONS.append(f"BUY_{name}"); #no need since extract.py bullshit

ECO_ACTIONS = sorted(list(set(ECO_ACTIONS)))
KEYBOARD_TO_BIT = {action: i for i, action in enumerate(KEYBOARD_ONLY_ACTIONS)}
ECO_TO_BIT = {action: i for i, action in enumerate(ECO_ACTIONS)}
ITEM_TO_INDEX = {item: i for i, item in enumerate(ITEM_NAMES)}

# =============================================================================
# DATA PROCESSING HELPERS
# =============================================================================

def merge_tick_data(tick1, tick2):
    """
    Merges data from two consecutive game ticks to represent one video frame.

    This robust version correctly handles two distinct issues:
    1.  Prevents TypeError by treating potential 'None' values in numeric fields.
    2.  Correctly averages position data only when two valid points exist,
        otherwise using the single available point.
    """
    # Handle cases where one or both ticks are missing
    if not tick1:
        return tick2
    if not tick2:
        return tick1

    m = tick1.copy()

    # --- FIX for Mouse Deltas ---
    m['mouse_x'] = (tick1.get('mouse_x') or 0) + (tick2.get('mouse_x') or 0)
    m['mouse_y'] = (tick1.get('mouse_y') or 0) + (tick2.get('mouse_y') or 0)

    # --- FIX for Position Averaging ---
    for coord in ['position_x', 'position_y', 'position_z']:
        positions = []
        pos1 = tick1.get(coord)
        if pos1 is not None:
            positions.append(pos1)
        pos2 = tick2.get(coord)
        if pos2 is not None:
            positions.append(pos2)
        if positions:
            m[coord] = sum(positions) / len(positions)
        else:
            m[coord] = 0.0

    # Union of keyboard and buy inputs is unaffected and already robust
    kb1 = set(filter(None, (tick1.get('keyboard_input') or '').split(',')))
    kb2 = set(filter(None, (tick2.get('keyboard_input') or '').split(',')))
    m['keyboard_input'] = ",".join(sorted(list(kb1.union(kb2))))

    b1 = set(filter(None, (tick1.get('buy_sell_input') or '').split(',')))
    b2 = set(filter(None, (tick2.get('buy_sell_input') or '').split(',')))
    m['buy_sell_input'] = ",".join(sorted(list(b1.union(b2))))

    return m

def get_bitmask(actions, mapping, name):
    """Converts a list of actions into a single integer bitmask."""
    mask = 0
    if not actions: return mask
    for a in actions:
        if a not in mapping: 
            LOG.warning(f"Unknown action '{a}' in '{name}' - skipping.")
            continue
        mask |= (1 << mapping[a])
    return mask

def get_bitmask_array(actions, mapping, name):
    """Converts a list of actions into a numpy array of uint64 bitmasks."""
    mask = np.zeros(4, dtype=np.uint64)
    if not actions: return mask
    for a in actions:
        if a not in mapping: 
            LOG.warning(f"Unknown action '{a}' in '{name}' - skipping.")
            continue
        idx, pi = mapping[a] // 64, mapping[a] % 64
        if idx < 4: mask[idx] |= (np.uint64(1) << np.uint64(pi))
    return mask

def get_inventory_bitmasks(inv_json, weapon, mapping):
    """Generates bitmasks for player inventory and active weapon."""
    im, wm = np.zeros(2, dtype=np.uint64), np.zeros(2, dtype=np.uint64)
    def set_bit(m, item):
        if not isinstance(item, str): return
        bp = mapping.get(item)
        if bp is None: 
            LOG.warning(f"Unknown item '{item}' - skipping.")
            return
        idx, pi = bp // 64, bp % 64
        if idx < 2: m[idx] |= (np.uint64(1) << np.uint64(pi))
    if weapon: set_bit(wm, weapon)
    try:
        for item in json.loads(inv_json) if inv_json else []:
            set_bit(im, item)
    except (json.JSONDecodeError, TypeError):
        pass # Ignore malformed inventory JSON
    return im, wm

# =============================================================================
# MAIN DRIVER
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Compile CS2 recordings into a model-ready LMDB.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--recdir", required=True, type=Path, help="Directory containing MP4 and WAV recording files.")
    parser.add_argument("--dbfile", required=True, type=Path, help="Path to the SQLite database with game metadata.")
    parser.add_argument("--outlmdb", required=True, type=Path, help="Path to write the output LMDB database.")
    parser.add_argument("--overwrite", action='store_true', help="If set, remove the output LMDB if it already exists.")
    parser.add_argument("--overridesql", action='store_true', help="If set, proceed even if DB flags a player as not recorded.")
    parser.add_argument("--debug", action="store_true", help="Enable debug-level logging and print final metadata to stdout.")
    args = parser.parse_args()

    # --- Setup Logging ---
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s-%(name)s-%(levelname)s-%(message)s')

    if args.outlmdb.exists():
        if args.overwrite:
            LOG.warning(f"Removing existing LMDB: {args.outlmdb}")
            shutil.rmtree(args.outlmdb)
        else:
            LOG.critical(f"Output path {args.outlmdb} exists. Use --overwrite to replace it.")
            sys.exit(1)

    # --- Define Numpy data structures ---
    gs_dtype = np.dtype([('tick', np.int32), ('round_state', np.uint8), ('team_alive', np.uint8), ('enemy_alive', np.uint8), ('enemy_pos', np.float32, (5, 3))])
    pi_dtype = np.dtype([('pos', np.float32, (3,)), ('mouse', np.float32, (2,)), ('health', np.uint8), ('armor', np.uint8), ('money', np.int32), ('keyboard_bitmask', np.uint32), ('eco_bitmask', np.uint64, (4,)), ('inventory_bitmask', np.uint64, (2,)), ('active_weapon_bitmask', np.uint64, (2,))])
    assert len(KEYBOARD_TO_BIT) <= 32, "Too many keyboard actions for uint32 bitmask"
    assert len(ECO_TO_BIT) <= 256, "Too many eco actions for 4x uint64 bitmask"
    assert len(ITEM_TO_INDEX) <= 128, "Too many items for 2x uint64 bitmask"

    # --- Phase 1: Load all metadata from database ---
    LOG.info("-> Phase 1: Validating database and loading all metadata into memory...")
    try:
        with t("db_connect"):
            conn = sqlite3.connect(f"file:{args.dbfile}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
        with t("db_load_rounds"):
            rounds_info = {r['round']: dict(r) for r in conn.execute("SELECT * FROM rounds").fetchall()}
        LOG.info(f"   - Loaded info for {len(rounds_info)} rounds.")
        with t("db_load_recordings"):
            db_recordings = conn.execute("SELECT * FROM recording ORDER BY roundnumber, team, playername").fetchall()
        with t("db_load_player_cache"):
            player_data_cache = {f"{r['playername']}:{r['tick']}": dict(r) for r in conn.execute("SELECT * FROM player").fetchall()}
        LOG.info(f"   - Cached data for {len(player_data_cache)} player-tick entries.")
    finally:
        if 'conn' in locals():
            conn.close()

    recordings_map = {}
    round_team_pov_paths = {}
    round_team_audio_paths = {}
    for rec_row in db_recordings:
        rec = dict(rec_row)
        round_num, team = rec['roundnumber'], rec['team']
        if round_num not in rounds_info or rounds_info[round_num].get('starttick') is None:
            continue
        if not rec['is_recorded'] and not args.overridesql:
            LOG.critical(f"DB indicates recording for {rec['playername']} round {round_num} is missing. Use --overridesql to ignore.")
            sys.exit(1)

        fname = f"{rec['roundnumber']:02d}_{rec['team']}_{sanitize_player_name(rec['playername'])}_{rec['starttick']}_{rec['stoptick']}"
        mp4, wav = args.recdir / f"{fname}.mp4", args.recdir / f"{fname}.wav"
        if not mp4.exists() or not wav.exists():
            LOG.critical(f"Media file not found: {mp4.name} or {wav.name} in {args.recdir}")
            sys.exit(1)

        rec['mp4_path'], rec['wav_path'] = str(mp4), str(wav)
        key = (round_num, team)
        recordings_map.setdefault(key, []).append(rec)
        round_team_pov_paths.setdefault(key, []).append(mp4.name)
        round_team_audio_paths.setdefault(key, []).append(wav.name)


    LOG.info(f"   - Validated {len(db_recordings)} recording entries.")
    for key, paths in round_team_pov_paths.items():
        if len(paths) != 5:
            LOG.warning(f"Round perspective {key} does not have 5 recordings ({len(paths)} found). It will be skipped.")

    # --- Phase 2: Process rounds sequentially and write to LMDB ---
    demoname = args.recdir.name
    tasks = sorted(recordings_map.keys())
    LOG.info(f"-> Phase 2: Processing {len(tasks)} round-perspectives...")

    env = lmdb.open(str(args.outlmdb), map_size=INITIAL_MAP_SIZE, writemap=True)
    try:
        pbar = tqdm(tasks, desc="Processing Perspectives")
        for round_num, team in pbar:
            round_data = recordings_map.get((round_num, team), [])
            if len(round_data) != 5:
                continue  # Skip perspectives that don't have all 5 players

            # Get rosters and cache death tick maps ONCE per perspective
            with t("json_rosters"):
                t_roster = [p[0] for p in json.loads(rounds_info[round_num]['t_team'])]
                ct_roster = [p[0] for p in json.loads(rounds_info[round_num]['ct_team'])]
                current_roster = t_roster if team == 'T' else ct_roster
                enemy_roster = ct_roster if team == 'T' else t_roster
            with t("json_team_maps"):
                team_deaths_map = {p[0]: p[1] for p in json.loads(rounds_info[round_num][f"{team.lower()}_team"])}
                enemy_deaths_map = {p[0]: p[1] for p in json.loads(rounds_info[round_num][f"{'ct' if team == 'T' else 't'}_team"])}

            start_tick = rounds_info[round_num]['starttick']
            end_tick = max(rec['stoptick'] for rec in round_data)
            results_for_round = []

            for tick in range(start_tick, end_tick + 1, TICKS_PER_FRAME):
                # --- Process Game State ---
                team_alive = sum(1 << i for i, n in enumerate(current_roster)
                                 if team_deaths_map.get(n, 0) == -1 or tick < team_deaths_map.get(n, 0))
                enemy_alive = sum(1 << i for i, n in enumerate(enemy_roster)
                                  if enemy_deaths_map.get(n, 0) == -1 or tick < enemy_deaths_map.get(n, 0))

                gs = np.zeros(1, dtype=gs_dtype)
                gs[0]['tick'], gs[0]['team_alive'], gs[0]['enemy_alive'] = tick, team_alive, enemy_alive

                rs_mask = 0
                if rounds_info[round_num].get('freezetime_endtick') and tick < rounds_info[round_num]['freezetime_endtick']: rs_mask |= (1 << 0) # In Freeze Time
                if tick >= rounds_info[round_num]['starttick'] and tick <= rounds_info[round_num]['endtick']: rs_mask |= (1 << 1) # Round Active
                if rounds_info[round_num].get('bomb_planted_tick', -1) != -1 and tick >= rounds_info[round_num]['bomb_planted_tick']: rs_mask |= (1 << 2) # Bomb Planted
                win_tick = rounds_info[round_num].get('win_tick')
                if win_tick and tick >= win_tick:
                    win_team = rounds_info[round_num].get('win_team')
                    if win_team == 't': rs_mask |= (1 << 3) # T Win
                    elif win_team == 'ct': rs_mask |= (1 << 4) # CT Win
                gs[0]['round_state'] = rs_mask

                for i, name in enumerate(enemy_roster):
                    if (enemy_alive >> i) & 1:
                        t1 = player_data_cache.get(f"{name}:{tick}")
                        t2 = player_data_cache.get(f"{name}:{tick+1}")
                        ed = merge_tick_data(t1, t2)
                        if ed: gs[0]['enemy_pos'][i] = [ed['position_x'], ed['position_y'], ed['position_z']]

                # --- Process Player Data for all 5 players ---
                pdl_unsorted = []
                for rec in round_data:
                    pn = rec['playername']
                    try:
                        p_idx = current_roster.index(pn)
                    except ValueError:
                        continue  # Player not in the official roster for this round

                    if not ((team_alive >> p_idx) & 1) or tick >= rec['stoptick']:
                        continue  # Player is dead or their recording stopped

                    # Process Player State
                    td = merge_tick_data(player_data_cache.get(f"{pn}:{tick}"), player_data_cache.get(f"{pn}:{tick+1}"))
                    pi = np.zeros(1, dtype=pi_dtype)
                    if td:
                        kb, bs = (td.get('keyboard_input') or ''), (td.get('buy_sell_input') or '')
                        akb = [a.replace("-", "_").replace(" ", "_") for a in kb.split(',') if a]
                        bs_a = [a.replace("-", "_").replace(" ", "_") for a in bs.split(',') if a]
                        k_a = [a for a in akb if not a.startswith('DROP_')]
                        pi[0]['keyboard_bitmask'] = get_bitmask(k_a, KEYBOARD_TO_BIT, "KEYBOARD_ONLY_ACTIONS")
                        d_a = [a for a in akb if a.startswith('DROP_')]
                        e_a = d_a + bs_a
                        if td.get('is_in_buyzone') == 1: e_a.append('IN_BUYZONE')
                        pi[0]['eco_bitmask'] = get_bitmask_array(e_a, ECO_TO_BIT, "ECO_ACTIONS")
                        pi[0]['pos'] = [td.get('position_x', 0), td.get('position_y', 0), td.get('position_z', 0)]
                        pi[0]['mouse'] = [td.get('mouse_x', 0), td.get('mouse_y', 0)]
                        pi[0]['health'], pi[0]['armor'], pi[0]['money'] = td.get('health', 0), td.get('armor', 0), td.get('money', 0)
                        im, wm = get_inventory_bitmasks(td.get('inventory'), td.get('active_weapon'), ITEM_TO_INDEX)
                        pi[0]['inventory_bitmask'], pi[0]['active_weapon_bitmask'] = im, wm

                    pdl_unsorted.append((p_idx, pi))

                if not pdl_unsorted:
                    continue  # No living, recorded players this tick

                # Sort player data by their index in the roster
                pdl_unsorted.sort(key=lambda item: item[0])
                pdl = [item[1] for item in pdl_unsorted]

                key = f"{demoname}_round_{round_num:03d}_team_{team}_tick_{tick:08d}".encode('utf-8')
                with t("payload_pack"):
                    payload = msgpack.packb({"game_state": gs, "player_data": pdl}, use_bin_type=True, default=mpnp.encode)
                results_for_round.append((key, payload))

            # --- Write accumulated results for the round to LMDB ---
            if results_for_round:
                # --- Robust "Retry on Failure" Loop for Windows Compatibility ---
                written_successfully = False
                attempts = 0
                max_attempts = 5 # A safety break to prevent infinite loops

                while not written_successfully and attempts < max_attempts:
                    try:
                        # Attempt to write the entire batch in a single transaction
                        with t("lmdb_write"):
                            with env.begin(write=True) as txn:
                                cursor = txn.cursor()
                                cursor.putmulti(results_for_round)
                        
                        # If the transaction completes without error, we're done.
                        written_successfully = True

                    except lmdb.MapFullError:
                        attempts += 1
                        LOG.warning(f"LMDB MapFullError caught on attempt {attempts}. Resizing database and retrying.")

                        # The transaction failed, so now we resize.
                        # This must be done OUTSIDE of a transaction.
                        current_info = env.info()
                        current_size = current_info['map_size']
                        
                        # Calculate a smart new size. We need to accommodate the batch that just failed.
                        failed_batch_size = sum(len(p) for _, p in results_for_round)
                        
                        # Grow by at least MAP_RESIZE_INCREMENT, but more if the failed batch was huge.
                        # The multiplier (e.g., 2) provides a buffer for LMDB overhead.
                        resize_amount = max(failed_batch_size * 2, MAP_RESIZE_INCREMENT)
                        new_size = current_size + resize_amount

                        LOG.info(f"Resizing LMDB map from {current_size/(1024**2):.2f}MB to {new_size/(1024**2):.2f}MB")
                        env.set_mapsize(new_size)

                    except Exception as e:
                        LOG.critical(f"An unexpected error occurred during LMDB write: {e}")
                        # Break the loop on any other kind of error
                        break
                
                if not written_successfully:
                    LOG.critical(f"Failed to write batch for round {round_num} after {max_attempts} attempts. Aborting.")
                    sys.exit(1)

        # --- Phase 3: Finalize by writing metadata ---
        LOG.info("-> Phase 3: Finalizing and writing metadata...")
        rounds_metadata = []
        for (round_num, team), pov_video_paths in sorted(round_team_pov_paths.items()):
            if round_num in rounds_info and len(pov_video_paths) == 5:
                round_info = rounds_info[round_num]
                team_roster = [p[0] for p in json.loads(round_info[f'{team.lower()}_team'])]

                # --- CORRECTED NAME PARSING LOGIC ---
                video_path_map = {}
                for p in pov_video_paths:
                    # Reconstruct player name by joining the middle parts of the filename
                    parts = Path(p).stem.split('_')
                    player_name = '_'.join(parts[2:-2])
                    video_path_map[player_name] = p
                sorted_videos = [video_path_map.get(sanitize_player_name(player_name)) for player_name in team_roster]

                pov_audio_paths = round_team_audio_paths.get((round_num, team), [])
                audio_path_map = {}
                for p in pov_audio_paths:
                    # Reconstruct player name by joining the middle parts of the filename
                    parts = Path(p).stem.split('_')
                    player_name = '_'.join(parts[2:-2])
                    audio_path_map[player_name] = p
                sorted_audio = [audio_path_map.get(sanitize_player_name(player_name)) for player_name in team_roster]
                # --- END OF CORRECTION ---

                if all(sorted_videos) and all(sorted_audio):
                    # --- MODIFICATION START ---
                    # Calculate the end tick based on the last moment any player
                    # on this team was recorded, rather than the absolute round end.
                    recordings_for_perspective = recordings_map.get((round_num, team), [])
                    team_pov_end_tick = 0
                    if recordings_for_perspective:
                        team_pov_end_tick = max(rec['stoptick'] for rec in recordings_for_perspective)
                    else:
                        # Fallback to the original round end tick if no recordings are found
                        team_pov_end_tick = round_info['endtick']
                    # --- MODIFICATION END ---

                    rounds_metadata.append({
                        "round_num": round_num,
                        "team": team,
                        "start_tick": round_info['starttick'],
                        "end_tick": team_pov_end_tick, # Use the calculated team-specific end tick
                        "pov_videos": sorted_videos,
                        "pov_audio": sorted_audio
                    })
                else:
                    LOG.warning(f"Could not form a complete 5-player POV list for round {round_num}, team {team} due to name mismatch. Skipping from metadata.")

        metadata_payload = json.dumps({"demoname": demoname, "rounds": rounds_metadata}, indent=2).encode('utf-8')
        with t("lmdb_write"):
            with env.begin(write=True) as txn:
                key = f"{demoname}_INFO".encode('utf-8')
                txn.put(key, metadata_payload)

        if args.debug:
            LOG.info("--- Final Metadata (_INFO) ---")
            print(metadata_payload.decode('utf-8'))

    except Exception as e:
        import traceback
        LOG.error(f"An unhandled exception occurred: {e}")
        LOG.error(traceback.format_exc())
        sys.exit(1)
    finally:
        if 'env' in locals():
            env.close()

    # --- Timing summary ---
    if TIMERS:
        total = sum(TIMERS.values())
        LOG.info("---- Timing (hot sections) ----")
        for k, v in sorted(TIMERS.items(), key=lambda kv: kv[1], reverse=True):
            LOG.info(f"{k:18s} {v:8.3f}s  ({(v/total*100):5.1f}%)")
        LOG.info(f"Total timed sections: {total:.3f}s")

    LOG.info("All processing steps completed successfully.")
    LOG.info(f"Final LMDB database is at: {args.outlmdb}")

if __name__ == '__main__':
    main()