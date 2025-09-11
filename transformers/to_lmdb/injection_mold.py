#!/usr/bin/env python3
"""
injection_mold.py - Compile CS2 recordings and database into a unified LMDB
for model training.

This version DOES NOT embed video frames. Instead, it processes all non-visual
data (player states, audio) and creates a final metadata key containing the
paths to the corresponding MP4 files for each round-perspective, to be used
by a video-aware training script.

This refactored version is single-threaded to eliminate multiprocessing-related
complexity and improve reliability.
"""
import argparse
import json
import logging
import os
import shutil
import sqlite3
import sys
from pathlib import Path

import lmdb
import numpy as np
from tqdm import tqdm

# Import msgpack and numpy extension for serialization
import msgpack
import msgpack_numpy as mpnp

try:
    import librosa
except ImportError:
    print("Error: The 'librosa' library is required for spectrogram generation.", file=sys.stderr)
    print("Please install it using: pip install librosa", file=sys.stderr)
    sys.exit(1)

# --- Configuration ---
GAME_TICKS_PER_SEC = 64
EXPECTED_VIDEO_FPS = 32
TICKS_PER_FRAME = GAME_TICKS_PER_SEC // EXPECTED_VIDEO_FPS
AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 2
AUDIO_BIT_DEPTH = 2
AUDIO_BYTES_PER_FRAME = (AUDIO_SAMPLE_RATE // EXPECTED_VIDEO_FPS) * AUDIO_CHANNELS * AUDIO_BIT_DEPTH
INITIAL_MAP_SIZE = 2 * 1024**3  # 2 GB
MAP_RESIZE_INCREMENT = 1 * 1024**3 # 1 GB

# --- Globals ---
LOG = logging.getLogger("InjectionMold")

# =============================================================================
# DATA ENCODING MAPPINGS (Canonical Source)
# =============================================================================
KEYBOARD_ONLY_ACTIONS = ["IN_ATTACK", "IN_JUMP", "IN_DUCK", "IN_FORWARD", "IN_BACK", "IN_USE", "IN_CANCEL", "IN_TURNLEFT", "IN_TURNRIGHT", "IN_MOVELEFT", "IN_MOVERIGHT", "IN_ATTACK2", "IN_RELOAD", "IN_ALT1", "IN_ALT2", "IN_SPEED", "IN_WALK", "IN_ZOOM", "IN_WEAPON1", "IN_WEAPON2", "IN_BULLRUSH", "IN_GRENADE1", "IN_GRENADE2", "IN_ATTACK3", "IN_SCORE", "IN_INSPECT", "SWITCH_1", "SWITCH_2", "SWITCH_3", "SWITCH_4", "SWITCH_5"]
ITEM_NAMES = sorted(list(set(["AK-47", "M4A4", "M4A1-S", "Galil AR", "FAMAS", "AUG", "SG 553", "AWP", "SSG 08", "G3SG1", "SCAR-20", "Glock-18", "USP-S", "P250", "P2000", "Dual Berettas", "Five-SeveN", "Tec-9", "CZ75-Auto", "R8 Revolver", "Desert Eagle", "MP9", "MAC-10", "MP7", "MP5-SD", "UMP-45", "P90", "PP-Bizon", "Nova", "XM1014", "MAG-7", "Sawed-Off", "M249", "Negev", "Knife", "knife_t", "knife_ct", "Bayonet", "Flip Knife", "Gut Knife", "Karambit", "M9 Bayonet", "Huntsman Knife", "Falchion Knife", "Bowie Knife", "Butterfly Knife", "Shadow Daggers", "Ursus Knife", "Navaja Knife", "Stiletto Knife", "Talon Knife", "Classic Knife", "Paracord Knife", "Survival Knife", "Nomad Knife", "Skeleton Knife", "High Explosive Grenade", "Flashbang", "Smoke Grenade", "Molotov", "Incendiary Grenade", "Decoy Grenade", "C4 Explosive", "Defuse Kit", "Zeus x27", "Kevlar Vest", "Kevlar and Helmet", "Helmet"])))
ECO_ACTIONS = ["IN_BUYZONE"]
safe_item_names = [name.replace(" ", "_").replace("-", "_") for name in ITEM_NAMES]
for name in safe_item_names: ECO_ACTIONS.append(f"BUY_{name}"); ECO_ACTIONS.append(f"SELL_{name}"); ECO_ACTIONS.append(f"DROP_{name}")
item_id_map_names = ["deagle", "elite", "fiveseven", "glock", "ak47", "aug", "awp", "famas", "g3sg1", "galilar", "m249", "m4a1", "mac10", "p90", "mp5sd", "ump45", "xm1014", "bizon", "mag7", "negev", "sawedoff", "tec9", "zeus","p2000", "mp7", "mp9", "nova", "p250", "scar20", "sg556", "ssg08", "knife", "flashbang", "hegrenade", "smokegrenade", "molotov", "decoy", "incgrenade", "c4", "knife_t", "m4a1_silencer", "usp_silencer", "cz75a", "revolver", "defuser", "vest", "vesthelm"]
for name in item_id_map_names: ECO_ACTIONS.append(f"BUY_{name}"); ECO_ACTIONS.append(f"SELL_{name}"); ECO_ACTIONS.append(f"DROP_{name}")
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

    This version correctly handles position averaging. If only one tick provides
    position data, that data is used directly instead of being incorrectly halved.
    """
    # Handle cases where one or both ticks are missing
    if not tick1:
        return tick2
    if not tick2:
        return tick1

    # Start with a copy of the first tick's data
    m = tick1.copy()

    # Mouse movements are summed over the frame, as they are deltas
    m['mouse_x'] = (tick1.get('mouse_x', 0)) + (tick2.get('mouse_x', 0))
    m['mouse_y'] = (tick1.get('mouse_y', 0)) + (tick2.get('mouse_y', 0))

    # --- CORRECTED POSITION MERGING ---
    # For each coordinate, collect all non-None values and find their mean.
    # This correctly averages when two values are present, and uses the single
    # value when only one is present.
    for coord in ['position_x', 'position_y', 'position_z']:
        positions = []
        pos1 = tick1.get(coord)
        if pos1 is not None:
            positions.append(pos1)

        pos2 = tick2.get(coord)
        if pos2 is not None:
            positions.append(pos2)

        # If the list is not empty, calculate the average; otherwise, default to 0.0
        if positions:
            m[coord] = sum(positions) / len(positions)
        else:
            m[coord] = 0.0

    # Union of keyboard and buy inputs remains the same
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
        if a not in mapping: raise ValueError(f"Unknown action '{a}' in '{name}'")
        mask |= (1 << mapping[a])
    return mask

def get_bitmask_array(actions, mapping, name):
    """Converts a list of actions into a numpy array of uint64 bitmasks."""
    mask = np.zeros(6, dtype=np.uint64)
    if not actions: return mask
    for a in actions:
        if a not in mapping: raise ValueError(f"Unknown action '{a}' in '{name}'")
        idx, pi = mapping[a] // 64, mapping[a] % 64
        if idx < 6: mask[idx] |= (np.uint64(1) << np.uint64(pi))
    return mask

def get_inventory_bitmasks(inv_json, weapon, mapping):
    """Generates bitmasks for player inventory and active weapon."""
    im, wm = np.zeros(2, dtype=np.uint64), np.zeros(2, dtype=np.uint64)
    def set_bit(m, item):
        if not isinstance(item, str): return
        # Handle knife variants by mapping them to a canonical "Knife"
        bp = mapping.get(item)
        if bp is None and ("knife" in item.lower() or "bayonet" in item.lower()):
            bp = mapping.get("Knife")
        if bp is None: raise ValueError(f"Unknown item '{item}'")
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
    parser.add_argument("--debug", action="store_true", help="Enable debug-level logging.")

    audio_group = parser.add_argument_group('Audio Spectrogram Parameters')
    audio_group.add_argument('--no-audio', action='store_true', help="Disable audio processing and store None for audio.")
    audio_group.add_argument('--n-mels', type=int, default=128, help="Number of Mel bands to generate (Default: 128).")
    audio_group.add_argument('--n-fft', type=int, default=2048, help="Length of the FFT window (Default: 2048).")
    audio_group.add_argument('--hop-length', type=int, default=512, help="Samples between frames (Default: 512).")
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
    pi_dtype = np.dtype([('pos', np.float32, (3,)), ('mouse', np.float32, (2,)), ('health', np.uint8), ('armor', np.uint8), ('money', np.int32), ('keyboard_bitmask', np.uint32), ('eco_bitmask', np.uint64, (6,)), ('inventory_bitmask', np.uint64, (2,)), ('active_weapon_bitmask', np.uint64, (2,))])
    assert len(KEYBOARD_TO_BIT) <= 32, "Too many keyboard actions for uint32 bitmask"
    assert len(ECO_TO_BIT) <= 384, "Too many eco actions for 6x uint64 bitmask"
    assert len(ITEM_TO_INDEX) <= 128, "Too many items for 2x uint64 bitmask"

    # --- Phase 1: Load all metadata from database ---
    LOG.info("-> Phase 1: Validating database and loading all metadata into memory...")
    try:
        conn = sqlite3.connect(f"file:{args.dbfile}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rounds_info = {r['round']: dict(r) for r in conn.execute("SELECT * FROM rounds").fetchall()}
        LOG.info(f"   - Loaded info for {len(rounds_info)} rounds.")

        db_recordings = conn.execute("SELECT * FROM recording ORDER BY roundnumber, team, playername").fetchall()
        player_data_cache = {f"{r['playername']}:{r['tick']}": dict(r) for r in conn.execute("SELECT * FROM player").fetchall()}
        LOG.info(f"   - Cached data for {len(player_data_cache)} player-tick entries.")
    finally:
        if 'conn' in locals():
            conn.close()

    recordings_map = {}
    round_team_pov_paths = {}
    for rec_row in db_recordings:
        rec = dict(rec_row)
        round_num, team = rec['roundnumber'], rec['team']
        if round_num not in rounds_info or rounds_info[round_num].get('starttick') is None:
            continue
        if not rec['is_recorded'] and not args.overridesql:
            LOG.critical(f"DB indicates recording for {rec['playername']} round {round_num} is missing. Use --overridesql to ignore.")
            sys.exit(1)

        fname = f"{rec['roundnumber']:02d}_{rec['team']}_{rec['playername']}_{rec['starttick']}_{rec['stoptick']}"
        mp4, wav = args.recdir / f"{fname}.mp4", args.recdir / f"{fname}.wav"
        if not mp4.exists() or not wav.exists():
            LOG.critical(f"Media file not found: {mp4.name} or {wav.name} in {args.recdir}")
            sys.exit(1)

        rec['mp4_path'], rec['wav_path'] = str(mp4), wav
        key = (round_num, team)
        recordings_map.setdefault(key, []).append(rec)
        round_team_pov_paths.setdefault(key, []).append(str(mp4))

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
                continue # Skip perspectives that don't have all 5 players

            # Open all audio files for this perspective
            audio_files = {rec['playername']: open(rec['wav_path'], 'rb') for rec in round_data}
            for aud in audio_files.values():
                aud.seek(44) # Skip WAV header

            # Get rosters
            t_roster = [p[0] for p in json.loads(rounds_info[round_num]['t_team'])]
            ct_roster = [p[0] for p in json.loads(rounds_info[round_num]['ct_team'])]
            current_roster = t_roster if team == 'T' else ct_roster
            enemy_roster = ct_roster if team == 'T' else t_roster

            start_tick = rounds_info[round_num]['starttick']
            end_tick = max(rec['stoptick'] for rec in round_data)
            results_for_round = []

            for tick in range(start_tick, end_tick + 1, TICKS_PER_FRAME):
                # --- Process Game State ---
                team_deaths = {p[0]: p[1] for p in json.loads(rounds_info[round_num][f"{team.lower()}_team"])}
                enemy_deaths = {p[0]: p[1] for p in json.loads(rounds_info[round_num][f"{'ct' if team == 'T' else 't'}_team"])}
                team_alive = sum(1 << i for i, n in enumerate(current_roster) if team_deaths.get(n, 0) == -1 or tick < team_deaths.get(n, 0))
                enemy_alive = sum(1 << i for i, n in enumerate(enemy_roster) if enemy_deaths.get(n, 0) == -1 or tick < enemy_deaths.get(n, 0))

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
                        continue # Player not in the official roster for this round

                    if not ((team_alive >> p_idx) & 1) or tick >= rec['stoptick']:
                        continue # Player is dead or their recording stopped

                    # Process Audio
                    mel_spectrogram = None
                    if not args.no_audio:
                        audio_bytes = audio_files[pn].read(AUDIO_BYTES_PER_FRAME)
                        if len(audio_bytes) == AUDIO_BYTES_PER_FRAME:
                            waveform_int = np.frombuffer(audio_bytes, dtype=np.int16)
                            waveform_float = waveform_int.astype(np.float32) / 32768.0
                            waveform_mono = waveform_float.reshape((-1, AUDIO_CHANNELS)).mean(axis=1)
                            
                            if len(waveform_mono) < args.n_fft:
                                pad_width = args.n_fft - len(waveform_mono)
                                waveform_mono = np.pad(waveform_mono, (pad_width // 2, pad_width - pad_width // 2), mode='constant')

                            S = librosa.feature.melspectrogram(y=waveform_mono, sr=AUDIO_SAMPLE_RATE, n_mels=args.n_mels, n_fft=args.n_fft, hop_length=args.hop_length)
                            mel_spectrogram = librosa.power_to_db(S, ref=np.max)
                        else:
                            continue # Not enough audio data, skip this tick for this player

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
                    
                    pdl_unsorted.append((p_idx, (pi, None, mel_spectrogram)))

                if not pdl_unsorted:
                    continue # No living, recorded players this tick

                # Sort player data by their index in the roster
                pdl_unsorted.sort(key=lambda item: item[0])
                pdl = [item[1] for item in pdl_unsorted]

                key = f"{demoname}_round_{round_num:03d}_team_{team}_tick_{tick:08d}".encode('utf-8')
                payload = msgpack.packb({"game_state": gs, "player_data": pdl}, use_bin_type=True, default=mpnp.encode)
                results_for_round.append((key, payload))

            # --- Write accumulated results for the round to LMDB ---
            if results_for_round:
                with env.begin(write=True) as txn:
                    # Check if DB needs resizing before writing
                    batch_size = sum(len(p) for _, p in results_for_round)
                    info = env.info()
                    stats = env.stat()
                    available = info['map_size'] - (info['last_pgno'] * stats['psize'])
                    if available < batch_size:
                        new_size = info['map_size'] + max(batch_size * 1.5, MAP_RESIZE_INCREMENT)
                        LOG.info(f"Resizing LMDB map from {info['map_size']/(1024**2):.2f}MB to {new_size/(1024**2):.2f}MB")
                        txn.commit() # Must commit before resizing
                        env.set_mapsize(new_size)
                        txn = env.begin(write=True) # Start new transaction

                    cursor = txn.cursor()
                    cursor.putmulti(results_for_round)

            # Close all audio files for this perspective
            for aud in audio_files.values():
                aud.close()

        # --- Phase 3: Finalize by writing metadata ---
        LOG.info("-> Phase 3: Finalizing and writing metadata...")
        rounds_metadata = []
        for (round_num, team), pov_paths in sorted(round_team_pov_paths.items()):
            if round_num in rounds_info and len(pov_paths) == 5:
                round_info = rounds_info[round_num]
                team_roster = [p[0] for p in json.loads(round_info[f'{team.lower()}_team'])]
                path_map = {Path(p).name.split('_')[2]: p for p in pov_paths}
                sorted_paths = [path_map.get(player_name) for player_name in team_roster]

                if all(sorted_paths):
                    rounds_metadata.append({
                        "round_num": round_num,
                        "team": team,
                        "start_tick": round_info['starttick'],
                        "end_tick": round_info['endtick'],
                        "pov_videos": sorted_paths
                    })
                else:
                    LOG.warning(f"Could not form a complete 5-player POV list for round {round_num}, team {team} due to name mismatch. Skipping from metadata.")

        metadata_payload = json.dumps({"demoname": demoname, "rounds": rounds_metadata}, indent=2).encode('utf-8')
        with env.begin(write=True) as txn:
            key = f"{demoname}_INFO".encode('utf-8')
            txn.put(key, metadata_payload)

    except Exception as e:
        import traceback
        LOG.error(f"An unhandled exception occurred: {e}")
        LOG.error(traceback.format_exc())
        sys.exit(1)
    finally:
        if 'env' in locals():
            env.close()

    LOG.info("All processing steps completed successfully.")
    LOG.info(f"Final LMDB database is at: {args.outlmdb}")

if __name__ == '__main__':
    main()