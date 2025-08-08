#!/usr/bin/env python3
"""
injection_mold.py - Compile CS2 recordings and database into a unified LMDB
for model training. (Robust Multiprocessing Version with a Managed Pool
of Shared Memory Buffers and pickle for IPC)
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
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import uuid
import pickle # <-- NEW: Import pickle

import cv2
import lmdb
import msgpack
import msgpack_numpy as m
import numpy as np
from tqdm import tqdm

# --- Configuration ---
GAME_TICKS_PER_SEC = 64; EXPECTED_VIDEO_FPS = 32
TICKS_PER_FRAME = GAME_TICKS_PER_SEC // EXPECTED_VIDEO_FPS
AUDIO_SAMPLE_RATE = 44100; AUDIO_CHANNELS = 2; AUDIO_BIT_DEPTH = 2
AUDIO_BYTES_PER_FRAME = (AUDIO_SAMPLE_RATE // EXPECTED_VIDEO_FPS) * AUDIO_CHANNELS * AUDIO_BIT_DEPTH
INITIAL_MAP_SIZE = 20 * 1024**3; MAP_RESIZE_INCREMENT = 5 * 1024**3
MAP_RESIZE_THRESHOLD = 500 * 1024**2
RESULT_BUFFER_SIZE = 500 * 1024**2

# --- Globals ---
LOG = logging.getLogger("InjectionMold"); LMDB_PATH_FOR_CLEANUP = None; m.patch()

# =============================================================================
# DATA ENCODING MAPPINGS (Canonical Source)
# =============================================================================
KEYBOARD_ONLY_ACTIONS = ["IN_ATTACK", "IN_JUMP", "IN_DUCK", "IN_FORWARD", "IN_BACK", "IN_USE", "IN_CANCEL", "IN_TURNLEFT", "IN_TURNRIGHT", "IN_MOVELEFT", "IN_MOVERIGHT", "IN_ATTACK2", "IN_RELOAD", "IN_ALT1", "IN_ALT2", "IN_SPEED", "IN_WALK", "IN_ZOOM", "IN_WEAPON1", "IN_WEAPON2", "IN_BULLRUSH", "IN_GRENADE1", "IN_GRENADE2", "IN_ATTACK3", "IN_SCORE", "IN_INSPECT", "SWITCH_1", "SWITCH_2", "SWITCH_3", "SWITCH_4", "SWITCH_5"]
ECO_ACTIONS = ["IN_BUYZONE", "DROP_deagle", "DROP_elite", "DROP_fiveseven", "DROP_glock", "DROP_ak47", "DROP_aug", "DROP_awp", "DROP_famas", "DROP_g3sg1", "DROP_galilar", "DROP_m249", "DROP_m4a1", "DROP_mac10", "DROP_p90", "DROP_mp5sd", "DROP_ump45", "DROP_xm1014", "DROP_bizon", "DROP_mag7", "DROP_negev", "DROP_sawedoff", "DROP_tec9", "DROP_p2000", "DROP_mp7", "DROP_mp9", "DROP_nova", "DROP_p250", "DROP_scar20", "DROP_sg556", "DROP_ssg08", "DROP_knife", "DROP_flashbang", "DROP_hegrenade", "DROP_smokegrenade", "DROP_molotov", "DROP_decoy", "DROP_incgrenade", "DROP_c4", "DROP_m4a1_silencer", "DROP_usp_silencer", "DROP_cz75a", "DROP_revolver", "DROP_defuser", "BUY_deagle", "BUY_elite", "BUY_fiveseven", "BUY_glock", "BUY_ak47", "BUY_aug", "BUY_awp", "BUY_famas", "BUY_g3sg1", "BUY_galilar", "BUY_m249", "BUY_m4a1", "BUY_mac10", "BUY_p90", "BUY_mp5sd", "BUY_ump45", "BUY_xm1014", "BUY_bizon", "BUY_mag7", "BUY_negev", "BUY_sawedoff", "BUY_tec9", "BUY_p2000", "BUY_mp7", "BUY_mp9", "BUY_nova", "BUY_p250", "BUY_scar20", "BUY_sg556", "BUY_ssg08", "BUY_knife", "BUY_flashbang", "BUY_hegrenade", "BUY_smokegrenade", "BUY_molotov", "BUY_decoy", "BUY_incgrenade", "BUY_c4", "BUY_m4a1_silencer", "BUY_usp_silencer", "BUY_cz75a", "BUY_revolver", "BUY_defuser", "BUY_vest", "BUY_vesthelm", "SELL_deagle", "SELL_fiveseven", "SELL_glock", "SELL_ak47", "SELL_aug", "SELL_awp", "SELL_famas", "SELL_galilar", "SELL_m4a1", "SELL_mac10", "SELL_p90", "SELL_ump45", "SELL_xm1014", "SELL_bizon", "SELL_mag7", "SELL_sawedoff", "SELL_tec9", "SELL_p2000", "SELL_mp7", "SELL_mp9", "SELL_nova", "SELL_p250", "SELL_ssg08", "SELL_flashbang", "SELL_hegrenade", "SELL_smokegrenade", "SELL_molotov", "SELL_decoy", "SELL_incgrenade"]
ITEM_NAMES = sorted(list(set(["AK-47", "M4A4", "M4A1-S", "Galil AR", "FAMAS", "AUG", "SG 553", "AWP", "SSG 08", "G3SG1", "SCAR-20", "Glock-18", "USP-S", "P250", "P2000", "Dual Berettas", "Five-SeveN", "Tec-9", "CZ75-Auto", "R8 Revolver", "Desert Eagle", "MP9", "MAC-10", "MP7", "MP5-SD", "UMP-45", "P90", "PP-Bizon", "Nova", "XM1014", "MAG-7", "Sawed-Off", "M249", "Negev", "Knife", "knife_t", "knife_ct", "Bayonet", "Flip Knife", "Gut Knife", "Karambit", "M9 Bayonet", "Huntsman Knife", "Falchion Knife", "Bowie Knife", "Butterfly Knife", "Shadow Daggers", "Ursus Knife", "Navaja Knife", "Stiletto Knife", "Talon Knife", "Classic Knife", "Paracord Knife", "Survival Knife", "Nomad Knife", "Skeleton Knife", "High Explosive Grenade", "Flashbang", "Smoke Grenade", "Molotov", "Incendiary Grenade", "Decoy Grenade", "C4 Explosive", "Defuse Kit", "Zeus x27", "Kevlar Vest", "Helmet"])))
KEYBOARD_TO_BIT = {action: i for i, action in enumerate(KEYBOARD_ONLY_ACTIONS)}
ECO_TO_BIT = {action: i for i, action in enumerate(ECO_ACTIONS)}
ITEM_TO_INDEX = {item: i for i, item in enumerate(ITEM_NAMES)}

# =============================================================================
# WORKER-SPECIFIC HELPERS AND PROCESSING
# =============================================================================
worker_data = {}

def get_bitmask(actions, mapping):
    mask = 0;
    if actions:
        for action in actions:
            if action in mapping: mask |= (1 << mapping[action])
    return mask

def get_bitmask_array(actions, mapping):
    mask = np.zeros(2, dtype=np.uint64);
    if actions:
        for action in actions:
            if action in mapping:
                bit_pos, idx, pos_in_idx = mapping[action], mapping[action] // 64, mapping[action] % 64
                if idx < 2: mask[idx] |= (np.uint64(1) << np.uint64(pos_in_idx))
    return mask

def get_inventory_bitmasks(inventory_json, active_weapon, mapping):
    inventory_mask, active_weapon_mask = np.zeros(2, dtype=np.uint64), np.zeros(2, dtype=np.uint64)
    def set_bit(mask, item_name):
        if not isinstance(item_name, str): return
        bit_pos = mapping.get(item_name)
        if bit_pos is None and ("knife" in item_name.lower() or "bayonet" in item_name.lower()): bit_pos = mapping.get("Knife")
        if bit_pos is None: raise ValueError(f"Unknown item: '{item_name}'. Add to ITEM_NAMES list.")
        idx, pos_in_idx = bit_pos // 64, bit_pos % 64
        if idx < 2: mask[idx] |= (np.uint64(1) << np.uint64(pos_in_idx))
    if active_weapon: set_bit(active_weapon_mask, active_weapon)
    try:
        for item in json.loads(inventory_json) if inventory_json else []: set_bit(inventory_mask, item)
    except (json.JSONDecodeError, TypeError): pass
    return inventory_mask, active_weapon_mask

def init_worker(shm_cache_name, shm_cache_size, free_q, result_q, gs_dtype, pi_dtype, rounds_info, recordings_map):
    m.patch()
    shm = SharedMemory(name=shm_cache_name)
    packed_cache_copy = bytes(shm.buf[:shm_cache_size])
    shm.close()
    # Use pickle to deserialize the initial cache
    worker_data['player_data_cache'] = pickle.loads(packed_cache_copy)
    worker_data['free_q'] = free_q
    worker_data['result_q'] = result_q
    worker_data['gs_dtype'] = gs_dtype
    worker_data['pi_dtype'] = pi_dtype
    worker_data['rounds_info'] = rounds_info
    worker_data['recordings_map'] = recordings_map

def process_round_perspective(task_args):
    try:
        round_num, team, demoname = task_args
        gs_dtype, pi_dtype = worker_data['gs_dtype'], worker_data['pi_dtype']
        rounds_info, recordings_map = worker_data['rounds_info'], worker_data['recordings_map']
        player_data_cache = worker_data['player_data_cache']
        free_q, result_q = worker_data['free_q'], worker_data['result_q']

        round_data = rounds_info[round_num]
        team_recordings = recordings_map[(round_num, team)]
        if len(team_recordings) != 5: return

        # ... (core processing logic remains identical) ...
        t_roster = [p[0] for p in json.loads(round_data['t_team'])]; ct_roster = [p[0] for p in json.loads(round_data['ct_team'])]
        current_roster = t_roster if team == 'T' else ct_roster; enemy_roster = ct_roster if team == 'T' else t_roster
        caps = {rec['playername']: cv2.VideoCapture(str(rec['mp4_path'])) for rec in team_recordings}
        auds = {rec['playername']: open(rec['wav_path'], 'rb') for rec in team_recordings}
        for aud_file in auds.values(): aud_file.seek(44)
        round_start_tick, round_end_tick = round_data['starttick'], max(rec['stoptick'] for rec in team_recordings)
        results = []
        for current_tick in range(round_start_tick, round_end_tick + 1, TICKS_PER_FRAME):
            game_state = np.zeros(1, dtype=gs_dtype); game_state[0]['tick'] = current_tick
            rs_mask = 0
            if round_data.get('freezetime_endtick') is not None and current_tick < round_data['freezetime_endtick']: rs_mask |= (1 << 0)
            if round_data.get('starttick') is not None and round_data.get('endtick') is not None and current_tick >= round_data['starttick'] and current_tick <= round_data['endtick']: rs_mask |= (1 << 1)
            if round_data.get('bomb_planted_tick', -1) != -1 and current_tick >= round_data['bomb_planted_tick']: rs_mask |= (1 << 2)
            game_state[0]['round_state'] = rs_mask
            team_deaths = {p[0]: p[1] for p in json.loads(round_data[f"{team.lower()}_team"])}; enemy_deaths = {p[0]: p[1] for p in json.loads(round_data[f"{'ct' if team == 'T' else 't'}_team"])}
            team_alive_mask = sum(1 << i for i, n in enumerate(current_roster) if team_deaths.get(n, 0) == -1 or current_tick < team_deaths.get(n, 0))
            enemy_alive_mask = sum(1 << i for i, n in enumerate(enemy_roster) if enemy_deaths.get(n, 0) == -1 or current_tick < enemy_deaths.get(n, 0))
            game_state[0]['team_alive'], game_state[0]['enemy_alive'] = team_alive_mask, enemy_alive_mask
            for i, name in enumerate(enemy_roster):
                if (enemy_alive_mask >> i) & 1:
                    lookup_key = f"{name}:{current_tick}"
                    if lookup_key in player_data_cache:
                        e_data = player_data_cache[lookup_key]; game_state[0]['enemy_pos'][i] = [e_data['position_x'], e_data['position_y'], e_data['position_z']]
            player_data_list = []
            for i, rec in enumerate(team_recordings):
                if not ((team_alive_mask >> i) & 1) or current_tick >= rec['stoptick']: continue
                playername = rec['playername']; ret, frame = caps[playername].read(); audio_chunk = auds[playername].read(AUDIO_BYTES_PER_FRAME)
                if not ret or len(audio_chunk) < AUDIO_BYTES_PER_FRAME: continue
                jpeg_bytes = cv2.imencode('.jpg', frame)[1].tobytes(); lookup_key = f"{playername}:{current_tick}"; tick_data = player_data_cache.get(lookup_key); player_input = np.zeros(1, dtype=pi_dtype)
                if tick_data:
                    kb_input_str, buy_sell_input_str = tick_data.get('keyboard_input', '') or '', tick_data.get('buy_sell_input', '') or ''
                    all_kb_actions = kb_input_str.split(','); keyboard_actions = [a for a in all_kb_actions if not a.startswith('DROP_')]; player_input[0]['keyboard_bitmask'] = get_bitmask(keyboard_actions, KEYBOARD_TO_BIT)
                    drop_actions = [a for a in all_kb_actions if a.startswith('DROP_')]; eco_actions_list = drop_actions + buy_sell_input_str.split(',')
                    if tick_data.get('is_in_buyzone') == 1: eco_actions_list.append('IN_BUYZONE')
                    player_input[0]['eco_bitmask'] = get_bitmask_array(filter(None, eco_actions_list), ECO_TO_BIT)
                    player_input[0]['pos'] = [tick_data.get('position_x', 0), tick_data.get('position_y', 0), tick_data.get('position_z', 0)]; player_input[0]['mouse'] = [tick_data.get('mouse_x', 0), tick_data.get('mouse_y', 0)]
                    player_input[0]['health'], player_input[0]['armor'], player_input[0]['money'] = tick_data.get('health', 0), tick_data.get('armor', 0), tick_data.get('money', 0)
                    inv_mask, wep_mask = get_inventory_bitmasks(tick_data.get('inventory'), tick_data.get('active_weapon'), ITEM_TO_INDEX); player_input[0]['inventory_bitmask'], player_input[0]['active_weapon_bitmask'] = inv_mask, wep_mask
                player_data_list.append((player_input, jpeg_bytes, audio_chunk))
            if not player_data_list: continue
            
            # For LMDB, we still use msgpack for efficiency
            final_payload_for_lmdb = msgpack.packb({"game_state": game_state, "player_data": player_data_list}, use_bin_type=True)
            key = f"{demoname}_round_{round_num:03d}_team_{team}_tick_{current_tick:08d}".encode('utf-8')
            results.append((key, final_payload_for_lmdb))
        
        for cap in caps.values(): cap.release()
        for aud in auds.values(): aud.close()
        if not results: return

        # --- CHANGE: Serialize results using pickle for IPC ---
        packed_results = pickle.dumps(results)
        
        if len(packed_results) > RESULT_BUFFER_SIZE:
            raise ValueError(f"Round {round_num}/{team} result size ({len(packed_results)/(1024**2):.2f}MB) exceeds buffer size. Increase RESULT_BUFFER_SIZE.")

        buffer_name = free_q.get()
        shm = SharedMemory(name=buffer_name)
        shm.buf[:len(packed_results)] = packed_results
        shm.close()
        result_q.put((buffer_name, len(packed_results)))

    except Exception:
        import traceback
        LOG.error(f"FATAL ERROR in worker for task {task_args}:\n{traceback.format_exc()}")
        result_q.put(("ERROR", None))

# =============================================================================
# MAIN DRIVER
# =============================================================================
if __name__ == '__main__':
    try: mp.set_start_method('spawn')
    except RuntimeError: pass

    class TqdmLoggingHandler(logging.Handler):
        def emit(self, record):
            try: msg = self.format(record); tqdm.write(msg, file=sys.stderr); self.flush()
            except Exception: self.handleError(record)
    def setup_logging(debug=False):
        level = logging.DEBUG if debug else logging.INFO; log = logging.getLogger(); log.setLevel(level)
        if log.hasHandlers(): log.handlers.clear()
        handler = TqdmLoggingHandler(); formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'); handler.setFormatter(formatter); log.addHandler(handler)
    def signal_handler(sig, frame):
        LOG.warning("Interruption detected. Forcing cleanup...")
        os._exit(1)
    def define_numpy_dtypes():
        gs_dtype = np.dtype([('tick', np.int32), ('round_state', np.uint8), ('team_alive', np.uint8), ('enemy_alive', np.uint8), ('enemy_pos', np.float32, (5, 3))])
        pi_dtype = np.dtype([('pos', np.float32, (3,)), ('mouse', np.float32, (2,)), ('health', np.uint8), ('armor', np.uint8), ('money', np.int32), ('keyboard_bitmask', np.uint32), ('eco_bitmask', np.uint64, (2,)), ('inventory_bitmask', np.uint64, (2,)), ('active_weapon_bitmask', np.uint64, (2,))])
        assert len(KEYBOARD_TO_BIT) <= 32 and len(ECO_TO_BIT) <= 128 and len(ITEM_TO_INDEX) <= 128
        return gs_dtype, pi_dtype

    parser = argparse.ArgumentParser(description="Compile CS2 recordings into an LMDB using multiprocessing.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--recdir", required=True, type=Path); parser.add_argument("--dbfile", required=True, type=Path); parser.add_argument("--outlmdb", required=True, type=Path)
    parser.add_argument("--overwrite", action='store_true'); parser.add_argument("--overridesql", action='store_true'); parser.add_argument("--debug", action="store_true")
    parser.add_argument("--workers", type=int, default=os.cpu_count()); parser.add_argument("--buffers", type=int, default=8); args = parser.parse_args()

    setup_logging(args.debug); LMDB_PATH_FOR_CLEANUP = args.outlmdb; signal.signal(signal.SIGINT, signal_handler)
    if args.outlmdb.exists():
        if args.overwrite: shutil.rmtree(args.outlmdb); LOG.warning(f"Removed existing LMDB: {args.outlmdb}")
        else: LOG.critical(f"Output path exists. Use --overwrite."); sys.exit(1)

    LOG.info("-> Phase 1: Validating database and loading metadata...")
    conn = sqlite3.connect(f"file:{args.dbfile}?mode=ro", uri=True); conn.row_factory = sqlite3.Row
    rounds_info = {r['round']: dict(r) for r in conn.execute("SELECT * FROM rounds").fetchall()}; LOG.info(f"   - Loaded info for {len(rounds_info)} rounds.")
    recordings_map = {}
    db_recordings = conn.execute("SELECT * FROM RECORDING ORDER BY roundnumber, team, playername").fetchall()
    for rec in db_recordings:
        rec = dict(rec); round_num, team = rec['roundnumber'], rec['team']
        if round_num not in rounds_info or rounds_info[round_num].get('starttick') is None: continue
        if not rec['is_recorded'] and not args.overridesql: LOG.critical(f"Rec for {rec['playername']} in round {round_num} not recorded."); sys.exit(1)
        fname = f"{rec['roundnumber']:02d}_{rec['team']}_{rec['playername']}_{rec['starttick']}_{rec['stoptick']}"
        mp4_path, wav_path = args.recdir / f"{fname}.mp4", args.recdir / f"{fname}.wav"
        if not mp4_path.exists() or not wav_path.exists(): LOG.critical(f"Media not found for {fname}"); sys.exit(1)
        rec['mp4_path'], rec['wav_path'] = mp4_path, wav_path
        key = (round_num, team)
        if key not in recordings_map: recordings_map[key] = []
        recordings_map[key].append(rec)
    LOG.info(f"   - Validated {len(db_recordings)} recording entries.")
    
    LOG.info("   - Caching player data into shared memory...")
    player_data_cache = { f"{r['playername']}:{r['tick']}": dict(r) for r in conn.execute("SELECT * FROM player").fetchall() }; conn.close()
    
    # --- CHANGE: Serialize initial cache with pickle ---
    packed_cache = pickle.dumps(player_data_cache)
    
    shm_cache = SharedMemory(create=True, size=len(packed_cache)); shm_cache.buf[:len(packed_cache)] = packed_cache
    LOG.info(f"   - Player data cache ({len(packed_cache)/(1024**2):.2f} MB) created in shared memory.")

    gs_dtype, pi_dtype = define_numpy_dtypes(); env = lmdb.open(str(args.outlmdb), map_size=INITIAL_MAP_SIZE, writemap=True)
    demoname = args.recdir.name; tasks = [(r, t, demoname) for r, t in sorted(recordings_map.keys())]
    
    LOG.info(f"-> Phase 2: Processing rounds with {args.workers} workers and {args.buffers} buffers...")
    
    manager = mp.Manager()
    free_buffers_q = manager.Queue()
    results_q = manager.Queue()
    
    num_buffers = args.buffers
    buffer_pool = [SharedMemory(create=True, size=RESULT_BUFFER_SIZE) for _ in range(num_buffers)]
    for buf in buffer_pool: free_buffers_q.put(buf.name)
    
    initargs = (shm_cache.name, len(packed_cache), free_buffers_q, results_q, gs_dtype, pi_dtype, rounds_info, recordings_map)
    
    try:
        with mp.Pool(processes=args.workers, initializer=init_worker, initargs=initargs) as pool:
            pool.map_async(process_round_perspective, tasks)
            
            for i in tqdm(range(len(tasks)), desc="Processing Round/Team Perspectives"):
                result = results_q.get()
                if result[0] == "ERROR":
                    raise RuntimeError("A worker process encountered a fatal error. Check logs above for details.")
                
                result_shm_name, result_size = result
                
                try:
                    result_shm = SharedMemory(name=result_shm_name)
                    packed_results = bytes(result_shm.buf[:result_size])
                    
                    # --- CHANGE: Deserialize results using pickle ---
                    round_results = pickle.loads(packed_results)
                    
                    result_shm.close()
                    free_buffers_q.put(result_shm_name)

                    # Note: batch_size is now the size of the *pickled* data, not msgpack.
                    batch_size = sum(len(payload) for key, payload in round_results)
                    info, stats = env.info(), env.stat()
                    available_space = info['map_size'] - (info['last_pgno'] * stats['psize'])
                    while available_space < batch_size:
                        LOG.warning(f"Resizing LMDB: Need {batch_size/(1024**2):.2f}MB, have {available_space/(1024**2):.2f}MB.")
                        new_size = info['map_size'] + MAP_RESIZE_INCREMENT
                        env.set_mapsize(new_size)
                        info = env.info(); available_space = info['map_size'] - (info['last_pgno'] * stats['psize'])
                    
                    with env.begin(write=True) as txn:
                        for key, payload in round_results:
                            txn.put(key, payload)
                except Exception as e:
                    LOG.error(f"Error handling result from buffer {result_shm_name}: {e}")
                    free_buffers_q.put(result_shm_name)
                    raise
    
    finally:
        LOG.info("-> Phase 3: Finalizing and cleaning up...")
        metadata = {"demoname": demoname, "rounds": [[r, rounds_info[r]['starttick'], rounds_info[r]['endtick']] for r in sorted(rounds_info.keys())]}
        with env.begin(write=True) as txn:
            key = f"{demoname}_INFO".encode('utf-8'); txn.put(key, json.dumps(metadata).encode('utf-8'))
        env.close()

        shm_cache.close(); shm_cache.unlink(); LOG.info("Shared memory cache unlinked.")
        
        # Cleanup buffer pool
        active_buffers = []
        # Drain the queue to find all buffer names
        while not free_buffers_q.empty():
            try: active_buffers.append(SharedMemory(name=free_buffers_q.get_nowait()))
            except (FileNotFoundError, InterruptedError): pass
        # Add original pool objects in case some were not in the queue
        for buf in buffer_pool:
            if buf.name not in [b.name for b in active_buffers]:
                active_buffers.append(buf)
        for buf in active_buffers:
            try: buf.close(); buf.unlink()
            except FileNotFoundError: pass

        LOG.info("Result buffer pool unlinked.")
        manager.shutdown()

        signal.signal(signal.SIGINT, signal.SIG_DFL)
        LOG.info("All processing steps completed successfully.")
        LOG.info(f"Final LMDB database is at: {args.outlmdb}")