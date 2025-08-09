#!/usr/bin/env python3
"""
injection_mold.py - Compile CS2 recordings and database into a unified LMDB
for model training. This version computes and stores Mel spectrograms instead
of raw audio.
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
import pickle
import time
import random

import cv2
import lmdb
import msgpack
import msgpack_numpy as m
import numpy as np
from tqdm import tqdm

try:
    import librosa
except ImportError:
    print("Error: The 'librosa' library is required for spectrogram generation.", file=sys.stderr)
    print("Please install it using: pip install librosa", file=sys.stderr)
    sys.exit(1)


# --- Configuration ---
GAME_TICKS_PER_SEC = 64; EXPECTED_VIDEO_FPS = 32
TICKS_PER_FRAME = GAME_TICKS_PER_SEC // EXPECTED_VIDEO_FPS
AUDIO_SAMPLE_RATE = 44100; AUDIO_CHANNELS = 2; AUDIO_BIT_DEPTH = 2
AUDIO_BYTES_PER_FRAME = (AUDIO_SAMPLE_RATE // EXPECTED_VIDEO_FPS) * AUDIO_CHANNELS * AUDIO_BIT_DEPTH
INITIAL_MAP_SIZE = 2 * 1024**3; MAP_RESIZE_INCREMENT = 1 * 1024**3

# --- Globals ---
LOG = logging.getLogger("InjectionMold"); LMDB_PATH_FOR_CLEANUP = None; m.patch()

# =============================================================================
# DATA ENCODING MAPPINGS (Canonical Source)
# =============================================================================
KEYBOARD_ONLY_ACTIONS = ["IN_ATTACK", "IN_JUMP", "IN_DUCK", "IN_FORWARD", "IN_BACK", "IN_USE", "IN_CANCEL", "IN_TURNLEFT", "IN_TURNRIGHT", "IN_MOVELEFT", "IN_MOVERIGHT", "IN_ATTACK2", "IN_RELOAD", "IN_ALT1", "IN_ALT2", "IN_SPEED", "IN_WALK", "IN_ZOOM", "IN_WEAPON1", "IN_WEAPON2", "IN_BULLRUSH", "IN_GRENADE1", "IN_GRENADE2", "IN_ATTACK3", "IN_SCORE", "IN_INSPECT", "SWITCH_1", "SWITCH_2", "SWITCH_3", "SWITCH_4", "SWITCH_5"]
ITEM_NAMES = sorted(list(set(["AK-47", "M4A4", "M4A1-S", "Galil AR", "FAMAS", "AUG", "SG 553", "AWP", "SSG 08", "G3SG1", "SCAR-20", "Glock-18", "USP-S", "P250", "P2000", "Dual Berettas", "Five-SeveN", "Tec-9", "CZ75-Auto", "R8 Revolver", "Desert Eagle", "MP9", "MAC-10", "MP7", "MP5-SD", "UMP-45", "P90", "PP-Bizon", "Nova", "XM1014", "MAG-7", "Sawed-Off", "M249", "Negev", "Knife", "knife_t", "knife_ct", "Bayonet", "Flip Knife", "Gut Knife", "Karambit", "M9 Bayonet", "Huntsman Knife", "Falchion Knife", "Bowie Knife", "Butterfly Knife", "Shadow Daggers", "Ursus Knife", "Navaja Knife", "Stiletto Knife", "Talon Knife", "Classic Knife", "Paracord Knife", "Survival Knife", "Nomad Knife", "Skeleton Knife", "High Explosive Grenade", "Flashbang", "Smoke Grenade", "Molotov", "Incendiary Grenade", "Decoy Grenade", "C4 Explosive", "Defuse Kit", "Zeus x27", "Kevlar Vest", "Kevlar and Helmet", "Helmet"])))
ECO_ACTIONS = ["IN_BUYZONE"]
safe_item_names = [name.replace(" ", "_").replace("-", "_") for name in ITEM_NAMES]
for name in safe_item_names: ECO_ACTIONS.append(f"BUY_{name}"); ECO_ACTIONS.append(f"SELL_{name}"); ECO_ACTIONS.append(f"DROP_{name}")
item_id_map_names = ["deagle", "elite", "fiveseven", "glock", "ak47", "aug", "awp", "famas", "g3sg1", "galilar", "m249", "m4a1", "mac10", "p90", "mp5sd", "ump45", "xm1014", "bizon", "mag7", "negev", "sawedoff", "tec9", "p2000", "mp7", "mp9", "nova", "p250", "scar20", "sg556", "ssg08", "knife", "flashbang", "hegrenade", "smokegrenade", "molotov", "decoy", "incgrenade", "c4", "knife_t", "m4a1_silencer", "usp_silencer", "cz75a", "revolver", "defuser", "vest", "vesthelm"]
for name in item_id_map_names: ECO_ACTIONS.append(f"BUY_{name}"); ECO_ACTIONS.append(f"SELL_{name}"); ECO_ACTIONS.append(f"DROP_{name}")
ECO_ACTIONS = sorted(list(set(ECO_ACTIONS)))
KEYBOARD_TO_BIT = {action: i for i, action in enumerate(KEYBOARD_ONLY_ACTIONS)}
ECO_TO_BIT = {action: i for i, action in enumerate(ECO_ACTIONS)}
ITEM_TO_INDEX = {item: i for i, item in enumerate(ITEM_NAMES)}

# =============================================================================
# WORKER-SPECIFIC HELPERS AND PROCESSING
# =============================================================================
worker_data = {}

def merge_tick_data(tick1, tick2):
    if not tick1: return tick2 or None
    if not tick2: return tick1
    m = tick1.copy()
    m['mouse_x']=(tick1.get('mouse_x')or 0)+(tick2.get('mouse_x')or 0); m['mouse_y']=(tick1.get('mouse_y')or 0)+(tick2.get('mouse_y')or 0)
    m['position_x']=((tick1.get('position_x')or 0)+(tick2.get('position_x')or 0))/2.0; m['position_y']=((tick1.get('position_y')or 0)+(tick2.get('position_y')or 0))/2.0; m['position_z']=((tick1.get('position_z')or 0)+(tick2.get('position_z')or 0))/2.0
    kb1=set(filter(None,(tick1.get('keyboard_input')or'').split(','))); kb2=set(filter(None,(tick2.get('keyboard_input')or'').split(','))); m['keyboard_input']=",".join(sorted(list(kb1.union(kb2))))
    b1=set(filter(None,(tick1.get('buy_sell_input')or'').split(','))); b2=set(filter(None,(tick2.get('buy_sell_input')or'').split(','))); m['buy_sell_input']=",".join(sorted(list(b1.union(b2))))
    return m

def get_bitmask(actions, mapping, name):
    mask=0
    if not actions: return mask
    for a in actions:
        if a not in mapping: raise ValueError(f"Unknown action '{a}' in '{name}'")
        mask|=(1<<mapping[a])
    return mask

def get_bitmask_array(actions, mapping, name):
    mask=np.zeros(6,dtype=np.uint64)
    if not actions: return mask
    for a in actions:
        if a not in mapping: raise ValueError(f"Unknown action '{a}' in '{name}'")
        bp,idx,pi=mapping[a],mapping[a]//64,mapping[a]%64
        if idx<6:mask[idx]|=(np.uint64(1)<<np.uint64(pi))
    return mask

def get_inventory_bitmasks(inv_json, weapon, mapping):
    im,wm=np.zeros(2,dtype=np.uint64),np.zeros(2,dtype=np.uint64)
    def set_bit(m, item):
        if not isinstance(item,str): return
        bp=mapping.get(item)
        if bp is None and("knife"in item.lower()or"bayonet"in item.lower()):bp=mapping.get("Knife")
        if bp is None: raise ValueError(f"Unknown item '{item}'")
        idx,pi=bp//64,bp%64
        if idx<2:m[idx]|=(np.uint64(1)<<np.uint64(pi))
    if weapon:set_bit(wm,weapon)
    try:
        for item in json.loads(inv_json)if inv_json else[]:set_bit(im,item)
    except(json.JSONDecodeError,TypeError):pass
    return im,wm

def init_worker(shm_cache_name, shm_cache_size, lmdb_path, lock, gs_dtype, pi_dtype, rounds_info, recordings_map, jpeg_quality, block_regions, audio_params):
    m.patch()
    shm = SharedMemory(name=shm_cache_name)
    packed_cache_copy = bytes(shm.buf[:shm_cache_size])
    shm.close()
    worker_data['player_data_cache'] = pickle.loads(packed_cache_copy)
    worker_data['lmdb_path'] = lmdb_path
    worker_data['lock'] = lock
    worker_data['gs_dtype'], worker_data['pi_dtype'] = gs_dtype, pi_dtype
    worker_data['rounds_info'], worker_data['recordings_map'] = rounds_info, recordings_map
    worker_data['jpeg_quality'] = jpeg_quality
    worker_data['block_regions'] = block_regions
    worker_data['audio_params'] = audio_params


def process_round_perspective(task_args):
    caps, auds = {}, {}
    try:
        round_num, team, demoname = task_args
        gs_dtype, pi_dtype = worker_data['gs_dtype'], worker_data['pi_dtype']
        rounds_info, recordings_map = worker_data['rounds_info'], worker_data['recordings_map']
        player_data_cache, lock, lmdb_path = worker_data['player_data_cache'], worker_data['lock'], worker_data['lmdb_path']
        audio_params = worker_data['audio_params']
        
        round_data = recordings_map[(round_num, team)]
        if len(round_data) != 5: return task_args
        t_roster=[p[0]for p in json.loads(rounds_info[round_num]['t_team'])];ct_roster=[p[0]for p in json.loads(rounds_info[round_num]['ct_team'])]
        current_roster=t_roster if team=='T'else ct_roster;enemy_roster=ct_roster if team=='T'else t_roster
        caps={rec['playername']:cv2.VideoCapture(str(rec['mp4_path']))for rec in round_data}
        auds={rec['playername']:open(rec['wav_path'],'rb')for rec in round_data}
        for aud in auds.values():aud.seek(44)
        start_tick,end_tick=rounds_info[round_num]['starttick'],max(rec['stoptick']for rec in round_data)
        results=[]
        for tick in range(start_tick,end_tick+1,TICKS_PER_FRAME):
            team_deaths={p[0]:p[1]for p in json.loads(rounds_info[round_num][f"{team.lower()}_team"])};enemy_deaths={p[0]:p[1]for p in json.loads(rounds_info[round_num][f"{'ct'if team=='T'else't'}_team"])}
            team_alive=sum(1<<i for i,n in enumerate(current_roster)if team_deaths.get(n,0)==-1 or tick<team_deaths.get(n,0))
            enemy_alive=sum(1<<i for i,n in enumerate(enemy_roster)if enemy_deaths.get(n,0)==-1 or tick<enemy_deaths.get(n,0))
            gs=np.zeros(1,dtype=gs_dtype);gs[0]['tick']=tick;gs[0]['team_alive'],gs[0]['enemy_alive']=team_alive,enemy_alive
            rs_mask=0
            if rounds_info[round_num].get('freezetime_endtick')is not None and tick<rounds_info[round_num]['freezetime_endtick']:rs_mask|=(1<<0)
            if rounds_info[round_num].get('starttick')is not None and rounds_info[round_num].get('endtick')is not None and tick>=rounds_info[round_num]['starttick']and tick<=rounds_info[round_num]['endtick']:rs_mask|=(1<<1)
            if rounds_info[round_num].get('bomb_planted_tick',-1)!=-1 and tick>=rounds_info[round_num]['bomb_planted_tick']:rs_mask|=(1<<2)
            win_tick = rounds_info[round_num].get('win_tick')
            if win_tick is not None and tick >= win_tick:
                win_team = rounds_info[round_num].get('win_team')
                if win_team == 't': rs_mask |= (1 << 3)
                elif win_team == 'ct': rs_mask |= (1 << 4)
            gs[0]['round_state']=rs_mask
            for i,name in enumerate(enemy_roster):
                if(enemy_alive>>i)&1:
                    lk=f"{name}:{tick}";t1=player_data_cache.get(lk);t2=player_data_cache.get(f"{name}:{tick+1}");ed=merge_tick_data(t1,t2)
                    if ed:gs[0]['enemy_pos'][i]=[ed['position_x'],ed['position_y'],ed['position_z']]
            pdl=[]
            for rec in round_data:
                pn=rec['playername']
                try:p_idx=current_roster.index(pn)
                except ValueError:continue
                is_alive=(team_alive>>p_idx)&1
                if not is_alive or tick>=rec['stoptick']:continue
                ret,frame=caps[pn].read()
                
                # --- START AUDIO PROCESSING CHANGE ---
                mel_spectrogram = None
                if not audio_params['no_audio']:
                    audio_bytes = auds[pn].read(AUDIO_BYTES_PER_FRAME)
                    if ret and len(audio_bytes) == AUDIO_BYTES_PER_FRAME:
                        # Convert 16-bit stereo PCM bytes to a mono float waveform
                        waveform_int = np.frombuffer(audio_bytes, dtype=np.int16)
                        waveform_float = waveform_int.astype(np.float32) / 32768.0
                        waveform_mono = waveform_float.reshape((-1, AUDIO_CHANNELS)).mean(axis=1)

                        # Compute Mel Spectrogram
                        S = librosa.feature.melspectrogram(
                            y=waveform_mono,
                            sr=AUDIO_SAMPLE_RATE,
                            n_mels=audio_params['n_mels'],
                            n_fft=audio_params['n_fft'],
                            hop_length=audio_params['hop_length']
                        )
                        # Convert to decibel scale
                        mel_spectrogram = librosa.power_to_db(S, ref=np.max)
                    else:
                        continue # Skip frame if video or audio is corrupted
                else: # if ret is True but audio is disabled
                    if not ret: continue
                # --- END AUDIO PROCESSING CHANGE ---

                if worker_data['block_regions']:
                    for (minX, minY, maxX, maxY) in worker_data['block_regions']:
                        cv2.rectangle(frame, (minX, minY), (maxX, maxY), (0, 0, 0), -1)

                jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, worker_data['jpeg_quality']]
                jpg=cv2.imencode('.jpg', frame, jpeg_params)[1].tobytes()

                lk=f"{pn}:{tick}";td=merge_tick_data(player_data_cache.get(lk),player_data_cache.get(f"{pn}:{tick+1}"));pi=np.zeros(1,dtype=pi_dtype)
                if td:
                    kb,bs=td.get('keyboard_input','')or'',td.get('buy_sell_input','')or''
                    akb=[a.replace("-","_").replace(" ","_") for a in kb.split(',')if a];bs_a=[a.replace("-","_").replace(" ","_")for a in bs.split(',')if a]
                    k_a=[a for a in akb if not a.startswith('DROP_')];pi[0]['keyboard_bitmask']=get_bitmask(k_a,KEYBOARD_TO_BIT,"KEYBOARD_ONLY_ACTIONS")
                    d_a=[a for a in akb if a.startswith('DROP_')];e_a=d_a+bs_a
                    if td.get('is_in_buyzone')==1:e_a.append('IN_BUYZONE')
                    pi[0]['eco_bitmask']=get_bitmask_array(e_a,ECO_TO_BIT,"ECO_ACTIONS")
                    pi[0]['pos']=[td.get('position_x',0),td.get('position_y',0),td.get('position_z',0)];pi[0]['mouse']=[td.get('mouse_x',0),td.get('mouse_y',0)]
                    pi[0]['health'],pi[0]['armor'],pi[0]['money']=td.get('health',0),td.get('armor',0),td.get('money',0)
                    im,wm=get_inventory_bitmasks(td.get('inventory'),td.get('active_weapon'),ITEM_TO_INDEX);pi[0]['inventory_bitmask'],pi[0]['active_weapon_bitmask']=im,wm
                pdl.append((pi,jpg,mel_spectrogram))
            if not pdl:continue
            key=f"{demoname}_round_{round_num:03d}_team_{team}_tick_{tick:08d}".encode('utf-8')
            payload=msgpack.packb({"game_state":gs,"player_data":pdl},use_bin_type=True);results.append((key,payload))
        if not results: return task_args
        
        with lock:
            batch_size = sum(len(payload) for _, payload in results)
            env = lmdb.open(str(lmdb_path), writemap=True)
            txn = env.begin(write=True)
            try:
                info = env.info(); stats = env.stat()
                used_bytes = info['last_pgno'] * stats['psize']
                available = info['map_size'] - used_bytes
                if available < batch_size:
                    txn.abort()
                    grow_by = max(batch_size, MAP_RESIZE_INCREMENT)
                    psize = stats['psize']
                    new_size = info['map_size'] + (((int(grow_by * 1.5) + psize - 1) // psize) * psize)
                    env.set_mapsize(new_size)
                    txn = env.begin(write=True)
                cursor = txn.cursor()
                cursor.putmulti(results)
                txn.commit()
            except:
                txn.abort()
                raise
            finally:
                env.close()

        return task_args
    except Exception as e:
        import traceback
        LOG.error(f"FATAL ERROR in worker for task {task_args}:\n{traceback.format_exc()}")
        return ("ERROR", f"Error in {task_args}: {e}")
    finally:
        for cap in caps.values(): cap.release()
        for aud in auds.values(): aud.close()

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
        level=logging.DEBUG if debug else logging.INFO; log=logging.getLogger(); log.setLevel(level)
        if log.hasHandlers():log.handlers.clear()
        handler=TqdmLoggingHandler();formatter=logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s');handler.setFormatter(formatter);log.addHandler(handler)
    def signal_handler(sig, frame): LOG.warning("Interruption detected..."); os._exit(1)
    def define_numpy_dtypes():
        gs_dtype=np.dtype([('tick',np.int32),('round_state',np.uint8),('team_alive',np.uint8),('enemy_alive',np.uint8),('enemy_pos',np.float32,(5,3))])
        pi_dtype=np.dtype([('pos',np.float32,(3,)),('mouse',np.float32,(2,)),('health',np.uint8),('armor',np.uint8),('money',np.int32),('keyboard_bitmask',np.uint32),('eco_bitmask',np.uint64,(6,)),('inventory_bitmask',np.uint64,(2,)),('active_weapon_bitmask',np.uint64,(2,))])
        assert len(KEYBOARD_TO_BIT)<=32 and len(ECO_TO_BIT)<=384 and len(ITEM_TO_INDEX)<=128
        return gs_dtype,pi_dtype
    parser = argparse.ArgumentParser(description="Compile CS2 recordings into a model-ready LMDB.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--recdir", required=True, type=Path); parser.add_argument("--dbfile", required=True, type=Path); parser.add_argument("--outlmdb", required=True, type=Path)
    parser.add_argument("--overwrite", action='store_true'); parser.add_argument("--overridesql", action='store_true'); parser.add_argument("--debug", action="store_true")
    parser.add_argument("--workers", type=int, default=os.cpu_count());
    parser.add_argument("--quality", type=int, default=80, help="JPEG compression quality (0-100).")
    parser.add_argument("--blockfile", type=Path, help="Path to a file with regions to black out (minX,minY,maxX,maxY per line).")

    # --- Audio Spectrogram Arguments with Solid Defaults ---
    audio_group = parser.add_argument_group('Audio Spectrogram Parameters')
    audio_group.add_argument('--no-audio', action='store_true', help="Disable audio processing entirely and store None for audio.")
    audio_group.add_argument(
        '--n-mels',
        type=int,
        default=128,
        help="Number of Mel bands to generate. This is the 'height' of the spectrogram. Default: 128."
    )
    audio_group.add_argument(
        '--n-fft',
        type=int,
        default=2048,
        help="Length of the FFT window, which determines frequency resolution. Default: 2048."
    )
    audio_group.add_argument(
        '--hop-length',
        type=int,
        default=512,
        help="Number of samples between successive frames (hop size). Determines time resolution. Default: 512."
    )

    args = parser.parse_args()
    setup_logging(args.debug); LMDB_PATH_FOR_CLEANUP = args.outlmdb; signal.signal(signal.SIGINT, signal_handler)
    if args.outlmdb.exists():
        if args.overwrite: shutil.rmtree(args.outlmdb); LOG.warning(f"Removed existing LMDB: {args.outlmdb}")
        else: LOG.critical(f"Output path exists. Use --overwrite."); sys.exit(1)

    # ... (rest of the main driver code is identical) ...
    
    block_regions = []
    if args.blockfile:
        if not args.blockfile.exists(): LOG.critical(f"Blockfile not found: {args.blockfile}"); sys.exit(1)
        with open(args.blockfile, 'r') as f:
            for line in f:
                try:
                    parts = [int(p.strip()) for p in line.split(',')];
                    if len(parts) == 4: block_regions.append(tuple(parts))
                    else: LOG.warning(f"Skipping malformed blockfile line: {line.strip()}")
                except ValueError: LOG.warning(f"Skipping non-integer blockfile line: {line.strip()}")
        LOG.info(f"   - Loaded {len(block_regions)} blockout regions.")

    LOG.info("-> Phase 1: Validating database and loading metadata...")
    conn = sqlite3.connect(f"file:{args.dbfile}?mode=ro", uri=True); conn.row_factory = sqlite3.Row
    rounds_info = {r['round']: dict(r) for r in conn.execute("SELECT * FROM rounds").fetchall()}; LOG.info(f"   - Loaded info for {len(rounds_info)} rounds.")
    recordings_map = {}
    db_recordings = conn.execute("SELECT * FROM RECORDING ORDER BY roundnumber, team, playername").fetchall()
    for rec in db_recordings:
        rec=dict(rec); round_num,team=rec['roundnumber'],rec['team']
        if round_num not in rounds_info or rounds_info[round_num].get('starttick') is None: continue
        if not rec['is_recorded'] and not args.overridesql: LOG.critical(f"Rec for {rec['playername']} round {round_num} not recorded."); sys.exit(1)
        fname=f"{rec['roundnumber']:02d}_{rec['team']}_{rec['playername']}_{rec['starttick']}_{rec['stoptick']}"
        mp4,wav=args.recdir/f"{fname}.mp4",args.recdir/f"{fname}.wav"
        if not mp4.exists()or not wav.exists():LOG.critical(f"Media not found: {fname}");sys.exit(1)
        rec['mp4_path'],rec['wav_path']=mp4,wav;key=(round_num,team)
        if key not in recordings_map:recordings_map[key]=[]
        recordings_map[key].append(rec)
    LOG.info(f"   - Validated {len(db_recordings)} recording entries.")
    
    LOG.info("   - Caching player data into shared memory...")
    player_data_cache={f"{r['playername']}:{r['tick']}":dict(r)for r in conn.execute("SELECT * FROM player").fetchall()};conn.close()
    packed_cache=pickle.dumps(player_data_cache)
    shm_cache=SharedMemory(create=True,size=len(packed_cache));shm_cache.buf[:len(packed_cache)]=packed_cache
    LOG.info(f"   - Player data cache ({len(packed_cache)/(1024**2):.2f} MB) created in SHM.")
    gs_dtype,pi_dtype=define_numpy_dtypes()
    
    env=lmdb.open(str(args.outlmdb),map_size=INITIAL_MAP_SIZE,writemap=True);env.close()
    
    demoname=args.recdir.name;tasks=[(r,t,demoname)for r,t in sorted(recordings_map.keys())]
    
    LOG.info(f"-> Phase 2: Processing rounds with {args.workers} worker processes...")
    
    lock = mp.Lock()
    audio_params = {'no_audio': args.no_audio, 'n_mels': args.n_mels, 'n_fft': args.n_fft, 'hop_length': args.hop_length}
    initargs = (shm_cache.name, len(packed_cache), args.outlmdb, lock, gs_dtype, pi_dtype, rounds_info, recordings_map, args.quality, block_regions, audio_params)

    try:
        with mp.Pool(processes=args.workers, initializer=init_worker, initargs=initargs) as pool:
            pbar = tqdm(pool.imap_unordered(process_round_perspective, tasks), total=len(tasks), desc="Processing Perspectives")
            for result in pbar:
                if isinstance(result, tuple) and result[0] == "ERROR":
                    raise RuntimeError(f"Worker process fatal error: {result[1]}")
    finally:
        LOG.info("-> Phase 3: Finalizing and cleaning up...")
        env = lmdb.open(str(args.outlmdb), writemap=True)
        metadata = {"demoname": demoname, "rounds": [[r, rounds_info[r]['starttick'], rounds_info[r]['endtick']] for r in sorted(rounds_info.keys())]}
        try:
            try:
                with env.begin(write=True) as txn:
                    key = f"{demoname}_INFO".encode('utf-8')
                    txn.put(key, json.dumps(metadata).encode('utf-8'))
            except lmdb.MapFullError:
                info = env.info(); stats = env.stat()
                psize = stats['psize']
                grow_by = max(64 * 1024**2, MAP_RESIZE_INCREMENT)
                new_size = info['map_size'] + ((grow_by + psize - 1)//psize)*psize
                env.set_mapsize(new_size)
                with env.begin(write=True) as txn:
                    key = f"{demoname}_INFO".encode('utf-8')
                    txn.put(key, json.dumps(metadata).encode('utf-8'))
        finally:
            env.close()
        shm_cache.close(); shm_cache.unlink(); LOG.info("Shared memory cache unlinked.")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        LOG.info("All processing steps completed successfully.")
        LOG.info(f"Final LMDB database is at: {args.outlmdb}")