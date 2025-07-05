#!/usr/bin/env python3
"""
aggregate.py - A unified script to process CS2 demo files.

This script merges the functionality of five separate processing scripts:
1. mouse.py: Extracts sensitivity-independent mouse movement data.
2. rounds.py: Extracts per-round metadata and player death information.
3. keyboard_location.py: Extracts detailed per-tick player state (inputs, position, etc.).
4. buy_sell_drop.py: Infers buy, sell, and drop actions.
5. merge.py: Combines all generated data into a final, unified database.

The script runs the full pipeline, taking a demo file as input and producing
a series of intermediate databases and a final `merged.db` in a specified
output directory. The `--optimize` feature from keyboard_location.py is
enabled by default.

Usage:
    python aggregate.py --demo /path/to/match.dem --out /path/to/output_directory
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
from demoparser2 import DemoParser
from tqdm import tqdm

# =============================================================================
# 1. IMPORTS FROM AWPY (CONDITIONAL)
# =============================================================================
try:  # AWPy ≥ 2.0
    from awpy.demo import Demo as _Demo

    _AWPY_VERSION = "2.x"
except ModuleNotFoundError:
    # Set to a value that will cause the rounds processing to fail gracefully
    _AWPY_VERSION = "1.x"

if _AWPY_VERSION.startswith("2"):
    import awpy.constants as _C
    import awpy.parsers.events as _ev
    import awpy.parsers.rounds as _rnd
    import polars as pl


# =============================================================================
# 2. GLOBAL CONSTANTS FROM ALL SCRIPTS
# =============================================================================

# --- From keyboard_location.py ---
WEAPON_CATEGORIES = {
    "AK-47": "SWITCH_1", "M4A4": "SWITCH_1", "M4A1-S": "SWITCH_1", "Galil AR": "SWITCH_1", "FAMAS": "SWITCH_1", "AUG": "SWITCH_1", "SG 553": "SWITCH_1", "AWP": "SWITCH_1", "SSG 08": "SWITCH_1", "G3SG1": "SWITCH_1", "SCAR-20": "SWITCH_1", "MP9": "SWITCH_1", "MAC-10": "SWITCH_1", "MP7": "SWITCH_1", "MP5-SD": "SWITCH_1", "UMP-45": "SWITCH_1", "P90": "SWITCH_1", "PP-Bizon": "SWITCH_1", "Nova": "SWITCH_1", "XM1014": "SWITCH_1", "MAG-7": "SWITCH_1", "Sawed-Off": "SWITCH_1", "M249": "SWITCH_1", "Negev": "SWITCH_1",
    "Glock-18": "SWITCH_2", "USP-S": "SWITCH_2", "P250": "SWITCH_2", "P2000": "SWITCH_2", "Dual Berettas": "SWITCH_2", "Five-SeveN": "SWITCH_2", "Tec-9": "SWITCH_2", "CZ75-Auto": "SWITCH_2", "Desert Eagle": "SWITCH_2", "R8 Revolver": "SWITCH_2",
    "knife":"SWITCH_3","knife_ct": "SWITCH_3", "knife_t": "SWITCH_3", "Bayonet": "SWITCH_3", "Flip Knife": "SWITCH_3", "Gut Knife": "SWITCH_3", "Karambit": "SWITCH_3", "M9 Bayonet": "SWITCH_3", "Huntsman Knife": "SWITCH_3", "Falchion Knife": "SWITCH_3", "Bowie Knife": "SWITCH_3", "Butterfly Knife": "SWITCH_3", "Shadow Daggers": "SWITCH_3", "Ursus Knife": "SWITCH_3", "Navaja Knife": "SWITCH_3", "Stiletto Knife": "SWITCH_3", "Talon Knife": "SWITCH_3", "Classic Knife": "SWITCH_3", "Paracord Knife": "SWITCH_3", "Survival Knife": "SWITCH_3", "Nomad Knife": "SWITCH_3", "Skeleton Knife": "SWITCH_3",
    "High Explosive Grenade": "SWITCH_4", "Flashbang": "SWITCH_4", "Smoke Grenade": "SWITCH_4", "Molotov": "SWITCH_4", "Incendiary Grenade": "SWITCH_4", "Decoy Grenade": "SWITCH_4",
    "C4 Explosive": "SWITCH_5", "Defuse Kit": "SWITCH_5", "Zeus x27": "SWITCH_3",
}
KEY_MAPPING = {
    "IN_ATTACK": 1<<0, "IN_JUMP": 1<<1, "IN_DUCK": 1<<2, "IN_FORWARD": 1<<3, "IN_BACK": 1<<4, "IN_USE": 1<<5, "IN_CANCEL": 1<<6, "IN_TURNLEFT": 1<<7, "IN_TURNRIGHT": 1<<8, "IN_MOVELEFT": 1<<9, "IN_MOVERIGHT": 1<<10, "IN_ATTACK2": 1<<11, "IN_RELOAD": 1<<13, "IN_ALT1": 1<<14, "IN_ALT2": 1<<15, "IN_SPEED": 1<<16, "IN_WALK": 1<<17, "IN_ZOOM": 1<<18, "IN_WEAPON1": 1<<19, "IN_WEAPON2": 1<<20, "IN_BULLRUSH": 1<<21, "IN_GRENADE1": 1<<22, "IN_GRENADE2": 1<<23, "IN_ATTACK3": 1<<24, "IN_SCORE": 1<<33, "IN_INSPECT": 1<<35,
}

# --- From buy_sell_drop.py ---
ITEM_ID_MAP = {
    1: "deagle", 2: "elite", 3: "fiveseven", 4: "glock", 7: "ak47", 8: "aug", 9: "awp",
    10: "famas", 11: "g3sg1", 13: "galilar", 14: "m249", 16: "m4a1", 17: "mac10", 19: "p90",
    23: "mp5sd", 24: "ump45", 25: "xm1014", 26: "bizon", 27: "mag7", 28: "negev", 29: "sawedoff",
    30: "tec9", 32: "p2000", 33: "mp7", 34: "mp9", 35: "nova", 36: "p250", 38: "scar20",
    39: "sg556", 40: "ssg08", 42: "knife", 43: "flashbang", 44: "hegrenade", 45: "smokegrenade",
    46: "molotov", 47: "decoy", 48: "incgrenade", 49: "c4", 59: "knife_t", 60: "m4a1_silencer",
    61: "usp_silencer", 63: "cz75a", 64: "revolver", 500: "knife_default_ct", 506: "knife_gut",
    507: "knife_flip", 508: "knife_bayonet", 509: "knife_m9_bayonet", 515: "knife_karambit",
    522: "knife_stiletto", 523: "knife_ursus", 80: "defuser", 81: "vest", 82: "vesthelm"
}
GRENADE_NAMES = {"flashbang", "hegrenade", "smokegrenade", "molotov", "decoy", "incgrenade"}


# =============================================================================
# 3. PROCESSING STEP 1: MOUSE DATA (from mouse.py)
# =============================================================================
def _mouse_setup_database(db_path: str, table_name: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    create_table_query = f"""
    CREATE TABLE {table_name} (
        tick INTEGER,
        player_name TEXT,
        x REAL,
        y REAL,
        PRIMARY KEY (tick, player_name)
    );
    """
    cursor.execute(create_table_query)
    return conn

def _mouse_normalize_angle_delta(delta: pd.Series) -> pd.Series:
    delta_copy = delta.copy()
    delta_copy[delta_copy > 180] -= 360
    delta_copy[delta_copy < -180] += 360
    return delta_copy

def _mouse_process_demo(parser: DemoParser, conn: sqlite3.Connection, table_name: str):
    print("    Parsing tick data... This may take a while.")
    props_to_parse = ["player_name", "pitch", "yaw", "aim_punch_angle"]
    
    try:
        ticks_df = parser.parse_ticks(props_to_parse)
        print("    Tick data parsed successfully.")
    except Exception as e:
        print(f"    An error occurred during parsing: {e}")
        return

    if 'aim_punch_angle' in ticks_df.columns:
        print("    Expanding 'aim_punch_angle' column...")
        aim_punch_components = ticks_df['aim_punch_angle'].apply(pd.Series)
        ticks_df['aim_punch_angle_pitch'] = aim_punch_components[0]
        ticks_df['aim_punch_angle_yaw'] = aim_punch_components[1]
        ticks_df.drop(columns=['aim_punch_angle'], inplace=True)
    else:
        print("    Error: 'aim_punch_angle' column not found. Cannot proceed.")
        return

    print("    Calculating mouse deltas with wrap-around correction...")
    ticks_df = ticks_df.sort_values(by=['player_name', 'tick']).reset_index(drop=True)
    grouped = ticks_df.groupby('player_name')
    
    with tqdm(total=4, desc="    Calculating Deltas", leave=False) as pbar:
        delta_pitch = grouped['pitch'].transform('diff')
        ticks_df['delta_pitch'] = _mouse_normalize_angle_delta(delta_pitch)
        pbar.update(1)
        
        delta_yaw = grouped['yaw'].transform('diff')
        ticks_df['delta_yaw'] = _mouse_normalize_angle_delta(delta_yaw)
        pbar.update(1)
        
        delta_aim_punch_pitch = grouped['aim_punch_angle_pitch'].transform('diff')
        ticks_df['delta_aim_punch_pitch'] = _mouse_normalize_angle_delta(delta_aim_punch_pitch)
        pbar.update(1)
        
        delta_aim_punch_yaw = grouped['aim_punch_angle_yaw'].transform('diff')
        ticks_df['delta_aim_punch_yaw'] = _mouse_normalize_angle_delta(delta_aim_punch_yaw)
        pbar.update(1)

    ticks_df['y'] = - (ticks_df['delta_pitch'] - ticks_df['delta_aim_punch_pitch']) 
    ticks_df['x'] = - (ticks_df['delta_yaw'] - ticks_df['delta_aim_punch_yaw'])
    
    result_df = ticks_df[['tick', 'player_name', 'x', 'y']].copy()
    result_df.dropna(inplace=True)

    print(f"    Writing {len(result_df)} records to the database...")
    try:
        result_df.to_sql(table_name, conn, if_exists='append', index=False)
    except Exception as e:
        print(f"    An error occurred while writing to the database: {e}")

def run_mouse_processing(demo_path: str, out_dir: str):
    print("\n[1/5] Processing mouse data...")
    db_path = os.path.join(out_dir, 'mouse.db')
    table_name = "MOUSE"

    try:
        parser = DemoParser(demo_path)
        db_connection = _mouse_setup_database(db_path, table_name)
    except Exception as e:
        print(f"    Failed to initialize parser or database. Error: {e}")
        return

    _mouse_process_demo(parser, db_connection, table_name)
    db_connection.close()
    print("    ✓ Mouse data processing finished.")

# =============================================================================
# 4. PROCESSING STEP 2: ROUNDS DATA (from rounds.py)
# =============================================================================
def _rounds_parse_2x(demo_path: Path) -> list[dict[str, Any]]:
    """Implementation for AWPy ≥ 2.0."""
    demo = _Demo(path=demo_path)
    demo.parse()

    rounds_df: pl.DataFrame = demo.rounds
    spawns = demo.events["player_spawn"]
    deaths_raw = demo.events["player_death"]
    deaths = _ev.parse_kills(deaths_raw)

    rows: list[dict[str, Any]] = []

    for rd in rounds_df.iter_rows(named=True):
        rn = rd["round_num"]
        s_tick, fz_end, e_tick = rd["start"], rd["freeze_end"], rd["official_end"]

        def team_list(side: str) -> list[list[Any]]:
            roster = (
                spawns.filter(
                    (pl.col("tick") >= s_tick)
                    & (pl.col("tick") <= fz_end)
                    & (pl.col("user_side") == side)
                )
                .select("user_name")
                .unique()
                .to_series()
                .to_list()
            )
            died = (
                deaths.filter(
                    (pl.col("tick") >= s_tick)
                    & (pl.col("tick") <= e_tick)
                )
                .select("victim_name", "tick")
            )
            death_map = dict(zip(died["victim_name"], died["tick"]))
            return [[p, death_map.get(p, -1)] for p in roster]

        rows.append(
            {
                "round": rn,
                "starttick": s_tick,
                "freezetime_endtick": fz_end,
                "endtick": e_tick,
                "t_team": team_list(_C.T_SIDE),
                "ct_team": team_list(_C.CT_SIDE),
            }
        )
    return rows

def _rounds_to_sql(rows: list[dict[str, Any]], db_file: Path) -> None:
    with sqlite3.connect(db_file) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS ROUNDS (
                   round               INTEGER PRIMARY KEY,
                   starttick           INTEGER,
                   freezetime_endtick  INTEGER,
                   endtick             INTEGER,
                   t_team              TEXT,
                   ct_team             TEXT
               )"""
        )
        conn.executemany(
            """INSERT OR REPLACE INTO ROUNDS
               VALUES (:round, :starttick, :freezetime_endtick,
                       :endtick, :t_team, :ct_team)""",
            (
                {
                    **r,
                    "t_team": json.dumps(r["t_team"]),
                    "ct_team": json.dumps(r["ct_team"]),
                }
                for r in rows
            ),
        )

def run_rounds_processing(demo_path: str, out_dir: str):
    print("\n[2/5] Processing rounds data...")
    if not _AWPY_VERSION.startswith("2"):
        print("    Error: AWPy version 2.x is required for round processing. Skipping.")
        # Create an empty DB to prevent downstream errors
        db_path = Path(os.path.join(out_dir, "rounds.db"))
        _rounds_to_sql([], db_path)
        return

    demo_path_obj = Path(demo_path)
    rows = _rounds_parse_2x(demo_path_obj)
    db_path = Path(os.path.join(out_dir, "rounds.db"))
    _rounds_to_sql(rows, db_path)
    print(f"    ✓ Wrote {len(rows)} rounds to {db_path.name}.")

# =============================================================================
# 5. PROCESSING STEP 3: KEYBOARD & LOCATION (from keyboard_location.py)
# =============================================================================
def _kl_get_weapon_switch_type(weapon_name: str | None) -> str | None:
    if not weapon_name: return None
    return WEAPON_CATEGORIES.get(weapon_name, f"SWITCH_UNDEFINED_{weapon_name}")

def _kl_extract_buttons(bits) -> list[str]:
    bits_int = int(bits)
    return [name for name, mask in KEY_MAPPING.items() if bits_int & mask]

def _kl_sanitize_inventory(inv):
    if isinstance(inv, (list, tuple, set)):
        try: return json.dumps(list(inv))
        except TypeError: return ",".join(map(str, inv))
    return str(inv)

def _kl_export_sqlite_inputs(db_path: Path, tick_df: pd.DataFrame) -> None:
    df = tick_df.copy()
    df["inventory"] = df["inventory"].apply(_kl_sanitize_inventory)
    df.rename(columns={"name": "playername", "balance": "money", "armor_value": "armor"}, inplace=True)
    df = df[["tick", "steamid", "playername", "keyboard_input", "inventory", "X", "Y", "Z", "active_weapon_name", "health", "armor", "money"]]
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS inputs (
            tick INTEGER NOT NULL, steamid INTEGER NOT NULL, playername TEXT,
            keyboard_input TEXT, inventory TEXT,
            x REAL, y REAL, z REAL, active_weapon TEXT,
            health INTEGER, armor INTEGER, money INTEGER,
            PRIMARY KEY (tick, steamid))""")
    cur.executemany("INSERT OR REPLACE INTO inputs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", df.to_records(index=False).tolist())
    con.commit()
    con.close()

def run_keyboard_location_processing(demo_path: str, out_dir: str):
    print("\n[3/5] Processing keyboard and location data...")
    sql_output_path = Path(os.path.join(out_dir, 'keyboard_location.db'))

    print("    [loading demo – one-off parse, please wait …]")
    dp = DemoParser(demo_path)

    tick_props = ["tick", "steamid", "name", "buttons", "inventory", "X", "Y", "Z", "active_weapon_name", "is_defusing", "health", "armor_value", "balance", "is_warmup_period", "total_rounds_played"]
    events_to_parse = ["player_death", "hegrenade_detonate", "flashbang_detonate", "smokegrenade_detonate", "molotov_detonate"]

    tick_df = pd.DataFrame(dp.parse_ticks(wanted_props=tick_props))
    events = dp.parse_events(events_to_parse)
    event_dfs = dict(events)
    
    if 'is_warmup_period' in tick_df.columns:
        print("    [filtering out warmup period...]")
        tick_df = tick_df[tick_df['is_warmup_period'] == False].copy()

    # --- This logic is now hardcoded as per the requirements ---
    print("    [optimizing dataset...]")
    initial_rows = len(tick_df)

    alive_mask = tick_df['health'] > 0
    tick_df = tick_df[alive_mask].copy()
    print(f"      - Removed {initial_rows - len(tick_df)} ticks where player health was 0.")
    
    initial_rows_after_health = len(tick_df)
    zero_pos_mask = (tick_df['X'] == 0) & (tick_df['Y'] == 0) & (tick_df['Z'] == 0)
    tick_df = tick_df[~zero_pos_mask].copy()
    print(f"      - Removed {initial_rows_after_health - len(tick_df)} ticks with invalid (0,0,0) positions.")
    print(f"      - Optimization complete. Total rows reduced from {initial_rows} to {len(tick_df)}.")

    print("    [gap-filling player data...]")
    if not tick_df.empty:
        tick_df.sort_values(["steamid", "tick"], inplace=True)
        player_steamids = tick_df['steamid'].unique()
        min_tick, max_tick = tick_df['tick'].min(), tick_df['tick'].max()
        all_ticks_range = range(min_tick, max_tick + 1)
        multi_index = pd.MultiIndex.from_product([player_steamids, all_ticks_range], names=['steamid', 'tick'])
        tick_df = tick_df.set_index(['steamid', 'tick']).reindex(multi_index)
        tick_df = tick_df.groupby(level='steamid').ffill()
        tick_df.dropna(subset=['name'], inplace=True)
        tick_df = tick_df.reset_index()
    else:
        print("    Warning: DataFrame is empty after filtering. No data to process.")
        return

    for col in ['health', 'armor_value', 'balance', 'total_rounds_played', 'buttons']:
        tick_df[col] = pd.to_numeric(tick_df[col], errors='coerce').fillna(0).astype(int)

    tick_df['prev_inventory'] = tick_df.groupby('steamid')['inventory'].shift(1)
    tick_df['inferred_drop'] = ''
    
    print("    [inferring weapon switches...]")
    tick_df['prev_weapon'] = tick_df.groupby('steamid')['active_weapon_name'].shift(1).fillna('')
    weapon_changed_mask = (tick_df['active_weapon_name'] != tick_df['prev_weapon']) & (tick_df['active_weapon_name'] != '')
    tick_df['switch_type'] = tick_df['active_weapon_name'].apply(_kl_get_weapon_switch_type)
    tick_df['inferred_switch'] = ''
    tick_df.loc[weapon_changed_mask, 'inferred_switch'] = tick_df.loc[weapon_changed_mask, 'switch_type']
    
    tick_df['real_keys_list'] = tick_df['buttons'].apply(_kl_extract_buttons)
    def combine_inputs(row):
        keys = row['real_keys_list']
        if row['inferred_switch'] and row['inferred_switch'] not in keys: keys.append(row['inferred_switch'])
        if row['inferred_drop']: keys.append(row['inferred_drop'])
        return ",".join(keys)
    tick_df['keyboard_input'] = tick_df.apply(combine_inputs, axis=1)

    tick_df.drop(columns=['prev_inventory', 'inferred_drop', 'prev_weapon', 'switch_type', 'inferred_switch', 'real_keys_list', 'buttons'], inplace=True, errors='ignore')

    print(f"    [exporting to {sql_output_path.name}...]")
    _kl_export_sqlite_inputs(sql_output_path, tick_df)
    print(f"    ✓ Wrote table 'inputs' to {sql_output_path.name}.")

# =============================================================================
# 6. PROCESSING STEP 4: BUY/SELL/DROP DATA (from buy_sell_drop.py)
# =============================================================================
def _bsd_get_item_name(item_id: int) -> str:
    return ITEM_ID_MAP.get(item_id, f"unknown_item_{item_id}")

class _bsd_DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.commit()
            self.conn.close()
    def init_db(self):
        cursor = self.conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS RAREACTIONS;")
        cursor.execute("DROP TABLE IF EXISTS BUYZONE;")
        cursor.execute("""
            CREATE TABLE RAREACTIONS (tick INTEGER, steamid TEXT, playername TEXT, action TEXT, item TEXT);
        """)
        cursor.execute("""
            CREATE TABLE BUYZONE (tick INTEGER, steamid TEXT, playername TEXT);
        """)
        cursor.execute("CREATE INDEX idx_reactions_tick_steamid ON RAREACTIONS (tick, steamid);")
        cursor.execute("CREATE INDEX idx_buyzone_tick_steamid ON BUYZONE (tick, steamid);")
    def batch_insert_actions(self, actions: List[Dict]):
        if not actions: return
        self.conn.executemany("INSERT INTO RAREACTIONS VALUES (:tick, :steamid, :playername, :action, :item)", actions)
    def insert_buyzone(self, tick, steamid, playername):
        self.conn.execute("INSERT INTO BUYZONE VALUES (?, ?, ?)", (int(tick), str(steamid), str(playername)))

class _bsd_InputVerifier:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.grenade_name_map = {
            "hegrenade": "High Explosive Grenade", "flashbang": "Flashbang", "smokegrenade": "Smoke Grenade",
            "molotov": "Molotov", "incgrenade": "Incendiary Grenade", "decoy": "Decoy Grenade"
        }
    def __enter__(self):
        self.conn = sqlite3.connect(f'file:{self.db_path}?mode=ro', uri=True)
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn: self.conn.close()

    def is_grenade_throw(self, steamid: str, drop_tick: int, item_name: str) -> bool:
        if item_name not in self.grenade_name_map: return False
        active_weapon_name = self.grenade_name_map[item_name]
        query_tick_start = drop_tick - 128
        cursor = self.conn.cursor()
        cursor.execute("SELECT tick, keyboard_input, active_weapon FROM inputs WHERE steamid = ? AND tick BETWEEN ? AND ? ORDER BY tick DESC", (int(steamid), query_tick_start, drop_tick))
        rows = cursor.fetchall()
        if not rows: return False
        held_grenade_before_drop = False
        if rows and rows[0][0] < drop_tick and rows[0][2] == active_weapon_name: held_grenade_before_drop = True
        if not held_grenade_before_drop and len(rows) > 1 and rows[1][0] < drop_tick and rows[1][2] == active_weapon_name: held_grenade_before_drop = True
        if not held_grenade_before_drop: return False
        for _, keyboard_input, active_weapon in rows:
            if active_weapon == active_weapon_name and keyboard_input and "IN_ATTACK" in keyboard_input: return True
        return False

def run_buy_sell_drop_processing(demo_path: str, out_dir: str):
    print("\n[4/5] Processing buy, sell, and drop data...")
    sql_output_path = os.path.join(out_dir, 'buy_sell_drop.db')
    sql_input_path = os.path.join(out_dir, 'keyboard_location.db')
    
    print(f"    Parsing demo with demoparser2: {os.path.basename(demo_path)}")
    parser = DemoParser(demo_path)

    print("    Step 1/3: Parsing high-certainty events...")
    event_names = ["item_purchase", "player_death"]
    events = parser.parse_events(event_names)
    events_by_tick: Dict[int, List[Tuple[str, Any]]] = {}
    for name, df in events:
        if not df.empty:
            for row in df.itertuples(index=False):
                if row.tick not in events_by_tick: events_by_tick[row.tick] = []
                events_by_tick[row.tick].append((name, row))

    print("    Step 2/3: Parsing tick-by-tick player states...")
    tick_props = ["tick", "player_steamid", "player_name", "balance", "inventory_as_ids", "in_buy_zone", "team_num", "ct_cant_buy", "terrorist_cant_buy"]
    all_ticks_df = parser.parse_ticks(tick_props)
    all_ticks_df.sort_values(by=["tick", "player_steamid"], inplace=True)
    all_ticks_df["inventory_as_ids"] = all_ticks_df["inventory_as_ids"].apply(lambda x: set(x) if x is not None else set())

    print(f"    Step 3/3: Analyzing {len(all_ticks_df)} tick states...")
    last_player_states: Dict[str, Dict[str, Any]] = {}
    potential_actions: List[Dict] = []
    
    with _bsd_DatabaseManager(sql_output_path) as db:
        db.init_db()
        for state in tqdm(all_ticks_df.itertuples(index=False), total=len(all_ticks_df), desc="    Analyzing Actions", leave=False):
            steamid = str(state.player_steamid)
            if steamid == '0': continue
            in_buyzone = state.in_buy_zone and not (state.team_num == 2 and state.terrorist_cant_buy) and not (state.team_num == 3 and state.ct_cant_buy)
            if in_buyzone: db.insert_buyzone(state.tick, steamid, state.player_name)
            last_state = last_player_states.get(steamid)
            if last_state:
                tick_events = events_by_tick.get(state.tick, [])
                if any(name == "player_death" and str(data.user_steamid) == steamid for name, data in tick_events):
                    last_player_states[steamid] = state._asdict()
                    continue
                for name, data in tick_events:
                    if name == "item_purchase" and str(data.steamid) == steamid:
                        potential_actions.append({"tick": state.tick, "steamid": steamid, "playername": state.player_name, "action": "BUY", "item": data.item_name})
                last_inv = last_state.get("inventory_as_ids", set())
                current_inv = state.inventory_as_ids
                if len(current_inv) < len(last_inv):
                    lost_items = last_inv - current_inv
                    if not lost_items: continue
                    lost_item_id = list(lost_items)[0]
                    lost_item_name = _bsd_get_item_name(lost_item_id)
                    if any(name == 'item_purchase' and str(data.steamid) == steamid for name, data in tick_events):
                        last_player_states[steamid] = state._asdict()
                        continue
                    if in_buyzone and state.balance > last_state["balance"]:
                        potential_actions.append({"tick": state.tick, "steamid": steamid, "playername": state.player_name, "action": "SELL", "item": lost_item_name})
                        last_player_states[steamid] = state._asdict()
                        continue
                    potential_actions.append({"tick": state.tick, "steamid": steamid, "playername": state.player_name, "action": "DROP", "item": lost_item_name})
            last_player_states[steamid] = state._asdict()
        
        final_actions = []
        print(f"    Post-processing {len(potential_actions)} potential actions...")
        with _bsd_InputVerifier(sql_input_path) as input_verifier:
            for action in tqdm(potential_actions, desc="    Filtering Grenades", leave=False):
                if action["action"] == "DROP" and action["item"] in GRENADE_NAMES:
                    if not input_verifier.is_grenade_throw(steamid=action["steamid"], drop_tick=action["tick"], item_name=action["item"]):
                        final_actions.append(action)
                else:
                    final_actions.append(action)

        print(f"    Finalizing database with {len(final_actions)} confirmed actions...")
        db.batch_insert_actions(final_actions)
    print(f"    ✓ Analysis complete. Results saved to '{os.path.basename(sql_output_path)}'")


# =============================================================================
# 7. PROCESSING STEP 5: MERGE DATABASES (from merge.py)
# =============================================================================
def _merge_create_merged_schema(cursor):
    cursor.execute("DROP TABLE IF EXISTS player")
    cursor.execute("DROP TABLE IF EXISTS rounds")
    cursor.execute("""
    CREATE TABLE player (
        tick INTEGER, steamid INTEGER, playername TEXT,
        position_x REAL, position_y REAL, position_z REAL,
        inventory TEXT, active_weapon TEXT, health INTEGER, armor INTEGER, money INTEGER,
        keyboard_input TEXT, mouse_x REAL, mouse_y REAL,
        is_in_buyzone INTEGER, buy_sell_input TEXT,
        PRIMARY KEY (tick, steamid)
    )
    """)
    cursor.execute("""
    CREATE TABLE rounds (
        round INTEGER PRIMARY KEY, starttick INTEGER, freezetime_endtick INTEGER,
        endtick INTEGER, t_team TEXT, ct_team TEXT
    )
    """)

def _merge_load_lookup_data(db_path):
    print("    Loading lookup data into memory...")
    mouse_positions, buy_sell_drop_actions, buyzone_presence, valid_round_ticks = {}, {}, set(), []
    try:
        with sqlite3.connect(os.path.join(db_path, 'mouse.db')) as conn:
            for tick, player_name, x, y in conn.cursor().execute("SELECT tick, player_name, x, y FROM MOUSE"):
                mouse_positions[(tick, player_name)] = (x, y)
    except sqlite3.OperationalError: print("    Warning: mouse.db or MOUSE table not found.")
    try:
        with sqlite3.connect(os.path.join(db_path, 'buy_sell_drop.db')) as conn:
            cursor = conn.cursor()
            for tick, playername, action, item in cursor.execute("SELECT tick, playername, action, item FROM RAREACTIONS"):
                key = (tick, playername)
                if key not in buy_sell_drop_actions: buy_sell_drop_actions[key] = []
                buy_sell_drop_actions[key].append((action, item))
            for tick, _, playername in cursor.execute("SELECT tick, steamid, playername FROM BUYZONE"):
                buyzone_presence.add((tick, playername))
    except sqlite3.OperationalError: print("    Warning: buy_sell_drop.db or its tables not found.")
    try:
        with sqlite3.connect(os.path.join(db_path, 'rounds.db')) as conn:
            for starttick, endtick in conn.cursor().execute("SELECT starttick, endtick FROM ROUNDS WHERE starttick IS NOT NULL AND endtick IS NOT NULL"):
                valid_round_ticks.append((starttick, endtick))
    except sqlite3.OperationalError:
        print("    Error: Could not load rounds.db. Cannot filter ticks by round. Aborting.")
        sys.exit(1)
    return mouse_positions, buy_sell_drop_actions, buyzone_presence, valid_round_ticks

def _merge_is_tick_in_valid_round(tick, round_intervals):
    for start, end in round_intervals:
        if start <= tick <= end: return True
    return False

def run_merge_processing(out_dir):
    print("\n[5/5] Merging all databases...")
    mouse_data, action_data, buyzone_data, round_intervals = _merge_load_lookup_data(out_dir)

    try:
        keyboard_conn = sqlite3.connect(os.path.join(out_dir, 'keyboard_location.db'))
    except sqlite3.OperationalError:
        print(f"    Error: Could not open keyboard_location.db in '{out_dir}'. Aborting.")
        sys.exit(1)
    keyboard_cursor = keyboard_conn.cursor()
    merged_conn = sqlite3.connect(os.path.join(out_dir, 'merged.db'))
    merged_cursor = merged_conn.cursor()
    _merge_create_merged_schema(merged_cursor)

    print("    Processing player data from keyboard_location.db...")
    player_rows_to_insert = []
    keyboard_cursor.execute("SELECT tick, steamid, playername, keyboard_input, inventory, x, y, z, active_weapon, health, armor, money FROM inputs")
    all_inputs = keyboard_cursor.fetchall()
    
    for row in tqdm(all_inputs, desc="    Merging Player Data", leave=False):
        tick, steamid, playername, kb_input, inventory, x, y, z, active_w, health, armor, money = row
        if health is None or health <= 0 or not _merge_is_tick_in_valid_round(tick, round_intervals): continue
        
        mouse_x, mouse_y = mouse_data.get((tick, playername), (None, None))
        is_in_buyzone = 1 if (tick, playername) in buyzone_data else 0
        actions = action_data.get((tick, playername), [])
        
        final_kb_inputs = kb_input.split(',') if kb_input else []
        for action, item in actions:
            if action == 'DROP': final_kb_inputs.append(f"DROP_{item.replace(' ', '_').replace('&', 'and')}")
        
        buy_sell_actions = []
        for action, item in actions:
            if action in ('BUY', 'SELL'): buy_sell_actions.append(f"{action}_{item.replace(' ', '_').replace('&', 'and')}")
        
        player_rows_to_insert.append((tick, steamid, playername, x, y, z, inventory, active_w, health, armor, money, ",".join(final_kb_inputs), mouse_x, mouse_y, is_in_buyzone, ",".join(buy_sell_actions)))

    print(f"    Inserting {len(player_rows_to_insert)} rows into 'player' table...")
    if player_rows_to_insert:
        merged_cursor.executemany("INSERT INTO player (tick, steamid, playername, position_x, position_y, position_z, inventory, active_weapon, health, armor, money, keyboard_input, mouse_x, mouse_y, is_in_buyzone, buy_sell_input) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", player_rows_to_insert)

    print("    Copying 'rounds' table...")
    try:
        with sqlite3.connect(os.path.join(out_dir, 'rounds.db')) as rounds_conn:
            all_rounds = rounds_conn.cursor().execute("SELECT round, starttick, freezetime_endtick, endtick, t_team, ct_team FROM ROUNDS").fetchall()
            merged_cursor.executemany("INSERT INTO rounds VALUES (?, ?, ?, ?, ?, ?)", all_rounds)
    except sqlite3.OperationalError: print("    Warning: Could not copy rounds.db data.")

    print("    Finalizing and closing databases...")
    merged_conn.commit()
    merged_conn.close()
    keyboard_conn.close()
    print(f"    ✓ Successfully created merged.db in '{out_dir}'")

# =============================================================================
# 8. MAIN DRIVER
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run full demo processing pipeline.")
    parser.add_argument('--demo', required=True, help='Path to demo file (.dem)')
    parser.add_argument('--out', default=None, help='Output directory (default: data_<demobasename>)')
    args = parser.parse_args()

    demo_path = os.path.abspath(args.demo)
    if not os.path.isfile(demo_path):
        print(f"Error: Demo file not found: {demo_path}", file=sys.stderr)
        sys.exit(1)

    demo_name = os.path.splitext(os.path.basename(demo_path))[0]
    out_dir = os.path.abspath(args.out or f"data_{demo_name}")

    os.makedirs(out_dir, exist_ok=True)
    print(f"Using output directory: {out_dir}")
    for db_file in glob.glob(os.path.join(out_dir, '*.db')):
        print(f"Removing existing DB: {db_file}")
        os.remove(db_file)
    
    try:
        run_mouse_processing(demo_path, out_dir)
        run_rounds_processing(demo_path, out_dir)
        run_keyboard_location_processing(demo_path, out_dir)
        run_buy_sell_drop_processing(demo_path, out_dir)
        run_merge_processing(out_dir)
    except Exception as e:
        print(f"\nAn unhandled error occurred during processing: {e}", file=sys.stderr)
        print("Pipeline aborted.", file=sys.stderr)
        sys.exit(1)

    print("\n-------------------------------------------")
    print("All processing steps completed successfully.")
    print(f"Final merged database is at: {os.path.join(out_dir, 'merged.db')}")
    print("-------------------------------------------")

if __name__ == '__main__':
    main()