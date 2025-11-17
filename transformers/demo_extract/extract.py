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
extract.py - A unified and optimized script to process CS2 demo files.

This script merges the functionality of multiple separate processing scripts into a
single, efficient pipeline that operates primarily in-memory. It avoids creating
intermediate disk files, writing only the final, unified database.

The pipeline includes the following steps:
1. Extracts sensitivity-independent mouse movement data.
2. Extracts per-round metadata and player death information.
3. Extracts detailed per-tick player state (inputs, position, etc.).
4. Infers buy, sell, and drop actions.
5. Combines all generated data into a final, unified database on disk.
6. Generates a list of recording candidates based on strict round validation.

The --optimize feature from the original keyboard_location.py is enabled by default.

Usage:
    python extract.py --demo /path/to/match.dem --out /path/to/final.db
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    import polars as pl


# =============================================================================
# 2. GLOBAL CONSTANTS FROM ALL SCRIPTS
# =============================================================================

# --- From keyboard_location.py ---
WEAPON_CATEGORIES = {
    "Desert Eagle": "SWITCH_2", "Dual Berettas": "SWITCH_2", "Five-SeveN": "SWITCH_2", "Glock-18": "SWITCH_2", "AK-47": "SWITCH_1",
    "AUG": "SWITCH_1", "AWP": "SWITCH_1", "FAMAS": "SWITCH_1", "G3SG1": "SWITCH_1", "Galil AR": "SWITCH_1", "M249": "SWITCH_1", "M4A4": "SWITCH_1", "MAC-10": "SWITCH_1",
    "P90": "SWITCH_1", "MP5-SD": "SWITCH_1", "UMP-45": "SWITCH_1", "XM1014": "SWITCH_1", "PP-Bizon": "SWITCH_1", "MAG-7": "SWITCH_1", "Negev": "SWITCH_1",
    "Sawed-Off": "SWITCH_1", "Tec-9": "SWITCH_2", "Zeus x27": "SWITCH_3", "P2000": "SWITCH_2", "MP7": "SWITCH_1", "MP9": "SWITCH_1", "Nova": "SWITCH_1",
    "P250": "SWITCH_2", "SCAR-20": "SWITCH_1", "SG 553": "SWITCH_1", "SSG 08": "SWITCH_1", "Knife": "SWITCH_3", "knife": "SWITCH_3", "Flashbang": "SWITCH_4",
    "High Explosive Grenade": "SWITCH_4", "Smoke Grenade": "SWITCH_4", "Molotov": "SWITCH_4", "Decoy Grenade": "SWITCH_4",
    "Incendiary Grenade": "SWITCH_4", "C4 Explosive": "SWITCH_5", "Kevlar Vest": "SWITCH_1", "Kevlar & Helmet": "SWITCH_1",
    "Heavy Assault Suit": "SWITCH_1", "item_nvg": "SWITCH_1", "Defuse Kit": "SWITCH_1", "Rescue Kit": "SWITCH_1",
    "Medi-Shot": "SWITCH_3", "knife_t": "SWITCH_3", "M4A1-S": "SWITCH_1", "USP-S": "SWITCH_2", "Trade Up Contract": "SWITCH_1",
    "CZ75-Auto": "SWITCH_2", "R8 Revolver": "SWITCH_2", "Charm Detachments": "SWITCH_1", "Bayonet": "SWITCH_3", "Classic Knife": "SWITCH_3",
    "Flip Knife": "SWITCH_3", "Gut Knife": "SWITCH_3", "Karambit": "SWITCH_3", "M9 Bayonet": "SWITCH_3", "Huntsman Knife": "SWITCH_3",
    "Falchion Knife": "SWITCH_3", "Bowie Knife": "SWITCH_3", "Butterfly Knife": "SWITCH_3", "Shadow Daggers": "SWITCH_3",
    "Paracord Knife": "SWITCH_3", "Survival Knife": "SWITCH_3", "Ursus Knife": "SWITCH_3", "Navaja Knife": "SWITCH_3",
    "Nomad Knife": "SWITCH_3", "Stiletto Knife": "SWITCH_3", "Talon Knife": "SWITCH_3", "Skeleton Knife": "SWITCH_3", "Kukri Knife": "SWITCH_3"
}
KEY_MAPPING = {
    "IN_ATTACK": 1<<0, "IN_JUMP": 1<<1, "IN_DUCK": 1<<2, "IN_FORWARD": 1<<3, "IN_BACK": 1<<4, "IN_USE": 1<<5, "IN_CANCEL": 1<<6, "IN_TURNLEFT": 1<<7, "IN_TURNRIGHT": 1<<8, "IN_MOVELEFT": 1<<9, "IN_MOVERIGHT": 1<<10, "IN_ATTACK2": 1<<11, "IN_RELOAD": 1<<13, "IN_ALT1": 1<<14, "IN_ALT2": 1<<15, "IN_SPEED": 1<<16, "IN_WALK": 1<<17, "IN_ZOOM": 1<<18, "IN_WEAPON1": 1<<19, "IN_WEAPON2": 1<<20, "IN_BULLRUSH": 1<<21, "IN_GRENADE1": 1<<22, "IN_GRENADE2": 1<<23, "IN_ATTACK3": 1<<24, "IN_SCORE": 1<<33, "IN_INSPECT": 1<<35,
}

# --- From buy_sell_drop.py ---
ITEM_ID_MAP = {
    1: "deagle", 2: "elite", 3: "fiveseven", 4: "glock", 7: "ak47", 8: "aug", 9: "awp", 10: "famas", 11: "g3sg1",
    13: "galilar", 14: "m249", 16: "m4a1", 17: "mac10", 19: "p90", 23: "mp5sd", 24: "ump45", 25: "xm1014",
    26: "ppbizon", 27: "mag7", 28: "negev", 29: "sawedoff", 30: "tec9", 31: "zeus", 32: "p2000", 33: "mp7",
    34: "mp9", 35: "nova", 36: "p250", 38: "scar20", 39: "sg556", 40: "ssg08", 41: "knife", 42: "knife",
    43: "flashbang", 44: "hegrenade", 45: "smokegrenade", 46: "molotov", 47: "decoy", 48: "incgrenade",
    49: "c4", 50: "vest", 51: "vesthelm", 52: "heavyassaultsuit", 54: "nvgs", 55: "defuser", 56: "rescue_kit",
    57: "medishot", 59: "knifet", 60: "m4a1_silencer", 61: "usp_silencer", 62: "tradeupcontract", 63: "cz75auto",
    64: "r8revolver", 65: "charmdetachments", 500: "bayonet", 503: "knife_default_ct", 505: "flipknife",
    506: "gutknife", 507: "karambit", 508: "knife_m9_bayonet", 509: "huntsmanknife", 512: "falchionknife",
    514: "bowieknife", 515: "butterflyknife", 516: "shadowdaggers", 517: "paracordknife", 518: "survivalknife",
    519: "ursusknife", 520: "navajaknife", 521: "nomadknife", 522: "stilettoknife", 523: "talonknife",
    525: "skeletonknife", 526: "kukriknife",
}

GRENADE_NAMES = {"flashbang", "hegrenade", "smokegrenade", "molotov", "decoy", "incgrenade"}


# =============================================================================
# 2.5 UTILITY FUNCTION FOR PLAYER VALIDATION
# =============================================================================
BAD_PLAYER_PREFIXES = ["Coach", "Spectator", "GOTV"]

def _is_valid_player(player_name: str | None) -> bool:
    """
    Checks if an entity is a valid player and not a coach, spectator, etc.
    """
    if not player_name:
        return False
    for prefix in BAD_PLAYER_PREFIXES:
        if player_name.startswith(prefix):
            return False
    return True


# =============================================================================
# 3. PROCESSING STEP 1: MOUSE DATA
# =============================================================================
def _mouse_setup_database(conn: sqlite3.Connection, table_name: str):
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    create_table_query = f"""
    CREATE TABLE {table_name} (
        tick INTEGER, player_name TEXT, x REAL, y REAL,
        PRIMARY KEY (tick, player_name)
    );
    """
    cursor.execute(create_table_query)

def _mouse_normalize_angle_delta(delta: pd.Series) -> pd.Series:
    delta_copy = delta.copy()
    delta_copy[delta_copy > 180] -= 360
    delta_copy[delta_copy < -180] += 360
    return delta_copy

def _mouse_process_demo(parser: DemoParser, conn: sqlite3.Connection, table_name: str):
    print("    Parsing tick data for mouse movement...")
    props_to_parse = ["player_name", "pitch", "yaw", "aim_punch_angle"]
    
    ticks_df = parser.parse_ticks(props_to_parse)
    ticks_df = ticks_df[ticks_df['player_name'].apply(_is_valid_player)].copy()

    if 'aim_punch_angle' in ticks_df.columns:
        aim_punch_components = ticks_df['aim_punch_angle'].apply(pd.Series)
        ticks_df['aim_punch_angle_pitch'] = aim_punch_components[0]
        ticks_df['aim_punch_angle_yaw'] = aim_punch_components[1]
        ticks_df.drop(columns=['aim_punch_angle'], inplace=True)
    else:
        print("    Warning: 'aim_punch_angle' column not found. Mouse data will be incomplete.")
        return

    print("    Calculating mouse deltas...")
    ticks_df = ticks_df.sort_values(by=['player_name', 'tick']).reset_index(drop=True)
    grouped = ticks_df.groupby('player_name')
    
    delta_pitch = grouped['pitch'].transform('diff')
    ticks_df['delta_pitch'] = _mouse_normalize_angle_delta(delta_pitch)
    delta_yaw = grouped['yaw'].transform('diff')
    ticks_df['delta_yaw'] = _mouse_normalize_angle_delta(delta_yaw)
    delta_aim_punch_pitch = grouped['aim_punch_angle_pitch'].transform('diff')
    ticks_df['delta_aim_punch_pitch'] = _mouse_normalize_angle_delta(delta_aim_punch_pitch)
    delta_aim_punch_yaw = grouped['aim_punch_angle_yaw'].transform('diff')
    ticks_df['delta_aim_punch_yaw'] = _mouse_normalize_angle_delta(delta_aim_punch_yaw)

    ticks_df['y'] = - (ticks_df['delta_pitch'] - ticks_df['delta_aim_punch_pitch']) 
    ticks_df['x'] = - (ticks_df['delta_yaw'] - ticks_df['delta_aim_punch_yaw'])
    
    result_df = ticks_df[['tick', 'player_name', 'x', 'y']].copy()
    result_df.dropna(inplace=True)
    result_df.to_sql(table_name, conn, if_exists='append', index=False)

def run_mouse_processing(demo_path: str, conn: sqlite3.Connection):
    print("\n[1/6] Processing mouse data...")
    table_name = "MOUSE"
    parser = DemoParser(demo_path)
    _mouse_setup_database(conn, table_name)
    _mouse_process_demo(parser, conn, table_name)
    print("    ✓ Mouse data processing finished.")

# =============================================================================
# 4. PROCESSING STEP 2: ROUNDS DATA
# =============================================================================
def _rounds_parse_2x(demo_path: Path) -> list[dict[str, Any]]:
    demo = _Demo(path=demo_path)
    demo.parse()
    rounds_df: pl.DataFrame = demo.rounds
    spawns = demo.events["player_spawn"]
    deaths_raw = demo.events["player_death"]
    bomb_plants = demo.events["bomb_planted"]
    deaths = _ev.parse_kills(deaths_raw)
    rows: list[dict[str, Any]] = []

    for rd in tqdm(rounds_df.iter_rows(named=True), total=len(rounds_df), desc="    Extracting round info", leave=False):
        rn, s_tick, fz_end, e_tick, win_tick, win_team = rd["round_num"], rd["start"], rd["freeze_end"], rd["official_end"], rd["end"], rd["winner"]
        # Find bomb plant tick for the current round
        plant_in_round = bomb_plants.filter((pl.col("tick") >= s_tick) & (pl.col("tick") <= e_tick))
        bomb_plant_tick = -1
        if not plant_in_round.is_empty():
            bomb_plant_tick = plant_in_round["tick"][0]
        def team_list(side: str) -> list[list[Any]]:
            roster_df = spawns.filter((pl.col("tick") >= s_tick) & (pl.col("tick") <= fz_end) & (pl.col("user_side") == side)).select("user_name").unique()
            filtered_roster_df = roster_df.filter(pl.col("user_name").map_elements(_is_valid_player, return_dtype=pl.Boolean))
            roster = filtered_roster_df.to_series().to_list()
            died = deaths.filter((pl.col("tick") >= s_tick) & (pl.col("tick") <= e_tick)).select("victim_name", "tick")
            death_map = dict(zip(died["victim_name"], died["tick"]))
            return [[p, death_map.get(p, -1)] for p in roster]
        rows.append({"round": rn, "starttick": s_tick, "freezetime_endtick": fz_end, "endtick": e_tick, "win_tick": win_tick, "win_team": win_team, "bomb_planted_tick": bomb_plant_tick, "t_team": team_list(_C.T_SIDE), "ct_team": team_list(_C.CT_SIDE)})
    return rows

def _rounds_to_sql(rows: list[dict[str, Any]], conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS ROUNDS (round INTEGER PRIMARY KEY, starttick INTEGER, freezetime_endtick INTEGER, endtick INTEGER, win_tick INTEGER, win_team TEXT, bomb_planted_tick INTEGER, t_team TEXT, ct_team TEXT)""")
    if not rows: return
    conn.executemany("""INSERT OR REPLACE INTO ROUNDS VALUES (:round, :starttick, :freezetime_endtick, :endtick, :win_tick, :win_team, :bomb_planted_tick, :t_team, :ct_team)""", ({**r, "t_team": json.dumps(r["t_team"]), "ct_team": json.dumps(r["ct_team"])} for r in rows))
    conn.commit()

def run_rounds_processing(demo_path: str, conn: sqlite3.Connection):
    print("\n[2/6] Processing rounds data...")
    if not _AWPY_VERSION.startswith("2"):
        print("    Warning: AWPy version 2.x is required for round processing. Skipping.")
        _rounds_to_sql([], conn) # Create empty table
        return
    rows = _rounds_parse_2x(Path(demo_path))
    _rounds_to_sql(rows, conn)
    print(f"    ✓ Processed {len(rows)} rounds into memory.")

# =============================================================================
# 5. PROCESSING STEP 3: KEYBOARD & LOCATION
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

def _kl_export_sqlite_inputs(conn: sqlite3.Connection, tick_df: pd.DataFrame) -> None:
    df = tick_df.copy()
    df["inventory"] = df["inventory"].apply(_kl_sanitize_inventory)
    df.rename(columns={"name": "playername", "balance": "money", "armor_value": "armor"}, inplace=True)
    df = df[["tick", "steamid", "playername", "keyboard_input", "inventory", "X", "Y", "Z", "active_weapon_name", "health", "armor", "money"]]
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS inputs (
            tick INTEGER NOT NULL, steamid INTEGER NOT NULL, playername TEXT, keyboard_input TEXT, inventory TEXT,
            x REAL, y REAL, z REAL, active_weapon TEXT, health INTEGER, armor INTEGER, money INTEGER,
            PRIMARY KEY (tick, steamid))""")
    cur.executemany("INSERT OR REPLACE INTO inputs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", df.to_records(index=False).tolist())
    conn.commit()

def run_keyboard_location_processing(demo_path: str, conn: sqlite3.Connection):
    print("\n[3/6] Processing keyboard and location data...")
    print("    Parsing detailed player states (this can be slow)...")
    dp = DemoParser(demo_path)

    tick_props = ["tick", "steamid", "name", "buttons", "inventory", "X", "Y", "Z", "active_weapon_name", "health", "armor_value", "balance", "is_warmup_period"]
    tick_df = pd.DataFrame(dp.parse_ticks(wanted_props=tick_props))
    tick_df = tick_df[tick_df['name'].apply(_is_valid_player)].copy()
    
    if 'is_warmup_period' in tick_df.columns:
        tick_df = tick_df[tick_df['is_warmup_period'] == False].copy()

    initial_rows = len(tick_df)
    tick_df = tick_df[tick_df['health'] > 0].copy()
    tick_df = tick_df[~((tick_df['X'] == 0) & (tick_df['Y'] == 0) & (tick_df['Z'] == 0))].copy()
    print(f"    Optimized dataset: {initial_rows} -> {len(tick_df)} rows.")

    if not tick_df.empty:
        print("    Gap-filling and processing player data...")
        tick_df.sort_values(["steamid", "tick"], inplace=True)
        player_steamids = tick_df['steamid'].unique()
        min_tick, max_tick = tick_df['tick'].min(), tick_df['tick'].max()
        all_ticks_range = range(min_tick, max_tick + 1)
        multi_index = pd.MultiIndex.from_product([player_steamids, all_ticks_range], names=['steamid', 'tick'])
        tick_df = tick_df.set_index(['steamid', 'tick']).reindex(multi_index).groupby(level='steamid').ffill().dropna(subset=['name']).reset_index()
    else:
        print("    Warning: DataFrame is empty after filtering. No data to process.")
        _kl_export_sqlite_inputs(conn, pd.DataFrame())
        return

    for col in ['health', 'armor_value', 'balance', 'buttons']:
        tick_df[col] = pd.to_numeric(tick_df[col], errors='coerce').fillna(0).astype(int)

    tick_df['prev_weapon'] = tick_df.groupby('steamid')['active_weapon_name'].shift(1).fillna('')
    weapon_changed_mask = (tick_df['active_weapon_name'] != tick_df['prev_weapon']) & (tick_df['active_weapon_name'] != '')
    tick_df['switch_type'] = tick_df['active_weapon_name'].apply(_kl_get_weapon_switch_type)
    tick_df['inferred_switch'] = ''
    tick_df.loc[weapon_changed_mask, 'inferred_switch'] = tick_df.loc[weapon_changed_mask, 'switch_type']
    
    tick_df['real_keys_list'] = tick_df['buttons'].apply(_kl_extract_buttons)
    def combine_inputs(row):
        keys = row['real_keys_list']
        if row['inferred_switch'] and row['inferred_switch'] not in keys: keys.append(row['inferred_switch'])
        return ",".join(keys)
    tick_df['keyboard_input'] = tick_df.apply(combine_inputs, axis=1)

    tick_df.drop(columns=['prev_weapon', 'switch_type', 'inferred_switch', 'real_keys_list', 'buttons'], inplace=True, errors='ignore')
    _kl_export_sqlite_inputs(conn, tick_df)
    print("    ✓ Processed keyboard and location data into memory.")

# =============================================================================
# 6. PROCESSING STEP 4: BUY/SELL/DROP DATA
# =============================================================================
def _bsd_get_item_name(item_id: int) -> str:
    return ITEM_ID_MAP.get(item_id, f"unknown_{item_id}")

class _bsd_DatabaseManager:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
    def init_db(self):
        cursor = self.conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS RAREACTIONS;")
        cursor.execute("DROP TABLE IF EXISTS BUYZONE;")
        cursor.execute("CREATE TABLE RAREACTIONS (tick INTEGER, steamid TEXT, playername TEXT, action TEXT, item TEXT);")
        cursor.execute("CREATE TABLE BUYZONE (tick INTEGER, steamid TEXT, playername TEXT);")
    def batch_insert_actions(self, actions: List[Dict]):
        if actions: self.conn.executemany("INSERT INTO RAREACTIONS VALUES (:tick, :steamid, :playername, :action, :item)", actions)
    def insert_buyzone(self, tick, steamid, playername):
        self.conn.execute("INSERT INTO BUYZONE VALUES (?, ?, ?)", (int(tick), str(steamid), str(playername)))

class _bsd_InputVerifier:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.grenade_name_map = {"hegrenade": "High Explosive Grenade", "flashbang": "Flashbang", "smokegrenade": "Smoke Grenade", "molotov": "Molotov", "incgrenade": "Incendiary Grenade", "decoy": "Decoy Grenade"}
    def is_grenade_throw(self, steamid: str, drop_tick: int, item_name: str) -> bool:
        if item_name not in self.grenade_name_map: return False
        active_weapon_name = self.grenade_name_map[item_name]
        cursor = self.conn.cursor()
        cursor.execute("SELECT tick, keyboard_input, active_weapon FROM inputs WHERE steamid = ? AND tick BETWEEN ? AND ? ORDER BY tick DESC", (int(steamid), drop_tick - 128, drop_tick))
        rows = cursor.fetchall()
        if not rows: return False
        held_grenade_before_drop = (rows and rows[0][0] < drop_tick and rows[0][2] == active_weapon_name) or (len(rows) > 1 and rows[1][0] < drop_tick and rows[1][2] == active_weapon_name)
        if not held_grenade_before_drop: return False
        for _, keyboard_input, active_weapon in rows:
            if active_weapon == active_weapon_name and keyboard_input and "IN_ATTACK" in keyboard_input: return True
        return False

def run_buy_sell_drop_processing(demo_path: str, input_conn: sqlite3.Connection, output_conn: sqlite3.Connection):
    print("\n[4/6] Processing buy, sell, and drop data...")
    parser = DemoParser(demo_path)
    
    events = parser.parse_events(["item_purchase", "player_death"])
    events_by_tick: Dict[int, List] = {}
    for name, df in events:
        if not df.empty:
            for row in df.itertuples(index=False):
                events_by_tick.setdefault(row.tick, []).append((name, row))

    tick_props = ["tick", "player_steamid", "player_name", "balance", "inventory_as_ids", "in_buy_zone", "team_num", "ct_cant_buy", "terrorist_cant_buy"]
    all_ticks_df = parser.parse_ticks(tick_props)
    all_ticks_df = all_ticks_df[all_ticks_df['player_name'].apply(_is_valid_player)].copy()

    all_ticks_df.sort_values(by=["tick", "player_steamid"], inplace=True)
    all_ticks_df["inventory_as_ids"] = all_ticks_df["inventory_as_ids"].apply(lambda x: set(x) if x is not None else set())

    last_player_states: Dict[str, Dict] = {}
    potential_actions: List[Dict] = []
    
    db = _bsd_DatabaseManager(output_conn)
    db.init_db()
    for state in tqdm(all_ticks_df.itertuples(index=False), total=len(all_ticks_df), desc="    Analyzing player actions", leave=False):
        steamid = str(state.player_steamid)
        if steamid == '0': continue
        if state.in_buy_zone and not (state.team_num == 2 and state.terrorist_cant_buy) and not (state.team_num == 3 and state.ct_cant_buy):
            db.insert_buyzone(state.tick, steamid, state.player_name)
        last_state = last_player_states.get(steamid)
        if last_state:
            tick_events = events_by_tick.get(state.tick, [])
            if any(name == "player_death" and str(data.user_steamid) == steamid for name, data in tick_events):
                last_player_states[steamid] = state._asdict(); continue
            for name, data in tick_events:
                if name == "item_purchase" and str(data.steamid) == steamid:
                    potential_actions.append({"tick": state.tick, "steamid": steamid, "playername": state.player_name, "action": "BUY", "item": data.item_name})
            if len(state.inventory_as_ids) < len(last_state.get("inventory_as_ids", set())):
                lost_items = last_state["inventory_as_ids"] - state.inventory_as_ids
                if not lost_items:
                    last_player_states[steamid] = state._asdict(); continue
                lost_item_name = _bsd_get_item_name(list(lost_items)[0])
                if any(name == 'item_purchase' and str(data.steamid) == steamid for name, data in tick_events):
                    last_player_states[steamid] = state._asdict(); continue
                if state.in_buy_zone and state.balance > last_state["balance"]:
                    potential_actions.append({"tick": state.tick, "steamid": steamid, "playername": state.player_name, "action": "SELL", "item": lost_item_name})
                else:
                    potential_actions.append({"tick": state.tick, "steamid": steamid, "playername": state.player_name, "action": "DROP", "item": lost_item_name})
        last_player_states[steamid] = state._asdict()
    
    final_actions = []
    input_verifier = _bsd_InputVerifier(input_conn)
    for action in tqdm(potential_actions, desc="    Filtering grenade throws", leave=False):
        if not(action["action"] == "DROP" and action["item"] in GRENADE_NAMES and input_verifier.is_grenade_throw(action["steamid"], action["tick"], action["item"])):
            final_actions.append(action)

    db.batch_insert_actions(final_actions)
    output_conn.commit()
    print(f"    ✓ Confirmed {len(final_actions)} actions into memory.")

# =============================================================================
# 7. PROCESSING STEP 5: MERGE DATABASES
# =============================================================================
def _merge_create_merged_schema(cursor):
    cursor.execute("DROP TABLE IF EXISTS player")
    cursor.execute("DROP TABLE IF EXISTS rounds")
    cursor.execute("""
    CREATE TABLE player (
        tick INTEGER, steamid INTEGER, playername TEXT, position_x REAL, position_y REAL, position_z REAL, inventory TEXT,
        active_weapon TEXT, health INTEGER, armor INTEGER, money INTEGER, keyboard_input TEXT, mouse_x REAL, mouse_y REAL,
        is_in_buyzone INTEGER, buy_sell_input TEXT, PRIMARY KEY (tick, steamid))""")
    cursor.execute("CREATE TABLE rounds (round INTEGER PRIMARY KEY, starttick INTEGER, freezetime_endtick INTEGER, endtick INTEGER, win_tick INTEGER, win_team TEXT, bomb_planted_tick INTEGER, t_team TEXT, ct_team TEXT)")

def _merge_load_lookup_data(mouse_conn, bsd_conn, rounds_conn):
    mouse_positions, buy_sell_drop_actions, buyzone_presence, valid_round_ticks = {}, {}, set(), []
    for tick, player_name, x, y in mouse_conn.cursor().execute("SELECT tick, player_name, x, y FROM MOUSE"): mouse_positions[(tick, player_name)] = (x, y)
    bsd_cursor = bsd_conn.cursor()
    for tick, playername, action, item in bsd_cursor.execute("SELECT tick, playername, action, item FROM RAREACTIONS"): buy_sell_drop_actions.setdefault((tick, playername), []).append((action, item))
    for tick, _, playername in bsd_cursor.execute("SELECT tick, steamid, playername FROM BUYZONE"): buyzone_presence.add((tick, playername))
    for starttick, endtick in rounds_conn.cursor().execute("SELECT starttick, endtick FROM ROUNDS WHERE starttick IS NOT NULL AND endtick IS NOT NULL"): valid_round_ticks.append((starttick, endtick))
    return mouse_positions, buy_sell_drop_actions, buyzone_presence, valid_round_ticks

def _merge_is_tick_in_valid_round(tick, round_intervals):
    for start, end in round_intervals:
        if start <= tick <= end: return True
    return False

def run_merge_processing(mouse_conn, rounds_conn, kl_conn, bsd_conn, merged_conn):
    print("\n[5/6] Merging all in-memory data...")
    mouse_data, action_data, buyzone_data, round_intervals = _merge_load_lookup_data(mouse_conn, bsd_conn, rounds_conn)

    merged_cursor = merged_conn.cursor()
    _merge_create_merged_schema(merged_cursor)

    player_rows_to_insert = []
    all_inputs = kl_conn.cursor().execute("SELECT tick, steamid, playername, keyboard_input, inventory, x, y, z, active_weapon, health, armor, money FROM inputs").fetchall()
    
    for row in tqdm(all_inputs, desc="    Assembling final rows   ", leave=False):
        tick, steamid, playername, kb_input, inventory, x, y, z, active_w, health, armor, money = row
        if health is None or health <= 0 or not _merge_is_tick_in_valid_round(tick, round_intervals): continue
        
        mouse_x, mouse_y = mouse_data.get((tick, playername), (None, None))
        is_in_buyzone = 1 if (tick, playername) in buyzone_data else 0
        final_kb_inputs, buy_sell_actions = kb_input.split(',') if kb_input else [], []
        
        for action, item in action_data.get((tick, playername), []):
            safe_item = item.replace(' ', '_').replace('&', 'and').replace('-','_')
            if action == 'DROP': final_kb_inputs.append(f"DROP_{safe_item}")
            elif action in ('BUY', 'SELL'): buy_sell_actions.append(f"{action}_{safe_item}")
        
        player_rows_to_insert.append((tick, steamid, playername, x, y, z, inventory, active_w, health, armor, money, ",".join(final_kb_inputs), mouse_x, mouse_y, is_in_buyzone, ",".join(buy_sell_actions)))

    print(f"    Writing {len(player_rows_to_insert)} rows to 'player' table...")
    if player_rows_to_insert:
        merged_cursor.executemany("INSERT INTO player VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", player_rows_to_insert)

    print("    Copying 'rounds' table...")
    all_rounds = rounds_conn.cursor().execute("SELECT * FROM ROUNDS").fetchall()
    merged_cursor.executemany("INSERT INTO rounds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", all_rounds)

    merged_conn.commit()
    print("    ✓ Final database merged and written to disk.")


# =============================================================================
# 8. PROCESSING STEP 6: RECORDING CANDIDATES (NEWLY INTEGRATED)
# =============================================================================
def _rc_prepare_and_create_table(conn: sqlite3.Connection):
    """Prepares the database by dropping and recreating the RECORDING table."""
    cursor = conn.cursor()
    print("    Preparing 'RECORDING' table...")
    cursor.execute("DROP TABLE IF EXISTS RECORDING;")
    cursor.execute("""
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
    """)
    conn.commit()

def _rc_fetch_and_process_rounds(conn: sqlite3.Connection) -> List[Tuple]:
    """Fetches, validates, and processes rounds to generate recording candidates."""
    print("    Processing and validating data from 'ROUNDS' table...")
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT round, starttick, freezetime_endtick, endtick, t_team, ct_team FROM ROUNDS")
    except sqlite3.OperationalError:
        print(f"    Error: Table 'ROUNDS' not found. Cannot generate candidates.")
        return []

    all_rounds_data = cursor.fetchall()
    to_be_recorded = []

    for round_num, starttick, freezetime_endtick, endtick, t_team_json, ct_team_json in all_rounds_data:
        # Stage 1: Parse and Basic Checks
        if freezetime_endtick is not None and starttick is not None:
            if (freezetime_endtick - starttick) > 2000:
                print(f"    Skipping round {round_num}: Extended freezetime duration ({freezetime_endtick - starttick} ticks).")
                continue
        
        if starttick is None or endtick is None:
            print(f"    Skipping round {round_num}: Missing start or end tick.")
            continue
            
        try:
            t_players = json.loads(t_team_json) if t_team_json else []
            ct_players = json.loads(ct_team_json) if ct_team_json else []
            if not isinstance(t_players, list) or not isinstance(ct_players, list):
                print(f"    Skipping round {round_num}: Team data is not a valid JSON list.")
                continue
        except json.JSONDecodeError:
            print(f"    Skipping round {round_num}: Failed to parse JSON data.")
            continue

        # Stage 2: Strict Validation
        is_round_valid = True
        # playercount = len(t_players) + len(ct_players)
        if len(t_players) != 5 or len(ct_players) != 5: 
            print(f"    Skipping round {round_num}: Invalid team sizes. T: {len(t_players)}, CT: {len(ct_players)}. Expected 5v5.")
            is_round_valid = False
        
        if is_round_valid:
            t_player_names = {p[0] for p in t_players if isinstance(p, list) and len(p) > 0}
            ct_player_names = {p[0] for p in ct_players if isinstance(p, list) and len(p) > 0}
            if t_player_names.intersection(ct_player_names):
                print(f"    Skipping round {round_num}: Player(s) found on both teams.")
                is_round_valid = False

        if is_round_valid and freezetime_endtick is not None:
            all_round_players_for_check = t_players + ct_players
            for player_data in all_round_players_for_check:
                if not isinstance(player_data, list) or len(player_data) < 2: continue
                _, death_tick = player_data[0], player_data[1]
                if death_tick != -1 and death_tick < (freezetime_endtick + 128):
                    print(f"    Skipping round {round_num}: Player died too early ({death_tick}).")
                    is_round_valid = False
                    break

        # Stage 3: Process the valid round
        if is_round_valid:
            all_round_players = [(*p, 'T') for p in t_players] + [(*p, 'CT') for p in ct_players]
            for player_data in all_round_players:
                if len(player_data) != 3: continue
                player_name, death_tick, team_name = player_data
                stop_tick = death_tick if death_tick != -1 else endtick
                record = (round_num, starttick, stop_tick, team_name, player_name)
                to_be_recorded.append(record)
    return to_be_recorded

def _rc_insert_records(conn: sqlite3.Connection, records: List[Tuple]):
    """Inserts the generated records into the 'RECORDING' table."""
    if not records:
        print("    No valid rounds found to generate recording candidates.")
        return

    records_to_insert = [(*rec, False, None) for rec in records]
    cursor = conn.cursor()
    sql = """
        INSERT INTO RECORDING (roundnumber, starttick, stoptick, team, playername, is_recorded, recording_filepath)
        VALUES (?, ?, ?, ?, ?, ?, ?);
    """
    try:
        cursor.executemany(sql, records_to_insert)
        conn.commit()
        print(f"    ✓ Successfully inserted {cursor.rowcount} records into 'RECORDING' table.")
    except sqlite3.IntegrityError as e:
        print(f"    Error during insertion: {e}. This should not happen with pre-validated data.")
        conn.rollback()

def run_recording_candidates_processing(db_path: str):
    """
    Main function for the recording candidate generation step.
    Connects to the final DB and populates the RECORDING table.
    """
    print(f"\n[6/6] Generating recording candidates...")
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        _rc_prepare_and_create_table(conn)
        records_to_add = _rc_fetch_and_process_rounds(conn)
        _rc_insert_records(conn, records_to_add)
    except sqlite3.Error as e:
        print(f"    A database error occurred during candidate generation: {e}")
        if conn: conn.rollback()
    finally:
        if conn:
            conn.close()


# =============================================================================
# 9. MAIN DRIVER
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run full, optimized demo processing pipeline.")
    parser.add_argument('--demo', required=True, help='Path to demo file (.dem)')
    parser.add_argument('--out', required=True, help='Path to the final output database file (e.g., output.db)')
    args = parser.parse_args()

    demo_path = os.path.abspath(args.demo)
    if not os.path.isfile(demo_path):
        print(f"Error: Demo file not found: {demo_path}", file=sys.stderr)
        sys.exit(1)

    # Prepare output path
    merged_db_path = os.path.abspath(args.out)
    out_dir = os.path.dirname(merged_db_path)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(merged_db_path):
        print(f"Removing existing output file: {merged_db_path}")
        os.remove(merged_db_path)
    
    # Setup in-memory database connections
    mouse_conn, rounds_conn, kl_conn, bsd_conn = (sqlite3.connect(':memory:') for _ in range(4))
    
    try:
        # Steps 1-4: Process into memory
        run_mouse_processing(demo_path, mouse_conn)
        run_rounds_processing(demo_path, rounds_conn)
        run_keyboard_location_processing(demo_path, kl_conn)
        run_buy_sell_drop_processing(demo_path, kl_conn, bsd_conn)

        # Step 5: Merge from memory to disk
        merged_conn = sqlite3.connect(merged_db_path)
        try:
            run_merge_processing(mouse_conn, rounds_conn, kl_conn, bsd_conn, merged_conn)
        finally:
            merged_conn.close()

        # Step 6: Post-process the on-disk file to add recording candidates
        run_recording_candidates_processing(merged_db_path)

    except Exception as e:
        print(f"\nAn unhandled error occurred during processing: {e}", file=sys.stderr)
        traceback.print_exc()
        print("Pipeline aborted.", file=sys.stderr)
        sys.exit(1)
    finally:
        # Close all in-memory connections
        mouse_conn.close()
        rounds_conn.close()
        kl_conn.close()
        bsd_conn.close()

    print("\n-------------------------------------------")
    print("All processing steps completed successfully.")
    print(f"Final database is at: {merged_db_path}")
    print("-------------------------------------------")

if __name__ == '__main__':
    main()