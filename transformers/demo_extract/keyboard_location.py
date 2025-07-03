#!/usr/bin/env python3
'''
- Parse demo,
- extract all player inputs/tick; table name PLAYERINPUT
- player recording timeframes (round#, starttick, endtick, isdeath?)
    - separate tables: playerinput, info for player pov recording (round#, starttick ,endtick, isdeath, (video_path, audio_path), (team)), separate table auxiliary information (money, roundcount)
'''
from __future__ import annotations
import argparse, sys, sqlite3, json
from pathlib import Path
import pandas as pd
from demoparser2 import DemoParser

# --- (Helper functions: WEAPON_CATEGORIES, get_weapon_switch_type, KEY_MAPPING, extract_buttons, _sanitize_inventory remain unchanged) ---
WEAPON_CATEGORIES = {
    "AK-47": "SWITCH_1", "M4A4": "SWITCH_1", "M4A1-S": "SWITCH_1", "Galil AR": "SWITCH_1", "FAMAS": "SWITCH_1", "AUG": "SWITCH_1", "SG 553": "SWITCH_1", "AWP": "SWITCH_1", "SSG 08": "SWITCH_1", "G3SG1": "SWITCH_1", "SCAR-20": "SWITCH_1", "MP9": "SWITCH_1", "MAC-10": "SWITCH_1", "MP7": "SWITCH_1", "MP5-SD": "SWITCH_1", "UMP-45": "SWITCH_1", "P90": "SWITCH_1", "PP-Bizon": "SWITCH_1", "Nova": "SWITCH_1", "XM1014": "SWITCH_1", "MAG-7": "SWITCH_1", "Sawed-Off": "SWITCH_1", "M249": "SWITCH_1", "Negev": "SWITCH_1",
    "Glock-18": "SWITCH_2", "USP-S": "SWITCH_2", "P250": "SWITCH_2", "P2000": "SWITCH_2", "Dual Berettas": "SWITCH_2", "Five-SeveN": "SWITCH_2", "Tec-9": "SWITCH_2", "CZ75-Auto": "SWITCH_2", "Desert Eagle": "SWITCH_2", "R8 Revolver": "SWITCH_2",
    "knife":"SWITCH_3","knife_ct": "SWITCH_3", "knife_t": "SWITCH_3", "Bayonet": "SWITCH_3", "Flip Knife": "SWITCH_3", "Gut Knife": "SWITCH_3", "Karambit": "SWITCH_3", "M9 Bayonet": "SWITCH_3", "Huntsman Knife": "SWITCH_3", "Falchion Knife": "SWITCH_3", "Bowie Knife": "SWITCH_3", "Butterfly Knife": "SWITCH_3", "Shadow Daggers": "SWITCH_3", "Ursus Knife": "SWITCH_3", "Navaja Knife": "SWITCH_3", "Stiletto Knife": "SWITCH_3", "Talon Knife": "SWITCH_3", "Classic Knife": "SWITCH_3", "Paracord Knife": "SWITCH_3", "Survival Knife": "SWITCH_3", "Nomad Knife": "SWITCH_3", "Skeleton Knife": "SWITCH_3",
    "High Explosive Grenade": "SWITCH_4", "Flashbang": "SWITCH_4", "Smoke Grenade": "SWITCH_4", "Molotov": "SWITCH_4", "Incendiary Grenade": "SWITCH_4", "Decoy Grenade": "SWITCH_4",
    "C4 Explosive": "SWITCH_5", "Defuse Kit": "SWITCH_5", "Zeus x27": "SWITCH_3",
}
def get_weapon_switch_type(weapon_name: str | None) -> str | None:
    if not weapon_name: return None
    return WEAPON_CATEGORIES.get(weapon_name, f"SWITCH_UNDEFINED_{weapon_name}")

KEY_MAPPING = {
    "IN_ATTACK": 1<<0, "IN_JUMP": 1<<1, "IN_DUCK": 1<<2, "IN_FORWARD": 1<<3, "IN_BACK": 1<<4, "IN_USE": 1<<5, "IN_CANCEL": 1<<6, "IN_TURNLEFT": 1<<7, "IN_TURNRIGHT": 1<<8, "IN_MOVELEFT": 1<<9, "IN_MOVERIGHT": 1<<10, "IN_ATTACK2": 1<<11, "IN_RELOAD": 1<<13, "IN_ALT1": 1<<14, "IN_ALT2": 1<<15, "IN_SPEED": 1<<16, "IN_WALK": 1<<17, "IN_ZOOM": 1<<18, "IN_WEAPON1": 1<<19, "IN_WEAPON2": 1<<20, "IN_BULLRUSH": 1<<21, "IN_GRENADE1": 1<<22, "IN_GRENADE2": 1<<23, "IN_ATTACK3": 1<<24, "IN_SCORE": 1<<33, "IN_INSPECT": 1<<35,
}
def extract_buttons(bits) -> list[str]:
    bits_int = int(bits)
    return [name for name, mask in KEY_MAPPING.items() if bits_int & mask]

def _sanitize_inventory(inv):
    if isinstance(inv, (list, tuple, set)):
        try: return json.dumps(list(inv))
        except TypeError: return ",".join(map(str, inv))
    return str(inv)

def _export_sqlite_inputs(db_path: Path, tick_df: pd.DataFrame) -> None:
    df = tick_df.copy()
    df["inventory"] = df["inventory"].apply(_sanitize_inventory)
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

def cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CS-2 demo inspector / exporter")
    p.add_argument("demofile", type=Path, help="Path to .dem file")
    p.add_argument("--sqlout", type=Path, metavar="FILE.db", help="Write results to SQLite DB")
    p.add_argument("--optimize", action="store_true", help="Remove ticks where players are dead (health=0) or have invalid positions (0,0,0)")
    return p

def main() -> None:
    args = cli().parse_args()
    if not args.demofile.exists():
        sys.exit(f"Demo not found: {args.demofile}")

    print("[loading demo – one-off parse, please wait …]")
    dp = DemoParser(args.demofile.as_posix())

    tick_props = [
        "tick", "steamid", "name", "buttons", "inventory", "X", "Y", "Z",
        "active_weapon_name", "is_defusing", "health", "armor_value", "balance",
        "is_warmup_period", "total_rounds_played"
    ]
    events_to_parse = [
        "player_death", "hegrenade_detonate", "flashbang_detonate",
        "smokegrenade_detonate", "molotov_detonate"
    ]

    tick_df = pd.DataFrame(dp.parse_ticks(wanted_props=tick_props))
    events = dp.parse_events(events_to_parse)
    event_dfs = dict(events)
    
    if 'is_warmup_period' in tick_df.columns:
        print("[filtering out warmup period...]")
        tick_df = tick_df[tick_df['is_warmup_period'] == False].copy()

    if args.optimize:
        print("[optimizing dataset...]")
        initial_rows = len(tick_df)

        alive_mask = tick_df['health'] > 0
        tick_df = tick_df[alive_mask].copy()
        print(f"  - Removed {initial_rows - len(tick_df)} ticks where player health was 0.")
        
        initial_rows_after_health = len(tick_df)
        zero_pos_mask = (tick_df['X'] == 0) & (tick_df['Y'] == 0) & (tick_df['Z'] == 0)
        tick_df = tick_df[~zero_pos_mask].copy()
        print(f"  - Removed {initial_rows_after_health - len(tick_df)} ticks with invalid (0,0,0) positions.")
        
        print(f"  - Optimization complete. Total rows reduced from {initial_rows} to {len(tick_df)}.")

    print("[gap-filling player data...]")
    if not tick_df.empty:
        tick_df.sort_values(["steamid", "tick"], inplace=True)
        player_steamids = tick_df['steamid'].unique()
        min_tick, max_tick = tick_df['tick'].min(), tick_df['tick'].max()
        all_ticks_range = range(min_tick, max_tick + 1)
        
        multi_index = pd.MultiIndex.from_product([player_steamids, all_ticks_range], names=['steamid', 'tick'])
        
        tick_df = tick_df.set_index(['steamid', 'tick']).reindex(multi_index)
        
        tick_df = tick_df.groupby(level='steamid').ffill()
        tick_df.dropna(subset=['name'], inplace=True) # Drop rows for players that never had data
        tick_df = tick_df.reset_index()
    else:
        print("Warning: DataFrame is empty after filtering. No data to process.")
        return

    for col in ['health', 'armor_value', 'balance', 'total_rounds_played', 'buttons']:
        tick_df[col] = pd.to_numeric(tick_df[col], errors='coerce').fillna(0).astype(int)

    print("[inferring weapon drops...]")
    tick_df['prev_inventory'] = tick_df.groupby('steamid')['inventory'].shift(1)
    tick_df['inferred_drop'] = ''

    grenade_throw_ticks = set()
    for name in ['hegrenade_detonate', 'flashbang_detonate', 'smokegrenade_detonate', 'molotov_detonate']:
        if event_dfs.get(name) is not None and not event_dfs[name].empty:
            df = event_dfs[name]
            # --- FIX: Iterate over rows to add individual (tick, steamid) tuples ---
            for event_row in df.itertuples(index=False):
                if pd.notna(event_row.tick) and pd.notna(event_row.user_steamid):
                    # Use a small tick window to account for network/event lag
                    grenade_throw_ticks.add((event_row.tick - 1, event_row.user_steamid))
                    grenade_throw_ticks.add((event_row.tick, event_row.user_steamid))
                    grenade_throw_ticks.add((event_row.tick + 1, event_row.user_steamid))
    
    death_ticks_set = set()
    if event_dfs.get('player_death') is not None and not event_dfs['player_death'].empty:
        df = event_dfs['player_death']
        death_ticks_set.update(zip(df['tick'], df['user_steamid']))

    def to_set(x): return set(x) if isinstance(x, list) else set()
    tick_df['inventory_set'] = tick_df['inventory'].apply(to_set)
    tick_df['prev_inventory_set'] = tick_df['prev_inventory'].apply(to_set)
    inventory_changed_mask = tick_df['inventory_set'] < tick_df['prev_inventory_set']

    print("[inferring weapon switches...]")
    tick_df['prev_weapon'] = tick_df.groupby('steamid')['active_weapon_name'].shift(1).fillna('')
    weapon_changed_mask = (tick_df['active_weapon_name'] != tick_df['prev_weapon']) & (tick_df['active_weapon_name'] != '')
    tick_df['switch_type'] = tick_df['active_weapon_name'].apply(get_weapon_switch_type)
    tick_df['inferred_switch'] = ''
    tick_df.loc[weapon_changed_mask, 'inferred_switch'] = tick_df.loc[weapon_changed_mask, 'switch_type']
    
    tick_df['real_keys_list'] = tick_df['buttons'].apply(extract_buttons)
    def combine_inputs(row):
        keys = row['real_keys_list']
        if row['inferred_switch'] and row['inferred_switch'] not in keys: keys.append(row['inferred_switch'])
        if row['inferred_drop']: keys.append(row['inferred_drop'])
        return ",".join(keys)
    tick_df['keyboard_input'] = tick_df.apply(combine_inputs, axis=1)

    tick_df.drop(columns=['prev_inventory', 'inferred_drop', 'inventory_set', 'prev_inventory_set', 'prev_weapon', 'switch_type', 'inferred_switch', 'real_keys_list', 'buttons'], inplace=True, errors='ignore')

    if args.sqlout:
        print(f"[exporting to {args.sqlout}...]")
        _export_sqlite_inputs(args.sqlout, tick_df)
        print(f"✓ wrote table 'inputs' to {args.sqlout} – done")
        return

    # --- REPL ---
    print("\nCommands:\n  seek <tick>   jump to that tick\n  quit / exit   leave\n")
    while True:
        try: cmd = input("> ").strip()
        except (EOFError, KeyboardInterrupt): print(); break
        if cmd in {"quit", "exit"}: break
        if not cmd.startswith("seek"): continue

        parts = cmd.split(maxsplit=1)
        if len(parts) != 2 or not parts[1].isdigit(): continue
        tgt_tick = int(parts[1])

        player_rows = tick_df.query("tick == @tgt_tick")

        if player_rows.empty:
            print(f"No player data at tick {tgt_tick}.")
            continue
        
        players = player_rows[["steamid", "name"]].drop_duplicates().reset_index(drop=True)
        for i, r in players.iterrows():
            print(f"{i+1}. {r['name']}  (sid {r['steamid']})")

        sel = input("Player # > ").strip()
        if not sel.isdigit() or not (1 <= int(sel) <= len(players)): continue
        sid = players.iloc[int(sel)-1]["steamid"]

        r = player_rows.loc[player_rows["steamid"] == sid].iloc[-1]
        print(f"\nTick {tgt_tick} – {r['name']}:\n"
              f"  Inputs      : {r['keyboard_input'] or '(none)'}\n"
              f"  Health/Armor: {int(r.get('health', 0))} / {int(r.get('armor_value', 0))}\n"
              f"  Money       : ${int(r.get('balance', 0))}\n"
              f"  Position    : x={r.get('X', 0):.2f}, y={r.get('Y', 0):.2f}, z={r.get('Z', 0):.2f}\n"
              f"  Inventory   : {_sanitize_inventory(r.get('inventory', []))}\n"
              f"  Active Weapon: {r.get('active_weapon_name', '') or '(none)'}")
        print()

if __name__ == "__main__":
    main()