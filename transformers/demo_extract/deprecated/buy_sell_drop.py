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
"""
import argparse
import sqlite3
import pandas as pd
from demoparser2 import DemoParser
from tqdm import tqdm
from typing import Dict, Any, Set, List, Tuple

# NOTE: This script no longer requires 'awpy'.
# It now requires an input database from button_loc.py
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

def get_item_name(item_id: int) -> str:
    return ITEM_ID_MAP.get(item_id, f"unknown_item_{item_id}")

class DatabaseManager:
    """Handles SQLite database connection and operations."""
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

class InputVerifier:
    """Queries the button_loc.py database to verify grenade throws."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        # Mapping from this script's item names to button_loc.py's active_weapon names
        self.grenade_name_map = {
            "hegrenade": "High Explosive Grenade",
            "flashbang": "Flashbang",
            "smokegrenade": "Smoke Grenade",
            "molotov": "Molotov",
            "incgrenade": "Incendiary Grenade",
            "decoy": "Decoy Grenade"
        }

    def __enter__(self):
        # Open in read-only mode to prevent accidental modification
        self.conn = sqlite3.connect(f'file:{self.db_path}?mode=ro', uri=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

    def is_grenade_throw(self, steamid: str, drop_tick: int, item_name: str) -> bool:
        """
        Checks if a grenade disappearing from inventory was a throw.
        A throw is identified by a recent IN_ATTACK press while holding the grenade.
        """
        if item_name not in self.grenade_name_map:
            return False

        active_weapon_name = self.grenade_name_map[item_name]
        # Look in a 2-second window (128 ticks @ 64tps) before the drop
        query_tick_start = drop_tick - 128
        
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT tick, keyboard_input, active_weapon
            FROM inputs
            WHERE steamid = ? AND tick BETWEEN ? AND ?
            ORDER BY tick DESC
            """,
            (int(steamid), query_tick_start, drop_tick)
        )
        rows = cursor.fetchall()

        if not rows:
            return False # No input data found for this player in this window

        # First, confirm the player was holding the grenade right before it disappeared.
        # The grenade disappears at 'drop_tick', so at 'drop_tick - 1' it was likely the active weapon.
        held_grenade_before_drop = False
        if rows and rows[0][0] < drop_tick: # Check first row if it's before the drop tick
             if rows[0][2] == active_weapon_name:
                 held_grenade_before_drop = True
        
        if not held_grenade_before_drop:
            # Sometimes there's a 1-tick delay, check the one before that too
            if len(rows) > 1 and rows[1][0] < drop_tick and rows[1][2] == active_weapon_name:
                held_grenade_before_drop = True

        if not held_grenade_before_drop:
            return False # Player wasn't holding the grenade, so it couldn't be a throw.

        # Now, look for the 'IN_ATTACK' press in the recent history while holding the grenade.
        for _, keyboard_input, active_weapon in rows:
            if active_weapon == active_weapon_name and keyboard_input and "IN_ATTACK" in keyboard_input:
                # Found the pattern: player pressed attack while holding the grenade. This is a throw.
                return True
        
        return False # Did not find the IN_ATTACK pattern, so it's likely a drop.

def main(demo_path: str, sql_output_path: str, sql_input_path: str):
    print(f"Parsing demo with demoparser2: {demo_path}")
    parser = DemoParser(demo_path)

    print("Step 1/3: Parsing high-certainty events...")
    event_names = ["item_purchase", "player_death"]
    events = parser.parse_events(event_names)
    events_by_tick: Dict[int, List[Tuple[str, Any]]] = {}
    for name, df in events:
        if not df.empty:
            for row in df.itertuples(index=False):
                if row.tick not in events_by_tick: events_by_tick[row.tick] = []
                events_by_tick[row.tick].append((name, row))

    print("Step 2/3: Parsing tick-by-tick player states...")
    tick_props = ["tick", "player_steamid", "player_name", "balance", "inventory_as_ids", "in_buy_zone", "team_num", "ct_cant_buy", "terrorist_cant_buy"]
    all_ticks_df = parser.parse_ticks(tick_props)
    all_ticks_df.sort_values(by=["tick", "player_steamid"], inplace=True)
    all_ticks_df["inventory_as_ids"] = all_ticks_df["inventory_as_ids"].apply(lambda x: set(x) if x is not None else set())

    print(f"Step 3/3: Analyzing {len(all_ticks_df)} tick states...")
    last_player_states: Dict[str, Dict[str, Any]] = {}
    potential_actions: List[Dict] = []
    
    with DatabaseManager(sql_output_path) as db:
        db.init_db()
        for state in tqdm(all_ticks_df.itertuples(index=False), total=len(all_ticks_df), desc="Analyzing Actions"):
            steamid = str(state.player_steamid)
            if steamid == '0': continue

            in_buyzone = state.in_buy_zone and not (state.team_num == 2 and state.terrorist_cant_buy) and not (state.team_num == 3 and state.ct_cant_buy)
            if in_buyzone:
                db.insert_buyzone(state.tick, steamid, state.player_name)

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
                    lost_item_name = get_item_name(lost_item_id)
                    
                    if any(name == 'item_purchase' and str(data.steamid) == steamid for name, data in tick_events):
                        last_player_states[steamid] = state._asdict()
                        continue
                        
                    if in_buyzone and state.balance > last_state["balance"]:
                        potential_actions.append({"tick": state.tick, "steamid": steamid, "playername": state.player_name, "action": "SELL", "item": lost_item_name})
                        last_player_states[steamid] = state._asdict()
                        continue

                    # Log as a potential drop for now, to be filtered later
                    potential_actions.append({"tick": state.tick, "steamid": steamid, "playername": state.player_name, "action": "DROP", "item": lost_item_name})

            last_player_states[steamid] = state._asdict()
        
        # Post-processing filter for grenade throws using the input database
        final_actions = []
        print(f"Post-processing {len(potential_actions)} potential actions...")
        with InputVerifier(sql_input_path) as input_verifier:
            for action in tqdm(potential_actions, desc="Filtering Grenade Throws"):
                if action["action"] == "DROP" and action["item"] in GRENADE_NAMES:
                    is_throw = input_verifier.is_grenade_throw(
                        steamid=action["steamid"],
                        drop_tick=action["tick"],
                        item_name=action["item"]
                    )
                    if not is_throw:
                        final_actions.append(action) # It's a real drop, not a throw
                else:
                    final_actions.append(action)

        print(f"Finalizing database with {len(final_actions)} confirmed actions...")
        db.batch_insert_actions(final_actions)

    print(f"\nAnalysis complete. Results saved to '{sql_output_path}'")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Analyze CS2 demo for BUY, SELL, and DROP actions, using detailed player input for accuracy.")
    arg_parser.add_argument("demofile", help="Path to the .dem file")
    arg_parser.add_argument("--sqlout", required=True, help="Path to the output SQLite database file.")
    arg_parser.add_argument("--sqlin", required=True, help="Path to the input SQLite database file generated by button_loc.py.")
    args = arg_parser.parse_args()
    main(args.demofile, args.sqlout, args.sqlin)