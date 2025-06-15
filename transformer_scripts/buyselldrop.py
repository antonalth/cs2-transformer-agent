import argparse
import sqlite3
import pandas as pd
from demoparser2 import DemoParser
from tqdm import tqdm
from typing import Dict, Any, Set, List, Tuple

# A partial mapping of item definition indices to names.
ITEM_ID_MAP = {
    1: "deagle", 2: "elite", 3: "fiveseven", 4: "glock",
    7: "ak47", 8: "aug", 9: "awp", 10: "famas", 11: "g3sg1",
    13: "galilar", 14: "m249", 16: "m4a1", 17: "mac10",
    19: "p90", 23: "mp5sd", 24: "ump45", 25: "xm1014",
    26: "bizon", 27: "mag7", 28: "negev", 29: "sawedoff",
    30: "tec9", 32: "p2000", 33: "mp7", 34: "mp9", 35: "nova",
    36: "p250", 38: "scar20", 39: "sg556", 40: "ssg08",
    42: "knife", 43: "flashbang", 44: "hegrenade", 45: "smokegrenade",
    46: "molotov", 47: "decoy", 48: "incgrenade", 49: "c4",
    59: "knife_t", 60: "m4a1_silencer", 61: "usp_silencer", 63: "cz75a",
    64: "revolver",
    500: "knife_default_ct", 506: "knife_gut", 507: "knife_flip", 508: "knife_bayonet",
    509: "knife_m9_bayonet", 515: "knife_karambit", 522: "knife_stiletto", 523: "knife_ursus",
    80: "defuser", 81: "vest", 82: "vesthelm"
}

def get_item_name(item_id: int) -> str:
    """Converts an item ID to its string name."""
    return ITEM_ID_MAP.get(item_id, f"unknown_item_{item_id}")

class DatabaseManager:
    """Handles SQLite database connection and operations."""
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.cursor = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.commit()
            self.conn.close()

    def init_db(self):
        """Creates the necessary tables if they don't exist."""
        self.cursor.execute("DROP TABLE IF EXISTS RAREACTIONS;")
        self.cursor.execute("DROP TABLE IF EXISTS BUYZONE;")
        
        self.cursor.execute("""
        CREATE TABLE RAREACTIONS (
            tick INTEGER,
            steamid TEXT,
            playername TEXT,
            action TEXT,
            item TEXT
        );
        """)
        self.cursor.execute("""
        CREATE TABLE BUYZONE (
            tick INTEGER,
            steamid TEXT,
            playername TEXT
        );
        """)
        self.cursor.execute("CREATE INDEX idx_reactions_tick_steamid ON RAREACTIONS (tick, steamid);")
        self.cursor.execute("CREATE INDEX idx_buyzone_tick_steamid ON BUYZONE (tick, steamid);")


    def insert_action(self, tick: int, steamid: str, playername: str, action: str, item: str):
        self.cursor.execute(
            "INSERT INTO RAREACTIONS (tick, steamid, playername, action, item) VALUES (?, ?, ?, ?, ?)",
            (int(tick), str(steamid), str(playername), str(action), str(item))
        )

    def insert_buyzone(self, tick: int, steamid: str, playername: str):
        self.cursor.execute(
            "INSERT INTO BUYZONE (tick, steamid, playername) VALUES (?, ?, ?)",
            (int(tick), str(steamid), str(playername))
        )

def main(demo_path: str, sql_output_path: str):
    print(f"Parsing demo: {demo_path}")
    parser = DemoParser(demo_path)

    print("Step 1/4: Parsing events...")
    event_names = ["item_purchase", "player_death", "weapon_fire"]
    events = parser.parse_events(event_names)

    events_by_tick: Dict[int, List[Tuple[str, Any]]] = {}
    for event_name, df in events:
        if df.empty:
            continue
        for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"Processing {event_name} events"):
            tick = row.tick
            if tick not in events_by_tick:
                events_by_tick[tick] = []
            events_by_tick[tick].append((event_name, row))

    print("Step 2/4: Parsing all ticks for player states...")
    tick_props = [
        "tick", "player_steamid", "player_name", "balance",
        "inventory_as_ids", "in_buy_zone", "team_num",
        "ct_cant_buy", "terrorist_cant_buy", "usercmd_impulse"
    ]
    with tqdm(total=1, desc="Parsing Ticks") as pbar:
        all_ticks_df = parser.parse_ticks(tick_props)
        pbar.update(1)

    all_ticks_df.sort_values(by=["tick", "player_steamid"], inplace=True)
    all_ticks_df["inventory_as_ids"] = all_ticks_df["inventory_as_ids"].apply(lambda x: set(x) if x is not None else set())

    print(f"Step 3/4: Analyzing {len(all_ticks_df)} tick states...")
    last_player_states: Dict[str, Dict[str, Any]] = {}
    
    with DatabaseManager(sql_output_path) as db:
        db.init_db()

        for current_state in tqdm(all_ticks_df.itertuples(index=False), total=len(all_ticks_df), desc="Analyzing Actions"):
            steamid = str(current_state.player_steamid)
            if steamid == '0':
                continue
            
            # -- BUYZONE LOGIC --
            # This logic now runs for every tick, not just on state change.
            is_in_usable_buyzone_now = (
                current_state.in_buy_zone and
                not (current_state.team_num == 2 and current_state.terrorist_cant_buy) and
                not (current_state.team_num == 3 and current_state.ct_cant_buy)
            )
            
            # ---- THE FIX IS HERE ----
            if is_in_usable_buyzone_now:
                 db.insert_buyzone(current_state.tick, steamid, current_state.player_name)
            
            # -- ACTION LOGIC --
            last_state = last_player_states.get(steamid)
            if last_state:
                tick_events = events_by_tick.get(current_state.tick, [])
                
                for event_name, event_data in tick_events:
                    if event_name == "item_purchase" and str(event_data.steamid) == steamid:
                        db.insert_action(current_state.tick, steamid, current_state.player_name, "BUY", event_data.item_name)

                is_dead_this_tick = any(
                    name == "player_death" and str(data.user_steamid) == steamid for name, data in tick_events
                )
                if is_dead_this_tick:
                    last_player_states[steamid] = current_state._asdict()
                    continue

                last_inv = last_state.get("inventory_as_ids", set())
                current_inv = current_state.inventory_as_ids
                
                if len(current_inv) < len(last_inv):
                    lost_items = last_inv - current_inv
                    
                    if current_state.usercmd_impulse == 201 and len(lost_items) == 1:
                        item_id = lost_items.pop()
                        item_name = get_item_name(item_id)
                        
                        is_grenade_throw = False
                        if "grenade" in item_name or "molotov" in item_name or "decoy" in item_name:
                            for tick_offset in range(-16, 17):
                                for event_name, event_data in events_by_tick.get(current_state.tick + tick_offset, []):
                                    if (event_name == "weapon_fire" and 
                                        str(event_data.user_steamid) == steamid and 
                                        event_data.weapon == item_name):
                                        is_grenade_throw = True
                                        break
                                if is_grenade_throw:
                                    break
                        
                        if not is_grenade_throw:
                            db.insert_action(current_state.tick, steamid, current_state.player_name, "DROP", item_name)

                    elif is_in_usable_buyzone_now and current_state.balance > last_state["balance"] and len(lost_items) == 1:
                        item_id = lost_items.pop()
                        item_name = get_item_name(item_id)
                        db.insert_action(current_state.tick, steamid, current_state.player_name, "SELL", item_name)

            last_player_states[steamid] = current_state._asdict()
        
        print("Step 4/4: Finalizing database...")

    print(f"\nAnalysis complete. Results saved to '{sql_output_path}'")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Analyze CS2 demo for BUY, SELL, and DROP actions.")
    arg_parser.add_argument("demofile", help="Path to the .dem file")
    arg_parser.add_argument("--sqlout", required=True, help="Path to the output SQLite database file.")

    args = arg_parser.parse_args()
    main(args.demofile, args.sqlout)