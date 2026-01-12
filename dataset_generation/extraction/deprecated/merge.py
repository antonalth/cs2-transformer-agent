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
import sqlite3
import os
import sys
import argparse
from tqdm import tqdm

def create_merged_schema(cursor):
    """Creates the necessary tables in the merged database."""
    print("Creating schema in merged.db...")
    
    # Drop tables if they exist to ensure a fresh start
    cursor.execute("DROP TABLE IF EXISTS player")
    cursor.execute("DROP TABLE IF EXISTS rounds")

    # Create the new player table as per the specification
    cursor.execute("""
    CREATE TABLE player (
        tick INTEGER,
        steamid INTEGER,
        playername TEXT,
        position_x REAL,
        position_y REAL,
        position_z REAL,
        inventory TEXT,
        active_weapon TEXT,
        health INTEGER,
        armor INTEGER,
        money INTEGER,
        keyboard_input TEXT,
        mouse_x REAL,
        mouse_y REAL,
        is_in_buyzone INTEGER,
        buy_sell_input TEXT,
        PRIMARY KEY (tick, steamid)
    )
    """)

    # Create the rounds table (schema will be copied from source)
    cursor.execute("""
    CREATE TABLE rounds (
        round INTEGER PRIMARY KEY,
        starttick INTEGER,
        freezetime_endtick INTEGER,
        endtick INTEGER,
        t_team TEXT,
        ct_team TEXT
    )
    """)
    print("Schema created successfully.")

def load_lookup_data(db_path):
    """
    Loads data from secondary databases into memory for efficient lookups.
    This is much faster than querying the databases inside the main loop.
    """
    print("Loading lookup data into memory...")
    
    # Data structures for quick lookups
    mouse_positions = {}
    buy_sell_drop_actions = {}
    buyzone_presence = set()
    valid_round_ticks = []

    # --- Load Mouse Data ---
    # Key: (tick, player_name), Value: (x, y)
    try:
        with sqlite3.connect(os.path.join(db_path, 'mouse.db')) as conn:
            cursor = conn.cursor()
            for tick, player_name, x, y in cursor.execute("SELECT tick, player_name, x, y FROM MOUSE"):
                mouse_positions[(tick, player_name)] = (x, y)
    except sqlite3.OperationalError:
        print("Warning: mouse.db or MOUSE table not found. Mouse data will be empty.")

    # --- Load Buy/Sell/Drop Data ---
    # Key: (tick, playername), Value: list of (action, item) tuples
    try:
        with sqlite3.connect(os.path.join(db_path, 'buy_sell_drop.db')) as conn:
            cursor = conn.cursor()
            for tick, playername, action, item in cursor.execute("SELECT tick, playername, action, item FROM RAREACTIONS"):
                key = (tick, playername)
                if key not in buy_sell_drop_actions:
                    buy_sell_drop_actions[key] = []
                buy_sell_drop_actions[key].append((action, item))

            # Key: (tick, playername)
            for tick, _, playername in cursor.execute("SELECT tick, steamid, playername FROM BUYZONE"):
                buyzone_presence.add((tick, playername))
    except sqlite3.OperationalError:
        print("Warning: buy_sell_drop.db or its tables not found. Buy/sell/drop/buyzone data will be empty.")

    # --- Load Rounds Data ---
    # A list of (starttick, endtick) tuples
    try:
        with sqlite3.connect(os.path.join(db_path, 'rounds.db')) as conn:
            cursor = conn.cursor()
            # Filter out rounds that might not have an endtick (e.g., live games)
            for starttick, endtick in cursor.execute("SELECT starttick, endtick FROM ROUNDS WHERE starttick IS NOT NULL AND endtick IS NOT NULL"):
                valid_round_ticks.append((starttick, endtick))
    except sqlite3.OperationalError:
        print("Error: Could not load rounds.db. Cannot filter ticks by round. Aborting.")
        sys.exit(1)

    print("Lookup data loaded.")
    return mouse_positions, buy_sell_drop_actions, buyzone_presence, valid_round_ticks

def is_tick_in_valid_round(tick, round_intervals):
    """Checks if a tick falls within any of the valid round start/end times."""
    # This could be optimized further with a binary search if intervals are sorted,
    # but for a few dozen rounds, a linear scan is perfectly fine.
    for start, end in round_intervals:
        if start <= tick <= end:
            return True
    return False

def main(db_dir):
    """Main function to perform the database merge."""
    if not os.path.isdir(db_dir):
        print(f"Error: Directory not found at '{db_dir}'")
        sys.exit(1)

    # --- 1. Load all non-primary data into memory for fast lookups ---
    mouse_data, action_data, buyzone_data, round_intervals = load_lookup_data(db_dir)

    # --- 2. Set up connections ---
    # The "driver" database that dictates which rows exist
    try:
        keyboard_conn = sqlite3.connect(os.path.join(db_dir, 'keyboard_location.db'))
    except sqlite3.OperationalError:
        print(f"Error: Could not open the main driver database keyboard_location.db in '{db_dir}'. Aborting.")
        sys.exit(1)
    
    keyboard_cursor = keyboard_conn.cursor()

    # The new merged database
    merged_conn = sqlite3.connect(os.path.join(db_dir, 'merged.db'))
    merged_cursor = merged_conn.cursor()

    # --- 3. Create the schema in the new database ---
    create_merged_schema(merged_cursor)

    # --- 4. Process the main 'inputs' table and merge data ---
    print("Processing player data from keyboard_location.db...")
    
    player_rows_to_insert = []
    
    # Fetch all rows from the driver table
    keyboard_cursor.execute("""
        SELECT tick, steamid, playername, keyboard_input, inventory, x, y, z, active_weapon, health, armor, money 
        FROM inputs
    """)
    
    all_inputs = keyboard_cursor.fetchall()
    
    for row in tqdm(all_inputs, desc="Merging Player Data"):
        tick, steamid, playername, kb_input, inventory, x, y, z, active_w, health, armor, money = row

        # FILTER 1: Player must be alive
        if health is None or health <= 0:
            continue

        # FILTER 2: Tick must be within a valid round
        if not is_tick_in_valid_round(tick, round_intervals):
            continue

        # --- Data Aggregation and Transformation ---

        # Get mouse data
        mouse_x, mouse_y = mouse_data.get((tick, playername), (None, None))
        
        # Check if in buyzone
        is_in_buyzone = 1 if (tick, playername) in buyzone_data else 0

        # Get actions for this tick and player
        actions = action_data.get((tick, playername), [])
        
        # Build keyboard_input field
        # Start with the base input from the table
        final_kb_inputs = kb_input.split(',') if kb_input else []
        # Add any DROP actions
        for action, item in actions:
            if action == 'DROP':
                # Sanitize item name to be a single token
                safe_item = item.replace(' ', '_').replace('&', 'and')
                final_kb_inputs.append(f"DROP_{safe_item}")

        # Build buy_sell_input field
        buy_sell_actions = []
        for action, item in actions:
            if action in ('BUY', 'SELL'):
                # Sanitize item name
                safe_item = item.replace(' ', '_').replace('&', 'and')
                buy_sell_actions.append(f"{action}_{safe_item}")
        
        # --- Assemble final row for insertion ---
        player_rows_to_insert.append((
            tick,
            steamid,
            playername,
            x, y, z,
            inventory,
            active_w,
            health, armor, money,
            ",".join(final_kb_inputs), # Join lists into a single string
            mouse_x, mouse_y,
            is_in_buyzone,
            ",".join(buy_sell_actions) # Join lists into a single string
        ))

    # --- 5. Batch insert all processed player rows ---
    print(f"Inserting {len(player_rows_to_insert)} rows into 'player' table...")
    if player_rows_to_insert:
        merged_cursor.executemany("""
            INSERT INTO player (tick, steamid, playername, position_x, position_y, position_z, 
                                inventory, active_weapon, health, armor, money, keyboard_input, 
                                mouse_x, mouse_y, is_in_buyzone, buy_sell_input)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, player_rows_to_insert)
    print("Player data inserted.")

    # --- 6. Copy the rounds table directly ---
    print("Copying 'rounds' table...")
    try:
        with sqlite3.connect(os.path.join(db_dir, 'rounds.db')) as rounds_conn:
            rounds_cursor = rounds_conn.cursor()
            all_rounds = rounds_cursor.execute("SELECT round, starttick, freezetime_endtick, endtick, t_team, ct_team FROM ROUNDS").fetchall()
            merged_cursor.executemany("INSERT INTO rounds VALUES (?, ?, ?, ?, ?, ?)", all_rounds)
        print("Rounds data copied.")
    except sqlite3.OperationalError:
        print("Warning: Could not copy rounds.db data.")


    # --- 7. Commit changes and close all connections ---
    print("Finalizing and closing databases...")
    merged_conn.commit()
    merged_conn.close()
    keyboard_conn.close()

    print("\n-----------------------------")
    print(f"Successfully created merged.db in '{db_dir}'")
    print("-----------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merges multiple CS:GO data SQLite databases into a single, unified 'merged.db' file."
    )
    parser.add_argument(
        "db_directory",
        type=str,
        help="The path to the directory containing the .db files (mouse.db, keyboard_location.db, etc.)"
    )
    
    args = parser.parse_args()
    main(args.db_directory)