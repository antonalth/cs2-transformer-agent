import sqlite3
import sys
import os
import json

def create_merged_db_structure(cursor):
    """Creates the new tables in the merged database."""
    print("Creating 'player' and 'rounds' tables in merged.db...")
    # Drop tables if they exist to ensure a fresh start
    cursor.execute("DROP TABLE IF EXISTS player")
    cursor.execute("DROP TABLE IF EXISTS rounds")

    # Create the new player table as per the specification
    cursor.execute("""
    CREATE TABLE player (
        tick INTEGER,
        steamid INTEGER,
        playername TEXT,
        position TEXT,
        inventory TEXT,
        active_weapon TEXT,
        health INTEGER,
        armor INTEGER,
        money INTEGER,
        keyboard_input TEXT,
        mouse_input TEXT,
        is_in_buyzone INTEGER, -- Using INTEGER for boolean (0 or 1)
        buy_sell_input TEXT
    )
    """)

    # Create the rounds table
    cursor.execute("""
    CREATE TABLE rounds (
        round INTEGER,
        starttick INTEGER,
        freezetime_endtick INTEGER,
        endtick INTEGER,
        t_team TEXT,
        ct_team TEXT
    )
    """)
    print("Table structure created successfully.")

def copy_rounds_data(dir_path, merged_cursor):
    """Copies all data from the original rounds.db into the new merged.db."""
    print("Copying data from rounds.db...")
    source_rounds_db_path = os.path.join(dir_path, 'rounds.db')
    if not os.path.exists(source_rounds_db_path):
        print(f"Warning: {source_rounds_db_path} not found. 'rounds' table will be empty.")
        return

    source_conn = sqlite3.connect(source_rounds_db_path)
    source_cursor = source_conn.cursor()

    source_cursor.execute("SELECT round, starttick, freezetime_endtick, endtick, t_team, ct_team FROM ROUNDS")
    all_rounds = source_cursor.fetchall()

    if all_rounds:
        merged_cursor.executemany("INSERT INTO rounds VALUES (?, ?, ?, ?, ?, ?)", all_rounds)
    
    source_conn.close()
    print(f"Copied {len(all_rounds)} rows to 'rounds' table.")

def pre_load_lookup_data(dir_path):
    """
    Pre-loads data from smaller tables into dictionaries for fast lookups.
    This is much more performant than querying the DB for every single row.
    """
    print("Pre-loading lookup data from mouse.db and buy_sell_drop.db...")
    mouse_data = {}
    actions_data = {}
    buyzone_data = set() # A set for fast 'in' checks

    # Load Mouse Data: {(tick, player_name): (x, y)}
    mouse_db_path = os.path.join(dir_path, 'mouse.db')
    if os.path.exists(mouse_db_path):
        conn = sqlite3.connect(mouse_db_path)
        cur = conn.cursor()
        for row in cur.execute("SELECT tick, player_name, x, y FROM MOUSE"):
            tick, player_name, x, y = row
            mouse_data[(tick, player_name)] = (x, y)
        conn.close()

    # Load Buy/Sell/Drop and Buyzone Data
    bsd_db_path = os.path.join(dir_path, 'buy_sell_drop.db')
    if os.path.exists(bsd_db_path):
        conn = sqlite3.connect(bsd_db_path)
        cur = conn.cursor()
        # Load Actions: {(tick, steamid): [(action, item), ...]}
        for row in cur.execute("SELECT tick, steamid, action, item FROM RAREACTIONS"):
            tick, steamid, action, item = row
            key = (tick, int(steamid)) # Ensure steamid is int for matching
            if key not in actions_data:
                actions_data[key] = []
            actions_data[key].append((action, item))

        # Load Buyzone Entries: {(tick, steamid)}
        for row in cur.execute("SELECT tick, steamid FROM BUYZONE"):
            tick, steamid = row
            buyzone_data.add((tick, int(steamid))) # Ensure steamid is int
        conn.close()
    
    print("Pre-loading complete.")
    return mouse_data, actions_data, buyzone_data


def merge_player_data(dir_path, merged_conn):
    """
    The main function to process keyboard_location data and merge other sources.
    """
    merged_cursor = merged_conn.cursor()
    
    # Pre-load data for efficiency
    mouse_data, actions_data, buyzone_data = pre_load_lookup_data(dir_path)

    # The 'keyboard_location.db' is the main driver
    kl_db_path = os.path.join(dir_path, 'keyboard_location.db')
    if not os.path.exists(kl_db_path):
        print(f"ERROR: The main driver file {kl_db_path} was not found. Cannot proceed.")
        return

    print("Processing main data from keyboard_location.db and merging...")
    kl_conn = sqlite3.connect(kl_db_path)
    # Use a dictionary-based row factory for easier access by column name
    kl_conn.row_factory = sqlite3.Row
    kl_cursor = kl_conn.cursor()

    # Start a transaction for much faster inserts
    merged_cursor.execute("BEGIN TRANSACTION")

    # Query all rows from the main 'inputs' table
    kl_cursor.execute("SELECT * FROM inputs ORDER BY tick")
    
    processed_rows = 0
    inserted_rows = 0
    while True:
        rows = kl_cursor.fetchmany(10000) # Process in chunks to manage memory
        if not rows:
            break

        player_rows_to_insert = []
        for row in rows:
            processed_rows += 1
            if processed_rows % 50000 == 0:
                print(f"  ... processed {processed_rows} source rows ...")

            # --- Rule: Skip players with 0 health ---
            if row['health'] == 0:
                continue

            # Extract base data
            tick = row['tick']
            steamid = row['steamid']
            playername = row['playername']

            # --- Prepare merged columns ---

            # Position
            position_str = f"{row['x']},{row['y']},{row['z']}"

            # Actions lookup key
            action_key = (tick, steamid)
            player_actions = actions_data.get(action_key, [])

            # Keyboard Input (base + DROP actions)
            keyboard_inputs = row['keyboard_input'].split(',') if row['keyboard_input'] else []
            for action, item in player_actions:
                if action == 'DROP':
                    # Sanitize item name just in case it contains commas
                    sanitized_item = item.replace(',', ';')
                    keyboard_inputs.append(f"DROP_{sanitized_item}")
            keyboard_input_str = ",".join(filter(None, keyboard_inputs))

            # Buy/Sell Input
            buy_sell_list = []
            for action, item in player_actions:
                if action in ('BUY', 'SELL'):
                    sanitized_item = item.replace(',', ';')
                    buy_sell_list.append(f"{action}_{sanitized_item}")
            buy_sell_input_str = ",".join(buy_sell_list)

            # Mouse Input
            mouse_pos = mouse_data.get((tick, playername))
            mouse_input_str = f"{mouse_pos[0]},{mouse_pos[1]}" if mouse_pos else ""

            # Is in Buyzone
            is_in_buyzone = 1 if action_key in buyzone_data else 0

            # Assemble the final row for insertion
            final_row = (
                tick,
                steamid,
                playername,
                position_str,
                row['inventory'],
                row['active_weapon'],
                row['health'],
                row['armor'],
                row['money'],
                keyboard_input_str,
                mouse_input_str,
                is_in_buyzone,
                buy_sell_input_str
            )
            player_rows_to_insert.append(final_row)
        
        if player_rows_to_insert:
            merged_cursor.executemany("""
                INSERT INTO player (tick, steamid, playername, position, inventory, active_weapon, health, armor, money, keyboard_input, mouse_input, is_in_buyzone, buy_sell_input) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, player_rows_to_insert)
            inserted_rows += len(player_rows_to_insert)
    
    # Commit the transaction
    merged_conn.commit()
    kl_conn.close()

    print("Merging complete.")
    print(f"Total rows processed from source: {processed_rows}")
    print(f"Total rows inserted into 'player' table: {inserted_rows}")


def main():
    """Main execution function."""
    if len(sys.argv) != 2:
        print("Usage: python extractstruct.py <directory_path_to_db_files>")
        sys.exit(1)

    dir_path = sys.argv[1]
    if not os.path.isdir(dir_path):
        print(f"Error: Directory not found at '{dir_path}'")
        sys.exit(1)

    print(f"Processing database files in: {dir_path}")

    # Define the path for the new merged database
    merged_db_path = os.path.join(dir_path, 'merged.db')

    # Connect to the new database (it will be created if it doesn't exist)
    with sqlite3.connect(merged_db_path) as merged_conn:
        merged_cursor = merged_conn.cursor()

        # 1. Create the table structure in the new DB
        create_merged_db_structure(merged_cursor)

        # 2. Copy the rounds data directly
        copy_rounds_data(dir_path, merged_cursor)
        
        # Commit the structure and rounds data
        merged_conn.commit()

        # 3. Process and merge all player-related data
        merge_player_data(dir_path, merged_conn)

    print(f"\n✅ Success! All data has been merged into '{merged_db_path}'")

if __name__ == "__main__":
    main()