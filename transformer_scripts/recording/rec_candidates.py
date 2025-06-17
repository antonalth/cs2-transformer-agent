'''

Database: input.db
  Table 'ROUNDS': round (INTEGER), starttick (INTEGER), freezetime_endtick (INTEGER), endtick (INTEGER), t_team (TEXT), ct_team (TEXT)

    sqlite> select * from rounds limit 5;
1|361|1641|9449|[["BTN", 8481], ["xellow", 4881], ["s0und", 3464], ["ragga", 3373], ["lauNX", -1]]|[["nota", -1], ["BELCHONOKK", 8268], ["Jame", 3029], ["TRAVIS", -1], ["Qikert", 8899]]
2|9449|10729|13999|[["BTN", 13551], ["ragga", 13073], ["s0und", 12905], ["xellow", 13210], ["lauNX", 12614]]|[["nota", -1], ["Jame", -1], ["TRAVIS", -1], ["BELCHONOKK", -1], ["Qikert", -1]]
3|13999|15279|17753|[["lauNX", 16898], ["BTN", 17240], ["s0und", 16713], ["xellow", 16970], ["ragga", 17305]]|[["BELCHONOKK", -1], ["Jame", -1], ["TRAVIS", -1], ["Qikert", -1], ["nota", 16894]]
4|17753|19033|26486|[["xellow", 24535], ["ragga", 26038], ["s0und", 22292], ["BTN", 21982], ["lauNX", 23259]]|[["BELCHONOKK", 23015], ["TRAVIS", 20733], ["Jame", -1], ["Qikert", 24885], ["nota", -1]]
5|26486|27766|35827|[["lauNX", -1], ["BTN", -1], ["ragga", -1], ["xellow", -1], ["s0und", -1]]|[["Jame", 29411], ["TRAVIS", -1], ["BELCHONOKK", -1], ["Qikert", -1], ["nota", 28868]]
sqlite>

#Pseudo-Script:
--sql input.db

if input.db has table RECORDING, ask before contining

For each round: get starttick, endtick, list of players with death tick (-1 if didnt die)
    ignore rounds where t_team or ct_team is not populated eg '' or '[]'
    for each player in list:
        store in to_be_recorded tuple: (roundnumber, starttick, deathtick/endtick if didnt die), playername, team)

create table RECORDING inside of input.db: {
    roundnumber INT,
    starttick INT,
    stoptick INT,
    team TEXT,
    playername TEXT,
    is_recorded bool,
    recording_filepath TEXT,
} //primary key is starttick, stoptick, playername

for each to_be_stored add an entry in RECORDING, is_recorded = FALSE, recording_filepath = NONE.

'''
#!/usr/bin/env python3

import sqlite3
import json
import os
import sys

def check_and_prepare_db(conn, db_path):
    """
    Checks if the 'RECORDING' table exists. If so, it prompts the user for
    permission to drop and recreate it.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='RECORDING';")
    if cursor.fetchone():
        print(f"Warning: Table 'RECORDING' already exists in '{db_path}'.")
        while True:
            choice = input("Do you want to drop it and recreate it? (y/n): ").lower().strip()
            if choice == 'y':
                print("Dropping existing 'RECORDING' table...")
                cursor.execute("DROP TABLE RECORDING;")
                conn.commit()
                return True
            elif choice == 'n':
                print("Aborting script as requested.")
                return False
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
    return True

def create_recording_table(conn):
    """
    Creates the 'RECORDING' table with the required schema.
    """
    cursor = conn.cursor()
    print("Creating table 'RECORDING'...")
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
    print("Table 'RECORDING' created successfully.")

def fetch_and_process_rounds(conn, db_path):
    """
    Fetches data from the 'ROUNDS' table, validates each round for a strict
    5v5 structure, and processes only the valid rounds.
    """
    print("Processing and validating data from 'ROUNDS' table...")
    cursor = conn.cursor()
    try:
        # Ensure we select freezetime_endtick
        cursor.execute("SELECT round, starttick, freezetime_endtick, endtick, t_team, ct_team FROM ROUNDS")
    except sqlite3.OperationalError:
        print(f"Error: Table 'ROUNDS' not found in '{db_path}'. Cannot proceed.")
        return None

    all_rounds_data = cursor.fetchall()
    to_be_recorded = []

    # Updated loop signature
    for round_num, starttick, freezetime_endtick, endtick, t_team_json, ct_team_json in all_rounds_data:
        # --- Stage 1: Parse and Basic Checks ---

        # Check for extended freezetime (e.g., from timeouts)
        if freezetime_endtick is not None and starttick is not None:
            if (freezetime_endtick - starttick) > 2000:
                print(f"Skipping round {round_num}: Extended freezetime duration ({freezetime_endtick - starttick} ticks).")
                continue
        
        if starttick is None or endtick is None:
            print(f"Skipping round {round_num}: Missing start or end tick.")
            continue
            
        try:
            t_players = json.loads(t_team_json) if t_team_json else []
            ct_players = json.loads(ct_team_json) if ct_team_json else []
            if not isinstance(t_players, list) or not isinstance(ct_players, list):
                print(f"Skipping round {round_num}: Team data is not a valid JSON list.")
                continue
        except json.JSONDecodeError as e:
            print(f"Skipping round {round_num}: Failed to parse JSON data. Error: {e}")
            continue

        # --- Stage 2: Strict Validation ---
        is_round_valid = True

        # Validation 1: Check for exactly 5 players per team
        if len(t_players) != 5 or len(ct_players) != 5:
            print(f"Skipping round {round_num}: Invalid team sizes. T: {len(t_players)}, CT: {len(ct_players)}. Expected 5v5.")
            is_round_valid = False
        
        # Validation 2: Check for players appearing on both teams
        if is_round_valid:
            t_player_names = {p[0] for p in t_players if isinstance(p, list) and len(p) > 0}
            ct_player_names = {p[0] for p in ct_players if isinstance(p, list) and len(p) > 0}
            
            overlapping_players = t_player_names.intersection(ct_player_names)
            
            if overlapping_players:
                print(f"Skipping round {round_num}: Player(s) {list(overlapping_players)} found on both teams.")
                is_round_valid = False

        # Validation 3: Check for deaths too close to freezetime end
        if is_round_valid and freezetime_endtick is not None:
            # Combine players from both teams for this specific check
            all_round_players_for_check = t_players + ct_players
            for player_data in all_round_players_for_check:
                if not isinstance(player_data, list) or len(player_data) < 2:
                    continue # Skip malformed entries like ["playername"]

                player_name, death_tick = player_data[0], player_data[1]

                # We only care about players who actually died
                if death_tick != -1:
                    if death_tick < (freezetime_endtick + 128):
                        print(f"Skipping round {round_num}: Player '{player_name}' died too early ({death_tick}), less than 128 ticks after freezetime ended ({freezetime_endtick}).")
                        is_round_valid = False
                        break # An early death invalidates the whole round

        # --- Stage 3: Process the round if it's valid ---
        if is_round_valid:
            # Combine all players for processing since the round is confirmed valid
            all_round_players = [(*p, 'T') for p in t_players] + [(*p, 'CT') for p in ct_players]
            
            for player_data in all_round_players:
                # Malformed player data check (e.g., ["playername"] without a tick)
                if len(player_data) != 3: 
                    print(f"Warning: Corrupt player entry {player_data} in otherwise valid round {round_num}. Skipping player.")
                    continue
                
                player_name, death_tick, team_name = player_data
                stop_tick = death_tick if death_tick != -1 else endtick
                record = (round_num, starttick, stop_tick, team_name, player_name)
                to_be_recorded.append(record)

    return to_be_recorded

def insert_records(conn, records):
    """
    Inserts the processed player records into the 'RECORDING' table.
    """
    if not records:
        print("No valid rounds found to generate records.")
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
        print(f"Successfully inserted {cursor.rowcount} records from valid rounds into 'RECORDING'.")
    except sqlite3.IntegrityError as e:
        print(f"Error during insertion: {e}. This should not happen with pre-validated data.")
        conn.rollback()

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} <path_to_database.db>")
        sys.exit(1)

    db_path = sys.argv[1]
    
    if not os.path.exists(db_path):
        print(f"Error: Database file '{db_path}' not found.")
        sys.exit(1)

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        print(f"Successfully connected to '{db_path}'.")

        if not check_and_prepare_db(conn, db_path): sys.exit(0)

        create_recording_table(conn)
        records_to_add = fetch_and_process_rounds(conn, db_path)
        
        if records_to_add is None: sys.exit(1)

        insert_records(conn, records_to_add)
        
        print("\nScript finished successfully.")

    except sqlite3.Error as e:
        print(f"\nA database error occurred: {e}")
        if conn: conn.rollback()
        sys.exit(1)
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main()