#!/usr/bin/env python3
import sqlite3
import argparse
import os
import sys

def run_query(conn, query):
    """Executes a query and returns the first column of the first row."""
    try:
        cur = conn.cursor()
        cur.execute(query)
        row = cur.fetchone()
        return row[0] if row else "N/A" # Return 'N/A' if query fails or returns no rows
    except sqlite3.OperationalError as e:
        # This handles cases where a table or column might not exist in one DB
        # print(f"Warning: Query failed - {e}", file=sys.stderr)
        return "Query Failed"

def main():
    parser = argparse.ArgumentParser(
        description="Run multiple SELECTs on two SQLite DBs and display side-by-side results."
    )
    parser.add_argument("db1", help="First .db file (e.g., the old merged.db)")
    parser.add_argument("db2", help="Second .db file (e.g., the new one from the fixed script)")
    args = parser.parse_args()

    db_paths = [args.db1, args.db2]
    db_names = [os.path.basename(p) for p in db_paths]

    # === EXPANDED QUERIES FOR DETAILED COMPARISON ===
    # Each query is a tuple: (Description, SQL Query)
    queries_with_desc = [
        # --- General Sanity Checks ---
        ("Total Player Ticks", "SELECT COUNT(1) FROM player"),
        ("Unique Players", "SELECT COUNT(DISTINCT steamid) FROM player"),
        ("Max Tick", "SELECT MAX(tick) FROM player"),
        ("Avg Health (Alive)", "SELECT AVG(health) FROM player WHERE health > 0"),
        ("Avg Money (Alive)", "SELECT AVG(money) FROM player WHERE health > 0"),

        # --- Buy/Sell/Drop Actions ---
        ("Total SELL Actions", "SELECT COUNT(1) FROM player WHERE buy_sell_input LIKE '%SELL%'"),
        ("Total BUY Actions", "SELECT COUNT(1) FROM player WHERE buy_sell_input LIKE '%BUY%'"),
        ("BUYs per 1000 Ticks", "SELECT (COUNT(1) * 1000.0) / (SELECT COUNT(1) FROM player) FROM player WHERE buy_sell_input LIKE '%BUY%'"),
        ("Total Drop Actions", "SELECT COUNT(1) FROM player WHERE keyboard_input LIKE '%DROP_%'"),
        ("Non-Grenade Drops", "SELECT COUNT(1) FROM player WHERE keyboard_input LIKE '%DROP_%' AND keyboard_input NOT LIKE '%DROP_hegrenade%' AND keyboard_input NOT LIKE '%DROP_flashbang%' AND keyboard_input NOT LIKE '%DROP_smokegrenade%' AND keyboard_input NOT LIKE '%DROP_molotov%' AND keyboard_input NOT LIKE '%DROP_incgrenade%' AND keyboard_input NOT LIKE '%DROP_decoy%'"),
        ("HE Grenade Drops", "SELECT COUNT(1) FROM player WHERE keyboard_input LIKE '%DROP_hegrenade%'"),
        ("Flashbang Drops", "SELECT COUNT(1) FROM player WHERE keyboard_input LIKE '%DROP_flashbang%'"),
        
        # --- Keyboard & Movement ---
        ("Total Weapon Switches", "SELECT COUNT(1) FROM player WHERE keyboard_input LIKE '%SWITCH_%'"),
        ("Total Jumps", "SELECT COUNT(1) FROM player WHERE keyboard_input LIKE '%IN_JUMP%'"),
        ("Total Ducks", "SELECT COUNT(1) FROM player WHERE keyboard_input LIKE '%IN_DUCK%'"),
        ("Ticks in Buyzone", "SELECT COUNT(1) FROM player WHERE is_in_buyzone = 1"),

        # --- Mouse Input Integrity ---
        ("Ticks with Mouse Input", "SELECT COUNT(1) FROM player WHERE mouse_x IS NOT NULL AND mouse_y IS NOT NULL"),
        ("Ticks with NULL Mouse", "SELECT COUNT(1) FROM player WHERE mouse_x IS NULL OR mouse_y IS NULL"),

        # --- Rounds Table Comparison ---
        ("Total Rounds", "SELECT COUNT(1) FROM rounds"),
        ("Avg Round Duration (s)", "SELECT AVG(endtick - starttick) / 128.0 FROM rounds"), # Assumes 128 tick for simplicity
        ("Total Deaths (from T)", "SELECT SUM(json_array_length(json_extract(t_team, '$'))) - SUM(CASE WHEN value = -1 THEN 1 ELSE 0 END) FROM rounds, json_each(json_extract(t_team, '$', '$[1]'))"),
        ("Total Deaths (from CT)", "SELECT SUM(json_array_length(json_extract(ct_team, '$'))) - SUM(CASE WHEN value = -1 THEN 1 ELSE 0 END) FROM rounds, json_each(json_extract(ct_team, '$', '$[1]'))"),
    ]
    
    # Connect to databases
    conns = []
    for p in db_paths:
        if not os.path.exists(p):
            print(f"Error: Database file not found at '{p}'", file=sys.stderr)
            sys.exit(1)
        # Use read-only mode to be safe
        conns.append(sqlite3.connect(f"file:{p}?mode=ro", uri=True))

    # Gather results: list of (description, query, result_db1, result_db2)
    results = []
    print("Running queries...")
    for desc, q in queries_with_desc:
        vals = [run_query(conn, q) for conn in conns]
        results.append((desc, q, *vals))

    # Close connections
    for conn in conns:
        conn.close()

    # --- Display Results ---
    print("\n--- Database Comparison Results ---")

    # Compute column widths
    desc_width = max(len("Metric"), *(len(r[0]) for r in results))
    db_widths = []
    for i, name in enumerate(db_names):
        # Check max length of the result strings for this DB column
        max_val_len = max(len(str(row[i+2])) for row in results)
        db_widths.append(max(len(name), max_val_len))

    # Print header
    header = f"{'Metric'.ljust(desc_width)} | " + " | ".join(
        name.rjust(db_widths[i]) for i, name in enumerate(db_names)
    )
    sep = "-" * desc_width + "-+-" + "-+-".join("-" * w for w in db_widths)
    print(header)
    print(sep)

    # Print each row
    for desc, q, *vals in results:
        # Format numerical results for better readability
        formatted_vals = []
        for v in vals:
            if isinstance(v, (int, float)):
                if abs(v) > 1000:
                    formatted_vals.append(f"{v:,.0f}")
                else:
                    formatted_vals.append(f"{v:,.2f}")
            else:
                formatted_vals.append(str(v))

        line = f"{desc.ljust(desc_width)} | " + " | ".join(
            formatted_vals[i].rjust(db_widths[i]) for i in range(len(vals))
        )
        print(line)

if __name__ == "__main__":
    main()