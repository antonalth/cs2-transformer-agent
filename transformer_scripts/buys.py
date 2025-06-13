#!/usr/bin/env python3
import argparse
import sqlite3
from demoparser2 import DemoParser

def main():
    p = argparse.ArgumentParser(description="List all weapon buys from a CS:GO demo")
    p.add_argument("demo_path", help="Path to the .dem file")
    p.add_argument(
        "--sqlout",
        metavar="filename.db",
        help="Write output to this SQLite DB (table BUYS) instead of printing"
    )
    args = p.parse_args()

    # load the demo
    dp = DemoParser(args.demo_path)

    # parse only the "item_purchase" events, and include the player 'name'
    events = dp.parse_events(
        ["item_purchase"],
        other=["name"]         # pull the friendly player‐name into the DataFrame
    )
    _, buys_df = events[0]

    if args.sqlout:
        # connect to (or create) the SQLite database
        conn = sqlite3.connect(args.sqlout)
        cur = conn.cursor()
        # ensure the BUYS table exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS BUYS (
                tick INTEGER,
                name TEXT,
                bought_item TEXT
            )
        """)

        # insert each buy into the database
        for _, row in buys_df.iterrows():
            tick        = int(row["tick"])
            name        = row["name"]
            bought_item = row["item_name"]
            cur.execute(
                "INSERT INTO BUYS (tick, name, bought_item) VALUES (?, ?, ?)",
                (tick, name, bought_item)
            )

        conn.commit()
        conn.close()
        print(f"Wrote {len(buys_df)} rows to {args.sqlout}")
    else:
        # print header
        print(f"{'Player':<20}  {'Weapon':<15}  {'Tick'}")
        print("-" * 50)

        # iterate and dump
        for _, row in buys_df.iterrows():
            player = row["name"]               # now this is e.g. "brnz4n"
            weapon = row["item_name"]          # e.g. "weapon_ak47"
            tick   = int(row["tick"])
            print(f"{player:<20}  {weapon:<15}  {tick}")

if __name__ == "__main__":
    main()
