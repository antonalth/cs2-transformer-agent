#!/usr/bin/env python3
'''
- Parse demo,
- extract all player inputs/tick; table name PLAYERINPUT
- round start - round stop;
- player recording timeframes (round#, starttick, endtick, isdeath?)
    - separate tables: playerinput, info for player pov recording (round#, starttick ,endtick, isdeath, (video_path, audio_path), (team)), separate table auxiliary information (money, roundcount)

'''

from __future__ import annotations
import argparse, sys, sqlite3, json
from pathlib import Path
import pandas as pd
from demoparser2 import DemoParser

# ── helper: decode "buttons" bit-field ────────────────────────────────────────
KEY_MAPPING = {
    "IN_ATTACK":    1 << 0,
    "IN_JUMP":      1 << 1,
    "IN_DUCK":      1 << 2,
    "IN_FORWARD":   1 << 3,
    "IN_BACK":      1 << 4,
    "IN_USE":       1 << 5,
    "IN_CANCEL":    1 << 6,
    "IN_TURNLEFT":  1 << 7,
    "IN_TURNRIGHT": 1 << 8,
    "IN_MOVELEFT":  1 << 9,
    "IN_MOVERIGHT": 1 << 10,
    "IN_ATTACK2":   1 << 11,
    "IN_RELOAD":    1 << 13,
    "IN_ALT1":      1 << 14,
    "IN_ALT2":      1 << 15,
    "IN_SPEED":     1 << 16,
    "IN_WALK":      1 << 17,
    "IN_ZOOM":      1 << 18,
    "IN_WEAPON1":   1 << 19,
    "IN_WEAPON2":   1 << 20,
    "IN_BULLRUSH":  1 << 21,
    "IN_GRENADE1":  1 << 22,
    "IN_GRENADE2":  1 << 23,
    "IN_ATTACK3":   1 << 24,
    "UNKNOWN_25":   1 << 25,
    "UNKNOWN_26":   1 << 26,
    "UNKNOWN_27":   1 << 27,
    "UNKNOWN_28":   1 << 28,
    "UNKNOWN_29":   1 << 29,
    "UNKNOWN_30":   1 << 30,
    "UNKNOWN_31":   1 << 31,
    "IN_SCORE":     1 << 33,
    "IN_INSPECT":   1 << 35,
}

def extract_buttons(bits) -> list[str]:
    """
    Given a bit-field (possibly passed as a JS decimal string),
    return the list of all key names currently held.
    """
    bits_int = int(bits)
    return [
        name
        for name, mask in KEY_MAPPING.items()
        if bits_int & mask
    ]

# ── helpers: SQLite export ────────────────────────────────────────────────────
def _sanitize_inventory(inv):
    """Convert *inv* to a JSON string SQLite can store (lists/tuples → JSON)."""
    if isinstance(inv, (list, tuple, set)):
        try:
            return json.dumps(list(inv))
        except TypeError:
            return ",".join(map(str, inv))
    return str(inv)

# Modified _export_sqlite function
def _export_sqlite(db_path: Path, tick_df: pd.DataFrame) -> None:
    """Write combined *tick_df* info into an SQLite DB at *db_path*."""
    df = tick_df.copy()

    # readable keyboard input
    df["keyboard_input"] = df["buttons"].apply(lambda b: ",".join(extract_buttons(int(b))))

    # stringify inventory lists
    df["inventory"] = df["inventory"].apply(_sanitize_inventory)

    # final shape - added 'active_weapon_name'
    df.rename(columns={"name": "playername"}, inplace=True)
    df = df[["tick", "steamid", "playername", "keyboard_input", "inventory", "X", "Y", "Z", "active_weapon_name"]] # ADDED active_weapon_name here

    # write to DB
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS inputs (
            tick INTEGER NOT NULL,
            steamid INTEGER NOT NULL,
            playername TEXT,
            keyboard_input TEXT,
            inventory TEXT,
            x REAL,
            y REAL,
            z REAL,
            active_weapon TEXT, -- ADDED active_weapon column
            PRIMARY KEY (tick, steamid)
        )
        """
    )
    # Updated column list for INSERT OR REPLACE INTO - now 9 columns
    cur.executemany(
        "INSERT OR REPLACE INTO inputs VALUES (?,?,?,?,?,?,?,?,?)", # ADDED another '?' for active_weapon
        df.to_records(index=False).tolist(),
    )
    con.commit()
    con.close()

# ── CLI setup ────────────────────────────────────────────────────────────────
def cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tiny CS-2 demo inspector / exporter")
    p.add_argument("demofile", type=Path, help="Path to .dem file")
    p.add_argument("--sqlout", type=Path, metavar="FILE.db",
                   help="Write results to SQLite DB instead of opening the CLI")
    return p

# ── main logic ───────────────────────────────────────────────────────────────
def main() -> None:
    args = cli().parse_args()
    if not args.demofile.exists():
        sys.exit(f"Demo not found: {args.demofile}")

    print("[loading demo – one-off parse, please wait …]")
    dp = DemoParser(args.demofile.as_posix())

    # 1) per-tick table - Updated wanted_props
    tick_df = (
        dp.parse_ticks(
            wanted_props=[
                "tick", "steamid", "userid", "name",
                "buttons", "inventory",
                "X", "Y", "Z", # Added player x, y, z coordinates
                "active_weapon_name", #"is_defusing", # Retained from original parsedemo.py
                # Removed: "team_num", "total_rounds_played", "pitch", "yaw",
                # "usercmd_mouse_dx", "usercmd_mouse_dy", "aim_punch_angle"
            ]
        ).pipe(pd.DataFrame)
    )

    # 2) gap-fill so every (tick,steamid) exists
    tick_df.sort_values(["steamid", "tick"], inplace=True)
    tick_df = (
        tick_df
        .set_index("tick")
        .groupby("steamid", group_keys=False)
        .apply(lambda g: g.reindex(range(g.index.min(), g.index.max() + 1)).ffill())
        .reset_index()
    )

    # Removed: 3) mouse deltas calculation (d_yaw, d_pitch)
    # Removed: 4) purchases parsing (get_buys function, raw_buys, buys DFs)

    # ── SQLite export path ────────────────────────────────────────────────────
    if args.sqlout:
        # Calls _export_sqlite without the 'buys' DataFrame
        _export_sqlite(args.sqlout, tick_df)
        print(f"✓ wrote {args.sqlout} – done")
        return

    # 5) tiny REPL - Updated for new features and removed old ones
    print("\nCommands:\n  seek <tick>   jump to that tick\n  quit / exit   leave\n")
    while True:
        try:
            cmd = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if cmd in {"quit", "exit"}:
            break
        if not cmd.startswith("seek"):
            print("Unknown command. Use 'seek <tick>' or 'quit'.")
            continue

        parts = cmd.split(maxsplit=1)
        if len(parts) != 2 or not parts[1].isdigit():
            print("Usage: seek <tick_number>")
            continue
        tgt_tick = int(parts[1])

        rows = tick_df.query("tick == @tgt_tick")
        if rows.empty:
            print(f"No data at tick {tgt_tick}.")
            continue

        players = rows[["steamid", "name"]].drop_duplicates().reset_index(drop=True)
        for i, r in players.iterrows():
            print(f"{i+1}. {r['name']}  (sid {r['steamid']})")

        sel = input("Player # > ").strip()
        if not sel.isdigit() or not (1 <= int(sel) <= len(players)):
            print("Invalid choice.")
            continue
        sid = players.iloc[int(sel)-1]["steamid"]

        r = rows.loc[rows["steamid"] == sid].iloc[-1]
        keys = extract_buttons(int(r["buttons"]))
        print(
            f"\nTick {tgt_tick} – {r['name']}:\n"
            f"  Keys held : {', '.join(keys) or '(none)'}\n"
            f"  Position  : x={r['X']:.2f}, y={r['Y']:.2f}, z={r['Z']:.2f}\n" # Added position
            f"  Inventory : {_sanitize_inventory(r['inventory'])}\n"
            f"  Active Weapon: {r['active_weapon_name']}"
        )
        # Removed: Mouse Δ display
        # Removed: Bought display
        print()

if __name__ == "__main__":
    main()