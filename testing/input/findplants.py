#!/usr/bin/env python3
"""
demo_cli.py – inspect keys / mouse Δ / purchases at any tick in a CS-2 demo

Prompt commands
---------------
seek <tick>   jump to that server tick, pick a player, view inputs & buys
quit | exit   leave

New option (2025-06-08)
----------------------
--sqlout <file.db>    parse demo and write a SQLite DB instead of opening the CLI
"""

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

def _export_sqlite(db_path: Path, tick_df: pd.DataFrame, buys: pd.DataFrame) -> None:
    """Write combined *tick_df* + *buys* info into an SQLite DB at *db_path*."""
    df = tick_df.copy()

    # readable keyboard input
    df["keyboard_input"] = df["buttons"].apply(lambda b: ",".join(extract_buttons(b)))

    # mouse delta as "dYaw,dPitch" string
    df["mouse"] = df.apply(lambda r: f"{r['d_yaw']:+.5f},{r['d_pitch']:+.5f}", axis=1)

    # stringify inventory lists
    df["inventory"] = df["inventory"].apply(_sanitize_inventory)

    # purchases per (tick, steamid)
    buy_col = next((c for c in ("weapon", "item", "weapon_name") if c in buys.columns), None)
    if buy_col and "steamid" in buys.columns and not buys.empty:
        buys_agg = (
            buys
            .dropna(subset=["steamid"])[["tick", "steamid", buy_col]]
            .groupby(["tick", "steamid"], as_index=False)
            .agg({buy_col: lambda s: ",".join(s.astype(str))})
            .rename(columns={buy_col: "buy"})
        )
        df = df.merge(buys_agg, on=["tick", "steamid"], how="left")
    else:
        df["buy"] = ""
    df["buy"] = df["buy"].fillna("")

    # final shape
    df.rename(columns={"name": "playername"}, inplace=True)
    df = df[["tick", "steamid", "playername", "keyboard_input", "inventory", "mouse", "buy"]]

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
            mouse TEXT,
            buy TEXT,
            PRIMARY KEY (tick, steamid)
        )
        """
    )
    cur.executemany(
        "INSERT OR REPLACE INTO inputs VALUES (?,?,?,?,?,?,?)",
        df.to_records(index=False).tolist(),
    )
    con.commit()
    con.close()

# ── CLI setup ───────────────────────────────────────────────────────────────
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

    # 1) per-tick table
    tick_df = (
        dp.parse_ticks(
            wanted_props=[
                "tick", "steamid", "userid", "name",
                "buttons", "pitch", "yaw", "inventory",
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

    # 3) mouse deltas
    tick_df["d_yaw"]   = tick_df.groupby("steamid")["yaw"].diff().fillna(0)
    tick_df["d_pitch"] = tick_df.groupby("steamid")["pitch"].diff().fillna(0)

    # 4) purchases – trial different API signatures
    def get_buys():
        try:
            return dp.parse_event("item_pickup", ["weapon", "price"], ["tick", "steamid"])
        except TypeError:
            try:
                return dp.parse_event("item_pickup", ["tick", "steamid", "weapon", "price"])
            except TypeError:
                raw = dp.parse_event("item_pickup")
                cols = [c for c in raw.columns if c in {"tick", "steamid", "userid", "weapon", "item", "price"}]
                return raw[cols]

    raw_buys = get_buys().pipe(pd.DataFrame)

    if "steamid" in raw_buys.columns:
        buys = raw_buys
    elif "userid" in raw_buys.columns:
        id_map = tick_df[["userid", "steamid"]].drop_duplicates()
        buys = raw_buys.merge(id_map, on="userid", how="left")
    else:
        print("Warning: no steam / user id columns in item_pickup events; purchase list will be empty.")
        buys = raw_buys.assign(steamid=pd.NA)

    # 5) output or REPL
    if args.sqlout:
        _export_sqlite(args.sqlout, tick_df, buys)
        print(f"✓ wrote {args.sqlout} – done")
        return

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
        keys = extract_buttons(r["buttons"])
        print(
            f"\nTick {tgt_tick} – {r['name']}:\n"
            f"  Keys held : {', '.join(keys) or '(none)'}\n"
            f"  Mouse Δ   : yaw {r['d_yaw']:+.2f}°, pitch {r['d_pitch']:+.2f}°\n"
            f"  Inventory : {_sanitize_inventory(r['inventory'])}"
        )

        b = buys.query("tick == @tgt_tick and steamid == @sid")
        buy_col = next((c for c in ("weapon", "item", "weapon_name") if c in b.columns), None)
        if not b.empty and buy_col:
            for _, ev in b.iterrows():
                price = ev.get("price", "??")
                print(f"  Bought    : {ev[buy_col]}  (-${price})")
        else:
            print("  Bought    : (nothing this tick)")
        print()

if __name__ == "__main__":
    main()
