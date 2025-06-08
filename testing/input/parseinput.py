#!/usr/bin/env python3
"""
demo_cli.py – jump to any tick in a CS-2 demo and inspect a player’s inputs
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
from demoparser2 import DemoParser
from examples.buttons_bitfield.main import extract_buttons   # same helper as before

def build_cli() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="CS-2 demo inspector")
    cli.add_argument("demofile", type=Path, help="*.dem to open")
    return cli

def main() -> None:
    args = build_cli().parse_args()
    if not args.demofile.exists():
        sys.exit("demo not found")

    print("[loading demo – this takes a few seconds …]")
    dp = DemoParser(args.demofile.as_posix())

    tick_df = (
        dp.parse_ticks(                          # <-- only the column name changed
            wanted_props=[
                "tick", "steamid", "name",
                "buttons", "pitch", "yaw", "inventory",
            ]
        )
        .pipe(pd.DataFrame)
    )

    # forward-fill sparse ticks so every (tick,steamid) exists once
    tick_df.sort_values(["steamid", "tick"], inplace=True)
    tick_df = (
        tick_df.set_index("tick")
        .groupby("steamid")
        .apply(lambda g: g.reindex(
            range(g.index.min(), g.index.max() + 1)
        ).ffill())
        .reset_index(level=0, drop=True)
        .reset_index()
    )

    tick_df["d_yaw"]   = tick_df.groupby("steamid")["yaw"].diff().fillna(0)
    tick_df["d_pitch"] = tick_df.groupby("steamid")["pitch"].diff().fillna(0)

    buys = dp.parse_event("item_pickup")   # default columns include tick & steamid

    # ── tiny REPL ──────────────────────────────────────────────────────────
    print("\nType  seek <tick>  to jump,  quit  to exit.\n")
    while True:
        try:
            cmd = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if cmd in {"quit", "exit"}:
            break
        if cmd.startswith("seek"):
            try:
                tgt_tick = int(cmd.split()[1])
            except (IndexError, ValueError):
                print("Usage: seek <tick>")
                continue

            rows = tick_df.query("tick == @tgt_tick")
            if rows.empty:
                print("No data at that tick.")
                continue

            players = rows[["steamid", "name"]].drop_duplicates().reset_index(drop=True)
            for i, r in players.iterrows():
                print(f"{i+1}. {r['name']}  (sid {r['steamid']})")

            try:
                sel = int(input("Player # > ")) - 1
                sid = players.iloc[sel]["steamid"]
            except Exception:
                print("Invalid choice.");  continue

            r = rows.loc[rows["steamid"] == sid].iloc[-1]
            print(f"\nTick {tgt_tick} – {r['name']}:")
            print(f"  Keys held : {', '.join(extract_buttons(int(r['buttons']))) or '(none)'}")
            print(f"  Mouse Δ   : yaw {r['d_yaw']:+.2f}°, pitch {r['d_pitch']:+.2f}°")
            print(f"  Inventory : {r['inventory']}")

            b = buys.query("tick == @tgt_tick and steamid == @sid")
            if not b.empty:
                for _, ev in b.iterrows():
                    print(f"  Bought    : {ev['weapon']}  (-${ev['price']})")
            else:
                print("  Bought    : nothing")
            print()
        else:
            print("Unknown command.")

if __name__ == "__main__":
    main()
