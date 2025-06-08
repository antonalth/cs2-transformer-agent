#!/usr/bin/env python3
"""
demo_cli.py  –  tiny interactive inspector for CS-2 demos

Usage:
    python demo_cli.py path/to/match.dem

Prompt commands:
    seek <tick>    – jump to an absolute server tick
    exit / quit    – leave
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd
from demoparser2 import DemoParser

# ------------------------------------------------------------------ helpers
try:
    # Works when run inside the demoparser checkout
    from examples.buttons_bitfield.main import extract_buttons
except ModuleNotFoundError:
    # Fallback: brutally decode the 64-bit field on our own
    from types import MappingProxyType
    _KEYS = MappingProxyType({
        0: "IN_ATTACK", 1: "IN_JUMP", 2: "IN_DUCK", 3: "IN_FORWARD",
        4: "IN_BACK", 5: "IN_USE", 6: "IN_CANCEL", 7: "IN_LEFT",
        8: "IN_RIGHT", 9: "IN_MOVELEFT", 10: "IN_MOVERIGHT",
        11: "IN_ATTACK2", 12: "IN_RUN", 13: "IN_RELOAD",
        16: "IN_SPEED", 17: "IN_WALK", 18: "IN_ZOOM",
        # …extend if you need more bits…
    })
    def extract_buttons(bits: int) -> list[str]:
        return [name for bit, name in _KEYS.items() if bits & (1 << bit)]

# ------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    cli.add_argument("demofile", type=Path, help="*.dem file to inspect")
    return cli


def main() -> None:
    args = build_parser().parse_args()
    if not args.demofile.exists():
        sys.exit("demo not found")

    print("[loading demo – this takes a few seconds …]")
    dp = DemoParser(args.demofile.as_posix())
    tick_df = dp.parse_ticks(
        wanted_props=[
            "tick", "steam_id", "name",          # identifiers
            "buttons", "pitch", "yaw", "inventory"
        ]
    ).pipe(pd.DataFrame)            # demoparser returns Polars; to Pandas

    # Fill gaps so every tick has a state row per player
    tick_df.sort_values(["steam_id", "tick"], inplace=True)
    tick_df = (
        tick_df
        .set_index("tick")
        .groupby("steam_id")
        .apply(lambda g: g.reindex(
            range(g.index.min(), g.index.max() + 1)
        ).ffill())
        .reset_index(level=0, drop=True)        # keep tick as index only
        .reset_index()
    )

    # Pre-compute mouse deltas
    tick_df["d_yaw"]   = tick_df.groupby("steam_id")["yaw"].diff().fillna(0)
    tick_df["d_pitch"] = tick_df.groupby("steam_id")["pitch"].diff().fillna(0)

    # Event table for purchases
    buys = dp.parse_event(
        "item_pickup",
        event_props=["weapon", "price"],
        wanted_props=["tick", "steam_id"]
    ).pipe(pd.DataFrame)

    # --------------- REPL ---------------
    current_tick: int | None = None
    print("\nType  seek <tick>  to jump,  quit  to exit.\n")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line in {"quit", "exit"}:
            break
        if line.startswith("seek"):
            try:
                current_tick = int(line.split()[1])
            except (IndexError, ValueError):
                print("Usage: seek <tick_number>")
                continue

            rows = tick_df.query("tick == @current_tick")
            if rows.empty:
                print("No data at this exact tick – try a nearby value.")
                continue

            players = (
                rows[["steam_id", "name"]]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            for idx, r in players.iterrows():
                print(f"{idx+1}. {r['name']} (sid {r['steam_id']})")
            sel = input("Player # to inspect > ").strip()
            try:
                sel_idx = int(sel) - 1
                steam_id = players.iloc[sel_idx]["steam_id"]
            except Exception:
                print("Invalid selection.")
                continue

            p_row = rows.loc[rows["steam_id"] == steam_id].iloc[-1]
            keys = extract_buttons(int(p_row["buttons"]))
            dx, dy = p_row["d_yaw"], p_row["d_pitch"]
            inv = p_row["inventory"]

            print(
                f"\nTick {current_tick} – {p_row['name']}:\n"
                f"  Keys held : {', '.join(keys) or '(none)'}\n"
                f"  Mouse Δ   : yaw {dx:+.2f}°, pitch {dy:+.2f}°\n"
                f"  Inventory : {inv}\n"
            )

            b = buys.query("tick == @current_tick and steam_id == @steam_id")
            if not b.empty:
                for _, ev in b.iterrows():
                    print(f"  Bought    : {ev['weapon']}  (-${ev['price']})")
            else:
                print("  Bought    : (nothing this tick)")
            print()  # blank line
            continue

        print("Unknown command.  Type  seek <tick>  or  quit.")

if __name__ == "__main__":
    main()
