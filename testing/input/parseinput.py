#!/usr/bin/env python3
"""
demo_cli.py – jump to any tick in a CS-2 demo and inspect a player’s inputs

Usage:
    python demo_cli.py path/to/match.dem
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd
from demoparser2 import DemoParser

# ------------------------------------------------------------------ helpers

# Valve’s IN_* bit constants (extend as needed)
_KEYS = {
    0: "IN_ATTACK",
    1: "IN_JUMP",
    2: "IN_DUCK",
    3: "IN_FORWARD",
    4: "IN_BACK",
    5: "IN_USE",
    6: "IN_CANCEL",
    7: "IN_LEFT",
    8: "IN_RIGHT",
    9: "IN_MOVELEFT",
    10: "IN_MOVERIGHT",
    11: "IN_ATTACK2",
    12: "IN_RUN",
    13: "IN_RELOAD",
    16: "IN_SPEED",
    17: "IN_WALK",
    18: "IN_ZOOM",
    # …add more bits if you need them…
}

def extract_buttons(bits: int) -> list[str]:
    """
    Decode the 64-bit 'buttons' bitfield into a list of IN_* names.
    """
    return [name for bit, name in _KEYS.items() if bits & (1 << bit)]


# ------------------------------------------------------------------ CLI setup

def build_cli() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="CS-2 demo inspector: seek a tick and view a player’s inputs"
    )
    cli.add_argument(
        "demofile",
        type=Path,
        help="Path to the .dem file to inspect"
    )
    return cli


def main() -> None:
    args = build_cli().parse_args()
    if not args.demofile.exists():
        sys.exit(f"Demo not found: {args.demofile!r}")

    print("[loading demo – this takes a few seconds …]")
    dp = DemoParser(args.demofile.as_posix())

    # parse_ticks: include 'steamid', 'tick', 'name', plus our extras
    tick_df = (
        dp.parse_ticks(
            wanted_props=[
                "tick", "steamid", "name",
                "buttons", "pitch", "yaw", "inventory",
            ]
        )
        .pipe(pd.DataFrame)  # convert polars → pandas
    )

    # forward-fill so every server tick has a row for each player
    tick_df.sort_values(["steamid", "tick"], inplace=True)
    tick_df = (
        tick_df
        .set_index("tick")
        .groupby("steamid")
        .apply(
            lambda g: g
                .reindex(range(g.index.min(), g.index.max() + 1))
                .ffill(),
            include_groups=False
        )
        .reset_index(level=0, drop=True)
        .reset_index()
    )

    # compute mouse deltas
    tick_df["d_yaw"]   = tick_df.groupby("steamid")["yaw"].diff().fillna(0)
    tick_df["d_pitch"] = tick_df.groupby("steamid")["pitch"].diff().fillna(0)

    # ── purchases: robust parse_event + inventory-diff fallback ──────
    try:
        # try the newer API: keyword-only wanted_props
        buys = (
            dp.parse_event(
                "item_pickup",
                wanted_props=["tick", "steamid"]
            )
            .pipe(pd.DataFrame)
        )
        # ensure columns exist
        if not {"tick", "steamid"}.issubset(buys.columns):
            raise ValueError

    except (TypeError, ValueError):
        # fallback to legacy or incomplete parse_event
        try:
            tmp = dp.parse_event("item_pickup").pipe(pd.DataFrame)
        except TypeError:
            tmp = pd.DataFrame()
        if {"tick", "steamid"}.issubset(tmp.columns):
            buys = tmp
        else:
            # inventory-diff fallback
            inv = (
                tick_df[["tick", "steamid", "inventory"]]
                .sort_values(["steamid", "tick"])
                .assign(prev=lambda df: df.groupby("steamid")["inventory"].shift(1))
            )
            buys = (
                inv.assign(new=lambda r: (
                    r["inventory"].apply(set) - r["prev"].apply(lambda x: set(x or []))
                ))
                .explode("new")
                .dropna(subset=["new"])
                .rename(columns={"new": "weapon"})
            )
            buys["price"] = None

    # ── interactive prompt ────────────────────────────────────────────────────────
    print("\nCommands:\n  seek <tick>   jump to that tick\n  quit / exit   leave\n")
    while True:
        try:
            cmd = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if cmd in ("quit", "exit"): 
            break

        if cmd.startswith("seek"):
            parts = cmd.split()
            if len(parts) != 2 or not parts[1].isdigit():
                print("Usage: seek <tick_number>")
                continue

            tgt_tick = int(parts[1])
            rows = tick_df.query("tick == @tgt_tick")
            if rows.empty:
                print(f"No data at tick {tgt_tick}.")
                continue

            players = (
                rows[["steamid", "name"]]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            for i, r in players.iterrows():
                print(f"{i+1}. {r['name']}  (sid {r['steamid']})")

            sel = input("Player # > ").strip()
            if not sel.isdigit() or not (1 <= int(sel) <= len(players)):
                print("Invalid choice.")
                continue
            sid = players.iloc[int(sel)-1]["steamid"]

            row = rows[rows["steamid"] == sid].iloc[-1]
            keys = extract_buttons(int(row["buttons"]))
            print(
                f"\nTick {tgt_tick} – {row['name']}:\n"
                f"  Keys held : {', '.join(keys) or '(none)'}\n"
                f"  Mouse Δ   : yaw {row['d_yaw']:+.2f}°, pitch {row['d_pitch']:+.2f}°\n"
                f"  Inventory : {row['inventory']}"
            )

            b = buys.query("tick == @tgt_tick and steamid == @sid")
            if not b.empty:
                for _, ev in b.iterrows():
                    print(f"  Bought    : {ev.get('weapon')}  (-${ev.get('price')})")
            else:
                print("  Bought    : (nothing this tick)")
            print()
        else:
            print("Unknown command. Use 'seek <tick>' or 'quit'.")


if __name__ == "__main__":
    main()
