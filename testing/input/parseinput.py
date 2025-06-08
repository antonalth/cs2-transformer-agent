#!/usr/bin/env python3
"""
demo_cli.py – tiny CLI to inspect player inputs in a CS-2 demo

Commands inside the prompt:
    seek <tick>   – jump to that server-tick, pick a player, see keys/mouse/buys
    quit | exit   – leave
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

import pandas as pd
from demoparser2 import DemoParser

# ────────────────────────────────────────────────────────────────────────────────
# Helper: decode the 64-bit “buttons” bit-field into IN_* strings.
# Extend the _KEYS map if you need more constants.
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
    16: "IN_SPEED",     # Shift (walk) in CS-2
    17: "IN_WALK",      # legacy name for older demos
    18: "IN_ZOOM",
}
def extract_buttons(bits: int) -> list[str]:
    return [name for bit, name in _KEYS.items() if bits & (1 << bit)]
# ────────────────────────────────────────────────────────────────────────────────


def build_cli() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        description="CS-2 demo inspector: seek a tick, pick a player, "
                    "see keys / mouse Δ / purchases",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    cli.add_argument("demofile", type=Path, help="Path to .dem file")
    return cli


def main() -> None:
    args = build_cli().parse_args()
    if not args.demofile.exists():
        sys.exit(f"Demo not found: {args.demofile}")

    print("[loading demo — this can take a few seconds …]")
    parser = DemoParser(args.demofile.as_posix())

    # 1)  Per-tick DataFrame ────────────────────────────────────────────────
    tick_df = (
        parser.parse_ticks(
            wanted_props=[
                "tick",      # always include to make queries easy
                "steamid",   # 64-bit Steam-ID  (no underscore!)
                "name",      # player nickname
                "buttons", "pitch", "yaw", "inventory",
            ]
        )
        .pipe(pd.DataFrame)      # polars → pandas
    )

    # 2)  Forward-fill so EVERY (tick, steamid) pair exists once
    tick_df.sort_values(["steamid", "tick"], inplace=True)
    tick_df = (
        tick_df
        .set_index("tick")
        .groupby("steamid")
        .apply(lambda g: g
               .reindex(range(g.index.min(), g.index.max() + 1))
               .ffill())
        .reset_index()           # brings back both “steamid” & “tick”
    )

    # 3)  Mouse movement deltas
    tick_df["d_yaw"]   = tick_df.groupby("steamid")["yaw"].diff().fillna(0)
    tick_df["d_pitch"] = tick_df.groupby("steamid")["pitch"].diff().fillna(0)

    # 4)  Purchases (“item_pickup” event) with essential columns
    buys = parser.parse_event(
        "item_pickup",
        ["weapon", "price"],         # event-specific fields
        ["tick", "steamid"],         # carry over for joins / filtering
    ).pipe(pd.DataFrame)

    # ── Interactive prompt ────────────────────────────────────────────────
    print("\nCommands:\n  seek <tick>   jump to that tick\n  quit / exit   leave\n")
    while True:
        try:
            cmd = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if cmd in {"quit", "exit"}:
            break

        if cmd.startswith("seek"):
            parts = cmd.split(maxsplit=1)
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
            steamid = players.iloc[int(sel) - 1]["steamid"]

            # last row for that player at this tick
            r = rows.loc[rows["steamid"] == steamid].iloc[-1]

            keys = extract_buttons(int(r["buttons"]))
            print(
                f"\nTick {tgt_tick} – {r['name']}:\n"
                f"  Keys held : {', '.join(keys) or '(none)'}\n"
                f"  Mouse Δ   : yaw {r['d_yaw']:+.2f}°, pitch {r['d_pitch']:+.2f}°\n"
                f"  Inventory : {r['inventory']}"
            )

            b = buys.query("tick == @tgt_tick and steamid == @steamid")
            if not b.empty:
                for _, ev in b.iterrows():
                    print(f"  Bought    : {ev['weapon']}  (-${ev['price']})")
            else:
                print("  Bought    : (nothing this tick)")
            print()
        else:
            print("Unknown command. Use 'seek <tick>' or 'quit'.")


if __name__ == "__main__":
    main()
