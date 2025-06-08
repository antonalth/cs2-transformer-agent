#!/usr/bin/env python3
"""
demo_cli.py – inspect a CS-2 demo from the terminal

Prompt commands
---------------
seek <tick>   jump to that server tick, choose a player, see keys / mouse Δ / buys
quit | exit   leave
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import inspect

import pandas as pd
from demoparser2 import DemoParser


# ── helper: decode 64-bit “buttons” bit-field ──────────────────────────────────
_KEYS = {
    0: "IN_ATTACK",   1: "IN_JUMP",    2: "IN_DUCK",   3: "IN_FORWARD",
    4: "IN_BACK",     5: "IN_USE",     6: "IN_CANCEL", 7: "IN_LEFT",
    8: "IN_RIGHT",    9: "IN_MOVELEFT", 10: "IN_MOVERIGHT", 11: "IN_ATTACK2",
    12: "IN_RUN",    13: "IN_RELOAD", 16: "IN_SPEED", 17: "IN_WALK",
    18: "IN_ZOOM",
}
def extract_buttons(bits: int) -> list[str]:
    """turn the buttons bit-field into IN_* names"""
    return [name for bit, name in _KEYS.items() if bits & (1 << bit)]


# ── CLI parsing ────────────────────────────────────────────────────────────────
def build_cli() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        description="Tiny CS-2 demo inspector (keys / mouse / buys)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    cli.add_argument("demofile", type=Path, help="Path to .dem file")
    return cli


def main() -> None:
    args = build_cli().parse_args()
    if not args.demofile.exists():
        sys.exit(f"Demo not found: {args.demofile}")

    print("[loading demo — this can take a few seconds …]")
    dp = DemoParser(args.demofile.as_posix())

    # ── 1) per-tick DataFrame ────────────────────────────────────────────────
    tick_df = (
        dp.parse_ticks(
            wanted_props=[
                "tick",
                "steamid",          # 64-bit ID (no underscore!)
                "name",
                "buttons", "pitch", "yaw", "inventory",
            ]
        )
        .pipe(pd.DataFrame)         # polars → pandas
    )

    # ── 2) forward-fill so EVERY (tick,steamid) exists once ─────────────────
    tick_df.sort_values(["steamid", "tick"], inplace=True)
    tick_df = (
        tick_df
        .set_index("tick")
        .groupby("steamid", group_keys=False)      # keep steamid as column
        .apply(lambda g: g
               .reindex(range(g.index.min(), g.index.max() + 1))
               .ffill())
        .reset_index()                             # only “tick” comes from index
    )

    # ── 3) mouse deltas ──────────────────────────────────────────────────────
    tick_df["d_yaw"]   = tick_df.groupby("steamid")["yaw"].diff().fillna(0)
    tick_df["d_pitch"] = tick_df.groupby("steamid")["pitch"].diff().fillna(0)

    # ── 4) purchases (“item_pickup”) — cope with 2-arg *or* 3-arg signature ─
    sig = inspect.signature(dp.parse_event)
    if len(sig.parameters) >= 3:          # new API: (name, event_props, wanted_props)
        buys = dp.parse_event(
            "item_pickup",
            ["weapon", "price"],          # event-specific columns
            ["tick", "steamid"],          # attach for joins/filters
        ).pipe(pd.DataFrame)
    else:                                 # old API: (name, wanted_props)
        raw = dp.parse_event("item_pickup").pipe(pd.DataFrame)
        buys = raw[["tick", "steamid", "weapon", "price"]]

    # ── 5) REPL ──────────────────────────────────────────────────────────────
    print("\nCommands:\n  seek <tick>   jump to that tick\n  quit | exit   leave\n")
    while True:
        try:
            cmd = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()             # newline before exiting
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
            sid = players.iloc[int(sel) - 1]["steamid"]

            r = rows.loc[rows["steamid"] == sid].iloc[-1]
            keys = extract_buttons(int(r["buttons"]))
            print(
                f"\nTick {tgt_tick} – {r['name']}:\n"
                f"  Keys held : {', '.join(keys) or '(none)'}\n"
                f"  Mouse Δ   : yaw {r['d_yaw']:+.2f}°, pitch {r['d_pitch']:+.2f}°\n"
                f"  Inventory : {r['inventory']}"
            )

            b = buys.query("tick == @tgt_tick and steamid == @sid")
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
