#!/usr/bin/env python3
"""
demo_cli.py – inspect keys / mouse Δ / purchases at any tick in a CS-2 demo

Prompt commands
---------------
seek <tick>   jump to that server tick, pick a player, view inputs & buys
quit | exit   leave
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
from demoparser2 import DemoParser


# ── helper: decode 'buttons' bit-field ─────────────────────────────────────────
_KEYS = {
    0: "IN_ATTACK", 1: "IN_JUMP", 2: "IN_DUCK", 3: "IN_FORWARD",
    4: "IN_BACK", 5: "IN_USE", 6: "IN_CANCEL", 7: "IN_LEFT",
    8: "IN_RIGHT", 9: "IN_MOVELEFT", 10: "IN_MOVERIGHT", 11: "IN_ATTACK2",
    12: "IN_RUN", 13: "IN_RELOAD", 16: "IN_SPEED", 17: "IN_WALK", 18: "IN_ZOOM",
}
def extract_buttons(bits: int) -> list[str]:
    return [name for bit, name in _KEYS.items() if bits & (1 << bit)]


# ── CLI setup ──────────────────────────────────────────────────────────────────
def cli() -> argparse.ArgumentParser:
    c = argparse.ArgumentParser(description="Tiny CS-2 demo inspector")
    c.add_argument("demofile", type=Path, help="Path to .dem file")
    return c


def main() -> None:
    args = cli().parse_args()
    if not args.demofile.exists():
        sys.exit(f"Demo not found: {args.demofile}")

    print("[loading demo – one-off parse, please wait …]")
    dp = DemoParser(args.demofile.as_posix())

    # 1) per-tick table ---------------------------------------------------------
    tick_df = (
        dp.parse_ticks(
            wanted_props=[
                "tick", "steamid", "name",
                "buttons", "pitch", "yaw", "inventory",
            ]
        )
        .pipe(pd.DataFrame)
    )

    # 2) gap-fill so every (tick,steamid) exists --------------------------------
    tick_df.sort_values(["steamid", "tick"], inplace=True)
    tick_df = (
        tick_df
        .set_index("tick")
        .groupby("steamid", group_keys=False)        # keep steamid as column
        .apply(lambda g: g
               .reindex(range(g.index.min(), g.index.max() + 1))
               .ffill())
        .reset_index()                               # only 'tick' comes from index
    )

    # 3) mouse deltas -----------------------------------------------------------
    tick_df["d_yaw"]   = tick_df.groupby("steamid")["yaw"].diff().fillna(0)
    tick_df["d_pitch"] = tick_df.groupby("steamid")["pitch"].diff().fillna(0)

    # 4) purchases – try ➊, fall back to ➋, then ➌ ------------------------------
    def get_buys():
        try:  # ➊ new style: (event, event_props, wanted_props)
            return dp.parse_event(
                "item_pickup",
                ["weapon", "price"],
                ["tick", "steamid"],
            )
        except TypeError:
            try:  # ➋ mid-old: (event, wanted_props)
                return dp.parse_event(
                    "item_pickup",
                    ["tick", "steamid", "weapon", "price"],
                )
            except TypeError:
                raw = dp.parse_event("item_pickup")  # ➌ oldest
                cols = [c for c in raw.columns
                        if c in {"tick", "steamid", "weapon", "price"}]
                return raw[cols]

    buys = get_buys().pipe(pd.DataFrame)

    # 5) tiny REPL --------------------------------------------------------------
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


if __name__ == "__main__":
    main()
