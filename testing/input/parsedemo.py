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
_KEYS = {
    0:  "IN_ATTACK",   1:  "IN_JUMP",    2:  "IN_DUCK",      3:  "IN_FORWARD",
    4:  "IN_BACK",     5:  "IN_USE",     6:  "IN_CANCEL",    7:  "IN_LEFT",
    8:  "IN_RIGHT",    9:  "IN_MOVELEFT",10: "IN_MOVERIGHT",11: "IN_ATTACK2",
    12: "IN_RUN",     13: "IN_RELOAD", 16: "IN_SPEED",    17: "IN_WALK",
    18: "IN_ZOOM",
}

def extract_buttons(bits: int) -> list[str]:
    """Return list of key names held in the bit-field *bits*."""
    return [name for bit, name in _KEYS.items() if bits & (1 << bit)]

# ── CLI setup ────────────────────────────────────────────────────────────────
def cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tiny CS-2 demo inspector / exporter")
    p.add_argument("demofile", type=Path, help="Path to .dem file")
    p.add_argument("--sqlout", type=Path, metavar="FILE.db",
                   help="Write results to SQLite DB instead of opening the CLI")
    return p

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
                "team_num","active_weapon_name", "total_rounds_played"
                "is_defusing","aim_punch_angle","usercmd_mouse_dx","usercmd_mouse_dy",#"usercmd_input_history",
                "buttons", "pitch", "yaw", "inventory",
            ]
        ).pipe(pd.DataFrame)
    )
# 5) tiny REPL
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
        print(f"{tgt_tick}: {r}")


if __name__ == "__main__":
    main()