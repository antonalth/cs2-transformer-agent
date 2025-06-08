#!/usr/bin/env python3
"""
demo_inspect_cli.py  –  super-light CLI explorer for CS:GO / CS2 demos

Usage:
    python demo_inspect_cli.py path/to/match.dem
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tabulate import tabulate
from demoparser2 import DemoParser


# ---- bit-field → list of key strings --------------------------------------

KEY_MAPPING = {
    0: "IN_ATTACK",   1: "IN_JUMP",     2: "IN_DUCK",      3: "IN_FORWARD",
    4: "IN_BACK",     5: "IN_USE",      6: "IN_CANCEL",    7: "IN_LEFT",
    8: "IN_RIGHT",    9: "IN_MOVELEFT", 10: "IN_MOVERIGHT",11: "IN_ATTACK2",
    12: "IN_RUN",     13: "IN_RELOAD",  14: "IN_ALT1",     15: "IN_ALT2",
    16: "IN_SCORE",   17: "IN_SPEED",   18: "IN_WALK",     19: "IN_ZOOM",
    20: "IN_WEAPON1", 21: "IN_WEAPON2", 22: "IN_BULLRUSH", 23: "IN_GRENADE1",
    24: "IN_GRENADE2",25: "IN_ATTACK3",
}

def extract_buttons(bitfield: int) -> List[str]:
    return [name for bit, name in KEY_MAPPING.items() if bitfield & (1 << bit)]

# ---- util ---------------------------------------------------------------
def canonicalise_steam_id(df):
    for cand in ("steam_id", "steamId", "steamid", "player_steamid"):
        if cand in df.columns:
            if cand != "steam_id":
                df = df.rename(columns={cand: "steam_id"})
            return df
    raise KeyError(
        "Could not find any steam-ID column. "
        "Run parser.list_updated_fields() to see what the demo exposes."
    )

# ---- one-off parsing -------------------------------------------------------

def parse_demo(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    parser = DemoParser(str(path))

    wanted = ["buttons", "pitch", "yaw", "inventory", "name", "steam_id"]
    tick_df = parser.parse_ticks(wanted_props=wanted)
    tick_df = canonicalise_steam_id(tick_df)        # ← ensures column exists/renamed

    tick_df["keys"]    = tick_df["buttons"].fillna(0).astype(int).apply(extract_buttons)
    tick_df["d_yaw"]   = tick_df.groupby("steam_id")["yaw"].diff()
    tick_df["d_pitch"] = tick_df.groupby("steam_id")["pitch"].diff()

    event_name = "item_purchase" if "item_purchase" in parser.event_types else "item_pickup"
    buy_df = parser.parse_event(
        event_name, ["weapon", "price"],
        ["tick", "steam_id", "is_freeze_period"],
    )
    buy_df = canonicalise_steam_id(buy_df)
    buy_df = buy_df[buy_df["is_freeze_period"] == True]

    return tick_df, buy_df



# ---- tiny REPL -------------------------------------------------------------

class DemoCLI:
    def __init__(self, tick_df: pd.DataFrame, buy_df: pd.DataFrame):
        self.ticks = tick_df
        self.buys  = buy_df
        self.players = (
            tick_df[["steam_id", "name"]]
            .drop_duplicates("steam_id")
            .sort_values("name", na_position="last")
            .set_index("steam_id")["name"]
            .to_dict()
        )
        self.current_player: int | None = None

    def cmd_players(self, *_):
        rows = [(sid, self.players.get(sid, "?")) for sid in self.players]
        print(tabulate(rows, headers=["SteamID", "Name"]))

    def cmd_select(self, args: List[str]):
        if not args:
            print("select <steam_id>")
            return
        try:
            sid = int(args[0])
        except ValueError:
            print("SteamID must be numeric"); return
        if sid not in self.players:
            print("No such SteamID in demo"); return
        self.current_player = sid
        print(f"Selected {self.players[sid]} ({sid})")

    def _print_row(self, row: pd.Series):
        print(f"\nTICK {int(row['tick'])}  |  {self.players.get(row['steam_id'])} ({int(row['steam_id'])})")
        print("-" * 60)
        print("Keys held :", ", ".join(row["keys"]) or "—")
        print(f"Mouse dYaw: {row['d_yaw']:+.2f}°   dPitch: {row['d_pitch']:+.2f}°")
        buys = self.buys.query("tick == @row.tick and steam_id == @row.steam_id")
        if len(buys):
            bought = ", ".join(buys["weapon"])
            print("Bought    :", bought)
        print()

    def cmd_seek(self, args: List[str]):
        if not args:
            print("seek <tick>"); return
        if self.current_player is None:
            print("Choose a player first with 'players' and 'select'"); return
        try:
            tick = int(args[0])
        except ValueError:
            print("tick must be integer"); return
        rows = self.ticks.query("tick == @tick and steam_id == @self.current_player")
        if rows.empty:
            print("No data on that tick (the demo only stores rows when props change)")
            return
        self._print_row(rows.iloc[0])

    def cmd_help(self, *_):
        print(
            "players                – list SteamIDs / names\n"
            "select <steam_id>      – choose a player\n"
            "seek   <tick>          – show inputs at that server-tick\n"
            "help                   – this help\n"
            "quit                   – exit"
        )

    def repl(self):
        self.cmd_help()
        while True:
            try:
                parts = input("» ").strip().split()
            except (EOFError, KeyboardInterrupt):
                print("\nbye"); break
            if not parts: continue
            cmd, *args = parts
            if cmd in ("quit", "exit"): break
            getattr(self, f"cmd_{cmd}", lambda *_: print("unknown command"))(args)


# ---- main ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("demo", type=Path, help="Path to .dem file")
    args = parser.parse_args()

    print("Parsing – this may take a moment…")
    tick_df, buy_df = parse_demo(args.demo)

    cli = DemoCLI(tick_df, buy_df)
    cli.repl()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__)
    else:
        main()
