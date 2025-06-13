#!/usr/bin/env python3
"""
extract_rounds.py  –  AWPy ≥ 2.0

Pull per-round meta plus per-player death ticks from a CS demo.

Columns
-------
round               int     round number (1-based)
starttick           int     first tick of the round
freezetime_endtick  int     tick when freeze-time ends
endtick             int     last tick of the round
t_team              JSON    [["name", death_tick], ...]  (death_tick = –1 ⇒ survived)
ct_team             JSON    same for CTs

Usage
-----
$ python extract_rounds.py match.dem                 # → JSONL on stdout
$ python extract_rounds.py match.dem --sqlout stats.db
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

# -----------------------------------------------------------------------------
# 1. Imports that differ between AWPy versions
try:                                # AWPy ≥ 2.0
    from awpy.demo import Demo as _Demo
    _AWPY_VERSION = "2.x"
except ModuleNotFoundError:         # AWPy 1.x (old layout) or demoparser2 directly
    try:
        from awpy.parser import DemoParser as _DemoParser
        _Demo = None
        _AWPY_VERSION = "1.x"
    except ModuleNotFoundError:
        try:
            from demoparser2 import DemoParser as _DemoParser
            _Demo = None
            _AWPY_VERSION = "demoparser2"
        except ModuleNotFoundError as exc:
            sys.exit(
                "[!] Neither AWPy ≥ 2.0, AWPy 1.x, nor demoparser2 could be imported.\n"
                "    pip-install one of them and try again."
            )

if _AWPY_VERSION.startswith("2"):
    import polars as pl
    import awpy.constants as _C
    import awpy.parsers.rounds as _rnd
    import awpy.parsers.events as _ev
# -----------------------------------------------------------------------------


def _parse_2x(demo_path: Path) -> list[dict[str, Any]]:
    """Implementation for AWPy ≥ 2.0."""
    demo = _Demo(path=demo_path)
    demo.parse()                                       # parse default events

    rounds_df: pl.DataFrame = demo.rounds              # start, freeze_end, end, …
    spawns = demo.events["player_spawn"]
    deaths_raw = demo.events["player_death"]

    # normalise kill events → victim_* columns
    deaths = _ev.parse_kills(deaths_raw)

    rows: list[dict[str, Any]] = []

    # Iterate round-wise (round_num is the row index after create_round_df)
    for rd in rounds_df.iter_rows(named=True):
        rn = rd["round_num"]
        s_tick, fz_end, e_tick = rd["start"], rd["freeze_end"], rd["end"]

        # helper to collect (side) roster & death ticks in the current round
        def team_list(side: str) -> list[list[Any]]:
            roster = (
                spawns.filter(
                    (pl.col("tick") >= s_tick)
                    & (pl.col("tick") <= fz_end)
                    & (pl.col("user_side") == side)
                )
                .select("user_name")
                .unique()
                .to_series()
                .to_list()
            )
            died = (
                deaths.filter(
                    (pl.col("tick") >= s_tick)
                    & (pl.col("tick") <= e_tick)
                )
                .select("victim_name", "tick")
            )
            death_map = dict(zip(died["victim_name"], died["tick"]))
            return [[p, death_map.get(p, -1)] for p in roster]

        rows.append(
            {
                "round": rn,
                "starttick": s_tick,
                "freezetime_endtick": fz_end,
                "endtick": e_tick,
                "t_team": team_list(_C.T_SIDE),
                "ct_team": team_list(_C.CT_SIDE),
            }
        )
    return rows


def _parse_1x_or_demoparser2(demo_path: Path) -> list[dict[str, Any]]:
    """Implementation for AWPy 1.x or demoparser2 – retains the original JSON route."""
    parser = _DemoParser(demofile=str(demo_path), parse_frames=False)
    match = parser.parse()

    # deaths by round
    death_map: dict[int, dict[str, int]] = {}
    for k in match.get("kills", []):
        death_map.setdefault(k["roundNum"], {})[k["victimName"]] = k["tick"]

    rows: list[dict[str, Any]] = []
    for rnd in match["gameRounds"]:
        rn = rnd["roundNum"]
        t_names = [p["playerName"] for p in rnd["tTeam"]["players"]]
        ct_names = [p["playerName"] for p in rnd["ctTeam"]["players"]]

        rows.append(
            {
                "round": rn,
                "starttick": rnd["startTick"],
                "freezetime_endtick": rnd["freezeTimeEndTick"],
                "endtick": rnd["endTick"],
                "t_team": [[n, death_map.get(rn, {}).get(n, -1)] for n in t_names],
                "ct_team": [[n, death_map.get(rn, {}).get(n, -1)] for n in ct_names],
            }
        )
    return rows


def _to_sql(rows: list[dict[str, Any]], db_file: Path) -> None:
    with sqlite3.connect(db_file) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS ROUNDS (
                   round               INTEGER PRIMARY KEY,
                   starttick           INTEGER,
                   freezetime_endtick  INTEGER,
                   endtick             INTEGER,
                   t_team              TEXT,
                   ct_team             TEXT
               )"""
        )
        conn.executemany(
            """INSERT OR REPLACE INTO ROUNDS
               VALUES (:round, :starttick, :freezetime_endtick,
                       :endtick, :t_team, :ct_team)""",
            (
                {
                    **r,
                    "t_team": json.dumps(r["t_team"]),
                    "ct_team": json.dumps(r["ct_team"]),
                }
                for r in rows
            ),
        )


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Extract per-round data from a CS demo.")
    ap.add_argument("demo", help=".dem file")
    ap.add_argument("--sqlout", metavar="DB", help="Write/replace rows in this SQLite DB")
    args = ap.parse_args()

    demo_path = Path(args.demo).expanduser()
    if not demo_path.is_file():
        sys.exit(f"[!] '{demo_path}' is not a file")

    if _AWPY_VERSION == "2.x":
        rows = _parse_2x(demo_path)
    else:
        rows = _parse_1x_or_demoparser2(demo_path)

    if args.sqlout:
        _to_sql(rows, Path(args.sqlout).expanduser())
    else:
        for r in rows:
            print(json.dumps(r, separators=(",", ":")))


if __name__ == "__main__":
    _cli()
