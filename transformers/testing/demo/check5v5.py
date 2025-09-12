#!/usr/bin/env python3
"""
check_5v5.py — scan .dem files and report if a demo is 5v5 or not (e.g., 6v6).

Usage:
  python check_5v5.py /path/to/demos            # non-recursive
  python check_5v5.py /path/to/demos -r         # recursive

Notes:
- Relies on AWPy. Install with: pip install awpy
- We try to read per-round team rosters first; if unavailable, we fall back to
  inferring from 'active' events (kills/damage/shots). We also show any names
  that look like coaches/observers, to explain 6v6 cases.
"""

import os
import sys
import json
import argparse
import re

IGNORE_NAME_PATTERNS = re.compile(r"(coach|observer|spectat|analyst|camera)", re.I)

def list_dems(root, recursive=False):
    if not recursive:
        return [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(".dem")]
    out = []
    for d, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".dem"):
                out.append(os.path.join(d, f))
    return sorted(out)

def safe_name(x):
    if isinstance(x, dict):
        return x.get("name") or x.get("playerName") or x.get("steamName") or x.get("steam_name")
    return str(x) if x is not None else None

def add_players_from_round(round_obj, t_names, ct_names):
    # Try common shapes from AWPy outputs across versions
    for key in ("tTeam", "teamT", "t_players", "tPlayers"):
        players = round_obj.get(key)
        if players:
            for p in players:
                n = safe_name(p)
                if n: t_names.add(n)
            break
    for key in ("ctTeam", "teamCT", "ct_players", "ctPlayers"):
        players = round_obj.get(key)
        if players:
            for p in players:
                n = safe_name(p)
                if n: ct_names.add(n)
            break

def analyze_with_awpy(path):
    try:
        from awpy.parser import DemoParser  # AWPy v2+
    except Exception as e:
        raise RuntimeError("AWPy not installed or import failed. Try: pip install awpy") from e

    # Parse at modest rate; we only need rounds/players
    dp = DemoParser(demofile=path, parse_rate=128, parse_kills=True, parse_frames=False, match_id=path)
    data = dp.parse()

    t_names, ct_names = set(), set()

    # 1) Prefer per-round rosters if present (most accurate for “6 on a side” detection)
    game_rounds = data.get("gameRounds") or data.get("game_rounds") or []
    for rnd in game_rounds:
        add_players_from_round(rnd, t_names, ct_names)

    # 2) Fallback: derive from activity if rosters missing
    if not t_names and not ct_names:
        # Look across common event collections
        for coll in ("kills", "damages", "weaponFires", "shots", "grenades", "flashes"):
            for ev in data.get(coll, []) or []:
                # try to infer side from attacker/player teams
                team = (ev.get("attackerTeam") or ev.get("playerTeam") or "").upper()
                nm = ev.get("attackerName") or ev.get("playerName")
                if not nm:
                    continue
                if team.startswith("T"):
                    t_names.add(nm)
                elif team.startswith("CT"):
                    ct_names.add(nm)

    # Heuristic: list potential observers/coaches
    suspects = sorted([n for n in (t_names | ct_names) if IGNORE_NAME_PATTERNS.search(n or "")])

    return {
        "t_count": len(t_names),
        "ct_count": len(ct_names),
        "t_names": sorted(t_names),
        "ct_names": sorted(ct_names),
        "suspects": suspects
    }

def main():
    ap = argparse.ArgumentParser(description="Check .dem files for 5v5 vs 6v6 (or other).")
    ap.add_argument("directory", help="Folder containing .dem files")
    ap.add_argument("-r", "--recursive", action="store_true", help="Recurse into subfolders")
    args = ap.parse_args()

    demos = list_dems(args.directory, args.recursive)
    if not demos:
        print("No .dem files found.")
        return

    ok = 0
    bad = 0
    for dem in demos:
        try:
            res = analyze_with_awpy(dem)
            t, ct = res["t_count"], res["ct_count"]
            verdict = "5v5 ✅" if (t == 5 and ct == 5) else f"NOT 5v5 ❌ (T={t}, CT={ct})"
            extra = ""
            if res["suspects"] and (t > 5 or ct > 5):
                extra = f" | suspects: {', '.join(res['suspects'][:4])}" + (" ..." if len(res['suspects']) > 4 else "")
            print(f"{os.path.basename(dem)} -> {verdict}{extra}")
            if t == 5 and ct == 5:
                ok += 1
            else:
                bad += 1
        except Exception as e:
            print(f"{os.path.basename(dem)} -> ERROR: {e}")
            bad += 1

    total = ok + bad
    print(f"\nSummary: {ok}/{total} look like 5v5.")

if __name__ == "__main__":
    main()
