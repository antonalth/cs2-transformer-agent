#!/usr/bin/env python3
"""
frame_inspector.py — extract all player info at a specified tick from a demo,
including which other players are in each player’s FOV and line-of-sight,
and generate a ‘spec_goto’ command with z adjusted by standing/crouching.

Usage:
    ./frame_inspector.py --demofile /path/to/match.dem --tick 12345

Output (JSON to stdout):
{
  "tick": 12345,
  "players": [
    {
      "steamid": 76561198000000000,
      "userId": 3,
      "team": "CT",
      "health": 100,
      "x": 1234.56,
      "y": 789.01,
      "z": 256.78,
      "yaw": 45.0,
      "pitch": -2.0,
      "crouching": false,
      "character_id_visible": [
        {
          "userId": 7,
          "steamid": 76561198000000007,
          "team": "T",
          "health": 100,
          "x": 1300.12,
          "y": 800.34,
          "z": 256.78,
          "yaw": 210.0,
          "pitch": 0.0,
          "crouching": true
        },
        …
      ],
      "spectate_command": "spec_goto 3 256.78"
    },
    …
  ]
}
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import polars as pl
from awpy import Demo
from awpy.vector import Vector3

# ────────── Constants ─────────────────────────────────────────────────────
# Minimal round events so Awpy’s round builder doesn’t KeyError
REQUIRED_ROUND_EVENTS = [
    "round_start",
    "round_end",
    "round_officially_ended",
    "round_freeze_end",
]

# Ask for steamId, any userId variants, plus duck state if available
PLAYER_PROPS = [
    "steamid",
    "userId",    # Awpy ≥ 2.x
    "user_id",   # alias
    "userid",    # legacy alias
    "side",      # newer Awpy team column
    "team_name", # older Awpy team column
    "team",      # numeric fallback
    "health",
    "X",
    "Y",
    "Z",
    "yaw",
    "pitch",
    # possible duck flags (Awpy may include one of these)
    "ducking",       # boolean
    "isDucked",      # boolean alias
    "duckAmount",    # float 0.0–1.0
]

V_FOV_DEG = 90.0
ASPECT = 16 / 9
H_FOV_DEG = math.degrees(2 * math.atan(math.tan(math.radians(V_FOV_DEG) / 2) * ASPECT))
EDGE_MARGIN_DEG = 3.0
PLAYER_HEIGHT_UU = 73.0
SCREEN_H = 1080

# Z-offsets for spectating (approximate eye height)
STAND_Z_OFFSET = 72.0
CROUCH_Z_OFFSET = 54.0
# ──────────────────────────────────────────────────────────────────────────

# Globals for uid lookup
DEMO: Demo | None = None
_UID_MAP: dict[int, int] | None = None


# ────────── math & helper functions ────────────────────────────────────────
def first_present(row: dict[str, Any], keys: Sequence[str]) -> Any:
    """Return the first key in `keys` present in `row` with a non-null value."""
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def yaw_pitch_to_vec(yaw_deg: float, pitch_deg: float) -> Vector3:
    """Convert yaw & pitch (degrees) to a forward unit vector."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    return Vector3(
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        -math.sin(pitch),
    )


def hv_error(eye: Vector3, fwd: Vector3, tgt: Vector3) -> tuple[float, float]:
    """
    Compute horizontal (yaw) and vertical (pitch) angular error between
    forward vector and target direction.
    """
    delta = (tgt - eye).normalize()
    # horizontal component
    f_h = Vector3(fwd.x, fwd.y, 0).normalize()
    d_h = Vector3(delta.x, delta.y, 0).normalize()
    yaw_err = math.degrees(math.acos(max(-1.0, min(1.0, f_h.dot(d_h)))))
    # total angle
    dot_full = max(-1.0, min(1.0, fwd.dot(delta)))
    tot_err = math.degrees(math.acos(dot_full))
    # vertical component
    pitch_err = math.sqrt(max(0.0, tot_err**2 - yaw_err**2))
    return yaw_err, pitch_err


def px_height(dist: float) -> float:
    """Estimate projected pixel height of a player at distance `dist`."""
    units_spanned = 2 * dist * math.tan(math.radians(V_FOV_DEG) / 2)
    return (PLAYER_HEIGHT_UU / units_spanned) * SCREEN_H


def load_visibility_checker(map_name: str):
    """
    Attempt to load Awpy's VisibilityChecker from a .tri mesh directory.
    Returns None if the .tri file is missing or loading fails.
    """
    try:
        from awpy.visibility import VisibilityChecker
    except ImportError:
        return None

    tri_dir = Path(__import__("os").environ.get("CS2_TRI_DIR", "map_tris"))
    tri_path = tri_dir / f"{map_name}.tri"
    if not tri_path.exists():
        sys.stderr.write(f"[frame_inspector] WARNING: no .tri mesh for {map_name}, skipping LOS checks.\n")
        return None
    try:
        return VisibilityChecker(path=tri_path)
    except Exception as e:
        sys.stderr.write(f"[frame_inspector] WARNING: failed to load VisibilityChecker: {e}\n")
        return None


def is_crouching(row: dict[str, Any]) -> bool:
    """
    Determine if a player is crouching from available fields:
    - If 'ducking' or 'isDucked' boolean field exists and is True.
    - Else if 'duckAmount' float > 0.5.
    Otherwise assume standing.
    """
    if "ducking" in row and row["ducking"] is not None:
        return bool(row["ducking"])
    if "isDucked" in row and row["isDucked"] is not None:
        return bool(row["isDucked"])
    if "duckAmount" in row and row["duckAmount"] is not None:
        try:
            return float(row["duckAmount"]) > 0.5
        except Exception:
            pass
    return False


def uid(row: dict[str, Any]) -> int:
    """
    Return the in-game userId for a player row (1–10).
    Fallback to steamId→userId mapping if no userId column is present.
    """
    # 1) Direct column if present
    for key in ("userId", "user_id", "userid"):
        if key in row and row[key] not in (None, "", -1):
            return int(row[key])

    # 2) Build steam→user map on demand
    global _UID_MAP, DEMO
    if _UID_MAP is None:
        _UID_MAP = {}
        for df in (DEMO.kills, DEMO.damages):
            # attacker columns
            if "attackerSteamId" in df.columns and "attackerUserId" in df.columns:
                for sid, uidv in zip(df["attackerSteamId"], df["attackerUserId"]):
                    _UID_MAP[int(sid)] = int(uidv)
            # victim columns
            if "victimSteamId" in df.columns and "victimUserId" in df.columns:
                for sid, uidv in zip(df["victimSteamId"], df["victimUserId"]):
                    _UID_MAP[int(sid)] = int(uidv)
            # player columns (some events)
            if "playerSteamId" in df.columns and "playerUserId" in df.columns:
                for sid, uidv in zip(df["playerSteamId"], df["playerUserId"]):
                    _UID_MAP[int(sid)] = int(uidv)

    steam = int(row.get("steamid", -1))
    return _UID_MAP.get(steam, -1)


# ────────── main routine ─────────────────────────────────────────────────────
def main() -> None:
    global DEMO

    ap = argparse.ArgumentParser(
        description="Dump player info and visible players at a given tick."
    )
    ap.add_argument("--demofile", required=True, type=Path, help="Path to a .dem file")
    ap.add_argument("--tick", required=True, type=int, help="Tick/frame number to inspect")
    args = ap.parse_args()

    if not args.demofile.exists():
        ap.error(f"Demo file not found: {args.demofile}")

    print(f"[frame_inspector] Parsing {args.demofile.name} …", file=sys.stderr)
    DEMO = Demo(path=args.demofile, verbose=False)
    # Supply round events to satisfy Awpy’s round builder, and player props for duck
    DEMO.parse(
        events=REQUIRED_ROUND_EVENTS,
        player_props=PLAYER_PROPS,
        other_props=None,
    )

    map_name = DEMO.header.get("map_name", "")
    vis_checker = load_visibility_checker(map_name)

    ticks_df = DEMO.ticks
    frame_df = ticks_df.filter(pl.col("tick") == args.tick)
    if frame_df.is_empty():
        sys.stderr.write(f"[frame_inspector] No data for tick {args.tick}.\n")
        sys.exit(1)

    rows: List[dict] = frame_df.iter_rows(named=True)

    # Build base player list
    players: List[Dict[str, Any]] = []
    for row in rows:
        steamid = row.get("steamid")
        user_id = first_present(row, ("userId", "user_id", "userid"))
        team = first_present(row, ("side", "team_name", "team"))
        health = row.get("health")
        x = row.get("X")
        y = row.get("Y")
        z = row.get("Z")
        yaw = first_present(row, ("yaw", "view_yaw", "view_x"))
        pitch = first_present(row, ("pitch", "view_pitch", "view_y"))

        # Skip dead or incomplete
        if health is None or int(health) <= 0:
            continue
        if None in (x, y, z, yaw, pitch) or user_id is None:
            continue

        crouch = is_crouching(row)
        players.append({
            "steamid": int(steamid),
            "userId": int(user_id),
            "team": team,
            "health": int(health),
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "yaw": float(yaw),
            "pitch": float(pitch),
            "crouching": crouch,
            "character_id_visible": [],
            "spectate_command": "",  # fill below
        })

    # Precompute eye and forward for each player
    info_list = []
    for p in players:
        eye = Vector3(p["x"], p["y"], p["z"] + (CROUCH_Z_OFFSET if p["crouching"] else STAND_Z_OFFSET))
        fwd = yaw_pitch_to_vec(p["yaw"], p["pitch"])
        info_list.append((p["userId"], eye, fwd))

    # Determine visibility: include all other players
    for idx, p in enumerate(players):
        pov_uid, eye, fwd = info_list[idx]
        for jdx, q in enumerate(players):
            if idx == jdx:
                continue
            tgt_uid, tgt_eye, _ = info_list[jdx]

            # FOV check
            yaw_err, pitch_err = hv_error(eye, fwd, tgt_eye)
            if yaw_err > (H_FOV_DEG / 2 - EDGE_MARGIN_DEG):
                continue
            if pitch_err > (V_FOV_DEG / 2 - EDGE_MARGIN_DEG):
                continue

            # LOS check
            if vis_checker and not vis_checker.is_visible(eye, tgt_eye):
                continue

            # Build a detailed entry for the visible player
            vis_entry = {
                "steamid": q["steamid"],
                "userId": q["userId"],
                "team": q["team"],
                "health": q["health"],
                "x": q["x"],
                "y": q["y"],
                "z": q["z"],
                "yaw": q["yaw"],
                "pitch": q["pitch"],
                "crouching": q["crouching"],
            }
            p["character_id_visible"].append(vis_entry)

        # spectate_command: “spec_goto <userId> <adjusted_z>”
        base_z = p["z"] + (CROUCH_Z_OFFSET if p["crouching"] else STAND_Z_OFFSET)
        p["spectate_command"] = f"spec_goto {p['userId']} {p['x']: .2f} {p['y']: 2.f} {base_z:.2f} {p['pitch']: .2f} {p['yaw']: .2f}"

    # Output JSON
    output: Dict[str, Any] = {
        "tick": args.tick,
        "players": players,
    }
    json.dump(output, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
