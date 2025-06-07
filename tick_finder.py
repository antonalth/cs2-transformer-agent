#!/usr/bin/env python3
"""
tick_finder.py — generate a “moments” JSON from a CS-2 demo
────────────────────────────────────────────────────────────
Flags
-----
--boundingbox        add 2-D bbox per visible target using Alg-5
--debugbbx           emit a line of JSON when bbox computation fails
--demofile PATH.dem  input demo
--out moments.json   output path (stdout if omitted)
--progress           show tqdm progress bar
--max-events N       cap number of candidate ticks
--optimize           drop POVs whose visible[] is empty

Notes
-----
* Each POV block carries a **spectate_command** of the form
  `spec_player "<ingame-name>"` so *tick_bbxdbg.py* can jump straight to that
  camera.  The former `spec_goto …` string has been removed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import polars as pl
from awpy import Demo
from awpy.data import TRIS_DIR
from awpy.vector import Vector3

# ── NEW: proven bounding-box algorithm ────────────────────────────────────
from bbox_algs.alg6 import DefaultBBoxAlg as BBoxAlg

# ── constants ─────────────────────────────────────────────────────────────
SIG_EVENTS = {"player_hurt", "player_death", "weapon_fire", "flashbang_detonate"}
ROUND_EVENTS = {
    "round_start",
    "round_end",
    "round_officially_ended",
    "round_freeze_end",
}
EVENTS = sorted(SIG_EVENTS | ROUND_EVENTS)

PLAYER_PROPS = [
    # identifiers
    "steamid",
    "userId",
    "user_id",
    "userid",
    # team / side
    "side",
    "team_name",
    "team",
    # vitals & world-pos
    "health",
    "X",
    "Y",
    "Z",
    # view angles
    "yaw",
    "pitch",
    # posture
    "ducking",
    "isDucked",
    "duckAmount",
    # names
    "name",
    "playerName",
    "player_name",
]

MARGIN = 6  # ± ticks around a signal event to capture
V_FOV_DEG = 90.0
ASPECT = 16 / 9
EDGE = 3.0  # degrees margin inside FOV for eligibility
H_FOV_DEG = math.degrees(
    2 * math.atan(math.tan(math.radians(V_FOV_DEG) / 2) * ASPECT)
)

STAND_Z, CROUCH_Z = 72.0, 54.0
SCR_W, SCR_H = 1280, 720  # assumed screenshot resolution

# ── instantiate Alg-5 once and reuse for every bbox calculation ───────────
bbox_alg = BBoxAlg(fov_deg=V_FOV_DEG, screen_w=SCR_W, screen_h=SCR_H)

# ══════════════════════════════════════════════════════════════════════════
# helper utilities
# ══════════════════════════════════════════════════════════════════════════
def first(row: dict[str, Any], keys: Sequence[str]) -> Any | None:
    """Return value of the first present key in *row* (or *None*)."""
    for k in keys:
        if k in row and row[k] not in (None, "", -1):
            return row[k]
    return None


def ypr_to_vec(yaw: float, pitch: float) -> Vector3:
    """Convert yaw & pitch (°) to a forward unit vector."""
    y, p = math.radians(yaw), math.radians(pitch)
    return Vector3(
        math.cos(p) * math.cos(y),
        math.cos(p) * math.sin(y),
        -math.sin(p),
    )


def angle_error(eye: Vector3, fwd: Vector3, tgt: Vector3) -> Tuple[float, float]:
    """
    Return absolute (yaw_err, pitch_err) between *fwd* and the vector *eye→tgt*.

    A quick, projection-free metric to decide FOV inclusion.
    """
    d = (tgt - eye).normalize()

    f_h = Vector3(fwd.x, fwd.y, 0).normalize()  # forward, flattened
    d_h = Vector3(d.x, d.y, 0).normalize()      # target dir, flattened
    yaw = math.degrees(math.acos(max(-1.0, min(1.0, f_h.dot(d_h)))))

    dot = max(-1.0, min(1.0, fwd.dot(d)))
    tot = math.degrees(math.acos(dot))
    pitch = math.sqrt(max(0.0, tot * tot - yaw * yaw))
    return yaw, pitch


def crouched(row: dict[str, Any]) -> bool:
    """Heuristic: consider player crouched if duck flags set or amount > .5."""
    if row.get("ducking") or row.get("isDucked"):
        return True
    amt = row.get("duckAmount")
    return isinstance(amt, (int, float)) and float(amt) > 0.5


def uid_of(row: dict[str, Any], lut: dict[int, int]) -> int:
    """Resolve userId even when the demo only carries SteamIDs."""
    for k in ("userId", "user_id", "userid"):
        if k in row and row[k] not in (None, -1, ""):
            return int(row[k])
    return lut.get(int(row["steamid"]), -1)


def load_vis(map_name: str):
    """Try to load Awpy's VisibilityChecker (.tri mesh); return *None* if absent."""
    try:
        from awpy.visibility import VisibilityChecker  # type: ignore
    except ImportError:
        return None

    tri = TRIS_DIR / f"{map_name}.tri"
    if not tri.exists():
        sys.stderr.write(
            f"[tick_finder] No .tri mesh for {map_name}; LOS disabled.\n"
        )
        return None

    try:
        return VisibilityChecker(path=tri)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[tick_finder] VisibilityChecker load failed: {exc}\n")
        return None


# ══════════════════════════════════════════════════════════════════════════
# main routine
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    cli = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cli.add_argument("--demofile", required=True, type=Path)
    cli.add_argument("--out", type=Path)
    cli.add_argument("--progress", action="store_true")
    cli.add_argument("--max-events", type=int)
    cli.add_argument(
        "--optimize",
        action="store_true",
        help="strip POVs whose visible list ends up empty",
    )
    cli.add_argument(
        "--boundingbox",
        action="store_true",
        help="add 2-D bbox for every visible target (Alg-5)",
    )
    cli.add_argument(
        "--debugbbx",
        action="store_true",
        help="emit debug JSON when bbox fails",
    )
    args = cli.parse_args()

    if not args.demofile.exists():
        cli.error("demo not found")

    demo = Demo(path=args.demofile, verbose=False)
    demo.parse(events=EVENTS, player_props=PLAYER_PROPS, other_props=None)

    # ── build a SteamID → userId lookup from kill/damage logs ──────────────
    uid_map: dict[int, int] = {}
    for df in (demo.kills, demo.damages):
        for sid_col, uid_col in (
            ("attackerSteamId", "attackerUserId"),
            ("victimSteamId", "victimUserId"),
            ("playerSteamId", "playerUserId"),
        ):
            if sid_col in df.columns and uid_col in df.columns:
                for sid, uidv in zip(df[sid_col], df[uid_col]):
                    uid_map[int(sid)] = int(uidv)

    # ── pick interesting ticks ─────────────────────────────────────────────
    cand: set[int] = set()
    for ev in SIG_EVENTS:
        ev_df = demo.events.get(ev)
        if ev_df is None or ev_df.is_empty():
            continue
        for t in ev_df["tick"].to_list():
            cand.update(range(t - MARGIN, t + MARGIN + 1))

    ticks = sorted(t for t in cand if t >= 0)
    if args.max_events and len(ticks) > args.max_events:
        random.seed(0)
        ticks = sorted(random.sample(ticks, args.max_events))

    # ── bucket demo rows by tick for quick look-ups ────────────────────────
    ticks_df = demo.ticks.filter(pl.col("tick").is_in(set(ticks)))
    buckets: dict[int, List[dict]] = defaultdict(list)
    for row in ticks_df.iter_rows(named=True):
        buckets[int(row["tick"])].append(row)

    vis_checker = load_vis(demo.header.get("map_name", ""))

    iterator = ticks
    if args.progress:
        try:
            from tqdm import tqdm  # noqa: WPS433
            iterator = tqdm(ticks, desc="ticks", unit="tick", leave=False)
        except ImportError:
            pass

    results: List[Dict[str, Any]] = []

    # ═══════════════════════════════════════════════════════════════════════
    # loop over each selected tick
    # ═══════════════════════════════════════════════════════════════════════
    for t in iterator:
        rows = buckets.get(t)
        if not rows:
            continue

        # ---- build a quick geometry table: uid → (eye, fwd) ---------------
        geom: Dict[int, Tuple[Vector3, Vector3]] = {}
        name_lut: Dict[int, str] = {}
        side_lut: Dict[int, str] = {}

        for r in rows:
            if int(r.get("health", 0)) <= 0:
                continue

            uid = uid_of(r, uid_map)
            if uid == -1:
                continue

            z_eye = STAND_Z if not crouched(r) else CROUCH_Z
            eye = Vector3(r["X"], r["Y"], r["Z"] + z_eye)
            fwd = ypr_to_vec(r["yaw"], r["pitch"])

            geom[uid] = (eye, fwd)
            name_lut[uid] = first(r, ("name", "playerName", "player_name")) or f"uid{uid}"
            side_lut[uid] = r.get("side") or r.get("team") or ""

        if not geom:
            continue

        # ---- build POV objects --------------------------------------------
        povs: List[Dict[str, Any]] = []
        for r in rows:
            if int(r.get("health", 0)) <= 0:
                continue
            pov_uid = uid_of(r, uid_map)
            if pov_uid not in geom:
                continue

            pov_entry: Dict[str, Any] = {
                "uid": pov_uid,
                "name": name_lut[pov_uid],
                "side": side_lut.get(pov_uid, ""),
                "x": r["X"],
                "y": r["Y"],
                "z": r["Z"],
                "yaw": r["yaw"],
                "pitch": r["pitch"],
                "crouching": crouched(r),
                "spectate_command": f'spec_player "{name_lut[pov_uid]}"',
                "visible": [],
            }

            eye_p, fwd_p = geom[pov_uid]

            # ---- iterate targets ------------------------------------------
            for tgt_uid, (eye_t, _) in geom.items():
                if tgt_uid == pov_uid:
                    continue

                yaw_err, pitch_err = angle_error(eye_p, fwd_p, eye_t)
                if (
                    yaw_err > (H_FOV_DEG / 2 - EDGE)
                    or pitch_err > (V_FOV_DEG / 2 - EDGE)
                ):
                    continue  # outside FOV cone

                # line-of-sight (if VisibilityChecker is available)
                if vis_checker and not vis_checker.is_visible(
                    start=(eye_p.x, eye_p.y, eye_p.z),
                    end=(eye_t.x, eye_t.y, eye_t.z),
                ):
                    continue

                tgt_entry: Dict[str, Any] = {
                    "uid": tgt_uid,
                    "side": side_lut.get(tgt_uid, ""),
                    "x": eye_t.x,
                    "y": eye_t.y,
                    "z": eye_t.z,
                    "yaw": rows[0]["yaw"],  # yaw/pitch of target not essential here
                    "pitch": rows[0]["pitch"],
                    "crouching": eye_t.z < r["Z"] + STAND_Z - 1,
                }

                # ── bounding-box (Alg-5) ───────────────────────────────────
                if args.boundingbox:
                    from_tuple = (
                        r["X"],
                        r["Y"],
                        r["Z"],
                        r["yaw"],
                        r["pitch"],
                        crouched(r),
                    )
                    to_tuple = (
                        tgt_entry["x"],
                        tgt_entry["y"],
                        tgt_entry["z"] - (STAND_Z if not tgt_entry["crouching"] else CROUCH_Z),
                        0.0,  # target viewangles do not matter for bbox
                        0.0,
                        tgt_entry["crouching"],
                    )
                    bb = bbox_alg.calcbb(from_tuple, to_tuple)
                    if bb is None:
                        if args.debugbbx:
                            dbg = {
                                "tick": t,
                                "pov_uid": pov_uid,
                                "tgt_uid": tgt_uid,
                                "reason": "calcbb_none",
                            }
                            print(json.dumps(dbg))
                        continue
                    tgt_entry["bbox"] = [round(v, 1) for v in bb]

                pov_entry["visible"].append(tgt_entry)

            if not (args.optimize and not pov_entry["visible"]):
                povs.append(pov_entry)

        if povs:
            results.append({"tick": t, "povs": povs})

    # ── write output ───────────────────────────────────────────────────────
    out_json = json.dumps(results, indent=2)
    if args.out:
        args.out.write_text(out_json, encoding="utf-8")
    else:
        print(out_json)


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
