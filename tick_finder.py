#!/usr/bin/env python3
"""
tick_finder.py — generate “moments” JSON from a CS-2 demo
---------------------------------------------------------
New flags:
  --boundingbox        # add 2-D bbox per visible target
  --debugbbx           # print debug info when bbox fails

Other flags (unchanged):
  --demofile PATH.dem
  --out moments.json
  --progress
  --max-events N
  --optimize           # drop POVs whose visible[] is empty

Example:
  ./tick_finder.py --demofile demos/match.dem --boundingbox --debugbbx --optimize --progress > moments_bb.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import polars as pl
from awpy import Demo
from awpy.vector import Vector3
from awpy.data import TRIS_DIR

# ───────── constants ─────────
SIG_EVENTS = {"player_hurt", "player_death", "weapon_fire", "flashbang_detonate"}
ROUND_EVENTS = {
    "round_start", "round_end", "round_officially_ended", "round_freeze_end"
}
EVENTS = sorted(SIG_EVENTS | ROUND_EVENTS)

PLAYER_PROPS = [
    "steamid", "userId", "user_id", "userid",
    "side", "team_name", "team",
    "health", "X", "Y", "Z",
    "yaw", "pitch",
    "ducking", "isDucked", "duckAmount",
]

MARGIN = 6
V_FOV_DEG = 90.0
ASPECT = 16 / 9
EDGE = 3.0  # degrees margin inside FOV
H_FOV_DEG = math.degrees(2 * math.atan(math.tan(math.radians(V_FOV_DEG) / 2) * ASPECT))

STAND_Z, CROUCH_Z = 72.0, 54.0
HITBOX_HALF_W = 16.0  # UU from centre in X & Y
HITBOX_H = 73.0       # height in UU

SCR_W, SCR_H = 1280, 720  # assumed screenshot resolution

# ───────── helpers ─────────
def first(row: dict[str, Any], keys: Sequence[str]) -> Any | None:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None

def ypr_to_vec(yaw: float, pitch: float) -> Vector3:
    """Convert yaw & pitch (degrees) to a forward unit vector."""
    y, p = math.radians(yaw), math.radians(pitch)
    return Vector3(
        math.cos(p) * math.cos(y),
        math.cos(p) * math.sin(y),
        -math.sin(p),
    )

def angle_error(eye: Vector3, fwd: Vector3, tgt: Vector3) -> Tuple[float, float]:
    """Compute horizontal (yaw) and vertical (pitch) angular error."""
    d = (tgt - eye).normalize()
    f_h = Vector3(fwd.x, fwd.y, 0).normalize()
    d_h = Vector3(d.x, d.y, 0).normalize()
    yaw = math.degrees(math.acos(max(-1, min(1, f_h.dot(d_h)))))
    dot = max(-1, min(1, fwd.dot(d)))
    tot = math.degrees(math.acos(dot))
    pitch = math.sqrt(max(0, tot * tot - yaw * yaw))
    return yaw, pitch

def crouched(row: dict[str, Any]) -> bool:
    """Determine if player is crouching."""
    if row.get("ducking") or row.get("isDucked"):
        return True
    amt = row.get("duckAmount")
    return isinstance(amt, (int, float)) and float(amt) > 0.5

def uid_of(row: dict[str, Any], lut: dict[int, int]) -> int:
    """Return in-game userId or fallback to lut mapping."""
    for k in ("userId", "user_id", "userid"):
        if k in row and row[k] not in (None, -1, ""):
            return int(row[k])
    return lut.get(int(row["steamid"]), -1)

def load_vis(map_name: str):
    """Attempt to load Awpy's VisibilityChecker from a .tri mesh."""
    try:
        from awpy.visibility import VisibilityChecker
    except ImportError:
        return None
    #tri = Path(os.getenv("CS2_TRI_DIR", "map_tris")) / f"{map_name}.tri"
    tri = TRIS_DIR / f"{map_name}.tri"
    if not tri.exists():
        sys.stderr.write(f"[tick_finder] No .tri mesh for {map_name}; LOS disabled.\n")
        return None
    try:
        return VisibilityChecker(path=tri)
    except Exception as e:
        sys.stderr.write(f"[tick_finder] VisibilityChecker load failed: {e}\n")
        return None

# ── 3-D → 2-D projection helpers (for --boundingbox) ────────────────────
ASPECT_INV = 1 / ASPECT
TAN_HALF_FOV = math.tan(math.radians(V_FOV_DEG) / 2)  # ==1 at 90°

def build_axes(yaw: float, pitch: float) -> Tuple[Vector3, Vector3, Vector3]:
    """Return (forward, right, up) basis from yaw/pitch."""
    f = ypr_to_vec(yaw, pitch).normalize()
    r = f.cross(Vector3(0, 0, 1)).normalize()
    u = r.cross(f)
    return f, r, u

def world_to_cam(pt: Vector3, eye: Vector3, r: Vector3, u: Vector3, f: Vector3) -> Tuple[float, float, float]:
    """Transform world point into camera-space coords (x_c, y_c, z_c)."""
    d = pt - eye
    x_c = d.dot(r)
    y_c = d.dot(u)
    z_c = d.dot(f)  # camera looks along -forward
    return x_c, y_c, z_c

def cam_to_ndc(x_c: float, y_c: float, z_c: float) -> Tuple[float | None, float | None]:
    """Project camera-space (x_c,y_c,z_c) to normalized device coords."""
    if z_c <= 0:
        return None, None
    x_ndc = (x_c / z_c) * ASPECT_INV / TAN_HALF_FOV
    y_ndc = (y_c / z_c) / TAN_HALF_FOV  # TAN_HALF_FOV ==1
    return x_ndc, y_ndc

def ndc_to_px(x_ndc: float, y_ndc: float) -> Tuple[float, float]:
    """Convert NDC coords to pixel coords on a SCR_W×SCR_H image."""
    x_px = (x_ndc + 1) * SCR_W / 2
    y_px = (1 - y_ndc) * SCR_H / 2
    return x_px, y_px

def bbox_for_target(
    pov_eye: Vector3,
    r: Vector3,
    u: Vector3,
    f: Vector3,
    tgt_x: float,
    tgt_y: float,
    tgt_z: float,
    want_debug: bool = False
) -> Tuple[Tuple[float, float, float, float] | None, List[dict]]:
    """
    Compute 2D bounding box for a target's axis-aligned 32×73 UU hitbox.
    Returns (bbox, debug_list). If bbox is None, debug_list contains per-corner info.
    """
    pts_px: List[Tuple[float, float]] = []
    dbg: List[dict] = []
    for dx in (-HITBOX_HALF_W, +HITBOX_HALF_W):
        for dy in (-HITBOX_HALF_W, +HITBOX_HALF_W):
            for dz in (0.0, HITBOX_H):
                world_pt = Vector3(tgt_x + dx, tgt_y + dy, tgt_z + dz)
                x_c, y_c, z_c = world_to_cam(world_pt, pov_eye, r, u, f)
                x_ndc, y_ndc = cam_to_ndc(x_c, y_c, z_c)
                if x_ndc is None or y_ndc is None:
                    if want_debug:
                        dbg.append({"cam": (x_c, y_c, z_c), "ndc": None})
                    continue
                px, py = ndc_to_px(x_ndc, y_ndc)
                pts_px.append((px, py))
                if want_debug:
                    dbg.append({"cam": (x_c, y_c, z_c), "ndc": (x_ndc, y_ndc), "px": (px, py)})

    if not pts_px:
        return None, dbg

    xs, ys = zip(*pts_px)
    xmin = max(0, min(xs))
    ymin = max(0, min(ys))
    xmax = min(SCR_W, max(xs))
    ymax = min(SCR_H, max(ys))
    return (xmin, ymin, xmax, ymax), dbg

# ───────── main ─────────
def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--demofile", required=True, type=Path)
    cli.add_argument("--out", type=Path)
    cli.add_argument("--progress", action="store_true")
    cli.add_argument("--max-events", type=int)
    cli.add_argument("--optimize", action="store_true",
                     help="strip POVs with empty visible list")
    cli.add_argument("--boundingbox", action="store_true",
                     help="add 2-D bbox for every visible target")
    cli.add_argument("--debugbbx", action="store_true",
                     help="emit debug JSON when bbox fails")
    args = cli.parse_args()

    if not args.demofile.exists():
        cli.error("demo not found")

    demo = Demo(path=args.demofile, verbose=False)
    demo.parse(events=EVENTS, player_props=PLAYER_PROPS, other_props=None)

    # Build SteamID → userId map from kills/damages
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

    # Collect candidate ticks (± MARGIN around events)
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

    # Group tick rows in one pass
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
    for t in iterator:
        rows = buckets.get(t)
        if not rows:
            continue

        povs: List[Dict[str, Any]] = []
        geom: List[tuple[int, Vector3, Vector3]] = []

        for r in rows:
            if int(r.get("health", 0)) <= 0:
                continue
            uid = uid_of(r, uid_map)
            if uid == -1:
                continue
            crouch = crouched(r)
            px, py, pz = float(r["X"]), float(r["Y"]), float(r["Z"])
            yaw = float(first(r, ("yaw", "view_yaw", "view_x")))
            pitch = float(first(r, ("pitch", "view_pitch", "view_y")))
            eye_z = pz + (CROUCH_Z if crouch else STAND_Z)
            pov_eye = Vector3(px, py, eye_z)

            pov = {
                "steamid": int(r["steamid"]),
                "userId": uid,
                "team": first(r, ("side", "team_name", "team")),
                "health": int(r["health"]),
                "x": px, "y": py, "z": pz,
                "yaw": yaw, "pitch": pitch,
                "crouching": crouch,
                "spectate_command": f"spec_goto {px:.1f} {py:.1f} {eye_z:.1f} {pitch:.1f} {yaw:.1f}",
                "visible": []
            }
            povs.append(pov)
            geom.append((uid, pov_eye, ypr_to_vec(yaw, pitch)))

        uid_to_pov = {p["userId"]: p for p in povs}

        for pov_uid, eye, fwd in geom:
            pov_entry = uid_to_pov[pov_uid]
            f_cam, r_cam, u_cam = build_axes(pov_entry["yaw"], pov_entry["pitch"])
            for tgt_uid, tgt_eye, _ in geom:
                if pov_uid == tgt_uid:
                    continue
                yaw_e, pitch_e = angle_error(eye, fwd, tgt_eye)
                if yaw_e > (H_FOV_DEG / 2 - EDGE) or pitch_e > (V_FOV_DEG / 2 - EDGE):
                    continue
                if vis_checker and not vis_checker.is_visible(eye, tgt_eye):
                    continue

                tgt = uid_to_pov[tgt_uid]
                vis_entry = {
                    "steamid": tgt["steamid"],
                    "userId": tgt_uid,
                    "team": tgt["team"],
                    "health": tgt["health"],
                    "x": tgt["x"], "y": tgt["y"], "z": tgt["z"],
                    "yaw": tgt["yaw"], "pitch": tgt["pitch"],
                    "crouching": tgt["crouching"],
                }
                if args.boundingbox:
                    bbox, dbg = bbox_for_target(
                        eye, r_cam, u_cam, f_cam,
                        tgt["x"], tgt["y"], tgt["z"],
                        args.debugbbx
                    )
                    if bbox:
                        vis_entry["bbox"] = [round(b, 1) for b in bbox]
                    elif args.debugbbx:
                        # print debug info to stdout
                        print(json.dumps({
                            "tick": t,
                            "pov": pov_uid,
                            "tgt": tgt_uid,
                            "debug": dbg
                        }))

                pov_entry["visible"].append(vis_entry)

        if args.optimize:
            povs = [p for p in povs if p["visible"]]

        if any(p["visible"] for p in povs):
            results.append({"tick": t, "players": povs})

    output = {"demofile": str(args.demofile.resolve()), "ticks": results}
    out_stream = open(args.out, "w") if args.out else sys.stdout
    json.dump(output, out_stream, indent=2)
    if args.out:
        out_stream.close()

    sys.stderr.write(f"[tick_finder] wrote {len(results):,} moments.\n")


if __name__ == "__main__":
    main()
