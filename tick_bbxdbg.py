#!/usr/bin/env python3
"""
tick_bbxdbg.py — interactive CLI to debug and compare multiple bounding-box algorithms
------------------------------------------------------------------------------------

Usage:
    ./tick_bbxdbg.py --demofile PATH.dem

Commands:
    seek TICKNUM    Seek to a specific tick in the demo; list POVs with non-empty visible lists.
    quit            Exit the tool.

For each selected POV:
    • Sends `demo_gototick` & `spec_goto` via MIRV,
    • Captures a screenshot of "Counter-Strike 2",
    • Runs each bounding‐box algorithm on every visible target,
      drawing boxes in different colors and labeling them with algorithm index,
    • Displays the overlaid image in a 1280×720 OpenCV window,
    • Press 'q' in the image window to return to the CLI.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import subprocess
import cv2
import polars as pl
from awpy import Demo
from awpy.vector import Vector3
from awpy.data import TRIS_DIR

from libs.window_capture import capture
from libs.mirv_client import connect
from bbox_algs.alg1 import BoundingBoxCS2 as bboxAlg1
from bbox_algs.alg2 import BoundingBoxCS2 as bboxAlg2
# add near the other imports
from bbox_algs.alg3 import DefaultBBoxAlg as bboxAlg3 
from bbox_algs.alg4 import CS2BBox as bboxAlg4
from bbox_algs.alg5 import DefaultBBoxAlg as bboxAlg5

BB_ALGORITHMS = [
    bboxAlg3,
    bboxAlg4,
    bboxAlg5
]

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

SCR_W, SCR_H = 1280, 720  # assumed screenshot resolution (Client area)

# ───────── Colors for drawing each algorithm’s boxes ─────────
COLORS: List[Tuple[int, int, int]] = [
    (0, 0, 255),    # A0: red
    (0, 255, 0),    # A1: green
    (255, 0, 0),    # A2: blue
    (0, 255, 255),  # A3: yellow
    (255, 255, 0),  # A4: cyan
    (255, 0, 255),  # A5: magenta
]

# ───────── helper functions ─────────
def ypr_to_vec(yaw: float, pitch: float) -> Vector3:
    """Convert yaw & pitch (degrees) to a forward unit vector."""
    y = math.radians(yaw)
    p = math.radians(pitch)
    return Vector3(
        math.cos(p) * math.cos(y),
        math.cos(p) * math.sin(y),
        -math.sin(p),
    )

def build_axes(yaw: float, pitch: float) -> Tuple[Vector3, Vector3, Vector3]:
    """Return (forward, right, up) basis from yaw/pitch."""
    f = ypr_to_vec(yaw, pitch).normalize()
    r = f.cross(Vector3(0, 0, 1)).normalize()
    u = r.cross(f)
    return f, r, u

def angle_error(eye: Vector3, fwd: Vector3, tgt: Vector3) -> Tuple[float, float]:
    """Compute horizontal (yaw) and vertical (pitch) angular error between forward vector and target."""
    d = (tgt - eye).normalize()
    f_h = Vector3(fwd.x, fwd.y, 0).normalize()
    d_h = Vector3(d.x, d.y, 0).normalize()
    yaw_err = math.degrees(math.acos(max(-1, min(1, f_h.dot(d_h)))))
    dot = max(-1, min(1, fwd.dot(d)))
    tot = math.degrees(math.acos(dot))
    pitch_err = math.sqrt(max(0, tot * tot - yaw_err * yaw_err))
    return yaw_err, pitch_err

def first(row: dict[str, Any], keys: Sequence[str]) -> Any | None:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None

def crouched(row: dict[str, Any]) -> bool:
    """Determine if player is crouching based on properties."""
    if row.get("ducking") or row.get("isDucked"):
        return True
    amt = row.get("duckAmount")
    return isinstance(amt, (int, float)) and float(amt) > 0.5

def uid_of(row: dict[str, Any], lut: dict[int, int]) -> int:
    """Return in-game userId or fallback to lut mapping from steamid to userId."""
    for k in ("userId", "user_id", "userid"):
        if k in row and row[k] not in (None, -1, ""):
            return int(row[k])
    return lut.get(int(row["steamid"]), -1)

def load_vis(map_name: str):
    """Attempt to load Awpy's VisibilityChecker from a .tri mesh; return None if unavailable."""
    try:
        from awpy.visibility import VisibilityChecker
    except ImportError:
        return None
    tri = TRIS_DIR / f"{map_name}.tri"
    if not tri.exists():
        sys.stderr.write(f"[bbxdbg] No .tri mesh for {map_name}; LOS disabled.\n")
        return None
    try:
        return VisibilityChecker(path=tri)
    except Exception as e:
        sys.stderr.write(f"[bbxdbg] VisibilityChecker load failed: {e}\n")
        return None


# ───────── main CLI logic ─────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive bounding-box debugger for CS-2 demo, supporting multiple algorithms."
    )
    parser.add_argument(
        "--demofile",
        required=True,
        type=Path,
        help="Path to the .dem file to debug.",
    )
    args = parser.parse_args()

    if not args.demofile.exists():
        parser.error(f"Demo file not found: {args.demofile}")

    
    subprocess.Popen("python s1a_restart_all.py", creationflags=subprocess.CREATE_NEW_CONSOLE)

    tload = time.time()
    # Parse the demo
    print(f"[bbxdbg] Loading demo: {args.demofile} ...")
    demo = Demo(path=args.demofile, verbose=False)
    demo.parse(events=EVENTS, player_props=PLAYER_PROPS, other_props=None)

    # Build SteamID → userId map from kills/damages
    uid_map: dict[int, int] = {}
    for df in (demo.kills, demo.damages):
        if df is None:
            continue
        for sid_col, uid_col in (
            ("attackerSteamId", "attackerUserId"),
            ("victimSteamId", "victimUserId"),
            ("playerSteamId", "playerUserId"),
        ):
            if sid_col in df.columns and uid_col in df.columns:
                for sid, uidv in zip(df[sid_col], df[uid_col]):
                    uid_map[int(sid)] = int(uidv)

    # Build buckets: tick → list of tick-row dicts
    print("[bbxdbg] Building tick buckets ...")
    ticks_df = demo.ticks  # Polars DataFrame
    buckets: dict[int, List[dict]] = defaultdict(list)
    for row in ticks_df.iter_rows(named=True):
        t = int(row["tick"])
        buckets[t].append(row)

    vis_checker = load_vis(demo.header.get("map_name", ""))

    tdone = time.time()
    if(tdone-tload < 25):
        #make sure that cs2 is actually loaded + node
        print("sleeping additionally")
        time.sleep(math.ceil(25-(tdone-tload)))

    # Initialize MIRV client (will block ~10s to connect)
    print("[bbxdbg] Connecting to MIRV WebSocket...")
    try:
        mirv = connect()
        print("[bbxdbg] MIRV connected.")
    except Exception as e:
        print(f"[bbxdbg] Failed to connect MIRV: {e}")
        mirv = None

    #load demofile (hacky sleep)
    full_path = str(args.demofile.resolve())
    mirv.sendCommand(f'playdemo "{full_path}"')
    time.sleep(25)
    mirv.sendCommand("demo_pause")
    time.sleep(1)

    # Instantiate each bounding‐box algorithm once
    algorithms = [
        cls(fov_deg=V_FOV_DEG, screen_w=SCR_W, screen_h=SCR_H)
        for cls in BB_ALGORITHMS
    ]

    print("[bbxdbg] Ready. Type 'seek TICKNUM' to inspect a tick, or 'quit' to exit.")
    while True:
        try:
            cmd = input("bbxdbg> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[bbxdbg] Exiting.")
            break

        if not cmd:
            continue

        if cmd.lower() == "quit":
            print("[bbxdbg] Quitting.")
            break

        if cmd.lower().startswith("seek "):
            parts = cmd.split()
            if len(parts) != 2 or not parts[1].isdigit():
                print("Usage: seek TICKNUM  (where TICKNUM is a non-negative integer)")
                continue
            tick = int(parts[1])
            if tick < 0:
                print("Tick must be non-negative.")
                continue

            rows = buckets.get(tick)
            if not rows:
                print(f"[bbxdbg] No data for tick {tick}.")
                continue

            # Build POV entries and geometry list for this tick
            povs: List[Dict[str, Any]] = []
            geom: List[tuple[int, Vector3, Vector3]] = []

            for r in rows:
                health = int(r.get("health", 0) or 0)
                if health <= 0:
                    continue
                uid = uid_of(r, uid_map)
                if uid == -1:
                    continue
                crouch = crouched(r)
                px, py, pz = float(r["X"]), float(r["Y"]), float(r["Z"])
                yaw = float(first(r, ("yaw", "view_yaw", "view_x")) or 0.0)
                pitch = float(first(r, ("pitch", "view_pitch", "view_y")) or 0.0)
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

            # Determine visibility for each (pov, target) pair
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
                    pov_entry["visible"].append(vis_entry)

            # Filter POVs with non-empty visible list
            povs = [p for p in povs if p["visible"]]
            if not povs:
                print(f"[bbxdbg] No POVs at tick {tick} have any visible targets.")
                continue

            mirv.sendCommand(f"demo_gototick {tick}")
            time.sleep(1)

            # Display demo_gototick once and list POV options
            print(f"\n--- demo_gototick {tick} ---")
            print("Choose which spectate command to send:")
            for i, p in enumerate(povs, start=1):
                print(f"  [{i}] {p['spectate_command']}")

            # Prompt user to select one
            sel = input(f"Select POV (1–{len(povs)}) or 'c' to cancel: ").strip()
            if sel.lower() == "c":
                print()
                continue
            if not sel.isdigit() or not (1 <= int(sel) <= len(povs)):
                print("Invalid selection. Returning to prompt.\n")
                continue
            sel_idx = int(sel) - 1
            chosen_pov = povs[sel_idx]

            # Send commands via MIRV
            if mirv:
                try:
                    mirv.sendCommand(chosen_pov["spectate_command"])
                    print("[bbxdbg] Sent demo_gototick and spectate_command to the game.")
                except Exception as e:
                    print(f"[bbxdbg] Error sending to MIRV: {e}")
            else:
                print("[bbxdbg] MIRV not connected; please run commands manually:")
                print(f"    demo_gototick {tick}")
                print(f"    {chosen_pov['spectate_command']}")

            # Wait for the game to update
            time.sleep(0.5)

            # Capture screenshot and draw bounding boxes
            tmp_path = Path("bbxdbg_capture.jpg")
            try:
                capture("Counter-Strike 2", str(tmp_path), 85)
            except Exception as e:
                print(f"[bbxdbg] Failed to capture window: {e}\n")
                continue

            # Recompute camera axes for drawing bboxes
            px, py, pz = chosen_pov["x"], chosen_pov["y"], chosen_pov["z"]
            yaw = chosen_pov["yaw"]
            pitch = chosen_pov["pitch"]
            crouch = chosen_pov["crouching"]
            eye_z = pz + (CROUCH_Z if crouch else STAND_Z)
            pov_eye = Vector3(px, py, eye_z)
            f_cam, r_cam, u_cam = build_axes(yaw, pitch)

            # Load the screenshot with OpenCV
            img_bgr = cv2.imread(str(tmp_path))
            if img_bgr is None:
                print(f"[bbxdbg] Could not read captured image at {tmp_path}\n")
                continue

            # Draw rectangles for each visible target and each algorithm
            for vis in chosen_pov["visible"]:
                tx, ty, tz = vis["x"], vis["y"], vis["z"]
                tgt_yaw, tgt_pitch, tgt_crouch = vis["yaw"], vis["pitch"], vis["crouching"]

                from_tuple = (px, py, pz, yaw, pitch, crouch)
                to_tuple   = (tx, ty, tz, tgt_yaw, tgt_pitch, tgt_crouch)

                for alg_idx, alg in enumerate(algorithms):
                    bb = alg.calcbb(from_tuple, to_tuple)
                    if bb is not None:
                        xmin, ymin, xmax, ymax = bb
                        color = COLORS[alg_idx % len(COLORS)]
                        cv2.rectangle(img_bgr, (xmin, ymin), (xmax, ymax), color, 2)
                        label = f"A{alg_idx}"
                        cv2.putText(
                            img_bgr,
                            label,
                            (xmin, max(0, ymin - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1,
                            cv2.LINE_AA
                        )

            # Display with OpenCV at fixed size 1280×720 and wait for 'q'
            window_name = f"Tick {tick}, POV {sel_idx}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, SCR_W, SCR_H)
            cv2.imshow(window_name, img_bgr)
            print("[bbxdbg] Press 'q' in the image window to return to CLI.")
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            cv2.destroyWindow(window_name)
            print("[bbxdbg] Returning to prompt.\n")

        else:
            print("Unknown command. Available commands:")
            print("  seek TICKNUM  — Inspect bounding boxes at a given tick")
            print("  quit          — Exit the tool\n")

    # Clean up MIRV connection on exit
    if mirv:
        mirv.close()


if __name__ == "__main__":
    main()
