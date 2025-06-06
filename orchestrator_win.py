#!/usr/bin/env python3
"""
orchestrator.py (refactored for Windows)
========================================

Three mutually–exclusive modes:

1.  --generate_jsons  : run tick_finder.py on every .dem file
2.  --generate-data   : drive CS-2 via MIRV, take screenshots via window_capture, write YOLO labels
3.  --visualize-data  : view / curate the dataset

All paths are **independent**:

    --demodir   /path/to/demos
    --jsondir   /path/to/jsons
    --datadir   /path/to/data       (per-demo IMG / LBL sub-dirs)

YOLO class-ids:
    0  -> T
    1  -> CT

Changes vs. legacy implementation
---------------------------------
* Control of CS-2 is now through libs.mirv_client instead of vconsole.
* Screen-grabs are done using libs.window_capture.capture.
* All vconsole drain logic has been removed.
* Code is Windows-centric: the tick_finder runner uses sys.executable to invoke Python directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import cv2  # noqa: WPS433

# ─────────── external libs (local) ──────────────────────────
try:
    from libs.mirv_client import connect as mirv_connect  # type: ignore[attr-defined]  # noqa: WPS433,E501
except Exception:  # noqa: BLE001
    mirv_connect = None

try:
    from libs.window_capture import capture as capture_window  # type: ignore[attr-defined]  # noqa: WPS433,E501
except Exception:  # noqa: BLE001
    capture_window = None

# ────────────────────────────────────────────────────────────
LOG = logging.getLogger("orc")
WINDOW_TITLE = "Counter-Strike 2"

# ───────────────────────── helpers ──────────────────────────

def run_tick_finder(demofile: Path, json_out: Path, extra: List[str]) -> None:
    """Invoke tick_finder.py via the same Python interpreter on Windows, writing output to JSON."""
    # Resolve tick_finder.py relative to this script
    script_dir = Path(__file__).parent.resolve()
    tick_finder_script = script_dir / "tick_finder.py"
    cmd = [
        sys.executable,
        str(tick_finder_script),
        "--demofile",
        str(demofile),
        "--out",
        str(json_out),
        "--optimize",
        "--boundingbox",
        "--progress",
    ] + extra
    LOG.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def grab_cs2_window(fname: Path, quality: int) -> None:
    """Capture the *client area* of the CS-2 window and save as JPEG."""
    if capture_window is None:
        LOG.error("libs.window_capture import failed; cannot capture screenshots.")
        return
    try:
        capture_window(WINDOW_TITLE, str(fname), quality)
    except Exception as exc:  # noqa: BLE001
        LOG.error("screenshot failed → %s", exc)


# ───────────────────────── bbox utils ───────────────────────

def bbox_to_yolo(bbox: List[float]) -> Tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) / 2 / 1280
    cy = (ymin + ymax) / 2 / 720
    w = (xmax - xmin) / 1280
    h = (ymax - ymin) / 720
    return cx, cy, w, h


def team_to_class(team: str) -> int:
    return 0 if team.upper().startswith("T") else 1


# ─────────────────────────── modes ──────────────────────────

def mode_generate_jsons(args: argparse.Namespace) -> None:
    demodir, jsondir = Path(args.demodir), Path(args.jsondir)
    jsondir.mkdir(parents=True, exist_ok=True)
    extras: List[str] = []
    if args.max_events:
        extras += ["--max-events", str(args.max_events)]

    demos = sorted(demodir.glob("*.dem"))
    for demo in demos:
        json_out = jsondir / f"{demo.name}.json"
        if json_out.exists() and json_out.stat().st_mtime >= demo.stat().st_mtime:
            LOG.info("skip %s (json fresh)", demo.name)
            continue
        try:
            run_tick_finder(demo, json_out, extras)
        except subprocess.CalledProcessError as exc:
            LOG.error("tick_finder failed for %s: %s", demo.name, exc)


def mode_generate_data(args: argparse.Namespace) -> None:
    if mirv_connect is None:
        LOG.error("libs.mirv_client not importable; cannot run generate-data mode.")
        return
    if capture_window is None:
        LOG.error("libs.window_capture not importable; cannot run generate-data mode.")
        return

    # Establish MIRV connection and wait for it to become active
    conn = mirv_connect()
    time.sleep(10)  # give WebSocket time to connect

    demodir, jsondir, datadir = map(Path, (args.demodir, args.jsondir, args.datadir))

    for jpath in sorted(jsondir.glob("*.json")):
        # derive demo filename by stripping ".json"
        demo_filename = jpath.name[:-5]  # "match1.dem.json" → "match1.dem"
        demo_file = demodir / demo_filename
        if not demo_file.exists():
            LOG.warning("demo %s referenced but not present", demo_filename)
            continue
        demo_name = Path(demo_filename).stem  # "match1"
        demo_out = datadir / demo_name
        (demo_out / "IMG").mkdir(parents=True, exist_ok=True)
        (demo_out / "LBL").mkdir(parents=True, exist_ok=True)

        # skip if already processed
        done_imgs = list((demo_out / "IMG").glob("*.jpg"))
        if done_imgs and not args.force:
            LOG.info("skip %s (existing data)", demo_name)
            continue

        with jpath.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        # play & pause (use full Windows path)
        full_path = str(demo_file.resolve())
        conn.sendCommand(f'playdemo "{full_path}"')
        if(args.debug): LOG.info(f"Ran command: 'playdemo {full_path}")
        LOG.info("loading demo %s", demo_filename)
        time.sleep(args.demo_load_wait)
        conn.sendCommand("demo_pause")
        if(args.debug): LOG.info(f"Ran command: 'demo_pause'")
        time.sleep(0.05)

        for tick_obj in meta["ticks"]:
            tick = tick_obj["tick"]
            conn.sendCommand(f"demo_gototick {tick}")
            if(args.debug): LOG.info(f"Ran command: 'demo_gototick {tick}")
            time.sleep(args.seek_wait)
            for pov in tick_obj["players"]:
                uid = pov["userId"]
                if not pov["visible"]:
                    continue
                conn.sendCommand(pov["spectate_command"])
                if(args.debug): LOG.info(f"Ran command: '{pov["spectate_command"]}")
                time.sleep(args.spectate_wait)
                img_name = f"{tick}_{uid}.jpg"
                lbl_name = f"{tick}_{uid}.txt"
                img_path = demo_out / "IMG" / img_name
                lbl_path = demo_out / "LBL" / lbl_name

                grab_cs2_window(img_path, args.jpeg_quality)

                # write YOLO label
                rows: List[str] = []
                for tgt in pov["visible"]:
                    if "bbox" not in tgt:
                        continue
                    cls = team_to_class(tgt["team"])
                    cx, cy, w, h = bbox_to_yolo(tgt["bbox"])
                    rows.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                lbl_path.write_text("\n".join(rows))

        shutil.copy2(jpath, demo_out / "meta.json")
        LOG.info("finished %s", demo_name)

    conn.close()


def draw_boxes(img, lbl_path: Path):
    if not lbl_path.exists():
        return img
    h, w = img.shape[:2]
    with lbl_path.open() as f:
        for ln in f:
            cls, cx, cy, bw, bh = map(float, ln.split())
            xmin = int((cx - bw / 2) * w)
            ymin = int((cy - bh / 2) * h)
            xmax = int((cx + bw / 2) * w)
            ymax = int((cy + bh / 2) * h)
            color = (0, 255, 0) if cls == 0 else (255, 0, 0)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            label = "T" if cls == 0 else "CT"
            cv2.putText(img, label, (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def mode_visualize_data(args: argparse.Namespace) -> None:
    imgs = sorted(Path(args.datadir).glob("**/IMG/*.jpg"))
    if not imgs:
        LOG.warning("no images found")
        return
    idx = 0
    while True:
        path = imgs[idx]
        lbl = path.parent.parent / "LBL" / f"{path.stem}.txt"
        frame = cv2.imread(str(path))
        frame = draw_boxes(frame, lbl)
        if(args.debug): LOG.info(f"Looking at frame: {path.stem}")
        cv2.imshow("dataset", frame)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("j"), 83, 32):  # right/space
            idx = (idx + 1) % len(imgs)
        elif key in (ord("k"), 81):  # left
            idx = (idx - 1) % len(imgs)
        elif key == ord("d"):
            path.unlink(missing_ok=True)
            lbl.unlink(missing_ok=True)
            imgs.pop(idx)
            idx %= max(1, len(imgs))


# ──────────────────────────── CLI parsing ───────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="CS-2 demo → YOLO dataset orchestrator (MIRV)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--generate_jsons", action="store_true")
    g.add_argument("--generate-data", action="store_true")
    g.add_argument("--visualize-data", action="store_true")

    ap.add_argument("--demodir", type=Path, default=Path("demos"))
    ap.add_argument("--jsondir", type=Path, default=Path("jsons"))
    ap.add_argument("--datadir", type=Path, default=Path("data"))

    # tick_finder flags
    ap.add_argument("--max-events", type=int)

    # generate-data tuning
    ap.add_argument("--port", type=int, default=31337, help="mirv socket port (default 31337)")
    ap.add_argument("--demo-load-wait", type=float, default=20.0)
    ap.add_argument("--seek-wait", type=float, default=0.15)
    ap.add_argument("--spectate-wait", type=float, default=0.05)
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("--force", action="store_true")

    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--debug",action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.generate_jsons:
        mode_generate_jsons(args)
    elif args.generate_data:
        mode_generate_data(args)
    elif args.visualize_data:
        mode_visualize_data(args)


if __name__ == "__main__":
    main()
