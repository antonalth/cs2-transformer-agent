#!/usr/bin/env python3
"""
orchestrator.py
===============

Three mutually–exclusive modes:

1.  --generate_jsons  : run tick_finder.py on every .dem file
2.  --generate-data   : drive CS-2, take screenshots, write YOLO labels
3.  --visualize-data  : view / curate the dataset

All paths are **independent**:

    --demodir   /path/to/demos
    --jsondir   /path/to/jsons
    --datadir   /path/to/data       (per-demo IMG / LBL sub-dirs)

YOLO class-ids:
    0  -> T
    1  -> CT
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import mss  # noqa: WPS433
from PIL import Image  # noqa: WPS433
import cv2  # noqa: WPS433

# vconsole.py must be importable (same dir or PYTHONPATH)
try:
    from libs.vconsole import connect as vconsole_connect  # noqa: WPS433
except Exception:  # noqa: BLE001
    vconsole_connect = None

# ───────────────────────────────────────────────────────────
LOG = logging.getLogger("orc")


def run_tick_finder(demofile: Path, json_out: Path, extra: List[str]) -> None:
    cmd = [
        "./tick_finder.py",
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


def get_cs2_bbox() -> Optional[Dict[str, int]]:
    """
    Use `xwininfo` to find the geometry of the "Counter-Strike 2" window.
    Returns a dict with {left, top, width, height} or None if not found.
    """
    try:
        proc = subprocess.run(
            ["xwininfo", "-name", "Counter-Strike 2"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return None

    left = top = width = height = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("Absolute upper-left X:"):
            left = int(line.split()[-1])
        elif line.startswith("Absolute upper-left Y:"):
            top = int(line.split()[-1])
        elif line.startswith("Width:"):
            width = int(line.split()[-1])
        elif line.startswith("Height:"):
            height = int(line.split()[-1])
    if None in (left, top, width, height):
        return None
    return {"left": left, "top": top, "width": width, "height": height}


def grab_cs2_window(fname: Path, quality: int) -> None:
    """
    Grab only the "Counter-Strike 2" window (using xwininfo to locate it)
    and save as JPEG. If window not found, fallback to full primary monitor.
    """
    bbox = get_cs2_bbox()
    with mss.mss() as sct:
        if bbox:
            img = sct.grab(bbox)
        else:
            LOG.warning("CS2 window not found; capturing entire monitor")
            monitor = sct.monitors[1]
            img = sct.grab(monitor)
        im = Image.frombytes("RGB", img.size, img.rgb)
        im.save(fname, format="JPEG", quality=quality)


def bbox_to_yolo(bbox: List[float]) -> Tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) / 2 / 1920
    cy = (ymin + ymax) / 2 / 1080
    w = (xmax - xmin) / 1920
    h = (ymax - ymin) / 1080
    return cx, cy, w, h


def team_to_class(team: str) -> int:
    return 0 if team.upper().startswith("T") else 1


def drain_vconsole(conn) -> None:
    """
    Read and discard all pending lines from vconsole, to keep socket alive.
    """
    try:
        while True:
            item = conn.read(block=False)
            if not item:
                break
            # Optional: LOG.debug("vconsole: %s", item)
    except Exception:
        pass


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
    if vconsole_connect is None:
        LOG.error("vconsole.py not importable; cannot run generate-data mode.")
        return

    conn = vconsole_connect(port=args.port)
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

        # play & pause
        conn.send(f'playdemo "{demo_file}"')
        drain_vconsole(conn)
        LOG.info("loading demo %s", demo_filename)
        time.sleep(args.demo_load_wait)
        conn.send("demo_pause")
        drain_vconsole(conn)
        time.sleep(0.05)

        for tick_obj in meta["ticks"]:
            tick = tick_obj["tick"]
            conn.send(f"demo_gototick {tick}")
            drain_vconsole(conn)
            time.sleep(args.seek_wait)
            for pov in tick_obj["players"]:
                uid = pov["userId"]
                if not pov["visible"]:
                    continue
                conn.send(pov["spectate_command"])
                drain_vconsole(conn)
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


# ──────────────────────────── CLI parsing ──────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="CS-2 demo → YOLO dataset orchestrator")
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
    ap.add_argument("--port", type=int, default=29000)
    ap.add_argument("--demo-load-wait", type=float, default=20.0)
    ap.add_argument("--seek-wait", type=float, default=0.15)
    ap.add_argument("--spectate-wait", type=float, default=0.05)
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("--force", action="store_true")

    ap.add_argument("--verbose", action="store_true")
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
