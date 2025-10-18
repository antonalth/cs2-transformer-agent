#!/usr/bin/env python3
r"""
ARP Viewer — interactive visualizer for CS2 autoregressive model

Features
- Loads a specified checkpoint and runs inference on a random T-frame window
  sampled from your dataset (train/val) **without** using DALI.
- Uses **pre-embedded** features when available (video/audio); otherwise falls
  back to on-the-fly vision path for the selected frames only (CPU/GPU OK).
- Displays all 5 player POVs in a compact **3x2 grid** (one empty slot is used
  to show global game-state labels — ground-truth vs prediction).
- Controls:  
    q – quit  
    j – next tick (frame)  
    k – previous tick  
    n – new random sample (reruns model on another random T-window)
- Overlays (per POV): green = ground-truth, red = prediction.  
  Lightweight HUD inspired by `testing/lmdb/lmdb_inspect_linux.py`.

Assumptions & integration
- LMDB contains meta + per-tick records (`*_INFO` and normal tick keys) as
  produced by your pipeline. We derive:
  \- alive mask, stats (health/armor/money), mouse delta, positions, keyboard
     bitmasks, eco/inventory bitmasks, active weapon, round state, enemy pos.
- Recordings (MP4) live under `<data_root>/recordings/<demoname>/...` as in
  your repo. We read frames for display via OpenCV (not DALI).
- Pre-embedded features (optional but preferred for inference):  
  `<data_root>/vit_embed/<demoname>/<something>.npy`  
  `<data_root>/aud_embed/<demoname>/<something>.npy`  
  If not found for the sampled window, we fall back to images path in model.

Place this file at: `transformers/testing/model/arp_viewer.py`
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import lmdb
import msgpack
import msgpack_numpy as mpnp
import numpy as np
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------------------
# Import model (uses your local tree). Adjust import if your path differs.
# --------------------------------------------------------------------------------------
# Expected at transformers/model/model.py in your repo
try:
    from model import model as cs2_model  # allow running from project root
except Exception:
    # Try relative to this file (testing/model/ -> ../../model)
    import sys
    THIS = Path(__file__).resolve()
    sys.path.append(str(THIS.parents[2] / "model"))
    import model as cs2_model  # type: ignore

CS2Transformer = cs2_model.CS2Transformer
CS2Config = cs2_model.CS2Config

# Optional: item/action label maps from injection_mold (for pretty HUD)
BIT_TO_KEYBOARD = None
BIT_TO_ECO = None
BIT_TO_ITEM = None
try:
    # Try to import sibling injection_mold in project
    import importlib.util, types
    cand_paths = [
        Path(__file__).resolve().parents[2] / "testing" / "lmdb" / "lmdb_inspect_linux.py",
        Path(__file__).resolve().parents[2] / "to_lmdb" / "injection_mold.py",
        Path(__file__).resolve().parents[2] / "injection_mold.py",
    ]
    for p in cand_paths:
        if p.is_file():
            spec = importlib.util.spec_from_file_location("_inj", str(p))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)  # type: ignore
                spec.loader.exec_module(mod)  # type: ignore
                # These names exist in your repo; fall back gracefully if missing
                BIT_TO_KEYBOARD = getattr(mod, "BIT_TO_KEYBOARD", None)
                BIT_TO_ECO = getattr(mod, "BIT_TO_ECO", None)
                BIT_TO_ITEM = getattr(mod, "BIT_TO_ITEM", None)
                break
except Exception:
    pass

if BIT_TO_KEYBOARD is None:
    BIT_TO_KEYBOARD = {i: f"K{i}" for i in range(31)}
if BIT_TO_ECO is None:
    BIT_TO_ECO = {i: f"ECO{i}" for i in range(224)}
if BIT_TO_ITEM is None:
    BIT_TO_ITEM = {i: f"IT{i}" for i in range(128)}

# --------------------------------------------------------------------------------------
# Constants (mirror your training setup)
# --------------------------------------------------------------------------------------
TICK_RATE = 64
EXPECTED_VIDEO_FPS = 32
TICKS_PER_FRAME = TICK_RATE // EXPECTED_VIDEO_FPS  # 2
NUM_POV = 5

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
COLOR_GT = (0, 255, 0)
COLOR_PRED = (0, 0, 255)
COLOR_TEXT = (230, 230, 230)
COLOR_SHADOW = (0, 0, 0)
LINE_TYPE = 1
LINE_HEIGHT = 16

# --------------------------------------------------------------------------------------
# Lightweight LMDB utilities (subset of train3.py)
# --------------------------------------------------------------------------------------
@dataclass
class TeamRound:
    demoname: str
    lmdb_path: str
    round_num: int
    team: str
    start_tick: int
    end_tick: int
    pov_videos: List[str]
    pov_audio: List[str]

@dataclass
class SampleRecord:
    sample_id: int
    demoname: str
    lmdb_path: str
    round_num: int
    team: str
    pov_videos: List[str]
    pov_audio: List[str]
    start_f: int
    start_tick_win: int
    T_frames: int

class LmdbStore:
    def __init__(self, max_readers: int = 512):
        self._envs: Dict[str, lmdb.Environment] = {}
        self._info_cache: Dict[str, Dict[str, Any]] = {}
        self._max_readers = max_readers

    def open(self, lmdb_path: str) -> lmdb.Environment:
        if lmdb_path not in self._envs:
            self._envs[lmdb_path] = lmdb.open(
                lmdb_path, readonly=True, lock=False, readahead=True, max_readers=self._max_readers
            )
        return self._envs[lmdb_path]

    def read_info(self, demoname: str, lmdb_path: str) -> Dict[str, Any]:
        """Reads `<demoname>_INFO` JSON describing rounds and media paths (relative)."""
        cache_key = (demoname, lmdb_path)
        if cache_key in self._info_cache:
            return self._info_cache[cache_key]
        with self.open(lmdb_path).begin(write=False) as txn:
            blob = txn.get(f"{demoname}_INFO".encode("utf-8"))
            if blob is None:
                raise FileNotFoundError(f"Missing _INFO for {demoname}")
            info = json.loads(blob.decode("utf-8"))
        self._info_cache[cache_key] = info
        return info

# Minimal metadata fetcher for ground-truth overlays
class LmdbMetaFetcher:
    """Robust LMDB meta fetcher that tolerates numpy structured scalars/arrays.

    Some pipelines store `game_state` and `player_data` as numpy structured
    scalars (np.void) rather than plain dicts. Access those via field-indexing
    instead of `.get(...)`.
    """

    @staticmethod
    def _key(d: str, r: int, t: str, tick: int) -> bytes:
        return f"{d}_round_{r:03d}_team_{t}_tick_{tick:08d}".encode("utf-8")

    @staticmethod
    def _bitmask_to_weapon_index(mask: np.ndarray) -> int:
        # mask shape [2] uint64 or compatible
        try:
            m = np.asarray(mask).astype(np.uint64).reshape(-1)
        except Exception:
            return -1
        if m.size < 2:
            m = np.pad(m, (0, max(0, 2 - m.size)), constant_values=0)
        if (m[0] | m[1]) == 0:
            return -1
        for i in range(128):
            if (m[i // 64] >> np.uint64(i % 64)) & np.uint64(1):
                return int(i)
        return -1

    @staticmethod
    def _get_field(obj, key, default=None):
        """Safely get a field from dict-like or numpy.void/structured array."""
        if obj is None:
            return default
        # dict-like
        if isinstance(obj, dict):
            return obj.get(key, default)
        # numpy structured scalar or array
        if isinstance(obj, np.void):
            fields = getattr(obj, "dtype", None).fields if hasattr(obj, "dtype") else None
            if fields and key in fields:
                try:
                    return obj[key]
                except Exception:
                    return default
            return default
        if isinstance(obj, np.ndarray) and obj.dtype.fields:
            # structured array: allow field selection
            try:
                return obj[key]
            except Exception:
                return default
        # object with attribute
        if hasattr(obj, key):
            try:
                return getattr(obj, key)
            except Exception:
                return default
        return default

    @staticmethod
    def _to_int(x, default=0):
        try:
            if isinstance(x, (np.generic, np.ndarray)):
                return int(np.asarray(x).item())
            return int(x)
        except Exception:
            return default

    @staticmethod
    def _to_tuple(x, n, default=0.0):
        try:
            arr = np.asarray(x).reshape(-1)
            if arr.size >= n:
                return tuple(float(v) for v in arr[:n])
        except Exception:
            pass
        return (default,) * n

    def fetch_window(self, env: lmdb.Environment, rec: SampleRecord) -> Dict[str, np.ndarray]:
        T = rec.T_frames
        alive = np.zeros((T, NUM_POV), dtype=np.bool_)
        stats = np.zeros((T, NUM_POV, 3), np.float32)            # health, armor, money
        mouse = np.zeros((T, NUM_POV, 2), np.float32)
        pos = np.zeros((T, NUM_POV, 3), np.float32)
        kbd = np.zeros((T, NUM_POV), np.uint32)
        eco = np.zeros((T, NUM_POV, 4), np.uint64)
        inv = np.zeros((T, NUM_POV, 2), np.uint64)
        wep = np.full((T, NUM_POV), -1, np.int32)
        rnd_num = np.full((T,), rec.round_num, np.int32)
        rnd_state = np.zeros((T,), np.uint8)
        enemy_pos = np.zeros((T, NUM_POV, 3), np.float32)

        ticks = rec.start_tick_win + (np.arange(T, dtype=np.int32) * TICKS_PER_FRAME)
        with env.begin(write=False) as txn:
            for f, tick in enumerate(ticks):
                blob = txn.get(self._key(rec.demoname, rec.round_num, rec.team, int(tick)))
                if not blob:
                    continue
                payload = msgpack.unpackb(blob, raw=False, object_hook=mpnp.decode)

                # --- game_state ---
                gs = payload.get("game_state") if isinstance(payload, dict) else None
                if gs is None:
                    continue
                # normalize gs0 to first entry / scalar
                if isinstance(gs, (list, tuple)):
                    gs0 = gs[0] if len(gs) else None
                elif isinstance(gs, np.ndarray):
                    gs0 = gs.flat[0] if gs.size else None
                else:
                    gs0 = gs
                if gs0 is None:
                    continue

                rs = self._get_field(gs0, "round_state", 0)
                rnd_state[f] = np.uint8(self._to_int(rs, 0))

                ep = self._get_field(gs0, "enemy_pos", np.zeros((NUM_POV, 3), np.float32))
                try:
                    enemy_pos[f] = np.asarray(ep, dtype=np.float32).reshape(NUM_POV, 3)
                except Exception:
                    pass

                team_alive = self._get_field(gs0, "team_alive", 0)
                team_alive_mask = self._to_int(team_alive, 0)
                alive_slots = [i for i in range(NUM_POV) if (team_alive_mask >> i) & 1]
                for slot in alive_slots:
                    alive[f, slot] = True

                # --- player_data ---
                pdl = payload.get("player_data") if isinstance(payload, dict) else None
                entries = []
                if isinstance(pdl, (list, tuple)):
                    entries = list(pdl)
                elif isinstance(pdl, np.ndarray):
                    entries = [p for p in pdl.flat]
                if not entries:
                    continue

                # Some pipelines store each entry as [player_struct] or the struct itself
                # Align lengths
                for p_idx, raw in zip(alive_slots, entries):
                    if isinstance(raw, (list, tuple)) and raw:
                        p = raw[0]
                    else:
                        p = raw
                    # Now p is dict-like or numpy.void
                    h = self._to_int(self._get_field(p, "health", 0), 0)
                    a = self._to_int(self._get_field(p, "armor", 0), 0)
                    mny = self._to_int(self._get_field(p, "money", 0), 0)
                    stats[f, p_idx] = [h, a, mny]

                    md = self._get_field(p, "mouse", (0.0, 0.0))
                    mdx, mdy = self._to_tuple(md, 2, 0.0)
                    mouse[f, p_idx] = [mdx, mdy]

                    ps = self._get_field(p, "pos", (0.0, 0.0, 0.0))
                    px, py, pz = self._to_tuple(ps, 3, 0.0)
                    pos[f, p_idx] = [px, py, pz]

                    kb = self._get_field(p, "keyboard_bitmask", 0)
                    kbd[f, p_idx] = np.uint32(self._to_int(kb, 0))

                    eco_mask = self._get_field(p, "eco_bitmask", np.zeros((4,), np.uint64))
                    try:
                        eco[f, p_idx] = np.asarray(eco_mask, dtype=np.uint64).reshape(4)
                    except Exception:
                        pass

                    inv_mask = self._get_field(p, "inventory_bitmask", np.zeros((2,), np.uint64))
                    try:
                        inv[f, p_idx] = np.asarray(inv_mask, dtype=np.uint64).reshape(2)
                    except Exception:
                        pass

                    aw = self._get_field(p, "active_weapon_bitmask", np.zeros((2,), np.uint64))
                    wep[f, p_idx] = self._bitmask_to_weapon_index(aw)

        return {
            "alive_mask": alive,
            "stats": stats,
            "mouse_delta": mouse,
            "position": pos,
            "keyboard_mask": kbd,
            "eco_mask": eco,
            "inventory_mask": inv,
            "active_weapon_idx": wep,
            "round_number": rnd_num,
            "round_state_mask": rnd_state,
            "enemy_positions": enemy_pos,
        }

class VideoWindow:
    def __init__(self, paths: List[str], fps: float = EXPECTED_VIDEO_FPS):
        assert len(paths) == NUM_POV
        self.caps = [cv2.VideoCapture(p) for p in paths]
        self.fps = fps
        for i, cap in enumerate(self.caps):
            if not cap.isOpened():
                print(f"[warn] Failed to open video for POV{i}: {paths[i]}")

    def read_frames(self, start_f: int, T: int) -> List[Optional[np.ndarray]]:
        frames_per_pov: List[Optional[np.ndarray]] = [None] * NUM_POV
        for i, cap in enumerate(self.caps):
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            buf = []
            ok = True
            for _ in range(T):
                ok, frm = cap.read()
                if not ok:
                    break
                buf.append(frm)
            frames_per_pov[i] = np.stack(buf, axis=0) if buf else None
        return frames_per_pov

    def release(self):
        for c in self.caps:
            try:
                c.release()
            except Exception:
                pass

# --------------------------------------------------------------------------------------
# Pre-embedded loader (optional). Tries to find contiguous [T,5,d] segment.
# If not resolvable, returns None and we will use raw images path.
# --------------------------------------------------------------------------------------
class PreembedResolver:
    def __init__(self, data_root: str, demoname: str):
        self.data_root = Path(data_root)
        self.demo = demoname
        self.vit_root = self.data_root / "vit_embed" / demoname
        self.aud_root = self.data_root / "aud_embed" / demoname

    def _glob_candidates(self, kind: str) -> List[Path]:
        root = self.vit_root if kind == "vit" else self.aud_root
        if not root.is_dir():
            return []
        # Be permissive — accept any .npy here; your pipeline typically saves
        # one array per POV segment. We'll try to match by round/team substring.
        return sorted(root.rglob("*.npy"))

    @staticmethod
    def _normalize_embed(arr: np.ndarray) -> Optional[np.ndarray]:
        """Return array as [T,5,D] if possible; otherwise None.

        Accepts common shapes like [T,5,D] or [5,T,D]. Collapses any remaining
        trailing dims into D. Ensures positive strides / contiguous memory.
        """
        if arr is None:
            return None
        a = np.asarray(arr)
        if a.ndim < 2:
            return None
        # If exactly 3D, try [T,5,D] or [5,T,D]
        if a.ndim == 3:
            if a.shape[1] == NUM_POV:
                # [T,5,D]
                T, P, D = a.shape
                a = a.reshape(T, P, D)
            elif a.shape[0] == NUM_POV:
                # [5,T,D] -> swap axes
                a = np.moveaxis(a, 0, 1)  # [T,5,D]
            else:
                # attempt to locate POV axis=5 and time as the other large axis
                pov_axis = next((i for i, s in enumerate(a.shape) if s == NUM_POV), None)
                if pov_axis is None:
                    return None
                time_axis = 0 if pov_axis != 0 else 1
                a = np.moveaxis(a, [time_axis, pov_axis], [0, 1])
                a = a.reshape(a.shape[0], a.shape[1], -1)
        else:
            # For >=4D: identify POV axis=5 and move time axis to 0, then flatten the rest
            pov_axis = next((i for i, s in enumerate(a.shape) if s == NUM_POV), None)
            if pov_axis is None:
                return None
            # choose time axis as the largest axis != pov_axis
            candidates = [(i, s) for i, s in enumerate(a.shape) if i != pov_axis]
            time_axis = max(candidates, key=lambda x: x[1])[0]
            a = np.moveaxis(a, [time_axis, pov_axis], [0, 1])
            T = a.shape[0]
            a = a.reshape(T, NUM_POV, -1)
        return np.ascontiguousarray(a)

    def load_window(self, round_num: int, team: str, start_f: int, T: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (video_embeddings [T,5,d], audio_embeddings [T,5,da]) or (None, None)."""
        vit_files = self._glob_candidates("vit")
        aud_files = self._glob_candidates("aud")
        if not vit_files:
            return None, None
        # Heuristic: pick first set whose filename mentions round/team.
        pick_v = [p for p in vit_files if f"round_{round_num:03d}" in p.stem and f"team_{team}" in p.stem]
        pick_a = [p for p in aud_files if f"round_{round_num:03d}" in p.stem and f"team_{team}" in p.stem]
        try:
            if pick_v:
                arr_v = np.load(pick_v[0], mmap_mode=None)
                arr_v = self._normalize_embed(arr_v)
                if arr_v is None:
                    return None, None
                Ttot = arr_v.shape[0]
                s = min(start_f, max(0, Ttot - T))
                arr_v = np.ascontiguousarray(arr_v[s : s + T])
            else:
                return None, None
            arr_a = None
            if pick_a:
                a = np.load(pick_a[0], mmap_mode=None)
                a = self._normalize_embed(a)
                if a is not None:
                    Ttot_a = a.shape[0]
                    s = min(start_f, max(0, Ttot_a - T))
                    arr_a = np.ascontiguousarray(a[s : s + T])
            return arr_v, arr_a
        except Exception:
            return None, None

# --------------------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------------------

def put_text(img: np.ndarray, text: str, xy: Tuple[int, int], color=COLOR_TEXT) -> Tuple[int, int]:
    x, y = xy
    # shadow
    cv2.putText(img, text, (x + 1, y + 1), FONT, FONT_SCALE, COLOR_SHADOW, LINE_TYPE, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, FONT_SCALE, color, LINE_TYPE, cv2.LINE_AA)
    return (x, y + LINE_HEIGHT)


def bitmask_to_keys(mask: int, mapping: Dict[int, str], top_k: int = 6) -> List[str]:
    if mask == 0:
        return []
    keys = []
    i = 0
    while mask and i < 64:
        if mask & 1:
            keys.append(mapping.get(i, str(i)))
        mask >>= 1
        i += 1
    return keys[:top_k]


def multi_hot_to_topk(logits: torch.Tensor, mapping: Dict[int, str], k: int = 3) -> List[str]:
    # logits [C]; apply sigmoid and take top-k
    probs = torch.sigmoid(logits.detach().float())
    topk = torch.topk(probs, k=min(k, probs.numel()))
    idx = topk.indices.cpu().tolist()
    return [mapping.get(i, str(i)) for i in idx]


def draw_pov_panel(frame: np.ndarray,
                   gt: Dict[str, Any], pred: Dict[str, torch.Tensor],
                   f_idx: int, pov_idx: int) -> np.ndarray:
    """Overlay compact GT (green) and prediction (red) for one POV on a single frame image."""
    if frame is None:
        frame = np.zeros((360, 640, 3), np.uint8)
        put_text(frame, "[missing frame]", (8, 18), (0, 200, 255))
        return frame

    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Alive check -> if dead, darken and label
    alive = bool(gt["alive_mask"][f_idx, pov_idx])
    if not alive:
        overlay[:] = 0
        put_text(overlay, "DEAD", (8, 18), (0, 200, 255))
        return overlay

    # --- GT in green ---
    y = 20
    s_h, s_a, s_m = gt["stats"][f_idx, pov_idx].tolist()
    y = put_text(overlay, f"GT hp:{int(s_h)} ar:{int(s_a)} $:{int(s_m)}", (8, y), COLOR_GT)[1]
    kbd_names = bitmask_to_keys(int(gt["keyboard_mask"][f_idx, pov_idx]), BIT_TO_KEYBOARD)
    y = put_text(overlay, f"GT keys: {', '.join(kbd_names)}", (8, y), COLOR_GT)[1]
    wep_idx = int(gt["active_weapon_idx"][f_idx, pov_idx])
    wep_name = BIT_TO_ITEM.get(wep_idx, str(wep_idx)) if wep_idx >= 0 else "None"
    y = put_text(overlay, f"GT wep: {wep_name}", (8, y), COLOR_GT)[1]
    mdx, mdy = gt["mouse_delta"][f_idx, pov_idx].tolist()
    y = put_text(overlay, f"GT mouse: ({mdx:+.3f}, {mdy:+.3f})", (8, y), COLOR_GT)[1]

    # --- Pred in red ---
    # Each model head may be absent; guard accordingly
    if pred:
        if "stats" in pred:
            s = pred["stats"][f_idx].detach().float().cpu().numpy().tolist()
            y = put_text(overlay, f"PR hp:{s[0]:.0f} ar:{s[1]:.0f} $:{s[2]:.0f}", (8, y), COLOR_PRED)[1]
        if "keyboard_logits" in pred:
            names = multi_hot_to_topk(pred["keyboard_logits"][f_idx], BIT_TO_KEYBOARD, k=4)
            y = put_text(overlay, f"PR keys: {', '.join(names)}", (8, y), COLOR_PRED)[1]
        if "active_weapon_idx" in pred:
            idx = int(torch.argmax(pred["active_weapon_idx"][f_idx]).item())
            y = put_text(overlay, f"PR wep: {BIT_TO_ITEM.get(idx, str(idx))}", (8, y), COLOR_PRED)[1]
        if "mouse_delta_deg" in pred:
            m = pred["mouse_delta_deg"][f_idx].detach().float().cpu().numpy().tolist()
            y = put_text(overlay, f"PR mouse: ({m[0]:+.3f}, {m[1]:+.3f})", (8, y), COLOR_PRED)[1]

    return overlay


def draw_global_panel(size: Tuple[int, int],
                      gt_round_state: int,
                      pred_round_state_logits: Optional[torch.Tensor]) -> np.ndarray:
    h, w = size
    img = np.zeros((h, w, 3), np.uint8)
    y = 20
    y = put_text(img, "Global", (8, y), (180, 180, 255))[1]
    y = put_text(img, f"GT round_state: {gt_round_state}", (8, y), COLOR_GT)[1]
    if pred_round_state_logits is not None:
        idx = int(torch.argmax(pred_round_state_logits).item())
        y = put_text(img, f"PR round_state: {idx}", (8, y), COLOR_PRED)[1]
    return img

# --------------------------------------------------------------------------------------
# Viewer core
# --------------------------------------------------------------------------------------

@dataclass
class ViewerState:
    env: lmdb.Environment
    rounds: List[TeamRound]
    data_root: str
    model: cs2_model.CS2Transformer
    device: torch.device
    use_preembedded: bool

    # Current sample
    rec: Optional[SampleRecord] = None
    frames_5x: Optional[List[Optional[np.ndarray]]] = None  # list of [T,H,W,3] or None per pov
    gt: Optional[Dict[str, np.ndarray]] = None
    preds: Optional[Dict[str, Any]] = None

    # Cursor
    t_idx: int = 0


def build_rounds_from_info(store: LmdbStore, data_root: str, demoname: str, lmdb_path: str) -> List[TeamRound]:
    """Build eligible rounds with valid media (5 pov videos + 5 audio)."""
    info = store.read_info(demoname, lmdb_path)
    base = Path(data_root) / "recordings" / demoname
    rounds: List[TeamRound] = []
    for r in info.get("rounds", []):
        pov_v = [str((base / p).resolve()) for p in r.get("pov_videos", [])]
        pov_a = [str((base / p).resolve()) for p in r.get("pov_audio", [])]
        if len(pov_v) != 5 or len(pov_a) != 5:
            continue
        if not all(os.path.exists(p) for p in pov_v + pov_a):
            continue
        rounds.append(TeamRound(
            demoname=demoname,
            lmdb_path=lmdb_path,
            round_num=int(r.get("round_num", 0)),
            team=str(r.get("team", "T")).upper(),
            start_tick=int(r.get("start_tick", 0)),
            end_tick=int(r.get("end_tick", 0)),
            pov_videos=pov_v,
            pov_audio=pov_a,
        ))
    return rounds


def sample_random_window(tr: TeamRound, T_frames: int, sample_id: int = 0) -> SampleRecord:
    # Choose a start_f within [0 .. frame_count - T]
    total_frames = max(0, (tr.end_tick - tr.start_tick) // TICKS_PER_FRAME)
    if total_frames <= T_frames:
        start_f = 0
    else:
        start_f = random.randint(0, total_frames - T_frames)
    start_tick_win = tr.start_tick + start_f * TICKS_PER_FRAME
    return SampleRecord(
        sample_id=sample_id,
        demoname=tr.demoname,
        lmdb_path=tr.lmdb_path,
        round_num=tr.round_num,
        team=tr.team,
        pov_videos=tr.pov_videos,
        pov_audio=tr.pov_audio,
        start_f=start_f,
        start_tick_win=start_tick_win,
        T_frames=T_frames,
    )


def run_model_on_record(state: ViewerState) -> None:
    """Build a CS2Batch with either preembedded or images path and run the model.
    Stores `state.preds`.
    """
    rec = state.rec
    assert rec is not None

    # Fetch GT meta for overlay
    fetcher = LmdbMetaFetcher()
    gt = fetcher.fetch_window(state.env, rec)
    state.gt = gt

    # Frames for display (OpenCV BGR -> RGB later if needed by model)
    vw = VideoWindow(rec.pov_videos)
    frames_5x = vw.read_frames(rec.start_f, rec.T_frames)  # list of [T,H,W,3] or None
    vw.release()
    state.frames_5x = frames_5x

    # Build batch
    device = state.device
    model_dtype = next(state.model.parameters()).dtype
    B, T, P = 1, rec.T_frames, NUM_POV

    # Alive mask
    alive_mask = torch.from_numpy(gt["alive_mask"]).unsqueeze(0).to(device)

    # Try preembedded first (video_embeddings path)
    vid_embeds = None
    mel_embeds = None
    if state.use_preembedded:
        resolver = PreembedResolver(state.data_root, rec.demoname)
        v_arr, a_arr = resolver.load_window(rec.round_num, rec.team, rec.start_f, rec.T_frames)
        if v_arr is not None and v_arr.shape[:2] == (T, P):
            v_arr = np.ascontiguousarray(v_arr)
            vid_embeds = torch.from_numpy(v_arr).unsqueeze(0).to(device)
        if a_arr is not None and a_arr.shape[:2] == (T, P):
            a_arr = np.ascontiguousarray(a_arr)
            mel_embeds = torch.from_numpy(a_arr).unsqueeze(0).to(device)

    batch: Dict[str, torch.Tensor] = {"alive_mask": alive_mask}

    if vid_embeds is not None:
        batch["video_embeddings"] = vid_embeds.to(dtype=model_dtype)  # [1,T,5,d_model_backbone]
        # Audio: if not embedded found, set zeros token so fuser still works
        if mel_embeds is not None:
            batch["mel_spectrogram"] = mel_embeds.to(dtype=model_dtype)  # allow audio encoder/fuser to accept
        else:
            # Zero mel — shape expected by audio encoder: [B,T,5,2,128,1] (stereo, 1 frame)
            batch["mel_spectrogram"] = torch.zeros((B, T, P, 2, 128, 1), device=device, dtype=model_dtype)
    else:
        # Fallback: use raw images (on-the-fly encoder path).
        # Convert frames to tensor shape [B,T,5,3,H,W] RGB and scale 0..1
        # If any POV missing, fill with black.
        H = 360
        W = 640
        buf = np.zeros((T, P, H, W, 3), dtype=np.uint8)
        for p in range(P):
            arr = frames_5x[p]
            if arr is None:
                continue
            # resize if needed
            ah, aw = arr.shape[1:3]
            if (ah, aw) != (H, W):
                arr = np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR) for f in arr], axis=0)
            buf[:, p] = arr
        # BGR->RGB without negative strides (avoid buf[..., ::-1])
        buf = buf[..., [2, 1, 0]]  # creates a positive-stride copy
        buf = np.ascontiguousarray(buf)
        images = torch.from_numpy(buf).permute(0, 1, 4, 2, 3).unsqueeze(0).float() / 255.0
        batch["images"] = images.to(device)
        batch["mel_spectrogram"] = torch.zeros((B, T, P, 2, 128, 1), device=device, dtype=model_dtype)

    state.model.eval()
    with torch.no_grad():
        preds = state.model(batch)

    # Normalize predictions into convenient [T]-major lists
    out: Dict[str, Any] = {"player": [], "game_strategy": {}}
    # Player heads per POV
    for i in range(NUM_POV):
        p: Dict[str, torch.Tensor] = {}
        for key in ["stats", "pos_heatmap_logits", "mouse_delta_deg", "keyboard_logits",
                    "eco_logits", "inventory_logits", "active_weapon_idx"]:
            if isinstance(preds["player"][i].get(key, None), torch.Tensor):
                # shape [B,T,...] => [T,...]
                p[key] = preds["player"][i][key][0]
        out["player"].append(p)
    # Global head
    gs = preds.get("game_strategy", {})
    if isinstance(gs, dict):
        for key in ["enemy_pos_heatmap_logits", "round_state_logits", "round_number_logits"]:
            if isinstance(gs.get(key, None), torch.Tensor):
                out.setdefault("game_strategy", {})[key] = gs[key][0]
    state.preds = out
    state.t_idx = 0


def compose_grid(state: ViewerState) -> np.ndarray:
    assert state.frames_5x is not None and state.gt is not None and state.preds is not None
    T = state.rec.T_frames  # type: ignore
    f = np.clip(state.t_idx, 0, T - 1)

    # Base cell size from actual frames if available
    # Try to use first available POV frame to set size; default 360x640 otherwise
    H, W = 360, 640
    for arr in state.frames_5x:
        if arr is not None and len(arr) > 0:
            H, W = arr.shape[1], arr.shape[2]
            break

    # Generate 5 POV panels
    panels: List[np.ndarray] = []
    for i in range(NUM_POV):
        frame = state.frames_5x[i][f] if state.frames_5x[i] is not None and f < len(state.frames_5x[i]) else None
        pred_dict = state.preds["player"][i]
        panel = draw_pov_panel(frame, state.gt, pred_dict, f, i)
        panels.append(panel)

    # Sixth panel = global: round_state
    pred_gs = state.preds.get("game_strategy", {})
    pred_round_state = pred_gs.get("round_state_logits", None)
    global_panel = draw_global_panel((H, W), int(state.gt["round_state_mask"][f]), pred_round_state[f] if pred_round_state is not None else None)

    # 3x2 grid: [0,1,2; 3,4,5]
    row1 = np.concatenate(panels[0:3], axis=1)
    row2 = np.concatenate(panels[3:5] + [global_panel], axis=1)
    grid = np.concatenate([row1, row2], axis=0)

    # Footer bar with controls/status
    bar_h = 28
    bar = np.zeros((bar_h, grid.shape[1], 3), np.uint8)
    txt = f"tick {state.rec.start_tick_win + f*TICKS_PER_FRAME}  |  frame {f+1}/{T}  |  j/k=step  n=new  q=quit"
    put_text(bar, txt, (10, 18), (200, 220, 255))
    return np.concatenate([grid, bar], axis=0)


# --------------------------------------------------------------------------------------
# Checkpoint loading helpers
# --------------------------------------------------------------------------------------

def load_model(ckpt_path: str, device: torch.device, dtype: str = "bf16") -> cs2_model.CS2Transformer:
    cfg = CS2Config()
    if dtype in ("bf16", "fp16", "fp32"):
        cfg.compute_dtype = dtype
        cfg.amp_autocast = (dtype != "fp32")
    model = CS2Transformer(cfg)
    # Robust state_dict loader
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = None
    for k in ["model_state_dict", "state_dict", "model", "module"]:
        if isinstance(ckpt, dict) and k in ckpt and isinstance(ckpt[k], dict):
            state = ckpt[k]
            break
    if state is None and isinstance(ckpt, dict):
        # Maybe raw state_dict
        state = {k: v for k, v in ckpt.items() if isinstance(v, (torch.Tensor, np.ndarray))}
    if state:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[load] missing: {len(missing)} params (non-fatal)")
        if unexpected:
            print(f"[load] unexpected: {len(unexpected)} params (non-fatal)")
    model.to(device)
    return model

# --------------------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="CS2 ARP viewer (no DALI)")
    p.add_argument("--ckpt", required=True, help="Path to model checkpoint .pt")
    p.add_argument("--lmdb", required=True, help="Path to a single game LMDB (e.g., /data/lmdb/demoname.lmdb)")
    p.add_argument("--demoname", required=True, help="Demoname key inside LMDB (e.g., match_2024_09_18_123456)")
    p.add_argument("--data-root", required=True, help="Root folder containing recordings/ and (optionally) vit_embed/, aud_embed/")
    p.add_argument("--T", type=int, default=64, help="Number of frames per sample window")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--cpu", action="store_true", help="Force CPU (debug)")
    p.add_argument("--no-preembedded", action="store_true", help="Disable preembedded and always use images path")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = load_model(args.ckpt, device, dtype=args.dtype)

    store = LmdbStore()
    env = store.open(args.lmdb)
    rounds = build_rounds_from_info(store, args.data_root, args.demoname, args.lmdb)
    if not rounds:
        print("No valid rounds with media found. Check --demoname/--lmdb/--data-root.")
        return

    state = ViewerState(
        env=env,
        rounds=rounds,
        data_root=args.data_root,
        model=model,
        device=device,
        use_preembedded=(not args.no_preembedded),
    )

    def new_random_sample():
        tr = random.choice(state.rounds)
        state.rec = sample_random_window(tr, args.T)
        run_model_on_record(state)

    new_random_sample()

    win_name = "CS2-ARP Viewer"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1920, 1080)

    while True:
        canvas = compose_grid(state)
        cv2.imshow(win_name, canvas)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('j'):
            state.t_idx = min(state.t_idx + 1, state.rec.T_frames - 1)  # type: ignore
        elif key == ord('k'):
            state.t_idx = max(state.t_idx - 1, 0)
        elif key == ord('n'):
            new_random_sample()
        # ignore others

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
