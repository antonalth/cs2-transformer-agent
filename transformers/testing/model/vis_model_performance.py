#!/usr/bin/env python3
# Visualize CS2Transformer model performance on raw video windows.
# - Loads a specified "best" checkpoint (EMA weights if present).
# - Randomly samples a T-frame window from train/val via the existing DALI + LMDB pipeline.
# - Runs the ViT (on-the-fly) and the autoregressive-masked transformer (single-shot forward over T).
# - Opens an OpenCV window that lets you step through frames and compare targets (green) vs. model predictions (red).
# - Displays the 5 POVs in a 3x2 grid; the sixth cell shows game-state labels (GT vs. Pred).
# - Keyboard: q=quit, j=next frame, k=prev frame, n=new random sample.
# This script plugs into your codebase (model.py / train3.py / injection_mold.py).

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import cv2

# --------------------------
# --- FIXED IMPORT SETUP ---
# --------------------------
# This section correctly locates the necessary project modules based on the
# script's file path, assuming it is run from `transformers/testing/model/`.

# This script's location: transformers/testing/model/viz_model_performance.py
THIS_FILE = Path(__file__).resolve()
# REPO_ROOT should resolve to the `transformers/` directory
REPO_ROOT = THIS_FILE.parents[2]
MODEL_DIR = REPO_ROOT / "model"        # Directory containing model.py, train3.py
TOLMDB_DIR = REPO_ROOT / "to_lmdb"     # Directory containing injection_mold.py

# Add the directories containing the modules to the Python system path
for p in (MODEL_DIR, TOLMDB_DIR):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
    elif not p.exists():
        print(f"[WARN] Expected module directory not found: {p}", file=sys.stderr)

# --------------------------
# --- PROJECT MODULES ---
# --------------------------
try:
    from model import CS2Transformer, CS2Config
    import train3 as T3  # we rely on Manifest, LmdbStore, build_team_rounds, build_epoch_loader
except ImportError as e:
    print(f"Fatal: Failed to import from {MODEL_DIR}. Error: {e}", file=sys.stderr)
    print("Please ensure the script is located at 'transformers/testing/model/viz_model_performance.py'", file=sys.stderr)
    sys.exit(1)

try:
    # For label name mappings (actions, items, etc.). Optional.
    import injection_mold as IM
except Exception:
    print("[INFO] 'injection_mold.py' not found or failed to import. Using fallback IDs for names.", file=sys.stderr)
    IM = None  # we will fall back to IDs if names are unavailable


# --------------------------
# Helpers: checkpoint loading
# --------------------------
def load_weights_from_checkpoint(model: torch.nn.Module, ckpt_path: str) -> Dict[str, Any]:
    """
    Load EMA weights if present, else model weights, from a train3.py checkpoint.
    Returns the full checkpoint dict.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = None
    # Prefer EMA for evaluation if available
    if "ema" in ckpt and ckpt["ema"]:
        try:
            model.load_state_dict(ckpt["ema"], strict=False)
            print(f"[ckpt] Loaded EMA weights from: {ckpt_path}")
            state_dict = ckpt["ema"]
        except Exception as e:
            print(f"[ckpt] EMA load failed ({e}); falling back to 'model' state_dict.")
    if state_dict is None:
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"[ckpt] Loaded model weights from: {ckpt_path}")
    return ckpt


def auto_find_best_ckpt(run_dir: str) -> str:
    """
    Try to find a 'best' checkpoint inside <run_dir>/checkpoints.
    Heuristics: prefer filenames containing 'best', else newest .pt/.pth.
    """
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoints directory found at {ckpt_dir}")
    # Gather candidates
    cands = list(ckpt_dir.glob("*.pt")) + list(ckpt_dir.glob("*.pth"))
    if not cands:
        raise FileNotFoundError(f"No checkpoint files found under {ckpt_dir}")
    # Prefer “best” in filename
    bestish = [p for p in cands if "best" in p.name.lower()]
    if bestish:
        # choose newest among them
        bestish.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(bestish[0])
    # else newest overall
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(cands[0])

def resolve_pov_paths(args) -> List[str]:
    paths = []
    if args.pov_pattern:
        # allow '{}' or '%d'
        for i in range(5):
            try:
                if '{}' in args.pov_pattern:
                    paths.append(args.pov_pattern.format(i))
                elif '%d' in args.pov_pattern:
                    paths.append(args.pov_pattern % i)
                else:
                    raise ValueError("Pattern must include '{}' or '%d'")
            except Exception as e:
                raise RuntimeError(f"Failed to format --pov-pattern for i={i}: {e}")
    else:
        for i in range(5):
            p = getattr(args, f"pov{i}")
            if p is None:
                raise RuntimeError(f"--pov{i} is required when --pov-pattern is not used")
            paths.append(p)
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"POV video not found: {p}")
    return paths

def read_frames_cv2(video_path: str, start_frame: int, T: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # clamp start to valid range
    start = max(0, min(start_frame, max(0, total - T)))
    # jump to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(T):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)  # BGR uint8
    cap.release()
    # if short, pad with black
    while len(frames) < T:
        if frames:
            h, w = frames[-1].shape[:2]
        else:
            h, w = 720, 1280
        frames.append(np.zeros((h, w, 3), np.uint8))
    return frames

def load_5pov_frames(pov_paths: List[str], start_frame: int, T: int, target_hw=(360, 640)) -> torch.Tensor:
    """Return [T, 5, 3, H, W] RGB uint8 tensor for viewer overlays."""
    Tlist = []
    for t in range(T):
        Tlist.append([])

    # Read each POV independently (keeps things simple/reliable)
    pov_frames = [read_frames_cv2(p, start_frame, T) for p in pov_paths]  # each: list of T BGR frames
    Ht, Wt = target_hw
    for t in range(T):
        for pov in range(5):
            bgr = pov_frames[pov][t]
            rgb = cv2.cvtColor(cv2.resize(bgr, (Wt, Ht)), cv2.COLOR_BGR2RGB)
            Tlist[t].append(torch.from_numpy(rgb).permute(2, 0, 1))  # [3,H,W] uint8
    # stack
    x = torch.stack([torch.stack(Tlist[t], dim=0) for t in range(T)], dim=0)  # [T,5,3,H,W]
    return x


# --------------------------
# Grid & overlay helpers
# --------------------------
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
LINE_TYPE = 1
TEXT_SHADOW = (0, 0, 0)
GREEN = (0, 255, 0)  # targets
RED = (0, 0, 255)    # predictions
WHITE = (255, 255, 255)
GRAY = (160, 160, 160)


def draw_text(img: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int], line_height: int = 16) -> int:
    """Draw text with a tiny shadow; returns updated y for next line."""
    cv2.putText(img, text, (x+1, y+1), FONT, FONT_SCALE, TEXT_SHADOW, LINE_TYPE, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, FONT_SCALE, color, LINE_TYPE, cv2.LINE_AA)
    return y + line_height


def blank_panel(w: int, h: int, text: str = "") -> np.ndarray:
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    if text:
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.8, 2)
        cx = (w - tw) // 2
        cy = (h + th) // 2
        cv2.putText(panel, text, (cx, cy), FONT, 0.8, GRAY, 2, cv2.LINE_AA)
    return panel


def img_from_tensor(x: torch.Tensor) -> np.ndarray:
    """
    x: [3, H, W], uint8/float32.
    Returns BGR uint8 for cv2.
    """
    if x.dtype != torch.uint8:
        x = x.clamp(0, 255).to(torch.uint8)
    arr = x.permute(1, 2, 0).cpu().numpy()  # HWC RGB
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def topk_multi_hot_from_logits(logits: torch.Tensor, names: List[str], k: int = 3) -> List[str]:
    """
    Apply a sigmoid-like ranking to get top-k 'active' names.
    Without calibration we just take the top-k logits.
    """
    logits = logits.detach().cpu().float().numpy()
    idx = np.argsort(-logits)[:k]
    labels = []
    for i in idx:
        if 0 <= i < len(names):
            labels.append(names[i])
        else:
            labels.append(f"id:{i}")
    return labels


def map_weapon_idx(idx: int) -> str:
    if IM is not None and hasattr(IM, "item_id_map_names"):
        try:
            return IM.item_id_map_names[int(idx)]
        except Exception:
            return f"weapon_id:{int(idx)}"
    return f"weapon_id:{int(idx)}"


def round_state_name(i: int) -> str:
    # If you have canonical names, add here or use IM. Otherwise show the integer.
    # (Common CS2 states include: 0-Live, 1-BombPlanted, 2-RoundEnd, etc.)
    return str(int(i))


def build_keyboard_names() -> List[str]:
    if IM is not None and hasattr(IM, "KEYBOARD_ONLY_ACTIONS"):
        return list(IM.KEYBOARD_ONLY_ACTIONS)
    # Fallback minimal set
    return [f"K{i}" for i in range(31)]


# --------------------------
# Prediction helpers
# --------------------------
@torch.no_grad()
def run_model_single_window(model: CS2Transformer, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    model.eval()
    moved = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device, non_blocking=True)
        else:
            moved[k] = v

    # If precomputed embeddings are present, ensure the model uses them
    # The key "video_embeddings" matches what is produced by train3.py's BatchAssembler
    if "video_embeddings" in moved:
        # Prevent the model from trying to re-encode raw images if they accidentally exist in the batch
        if "images" in moved:
            moved["images"] = None
    
    preds = model(moved)
    
    # move to CPU for viewer
    def to_cpu_tree(obj):
        if isinstance(obj, torch.Tensor): return obj.detach().cpu()
        if isinstance(obj, dict):         return {k: to_cpu_tree(v) for k, v in obj.items()}
        if isinstance(obj, list):         return [to_cpu_tree(v) for v in obj]
        return obj
    return to_cpu_tree(preds)


def clamp_int(x, lo, hi):
    return max(lo, min(hi, int(x)))


def render_grid_for_time(
    t: int,
    images_t: torch.Tensor,           # [5, 3, H, W] (uint8 or float)
    alive_mask_t: torch.Tensor,       # [5] bool
    preds_t: Dict[str, Any],          # per time t, B=1 removed already
    targets_t: Dict[str, Any],        # per time t, B=1 removed already
    panel_size: Tuple[int, int] = (640, 360),
) -> np.ndarray:
    """
    Compose a 3x2 grid (WxH per panel) showing 5 POVs + a summary panel.
    """
    W, H = panel_size
    # Build six panels in row-major order: (0,0),(1,0),(2,0) / (0,1),(1,1),(2,1)
    panels: List[np.ndarray] = []
    kb_names = build_keyboard_names()

    for p_idx in range(5):
        if bool(alive_mask_t[p_idx].item()):
            frame_bgr = img_from_tensor(images_t[p_idx])
        else:
            frame_bgr = blank_panel(W, H, "DEAD")

        # Overlays
        y = 18
        # --- Targets (green) ---
        tgt_stats = targets_t["player"][p_idx].get("stats", None)
        if tgt_stats is not None:
            hp, armor, money = [float(x) for x in tgt_stats]
            y = draw_text(frame_bgr, f"TGT: HP {hp:.0f}  AR {armor:.0f}  $ {money:.0f}", 8, y, GREEN)
        tgt_mouse = targets_t["player"][p_idx].get("mouse_delta_deg", None)
        if tgt_mouse is not None:
            y = draw_text(frame_bgr, f"TGT: mouse(d,p) {tgt_mouse[0]:.2f}, {tgt_mouse[1]:.2f}", 8, y, GREEN)
        tgt_weapon = targets_t["player"][p_idx].get("active_weapon_idx", None)
        if tgt_weapon is not None:
            y = draw_text(frame_bgr, f"TGT: weapon {map_weapon_idx(int(tgt_weapon))}", 8, y, GREEN)
        tgt_kb = targets_t["player"][p_idx].get("keyboard_logits", None)
        if tgt_kb is not None:
            # target is multi-hot 0/1; show up to 3 active actions
            import numpy as _np
            kb = _np.where(_np.asarray(tgt_kb) > 0.5)[0].tolist()
            kb = kb[:3]
            names = [kb_names[i] if 0 <= i < len(kb_names) else f"id:{i}" for i in kb]
            y = draw_text(frame_bgr, "TGT: " + ", ".join(names), 8, y, GREEN)

        # --- Predictions (red) ---
        pr_stats = preds_t["player"][p_idx].get("stats", None)
        if pr_stats is not None:
            hp, armor, money = [float(x) for x in pr_stats]
            y = draw_text(frame_bgr, f"PRD: HP {hp:.0f}  AR {armor:.0f}  $ {money:.0f}", 8, y, RED)
        pr_mouse = preds_t["player"][p_idx].get("mouse_delta_deg", None)
        if pr_mouse is not None:
            y = draw_text(frame_bgr, f"PRD: mouse(d,p) {pr_mouse[0]:.2f}, {pr_mouse[1]:.2f}", 8, y, RED)
        pr_weapon = preds_t["player"][p_idx].get("active_weapon_idx", None)
        if pr_weapon is not None:
            # categorical logits; show top-1 id/name
            if isinstance(pr_weapon, torch.Tensor):
                i = int(torch.argmax(pr_weapon).item())
            else:
                try:
                    i = int(pr_weapon)
                except Exception:
                    i = -1
            y = draw_text(frame_bgr, f"PRD: weapon {map_weapon_idx(i)}", 8, y, RED)
        pr_kb = preds_t["player"][p_idx].get("keyboard_logits", None)
        if pr_kb is not None:
            # multi-label logits; show top-3
            names = topk_multi_hot_from_logits(pr_kb, kb_names, k=3)
            y = draw_text(frame_bgr, "PRD: " + ", ".join(names), 8, y, RED)

        panels.append(frame_bgr)

    # Sixth panel = summary (game state, etc.)
    summary = blank_panel(W, H)
    y = 24
    # Round number
    rn_tgt = targets_t["game_strategy"].get("round_number", None)
    rn_prd = preds_t["game_strategy"].get("round_number", None) or preds_t["game_strategy"].get("round_number_logits", None)
    if rn_tgt is not None:
        try:
            y = draw_text(summary, f"Round#:  TGT {int(rn_tgt)}", 12, y, GREEN)
        except Exception:
            pass
    if rn_prd is not None:
        # Either regression or logits; try both ways
        if isinstance(rn_prd, torch.Tensor):
            if rn_prd.numel() == 1:
                rn_hat = int(torch.round(rn_prd).item())
            else:
                rn_hat = int(torch.argmax(rn_prd).item()) + 1 # Convert 0-indexed to 1-indexed
        else:
            try:
                rn_hat = int(rn_prd)
            except Exception:
                rn_hat = -1
        y = draw_text(summary, f"Round#:  PRD {rn_hat}", 12, y, RED)

    # Round state
    rs_tgt = targets_t["game_strategy"].get("round_state_logits", None)
    if isinstance(rs_tgt, torch.Tensor):
        rs_tgt_id = int(torch.argmax(rs_tgt).item())
    elif isinstance(rs_tgt, (list, np.ndarray)):
        rs_tgt_id = int(np.argmax(rs_tgt))
    else:
        rs_tgt_id = None
    if rs_tgt_id is not None:
        y = draw_text(summary, f"State:   TGT {round_state_name(rs_tgt_id)}", 12, y, GREEN)

    rs_prd = preds_t["game_strategy"].get("round_state_logits", None)
    if isinstance(rs_prd, torch.Tensor):
        rs_prd_id = int(torch.argmax(rs_prd).item())
    elif isinstance(rs_prd, (list, np.ndarray)):
        rs_prd_id = int(np.argmax(rs_prd))
    else:
        rs_prd_id = None
    if rs_prd_id is not None:
        y = draw_text(summary, f"State:   PRD {round_state_name(rs_prd_id)}", 12, y, RED)

    panels.append(summary)

    # Tile into a 3x2 grid
    row1 = np.concatenate(panels[0:3], axis=1)
    row2 = np.concatenate(panels[3:6], axis=1)
    grid = np.concatenate([row1, row2], axis=0)
    return grid


def slice_time(preds: Dict[str, Any], t: int) -> Dict[str, Any]:
    """
    Reduce predictions/targets to time index t for B=1 and return a simple dict:
    { 'player': [ {...heads...} *5 ], 'game_strategy': {...} }
    """
    out: Dict[str, Any] = {"player": [], "game_strategy": {}}
    # Players
    for i in range(5):
        pi: Dict[str, Any] = {}
        P = preds["player"][i]
        for k, v in P.items():
            # expected shape [B, T, ...]
            if isinstance(v, torch.Tensor):
                pi[k] = v[0, t]
            else:
                pi[k] = v
        out["player"].append(pi)
    # Strategy
    for k, v in preds["game_strategy"].items():
        if isinstance(v, torch.Tensor):
            out["game_strategy"][k] = v[0, t]
        else:
            out["game_strategy"][k] = v
    return out


def slice_targets(targets: Dict[str, Any], t: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {"player": [], "game_strategy": {}}
    for i in range(5):
        pi: Dict[str, Any] = {}
        P = targets["player"][i]
        for k, v in P.items():
            # targets tensors are shape [B, T, ...] already from assembler
            if isinstance(v, torch.Tensor):
                pi[k] = v[0, t]
            else:
                pi[k] = v
        out["player"].append(pi)
    for k, v in targets["game_strategy"].items():
        if isinstance(v, torch.Tensor):
            out["game_strategy"][k] = v[0, t]
        else:
            out["game_strategy"][k] = v
    return out


# --------------------------
# Sampling & UI loop
# --------------------------
def build_min_args(
    data_root: str,
    manifest: str,
    run_dir: str,
    split: str,
    T_frames: int,
    batch_size: int = 1,
    dali_threads: int = 4,
    windows_per_round: int = 1,
    use_precomputed: bool = False,
    seed: int = 123,
):
    """A tiny object with the fields build_epoch_loader expects."""
    class A:
        pass
    a = A()
    a.data_root = data_root
    a.manifest = manifest
    a.run_dir = run_dir
    a.mode = "smoke"
    a.split = split
    a.T_frames = T_frames
    a.batch_size = batch_size
    a.use_precomputed_embeddings = use_precomputed
    a.dali_threads = dali_threads
    a.windows_per_round = windows_per_round
    # extras used downstream (harmless defaults)
    a.epochs = 1
    a.lr = 3e-4
    a.min_lr = 1e-5
    a.weight_decay = 0.1
    a.warmup_steps = 2000
    a.grad_clip = 1.0
    a.accum_steps = 1
    a.optim = "adamw"
    a.lr_schedule = "cosine"
    a.warmup_updates = 1500
    a.cycle_updates = 0
    a.cycle_mult = 2.0
    a.onecycle_div_factor = 100.0
    a.onecycle_final_div_factor = 10000.0
    a.log_every = 50
    a.eval_every = 1000
    a.save_every = 1000
    a.resume = ""
    a.compile = False
    a.ema_decay = 0.999
    a.seed = seed
    a.dist_backend = "nccl"
    a.balance_losses_after_updates = 1500
    a.balance_momentum = 0.99
    a.balance_every_evals = 1
    a.detailed_loss = False
    a.num_steps=10

    return a


def build_random_iterator_and_assembler(args) -> Tuple[Any, Any]:
    """Build the input pipeline for a single epoch and return (iterator, assembler)."""
    # Manifest & games
    manifest = T3.Manifest(args.data_root, args.manifest)
    games = manifest.get_games(args.split)
    if not games:
        raise RuntimeError(f"No games for split '{args.split}'. Check manifest & data_root.")
    store = T3.LmdbStore()
    team_rounds = T3.build_team_rounds(args.data_root, games, store)

    # Iterator for one "epoch" (just need it to fetch samples)
    d_iter, assembler, _ = T3.build_epoch_loader(
        args, epoch=np.random.randint(0, 100000), store=store, team_rounds=team_rounds,
        last_batch_policy=("partial" if args.split.lower() == "val" else "drop"),
        rank=0, world_size=1
    )
    return d_iter, assembler

def build_preembedded_iter(args) -> Tuple[Any, Any]:
    # Use your existing machinery but ask for precomputed embeddings only.
    args.use_precomputed_embeddings = True
    manifest = T3.Manifest(args.data_root, args.manifest)
    games = manifest.get_games(args.split)
    if not games:
        raise RuntimeError(f"No games for split '{args.split}'. Check manifest & data_root.")

    store = T3.LmdbStore()
    team_rounds = T3.build_team_rounds(args.data_root, games, store)

    # Build the epoch loader; this should yield ONLY cached token batches
    d_iter, assembler, _ = T3.build_epoch_loader(
        args, epoch=np.random.randint(0, 100000), store=store, team_rounds=team_rounds,
        last_batch_policy=("partial" if args.split.lower() == "val" else "drop"),
        rank=0, world_size=1
    )
    return d_iter, assembler

def sample_one_batch(d_iter, assembler, use_precomputed: bool = False) -> Dict[str, torch.Tensor]:
    """Pull a single batch from DALI and assemble it into the model's input dict."""
    dali_out = next(iter(d_iter))  # DALIGenericIterator yields a list; with auto_reset=True
    if isinstance(dali_out, list) and len(dali_out) == 1:
        dali_out = dali_out[0]
    batch = assembler.assemble(dali_out, use_precomputed)
    return batch

def sample_preembedded_batch(d_iter, assembler) -> Dict[str, torch.Tensor]:
    """Yield a batch with precomputed tokens (no raw images)."""
    # The iterator returns one list per device; unwrap
    dali_out = next(iter(d_iter))
    if isinstance(dali_out, list) and len(dali_out) == 1:
        dali_out = dali_out[0]
    # assemble with use_precomputed=True
    batch = assembler.assemble(dali_out, use_precomputed=True)
    return batch


def ui_loop(
    model: CS2Transformer,
    device: torch.device,
    args,
):
    if args.preembedded:
        # 1) Build an iterator that yields preembedded batches only
        d_iter, assembler = build_preembedded_iter(args)

        # 2) Resolve 5 POV video files for external frame rendering
        pov_paths = resolve_pov_paths(args)

        # 3) Pull first preembedded sample
        batch = sample_preembedded_batch(d_iter, assembler)
        
        # Figure out sequence length T from embeddings/targets
        if "video_embeddings" in batch:
            T = int(batch["video_embeddings"].shape[1])
        else:
            # fallback if embeddings not in batch; try targets
            any_head = next(iter(batch["targets"]["game_strategy"].values()))
            T = int(any_head.shape[1])
        
        # Use a start frame of 0, as metadata is not passed in the batch
        start_frame = 0

        # 4) Load frames externally for *display* only
        frames = load_5pov_frames(pov_paths, start_frame, T, target_hw=(360, 640))  # [T,5,3,H,W] RGB uint8
        
        # Use alive mask from batch if available, otherwise assume all alive
        if "alive_mask" in batch:
            alive = batch["alive_mask"]  # [B,T,5]
        else:
            alive = torch.ones((1, T, 5), dtype=torch.bool)

        preds = run_model_single_window(model, batch, device)

        images = frames.unsqueeze(0)  # [B=1,T,5,3,H,W] RGB uint8 (matches rest of viewer)
        targets = batch["targets"]

    else:
        # original path (raw images via DALI)
        d_iter, assembler = build_random_iterator_and_assembler(args)
        batch = sample_one_batch(d_iter, assembler, use_precomputed=False)
        preds = run_model_single_window(model, batch, device)
        images = batch["images"]
        alive = batch["alive_mask"]
        targets = batch["targets"]

    B, T, P, C, H, W = images.shape
    assert B == 1 and P == 5, f"Expected B=1, P=5; got {images.shape}"

    # Window/Panel geometry
    PANEL_W, PANEL_H = 640, 360

    # cv2 window
    win_name = "CS2 Model Viewer"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, PANEL_W * 3, PANEL_H * 2)

    t = 0  # current time index
    while True:
        t = clamp_int(t, 0, T - 1)

        # Slice preds/targets at time t
        preds_t = slice_time(preds, t)
        targets_t = slice_targets(targets, t)

        # Compose grid
        grid = render_grid_for_time(
            t,
            images_t=images[0, t],
            alive_mask_t=alive[0, t],
            preds_t=preds_t,
            targets_t=targets_t,
            panel_size=(PANEL_W, PANEL_H),
        )
        # Overlay headline
        head = f"split={args.split}  T={T}  frame={t+1}/{T}   ckpt={Path(args.ckpt).name if args.ckpt else 'auto'}"
        cv2.putText(grid, head, (12, 24), FONT, 0.6, WHITE, 2, cv2.LINE_AA)

        cv2.imshow(win_name, grid)
        key = cv2.waitKey(0) & 0xFF  # wait for next key

        if key == ord('q'):
            break
        elif key == ord('j'):
            t = min(t + 1, T - 1)
        elif key == ord('k'):
            t = max(t - 1, 0)
        elif key == ord('n'):
            # New random window
            print("Sampling new random window...")
            if args.preembedded:
                batch = sample_preembedded_batch(d_iter, assembler)
                frames = load_5pov_frames(pov_paths, 0, T, target_hw=(360, 640))
                images = frames.unsqueeze(0)
                alive = batch.get("alive_mask", torch.ones((1, T, 5), dtype=torch.bool))
            else:
                batch = sample_one_batch(d_iter, assembler, use_precomputed=False)
                images = batch["images"]
                alive = batch["alive_mask"]

            preds = run_model_single_window(model, batch, device)
            targets = batch["targets"]
            B, T, P, C, H, W = images.shape
            t = 0  # reset to first frame

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Interactive CS2 model performance viewer")
    parser.add_argument("--run-dir", type=str, required=False, default="runs/exp1",
                        help="Training run directory containing checkpoints/ and filelists.")
    parser.add_argument("--ckpt", type=str, default="",
                        help="Path to .pt/.pth checkpoint. If empty, auto-picks best from run-dir.")
    parser.add_argument("--data-root", type=str, default=os.environ.get("DATA_ROOT", "data"))
    parser.add_argument("--manifest", type=str, default=None,
                        help="Manifest JSON listing games by split. Defaults to <data-root>/manifest.json")
    parser.add_argument("--split", type=str, choices=["train", "val"], default="val")
    parser.add_argument("--T-frames", dest="T_frames", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--preembedded", action="store_true",
                        help="Use precomputed embeddings from LMDB for model input (no DALI).")
    parser.add_argument("--pov0", type=str, default=None, help="Video path for POV 0")
    parser.add_argument("--pov1", type=str, default=None, help="Video path for POV 1")
    parser.add_argument("--pov2", type=str, default=None, help="Video path for POV 2")
    parser.add_argument("--pov3", type=str, default=None, help="Video path for POV 3")
    parser.add_argument("--pov4", type=str, default=None, help="Video path for POV 4")
    parser.add_argument("--pov-pattern", type=str, default=None,
                        help="printf-style or python-format pattern for 5 POVs, e.g. '/path/pov_%d.mp4' or '/path/pov_{}.mp4'")
    args = parser.parse_args()

    if args.manifest is None:
        args.manifest = str(Path(args.data_root) / "manifest.json")

    # Build model config (match training where possible)
    cfg = CS2Config(context_frames=args.T_frames)
    # Use standard causal (token-by-token) mask at inference by default
    cfg.inference_use_standard_causal = True

    device = torch.device(args.device)
    if args.dtype == "bf16" and not (device.type == "cuda" and torch.cuda.is_bf16_supported()):
        print("[warn] bf16 not supported; falling back to fp32.")
        args.dtype = "fp32"
    if args.dtype == "fp16" and device.type == "cpu":
        print("[warn] fp16 on CPU unsupported; falling back to fp32.")
        args.dtype = "fp32"

    model = CS2Transformer(cfg).to(device)
    # Mixed precision autocast in the model is already guarded by cfg.amp_autocast
    if args.dtype == "fp16":
        pass  # runtime autocast uses fp16
    elif args.dtype == "bf16":
        pass
    else:
        # Force fp32 compute if requested
        for p in model.parameters():
            p.data = p.data.float()

    # Resolve checkpoint path
    ckpt_path = args.ckpt or auto_find_best_ckpt(args.run_dir)
    _ = load_weights_from_checkpoint(model, ckpt_path)
    # Add resolved path to args for display
    args.ckpt = ckpt_path

    # Minimal args object for data pipeline
    min_args = build_min_args(
        data_root=args.data_root,
        manifest=args.manifest,
        run_dir=args.run_dir,
        split=args.split,
        T_frames=args.T_frames,
        batch_size=1,
        dali_threads=4,
        windows_per_round=1,
        use_precomputed=args.preembedded,
        seed=int(time.time()), # use different seed each run
    )
    # Pass through pov arguments for ui_loop to use
    min_args.preembedded = args.preembedded
    min_args.pov_pattern = args.pov_pattern
    for i in range(5):
        setattr(min_args, f"pov{i}", getattr(args, f"pov{i}"))


    # Enter UI
    ui_loop(model, device, min_args)


if __name__ == "__main__":
    main()