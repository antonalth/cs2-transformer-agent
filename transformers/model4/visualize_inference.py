#!/usr/bin/env python3
"""
Generate a side-by-side GT vs prediction video for model4.

model4 is single-player and action-conditioned, so the visualization follows the
actual training semantics:
- one POV only
- teacher-forced previous keyboard/mouse/eco inputs by default
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DatasetConfig, GlobalConfig, ModelConfig, TrainConfig
from dataset import DatasetRoot, GroundTruth, TrainingSample, cs2_collate_fn
from lightning_module import CS2PredictorModule, recursive_apply_to_floats, recursive_to_device
from model import ModelPrediction
from model_loss import mu_law_decode, mu_law_encode


ITEM_NAMES = sorted(
    list(
        set(
            [
                "Desert Eagle",
                "Dual Berettas",
                "Five-SeveN",
                "Glock-18",
                "AK-47",
                "AUG",
                "AWP",
                "FAMAS",
                "G3SG1",
                "Galil AR",
                "M249",
                "M4A4",
                "MAC-10",
                "P90",
                "MP5-SD",
                "UMP-45",
                "XM1014",
                "PP-Bizon",
                "MAG-7",
                "Negev",
                "Sawed-Off",
                "Tec-9",
                "Zeus x27",
                "P2000",
                "MP7",
                "MP9",
                "Nova",
                "P250",
                "SCAR-20",
                "SG 553",
                "SSG 08",
                "Knife",
                "knife",
                "Flashbang",
                "High Explosive Grenade",
                "Smoke Grenade",
                "Molotov",
                "Decoy Grenade",
                "Incendiary Grenade",
                "C4 Explosive",
                "Kevlar Vest",
                "Kevlar & Helmet",
                "Heavy Assault Suit",
                "item_nvg",
                "Defuse Kit",
                "Rescue Kit",
                "Medi-Shot",
                "knife_t",
                "M4A1-S",
                "USP-S",
                "Trade Up Contract",
                "CZ75-Auto",
                "R8 Revolver",
                "Charm Detachments",
                "Bayonet",
                "Classic Knife",
                "Flip Knife",
                "Gut Knife",
                "Karambit",
                "M9 Bayonet",
                "Huntsman Knife",
                "Falchion Knife",
                "Bowie Knife",
                "Butterfly Knife",
                "Shadow Daggers",
                "Paracord Knife",
                "Survival Knife",
                "Ursus Knife",
                "Navaja Knife",
                "Nomad Knife",
                "Stiletto Knife",
                "Talon Knife",
                "Skeleton Knife",
                "Kukri Knife",
            ]
        )
    )
)

KEYBOARD_ONLY_ACTIONS = [
    "IN_ATTACK",
    "IN_JUMP",
    "IN_DUCK",
    "IN_FORWARD",
    "IN_BACK",
    "IN_USE",
    "IN_CANCEL",
    "IN_TURNLEFT",
    "IN_TURNRIGHT",
    "IN_MOVELEFT",
    "IN_MOVERIGHT",
    "IN_ATTACK2",
    "IN_RELOAD",
    "IN_ALT1",
    "IN_ALT2",
    "IN_SPEED",
    "IN_WALK",
    "IN_ZOOM",
    "IN_WEAPON1",
    "IN_WEAPON2",
    "IN_BULLRUSH",
    "IN_GRENADE1",
    "IN_GRENADE2",
    "IN_ATTACK3",
    "IN_SCORE",
    "IN_INSPECT",
    "SWITCH_1",
    "SWITCH_2",
    "SWITCH_3",
    "SWITCH_4",
    "SWITCH_5",
]

COLORS = {
    "gt": (0, 255, 0),
    "pred_tf": (0, 0, 255),
    "pred_ar": (255, 200, 0),
    "white": (255, 255, 255),
    "outline": (0, 0, 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="./dataset0")
    parser.add_argument("--output", type=str, default="tmp/model4_vis.mp4")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--fast_preprocess", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_global_config(checkpoint_path: Path, data_root: str) -> GlobalConfig:
    config_path = checkpoint_path.parent / "config.json"
    if not config_path.exists():
        config_path = checkpoint_path.parent.parent / "config.json"

    if config_path.exists():
        global_cfg = GlobalConfig.from_file(config_path)
    else:
        d_cfg = DatasetConfig(data_root=data_root, run_dir="./runs")
        m_cfg = ModelConfig()
        t_cfg = TrainConfig()
        global_cfg = GlobalConfig(d_cfg, m_cfg, t_cfg)

    global_cfg.dataset.data_root = data_root
    return global_cfg


def decode_keyboard(mask: int) -> str:
    names: list[str] = []
    short = {
        "IN_FORWARD": "W",
        "IN_BACK": "S",
        "IN_MOVELEFT": "A",
        "IN_MOVERIGHT": "D",
        "IN_JUMP": "JUMP",
        "IN_DUCK": "DUCK",
        "IN_WALK": "WALK",
        "IN_ATTACK": "ATK1",
        "IN_ATTACK2": "ATK2",
        "IN_RELOAD": "R",
        "IN_USE": "E",
        "IN_SCORE": "TAB",
    }
    for idx, action in enumerate(KEYBOARD_ONLY_ACTIONS):
        if (mask >> idx) & 1:
            names.append(short.get(action, action.replace("IN_", "")))
    return " ".join(names) if names else "-"


def weapon_name(idx: int) -> str:
    if 0 <= idx < len(ITEM_NAMES):
        return ITEM_NAMES[idx]
    return "-"


def draw_text_lines(
    image: np.ndarray,
    lines: list[str],
    x: int,
    y: int,
    color: tuple[int, int, int],
    *,
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    line_h = int(24 * (scale / 0.55))
    for i, line in enumerate(lines):
        yy = y + i * line_h
        cv2.putText(image, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, COLORS["outline"], thickness + 2, cv2.LINE_AA)
        cv2.putText(image, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def frame_to_bgr(frame: torch.Tensor) -> np.ndarray:
    image = frame.detach().cpu()
    if image.dtype.is_floating_point:
        if image.max() <= 1.0:
            image = image * 255.0
        image = image.clamp(0, 255).to(torch.uint8)
    else:
        image = image.clamp(0, 255).to(torch.uint8)
    image = image.permute(1, 2, 0).numpy()
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def render_panel(
    frame_bgr: np.ndarray,
    gt: dict[str, Any],
    pred_tf: dict[str, Any],
    pred_ar: dict[str, Any],
    meta_lines: list[str],
) -> np.ndarray:
    panel = frame_bgr.copy()

    draw_text_lines(panel, meta_lines, 12, 24, COLORS["white"], scale=0.5)
    draw_text_lines(
        panel,
        [
            f"GT  alive={int(gt['alive'])}  hp={gt['hp']} ap={gt['armor']} money={gt['money']}",
            f"GT  weapon={gt['weapon']}",
            f"GT  mouse=({gt['mouse_x']:.2f}, {gt['mouse_y']:.2f}) bins=[{gt['mouse_x_bin']},{gt['mouse_y_bin']}]",
            f"GT  keys={gt['keys']}",
            f"GT  buy={gt['eco_buy']}  buy_now={int(gt['eco_purchase'])}",
        ],
        12,
        94,
        COLORS["gt"],
        scale=0.5,
    )
    draw_text_lines(
        panel,
        [
            f"TF  mouse=({pred_tf['mouse_x']:.2f}, {pred_tf['mouse_y']:.2f}) bins=[{pred_tf['mouse_x_bin']},{pred_tf['mouse_y_bin']}]",
            f"TF  keys={pred_tf['keys']}",
            f"TF  buy={pred_tf['eco_buy']}  buy_now={int(pred_tf['eco_purchase'])} ({pred_tf['eco_purchase_prob']:.2f})",
        ],
        12,
        224,
        COLORS["pred_tf"],
        scale=0.5,
    )
    draw_text_lines(
        panel,
        [
            f"AR  mouse=({pred_ar['mouse_x']:.2f}, {pred_ar['mouse_y']:.2f}) bins=[{pred_ar['mouse_x_bin']},{pred_ar['mouse_y_bin']}]",
            f"AR  keys={pred_ar['keys']}",
            f"AR  buy={pred_ar['eco_buy']}  buy_now={int(pred_ar['eco_purchase'])} ({pred_ar['eco_purchase_prob']:.2f})",
        ],
        12,
        302,
        COLORS["pred_ar"],
        scale=0.5,
    )
    return panel


def gt_frame_data(gt: GroundTruth, t: int) -> dict[str, Any]:
    weapon_idx = int(gt.active_weapon_idx[0, t].item())
    eco_idx = int(gt.eco_buy_idx[0, t].item())
    did_buy = bool((gt.eco_mask[0, t] != 0).any().item())
    mouse_x = float(gt.mouse_delta[0, t, 0].item())
    mouse_y = float(gt.mouse_delta[0, t, 1].item())
    return {
        "alive": bool(gt.alive_mask[0, t].item()),
        "hp": int(gt.stats[0, t, 0].item()),
        "armor": int(gt.stats[0, t, 1].item()),
        "money": int(gt.stats[0, t, 2].item()),
        "weapon": weapon_name(weapon_idx),
        "mouse_x": mouse_x,
        "mouse_y": mouse_y,
        "mouse_x_bin": -1,
        "mouse_y_bin": -1,
        "keys": decode_keyboard(int(gt.keyboard_mask[0, t].item())),
        "eco_buy": weapon_name(eco_idx),
        "eco_purchase": did_buy,
    }


def pred_frame_data(pred: ModelPrediction, t: int, cfg: ModelConfig) -> dict[str, Any]:
    mouse_x_bin = int(torch.argmax(pred.mouse_x[0, t]).item())
    mouse_y_bin = int(torch.argmax(pred.mouse_y[0, t]).item())

    keyboard_logits = pred.keyboard_logits[0, t]
    keyboard_mask = 0
    for idx in range(cfg.keyboard_dim):
        if torch.sigmoid(keyboard_logits[idx]).item() >= 0.5:
            keyboard_mask |= 1 << idx

    eco_buy_idx = int(torch.argmax(pred.eco_buy_logits[0, t]).item())
    eco_purchase_prob = float(torch.sigmoid(pred.eco_purchase_logits[0, t, 0]).item())
    eco_purchase = eco_purchase_prob >= 0.5

    return {
        "mouse_x": float(mu_law_decode(torch.tensor(mouse_x_bin), cfg.mouse_mu, cfg.mouse_max, cfg.mouse_bins_count).item()),
        "mouse_y": float(mu_law_decode(torch.tensor(mouse_y_bin), cfg.mouse_mu, cfg.mouse_max, cfg.mouse_bins_count).item()),
        "mouse_x_bin": mouse_x_bin,
        "mouse_y_bin": mouse_y_bin,
        "keys": decode_keyboard(keyboard_mask),
        "eco_buy": weapon_name(eco_buy_idx),
        "eco_purchase": eco_purchase,
        "eco_purchase_prob": eco_purchase_prob,
    }


def teacher_forced_prev_actions(truth: GroundTruth, cfg: ModelConfig) -> dict[str, torch.Tensor]:
    B, T = truth.keyboard_mask.shape
    device = truth.keyboard_mask.device
    no_buy_idx = cfg.eco_dim

    prev_keyboard_mask = torch.zeros_like(truth.keyboard_mask)
    prev_keyboard_mask[:, 1:] = truth.keyboard_mask[:, :-1]

    from model_loss import mu_law_encode

    prev_mouse_x_bin = torch.zeros(B, T, dtype=torch.long, device=device)
    prev_mouse_y_bin = torch.zeros(B, T, dtype=torch.long, device=device)
    prev_mouse_x_bin[:, 1:] = mu_law_encode(
        truth.mouse_delta[:, :-1, 0],
        cfg.mouse_mu,
        cfg.mouse_max,
        cfg.mouse_bins_count,
    )
    prev_mouse_y_bin[:, 1:] = mu_law_encode(
        truth.mouse_delta[:, :-1, 1],
        cfg.mouse_mu,
        cfg.mouse_max,
        cfg.mouse_bins_count,
    )

    prev_eco_buy_idx = torch.full((B, T), no_buy_idx, dtype=torch.long, device=device)
    prev_did_buy = (truth.eco_mask[:, :-1] != 0).any(dim=-1)
    prev_buy_idx = truth.eco_buy_idx[:, :-1].long().clamp(0, cfg.eco_dim - 1)
    prev_eco_buy_idx[:, 1:] = torch.where(
        prev_did_buy,
        prev_buy_idx,
        torch.full_like(prev_buy_idx, no_buy_idx),
    )

    return {
        "prev_keyboard_mask": prev_keyboard_mask,
        "prev_mouse_x_bin": prev_mouse_x_bin,
        "prev_mouse_y_bin": prev_mouse_y_bin,
        "prev_eco_buy_idx": prev_eco_buy_idx,
    }


def attach_gt_mouse_bins(data: dict[str, Any], cfg: ModelConfig) -> dict[str, Any]:
    data = dict(data)
    data["mouse_x_bin"] = int(
        mu_law_encode(
            torch.tensor(data["mouse_x"]),
            cfg.mouse_mu,
            cfg.mouse_max,
            cfg.mouse_bins_count,
        ).item()
    )
    data["mouse_y_bin"] = int(
        mu_law_encode(
            torch.tensor(data["mouse_y"]),
            cfg.mouse_mu,
            cfg.mouse_max,
            cfg.mouse_bins_count,
        ).item()
    )
    return data


def run_inference_and_video(
    model: CS2PredictorModule,
    loader: DataLoader,
    output_path: str,
    global_cfg: GlobalConfig,
    num_samples: int,
    device: torch.device,
) -> None:
    writer = None
    processed = 0

    target_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "no": torch.float32,
    }.get(global_cfg.train.mixed_precision, torch.float32)

    model.eval()
    if global_cfg.dataset.epoch_video_decoding_device == "cpu":
        model.model.video.set_fast_preprocess(True)

    with torch.no_grad():
        for sample in loader:
            if processed >= num_samples:
                break

            sample = recursive_to_device(sample, device)
            sample.truth = recursive_apply_to_floats(sample.truth, lambda t: t.to(dtype=target_dtype))

            prev_actions = teacher_forced_prev_actions(sample.truth, global_cfg.model)
            preds_tf = ModelPrediction(**model(sample.images, sample.audio, **prev_actions))

            B, T, C, H, W = sample.images.shape
            if B != 1:
                raise ValueError(f"visualizer expects batch size 1, got {B}")

            if writer is None:
                out_w = W
                out_h = H
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    32,
                    (out_w, out_h),
                )

            roundsample = sample._roundsample
            player_name = roundsample.player_name if roundsample is not None else "unknown"
            round_num = roundsample.round.round_num if roundsample is not None else -1
            split_name = roundsample.round.game.demo_name if roundsample is not None else "unknown"
            samples_per_frame = sample.audio.shape[-1] // T
            ar_state = model.model.init_autoregressive_state()
            preds_ar_frames: list[dict[str, Any]] = []

            for t in range(T):
                audio_start = t * samples_per_frame
                audio_end = (t + 1) * samples_per_frame if t < T - 1 else sample.audio.shape[-1]
                step_pred_dict, ar_state = model.model.forward_step(
                    sample.images[:, t : t + 1],
                    sample.audio[:, :, audio_start:audio_end],
                    state=ar_state,
                )
                step_pred = ModelPrediction(**step_pred_dict)
                preds_ar_frames.append(pred_frame_data(step_pred, 0, global_cfg.model))

            for t in range(T):
                frame_bgr = frame_to_bgr(sample.images[0, t])
                gt = attach_gt_mouse_bins(gt_frame_data(sample.truth, t), global_cfg.model)
                pred_tf = pred_frame_data(preds_tf, t, global_cfg.model)
                pred_ar = preds_ar_frames[t]
                meta_lines = [
                    f"sample={processed + 1}/{num_samples}  demo={split_name}",
                    f"round={round_num}  player={player_name}  frame={t}",
                ]
                panel = render_panel(frame_bgr, gt, pred_tf, pred_ar, meta_lines)
                writer.write(panel)

            processed += 1
            print(f"visualized sample {processed}/{num_samples}")

    if writer is not None:
        writer.release()
        print(f"saved video to {output_path}")


def select_random_indices(ds, num_samples: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    grouped: dict[tuple[str, int, str, int], list[int]] = {}

    for idx, sample in enumerate(ds.samples):
        round_obj = sample.round
        key = (
            round_obj.game.demo_name,
            round_obj.round_num,
            round_obj.team,
            sample.player_idx,
        )
        grouped.setdefault(key, []).append(idx)

    groups = list(grouped.items())
    rng.shuffle(groups)

    selected: list[int] = []
    for _, indices in groups[:num_samples]:
        selected.append(rng.choice(indices))

    if len(selected) < num_samples:
        remaining = [idx for idx in range(len(ds)) if idx not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[: num_samples - len(selected)])

    return selected


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    global_cfg = load_global_config(ckpt_path, args.data_root)

    print(f"loading model from {args.checkpoint}")
    model = CS2PredictorModule.load_from_checkpoint(args.checkpoint, global_cfg=global_cfg, strict=False)
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    ds_root = DatasetRoot(global_cfg.dataset)
    ds = ds_root.build_dataset(args.split)
    if len(ds) == 0 and args.split == "val":
        print("validation set empty, falling back to train")
        ds = ds_root.build_dataset("train")
    if len(ds) == 0:
        raise RuntimeError("dataset is empty")

    selected_indices = select_random_indices(ds, args.num_samples, args.seed)
    subset = torch.utils.data.Subset(ds, selected_indices)

    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        collate_fn=cs2_collate_fn,
        num_workers=0,
    )
    run_inference_and_video(model, loader, args.output, global_cfg, args.num_samples, device)


if __name__ == "__main__":
    main()
