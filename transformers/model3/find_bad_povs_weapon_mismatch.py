#!/usr/bin/env python3
"""
Scan train/val rounds with a trained model3 checkpoint and flag POV videos whose
active-weapon predictions disagree with the recorded labels across multiple
sampled windows. This is intended to catch perspective-shifted/bad POV clips.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from config import GlobalConfig, DatasetConfig, ModelConfig, TrainConfig
from dataset import DatasetRoot, Epoch, Round, RoundSample, TrainingSample, cs2_collate_fn
from lightning_module import CS2PredictorModule


ITEM_NAMES = sorted(
    list(
        set(
            [
                "Desert Eagle", "Dual Berettas", "Five-SeveN", "Glock-18", "AK-47",
                "AUG", "AWP", "FAMAS", "G3SG1", "Galil AR", "M249", "M4A4", "MAC-10",
                "P90", "MP5-SD", "UMP-45", "XM1014", "PP-Bizon", "MAG-7", "Negev",
                "Sawed-Off", "Tec-9", "Zeus x27", "P2000", "MP7", "MP9", "Nova",
                "P250", "SCAR-20", "SG 553", "SSG 08", "Knife", "knife", "Flashbang",
                "High Explosive Grenade", "Smoke Grenade", "Molotov", "Decoy Grenade",
                "Incendiary Grenade", "C4 Explosive", "Kevlar Vest", "Kevlar & Helmet",
                "Heavy Assault Suit", "item_nvg", "Defuse Kit", "Rescue Kit",
                "Medi-Shot", "knife_t", "M4A1-S", "USP-S", "Trade Up Contract",
                "CZ75-Auto", "R8 Revolver", "Charm Detachments", "Bayonet", "Classic Knife",
                "Flip Knife", "Gut Knife", "Karambit", "M9 Bayonet", "Huntsman Knife",
                "Falchion Knife", "Bowie Knife", "Butterfly Knife", "Shadow Daggers",
                "Paracord Knife", "Survival Knife", "Ursus Knife", "Navaja Knife",
                "Nomad Knife", "Stiletto Knife", "Talon Knife", "Skeleton Knife", "Kukri Knife",
            ]
        )
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints_fsdp/model3_ga8_perceiver_learnedpe_noenemypos_20260322/checkpoint_stepstep=9000.ckpt",
    )
    parser.add_argument("--data_root", type=str, default="/workspace/dataset0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--splits", type=str, default="train,val")
    parser.add_argument("--num_points", type=int, default=4)
    parser.add_argument("--window_frames", type=int, default=20)
    parser.add_argument("--max_rounds_per_split", type=int, default=None)
    parser.add_argument("--max_games_per_split", type=int, default=None)
    parser.add_argument("--summary_csv", type=str, default="/workspace/tmp/bad_pov_weapon_scan_summary.csv")
    parser.add_argument("--details_csv", type=str, default="/workspace/tmp/bad_pov_weapon_scan_details.csv")
    parser.add_argument("--min_valid_frames", type=int, default=8)
    parser.add_argument("--window_mismatch_threshold", type=float, default=0.75)
    parser.add_argument("--overall_mismatch_threshold", type=float, default=0.60)
    parser.add_argument("--min_bad_windows", type=int, default=2)
    parser.add_argument("--gt_prob_threshold", type=float, default=0.10)
    parser.add_argument("--fast_preprocess", action="store_true")
    parser.add_argument("--zero_audio", action="store_true")
    parser.add_argument("--devices", type=str, default=None, help="Comma-separated devices, e.g. cuda:0,cuda:1,cuda:2")
    parser.add_argument("--worker_index", type=int, default=0)
    parser.add_argument("--worker_count", type=int, default=1)
    parser.add_argument("--worker_tag", type=str, default=None)
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
        t_cfg = TrainConfig(data_root=data_root)
        global_cfg = GlobalConfig(d_cfg, m_cfg, t_cfg)
    global_cfg.dataset.data_root = data_root
    global_cfg.train.data_root = data_root
    return global_cfg


def player_name_from_video_path(video_path: str) -> str:
    parts = Path(video_path).stem.split("_")
    if len(parts) >= 5:
        return "_".join(parts[2:-2])
    return Path(video_path).stem


def weapon_name(idx: int) -> str:
    if 0 <= idx < len(ITEM_NAMES):
        return ITEM_NAMES[idx]
    return f"UNK({idx})"


def build_round_samples(round_obj: Round, num_points: int, window_frames: int) -> list[RoundSample]:
    max_start = max(0, round_obj.frame_count - window_frames)
    starts = np.linspace(0, max_start, num=max(1, num_points), dtype=int)
    starts = sorted({int(v) for v in starts})
    samples = []
    for start_f in starts:
        samples.append(
            RoundSample(
                round=round_obj,
                start_tick=round_obj.start_tick + (start_f * 2),
                start_frame=start_f,
                length_frames=min(window_frames, round_obj.frame_count),
            )
        )
    return samples


def iter_rounds(ds_root: DatasetRoot, split: str, max_games: int | None, max_rounds: int | None) -> Iterable[Round]:
    games = ds_root.train if split == "train" else ds_root.val
    round_count = 0
    for game_idx, game in enumerate(games):
        if max_games is not None and game_idx >= max_games:
            break
        for round_obj in game.rounds:
            yield round_obj
            round_count += 1
            if max_rounds is not None and round_count >= max_rounds:
                return


def round_key(split: str, round_obj: Round) -> str:
    return f"{split}|{round_obj.game.demo_name}|{round_obj.round_num}|{round_obj.team}"


def worker_output_path(base_path: str, worker_tag: str | None) -> str:
    if not worker_tag:
        return base_path
    path = Path(base_path)
    return str(path.with_name(f"{path.stem}.{worker_tag}{path.suffix}"))


def load_processed_rounds(summary_csv: str) -> set[str]:
    processed: Counter[str] = Counter()
    path = Path(summary_csv)
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['split']}|{row['demo_name']}|{row['round_num']}|{row['team']}"
            processed[key] += 1
    return {key for key, count in processed.items() if count >= 5}


def append_csv_rows(path_str: str, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    if not rows:
        return
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def shard_rounds(rounds: list[Round], split: str, worker_index: int, worker_count: int) -> list[Round]:
    if worker_count <= 1:
        return rounds
    selected = []
    for round_obj in rounds:
        key = round_key(split, round_obj).encode("utf-8")
        shard = int(hashlib.md5(key).hexdigest()[:8], 16) % worker_count
        if shard == worker_index:
            selected.append(round_obj)
    return selected


def merge_csvs(source_paths: list[str], dest_path: str, sort_keys: list[str] | None = None) -> None:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for source in source_paths:
        path = Path(source)
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            for row in reader:
                rows.append(row)
    if fieldnames is None:
        return
    if sort_keys:
        rows.sort(key=lambda row: tuple(row[k] for k in sort_keys))
    path = Path(dest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_spawn_multi_gpu(args: argparse.Namespace) -> bool:
    if args.devices is None:
        return False
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    if len(devices) <= 1:
        args.device = devices[0] if devices else args.device
        return False
    if args.worker_count != 1:
        return False

    print(f"Launching {len(devices)} workers across devices: {devices}")
    procs = []
    for worker_index, device in enumerate(devices):
        worker_tag = f"worker{worker_index:02d}"
        cmd = [
            "python",
            Path(__file__).name,
            "--checkpoint", args.checkpoint,
            "--data_root", args.data_root,
            "--device", device,
            "--splits", args.splits,
            "--num_points", str(args.num_points),
            "--window_frames", str(args.window_frames),
            "--summary_csv", args.summary_csv,
            "--details_csv", args.details_csv,
            "--min_valid_frames", str(args.min_valid_frames),
            "--window_mismatch_threshold", str(args.window_mismatch_threshold),
            "--overall_mismatch_threshold", str(args.overall_mismatch_threshold),
            "--min_bad_windows", str(args.min_bad_windows),
            "--gt_prob_threshold", str(args.gt_prob_threshold),
            "--worker_index", str(worker_index),
            "--worker_count", str(len(devices)),
            "--worker_tag", worker_tag,
        ]
        if args.max_rounds_per_split is not None:
            cmd.extend(["--max_rounds_per_split", str(args.max_rounds_per_split)])
        if args.max_games_per_split is not None:
            cmd.extend(["--max_games_per_split", str(args.max_games_per_split)])
        if args.fast_preprocess:
            cmd.append("--fast_preprocess")
        if args.zero_audio:
            cmd.append("--zero_audio")
        proc = subprocess.Popen(cmd, cwd=Path(__file__).resolve().parent)
        procs.append(proc)

    exit_code = 0
    for proc in procs:
        code = proc.wait()
        if code != 0:
            exit_code = code

    summary_paths = [worker_output_path(args.summary_csv, f"worker{i:02d}") for i in range(len(devices))]
    detail_paths = [worker_output_path(args.details_csv, f"worker{i:02d}") for i in range(len(devices))]
    merge_csvs(summary_paths, args.summary_csv, sort_keys=["split", "demo_name", "round_num", "team", "player_idx"])
    merge_csvs(
        detail_paths,
        args.details_csv,
        sort_keys=["split", "demo_name", "round_num", "team", "player_idx", "sample_start_frame"],
    )
    raise SystemExit(exit_code)


def summarize_modes(values: list[int]) -> str:
    if not values:
        return ""
    counts = Counter(values)
    top = counts.most_common(3)
    return ";".join(f"{weapon_name(idx)}:{count}" for idx, count in top)


def main() -> None:
    args = parse_args()
    maybe_spawn_multi_gpu(args)

    checkpoint_path = Path(args.checkpoint)
    global_cfg = load_global_config(checkpoint_path, args.data_root)
    global_cfg.dataset.epoch_round_sample_length = args.window_frames
    summary_csv = worker_output_path(args.summary_csv, args.worker_tag)
    details_csv = worker_output_path(args.details_csv, args.worker_tag)

    device = torch.device(args.device)
    print(f"Loading checkpoint: {checkpoint_path} on {device}")
    module = CS2PredictorModule.load_from_checkpoint(str(checkpoint_path), global_cfg=global_cfg, strict=False)
    module.eval()
    module.to(device)
    model = module.model
    if args.fast_preprocess and hasattr(model.video, "set_fast_preprocess"):
        model.video.set_fast_preprocess(True)

    ds_root = DatasetRoot(global_cfg.dataset)
    ds_root.store.close_all()
    epoch = Epoch(global_cfg.dataset, [])

    processed_rounds = load_processed_rounds(summary_csv)
    print(f"Resuming with {len(processed_rounds)} completed rounds already recorded in {summary_csv}")

    summary_fieldnames: list[str] | None = None
    detail_fieldnames: list[str] | None = None

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    with torch.inference_mode():
        for split in splits:
            split_rounds = list(iter_rounds(ds_root, split, args.max_games_per_split, args.max_rounds_per_split))
            split_rounds = shard_rounds(split_rounds, split, args.worker_index, args.worker_count)
            for round_idx, round_obj in enumerate(
                split_rounds,
                start=1,
            ):
                rkey = round_key(split, round_obj)
                if rkey in processed_rounds:
                    continue
                samples = build_round_samples(round_obj, args.num_points, args.window_frames)
                training_samples: list[TrainingSample] = []
                for sample in samples:
                    training_samples.append(
                        TrainingSample(
                            images=epoch._decode_video(sample),
                            audio=epoch._decode_audio(sample),
                            truth=epoch._get_truth(sample),
                            _roundsample=sample,
                        )
                    )

                batch = cs2_collate_fn(training_samples)
                batch = module.transfer_batch_to_device(batch, device, 0)
                if args.zero_audio:
                    batch.audio.zero_()
                preds_dict = module(batch.images, batch.audio)

                weapon_logits = preds_dict["active_weapon_logits"]
                weapon_probs = torch.softmax(weapon_logits, dim=-1)
                pred_idx = weapon_logits.argmax(dim=-1)
                pred_conf = weapon_probs.amax(dim=-1)

                gt_idx = batch.truth.active_weapon_idx.long()
                alive = batch.truth.alive_mask.bool()
                valid = alive & (gt_idx >= 0)
                safe_gt_idx = gt_idx.clamp(min=0, max=global_cfg.model.weapon_dim - 1)
                gt_prob = weapon_probs.gather(-1, safe_gt_idx.unsqueeze(-1)).squeeze(-1)

                if round_idx % 25 == 0:
                    print(
                        f"[{args.worker_tag or 'worker0'}][{split}] processed shard rounds: "
                        f"{round_idx}/{len(split_rounds)}"
                    )

                summary_rows: list[dict[str, object]] = []
                detail_rows: list[dict[str, object]] = []

                for player_idx in range(5):
                    video_path = round_obj.pov_video[player_idx]
                    audio_path = round_obj.pov_audio[player_idx]
                    player_name = player_name_from_video_path(video_path)

                    total_valid = 0
                    total_mismatch = 0
                    total_gt_prob = 0.0
                    total_pred_conf = 0.0
                    bad_windows = 0
                    valid_windows = 0
                    all_gt_weapons: list[int] = []
                    all_pred_weapons: list[int] = []
                    window_mismatch_rates: list[str] = []

                    for batch_idx, sample in enumerate(samples):
                        valid_mask = valid[batch_idx, :, player_idx]
                        valid_count = int(valid_mask.sum().item())
                        mismatch_count = 0
                        mismatch_rate = ""
                        mean_gt_prob = ""
                        mean_pred_conf = ""
                        gt_mode = ""
                        pred_mode = ""
                        is_bad_window = False

                        if valid_count > 0:
                            valid_windows += 1
                            valid_gt = gt_idx[batch_idx, :, player_idx][valid_mask]
                            valid_pred = pred_idx[batch_idx, :, player_idx][valid_mask]
                            valid_gt_prob = gt_prob[batch_idx, :, player_idx][valid_mask]
                            valid_pred_conf = pred_conf[batch_idx, :, player_idx][valid_mask]

                            mismatch_mask = valid_pred != valid_gt
                            mismatch_count = int(mismatch_mask.sum().item())
                            mismatch_rate_value = mismatch_count / valid_count
                            mean_gt_prob_value = float(valid_gt_prob.mean().item())
                            mean_pred_conf_value = float(valid_pred_conf.mean().item())

                            mismatch_rate = f"{mismatch_rate_value:.4f}"
                            mean_gt_prob = f"{mean_gt_prob_value:.4f}"
                            mean_pred_conf = f"{mean_pred_conf_value:.4f}"
                            gt_values = [int(v) for v in valid_gt.tolist()]
                            pred_values = [int(v) for v in valid_pred.tolist()]
                            gt_mode = summarize_modes(gt_values)
                            pred_mode = summarize_modes(pred_values)

                            total_valid += valid_count
                            total_mismatch += mismatch_count
                            total_gt_prob += float(valid_gt_prob.sum().item())
                            total_pred_conf += float(valid_pred_conf.sum().item())
                            all_gt_weapons.extend(gt_values)
                            all_pred_weapons.extend(pred_values)
                            window_mismatch_rates.append(f"{sample.start_frame}:{mismatch_rate_value:.3f}")

                            is_bad_window = (
                                valid_count >= args.min_valid_frames
                                and mismatch_rate_value >= args.window_mismatch_threshold
                                and mean_gt_prob_value <= args.gt_prob_threshold
                            )
                            if is_bad_window:
                                bad_windows += 1

                        detail_rows.append(
                            {
                                "split": split,
                                "demo_name": round_obj.game.demo_name,
                                "round_num": round_obj.round_num,
                                "team": round_obj.team,
                                "player_idx": player_idx,
                                "player_name": player_name,
                                "video_path": video_path,
                                "audio_path": audio_path,
                                "video_relpath": os.path.relpath(video_path, args.data_root),
                                "audio_relpath": os.path.relpath(audio_path, args.data_root),
                                "frame_count": round_obj.frame_count,
                                "sample_start_frame": sample.start_frame,
                                "sample_length_frames": sample.length_frames,
                                "valid_frames": valid_count,
                                "mismatch_frames": mismatch_count,
                                "mismatch_rate": mismatch_rate,
                                "mean_gt_prob": mean_gt_prob,
                                "mean_pred_conf": mean_pred_conf,
                                "gt_weapons_mode": gt_mode,
                                "pred_weapons_mode": pred_mode,
                                "bad_window": int(is_bad_window),
                            }
                        )

                    overall_mismatch_rate = (total_mismatch / total_valid) if total_valid else float("nan")
                    overall_mean_gt_prob = (total_gt_prob / total_valid) if total_valid else float("nan")
                    overall_mean_pred_conf = (total_pred_conf / total_valid) if total_valid else float("nan")

                    flag_bad = (
                        total_valid >= args.min_valid_frames * args.min_bad_windows
                        and bad_windows >= args.min_bad_windows
                        and overall_mismatch_rate >= args.overall_mismatch_threshold
                        and overall_mean_gt_prob <= args.gt_prob_threshold
                    )

                    summary_rows.append(
                        {
                            "split": split,
                            "demo_name": round_obj.game.demo_name,
                            "round_num": round_obj.round_num,
                            "team": round_obj.team,
                            "player_idx": player_idx,
                            "player_name": player_name,
                            "video_path": video_path,
                            "audio_path": audio_path,
                            "video_relpath": os.path.relpath(video_path, args.data_root),
                            "audio_relpath": os.path.relpath(audio_path, args.data_root),
                            "frame_count": round_obj.frame_count,
                            "windows_evaluated": len(samples),
                            "windows_with_valid_frames": valid_windows,
                            "bad_windows": bad_windows,
                            "valid_frames": total_valid,
                            "mismatch_frames": total_mismatch,
                            "mismatch_rate": "" if total_valid == 0 else f"{overall_mismatch_rate:.4f}",
                            "mean_gt_prob": "" if total_valid == 0 else f"{overall_mean_gt_prob:.4f}",
                            "mean_pred_conf": "" if total_valid == 0 else f"{overall_mean_pred_conf:.4f}",
                            "gt_weapons_mode": summarize_modes(all_gt_weapons),
                            "pred_weapons_mode": summarize_modes(all_pred_weapons),
                            "sample_points": ";".join(str(s.start_frame) for s in samples),
                            "window_mismatch_rates": ";".join(window_mismatch_rates),
                            "flag_bad": int(flag_bad),
                        }
                    )

                if summary_rows and summary_fieldnames is None:
                    summary_fieldnames = list(summary_rows[0].keys())
                if detail_rows and detail_fieldnames is None:
                    detail_fieldnames = list(detail_rows[0].keys())
                if summary_fieldnames:
                    append_csv_rows(summary_csv, summary_rows, summary_fieldnames)
                if detail_fieldnames:
                    append_csv_rows(details_csv, detail_rows, detail_fieldnames)

                flagged = sum(int(row["flag_bad"]) for row in summary_rows)
                top_score = max((float(row["mismatch_rate"] or 0.0) for row in summary_rows), default=0.0)
                print(
                    f"[{args.worker_tag or 'worker0'}] {rkey} | flagged={flagged}/5 "
                    f"| top_mismatch={top_score:.3f}"
                )
                processed_rounds.add(rkey)

    print(f"Wrote summary CSV shard: {summary_csv}")
    print(f"Wrote details CSV shard: {details_csv}")

    for env in epoch.lmdb_envs.values():
        env.close()
    ds_root.store.close_all()


if __name__ == "__main__":
    main()
