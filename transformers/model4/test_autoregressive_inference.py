#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

from config import GlobalConfig, DatasetConfig, ModelConfig, TrainConfig
from dataset import DatasetRoot, FRAME_RATE, cs2_collate_fn
from lightning_module import CS2PredictorModule, recursive_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate model4 autoregressive inference and cache cropping on a real dataset sample."
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional Lightning checkpoint to load.")
    parser.add_argument("--data_root", type=str, default="dataset0")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--crop-frames", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    return parser.parse_args()


def load_global_config(checkpoint: str | None, data_root: str) -> GlobalConfig:
    if checkpoint:
        ckpt_path = Path(checkpoint)
        for config_path in (ckpt_path.parent / "config.json", ckpt_path.parent.parent / "config.json"):
            if config_path.exists():
                cfg = GlobalConfig.from_file(config_path)
                cfg.dataset.data_root = data_root
                return cfg
    return GlobalConfig(
        dataset=DatasetConfig(data_root=data_root, run_dir="./runs"),
        model=ModelConfig(),
        train=TrainConfig(data_root=data_root),
    )


def load_module(global_cfg: GlobalConfig, checkpoint: str | None, device: torch.device) -> CS2PredictorModule:
    if checkpoint:
        module = CS2PredictorModule.load_from_checkpoint(checkpoint, global_cfg=global_cfg, strict=False)
    else:
        module = CS2PredictorModule(global_cfg)
    module.to(device)
    module.eval()
    return module


def load_batch(global_cfg: GlobalConfig, split: str, sample_index: int, num_frames: int):
    ds_cfg = DatasetConfig(
        data_root=global_cfg.dataset.data_root,
        run_dir=global_cfg.dataset.run_dir,
        warn_skip=global_cfg.dataset.warn_skip,
        sample_stride=global_cfg.dataset.sample_stride,
        epoch_round_sample_length=max(num_frames, 1),
        epoch_video_decoding_device=global_cfg.dataset.epoch_video_decoding_device,
        audio_sample_rate=global_cfg.dataset.audio_sample_rate,
    )
    ds_root = DatasetRoot(ds_cfg)
    dataset = ds_root.build_dataset(split)
    if len(dataset) == 0 and split == "val":
        dataset = ds_root.build_dataset("train")
        split = "train"
    if len(dataset) == 0:
        raise RuntimeError("dataset is empty")

    actual_index = sample_index % len(dataset)
    sample = dataset[actual_index]
    batch = cs2_collate_fn([sample])
    return batch, split, actual_index


def concat_predictions(predictions: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = predictions[0].keys()
    return {key: torch.cat([pred[key] for pred in predictions], dim=1) for key in keys}


def compare_prediction_dicts(
    reference: Dict[str, torch.Tensor],
    candidate: Dict[str, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> tuple[bool, list[str]]:
    ok = True
    lines: list[str] = []
    for key in sorted(reference.keys()):
        ref = reference[key].detach().float().cpu()
        cand = candidate[key].detach().float().cpu()
        diff = (ref - cand).abs()
        max_diff = float(diff.max().item())
        mean_diff = float(diff.mean().item())
        allclose = torch.allclose(ref, cand, atol=atol, rtol=rtol)
        status = "ok" if allclose else "FAIL"
        lines.append(f"{status:4s} {key:24s} max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")
        ok = ok and allclose
    return ok, lines


def ensure_finite(prediction: Dict[str, torch.Tensor], label: str) -> None:
    for key, value in prediction.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"{label}: non-finite values in {key}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    global_cfg = load_global_config(args.checkpoint, args.data_root)
    module = load_module(global_cfg, args.checkpoint, device)
    backbone = module.model

    batch, actual_split, actual_index = load_batch(global_cfg, args.split, args.sample_index, args.num_frames)
    batch = recursive_to_device(batch, device)
    num_frames = min(args.num_frames, batch.images.shape[1])
    if num_frames <= 0:
        raise RuntimeError("sample has no frames")

    audio_samples_per_frame = global_cfg.dataset.audio_sample_rate // FRAME_RATE
    total_audio_samples = num_frames * audio_samples_per_frame
    if batch.audio.shape[-1] < total_audio_samples:
        raise RuntimeError(
            f"sample audio is shorter than requested window: have {batch.audio.shape[-1]}, "
            f"need {total_audio_samples}"
        )

    batch.images = batch.images[:, :num_frames]
    batch.audio = batch.audio[..., :total_audio_samples]
    for field_name, value in batch.truth.__dict__.items():
        if torch.is_tensor(value):
            setattr(batch.truth, field_name, value[:, :num_frames] if value.ndim >= 2 else value)

    print(f"Loaded split={actual_split} sample_index={actual_index} frames={num_frames}")
    print(f"Device={device} dtype={backbone.cfg.dtype} crop_frames={args.crop_frames}")

    with torch.inference_mode():
        prev_actions = module._teacher_forced_prev_actions(batch.truth)
        full_preds = backbone(batch.images, batch.audio, **prev_actions)

        exact_state = backbone.init_autoregressive_state()
        exact_predictions = []
        for t in range(num_frames):
            img_step = batch.images[:, t : t + 1]
            audio_step = batch.audio[:, :, t * audio_samples_per_frame : (t + 1) * audio_samples_per_frame]
            if t == 0:
                pred_t, exact_state = backbone.forward_step(img_step, audio_step, exact_state)
            else:
                pred_t, exact_state = backbone.forward_step(
                    img_step,
                    audio_step,
                    exact_state,
                    prev_keyboard_mask=prev_actions["prev_keyboard_mask"][:, t],
                    prev_mouse_x_bin=prev_actions["prev_mouse_x_bin"][:, t],
                    prev_mouse_y_bin=prev_actions["prev_mouse_y_bin"][:, t],
                    prev_eco_buy_idx=prev_actions["prev_eco_buy_idx"][:, t],
                )
            ensure_finite(pred_t, f"teacher_forced_step(frame={t})")
            exact_predictions.append(pred_t)
        ar_preds = concat_predictions(exact_predictions)

        exact_ok, exact_lines = compare_prediction_dicts(
            full_preds,
            ar_preds,
            atol=args.atol,
            rtol=args.rtol,
        )
        print("\nTeacher-forced full-vs-step equivalence:")
        for line in exact_lines:
            print(line)

        crop_frames = min(args.crop_frames, num_frames)
        if crop_frames <= 0:
            raise RuntimeError("crop_frames must be positive")
        crop_state = backbone.init_autoregressive_state(max_cache_frames=crop_frames)
        crop_ok = True
        print("\nRolling cache mechanics:")
        for t in range(num_frames):
            img_step = batch.images[:, t : t + 1]
            audio_step = batch.audio[:, :, t * audio_samples_per_frame : (t + 1) * audio_samples_per_frame]
            if t == 0:
                pred_t, crop_state = backbone.forward_step(
                    img_step,
                    audio_step,
                    crop_state,
                    max_cache_frames=crop_frames,
                )
            else:
                pred_t, crop_state = backbone.forward_step(
                    img_step,
                    audio_step,
                    crop_state,
                    max_cache_frames=crop_frames,
                    prev_keyboard_mask=prev_actions["prev_keyboard_mask"][:, t],
                    prev_mouse_x_bin=prev_actions["prev_mouse_x_bin"][:, t],
                    prev_mouse_y_bin=prev_actions["prev_mouse_y_bin"][:, t],
                    prev_eco_buy_idx=prev_actions["prev_eco_buy_idx"][:, t],
                )
            ensure_finite(pred_t, f"cropped_step(frame={t})")

            expected_cached_frames = min(t + 1, crop_frames)
            expected_offset = max(0, t + 1 - crop_frames)
            actual_offset = 0 if crop_state.cache is None else int(crop_state.cache.position_offset)
            step_ok = (
                crop_state.cached_frames == expected_cached_frames
                and crop_state.total_frames_processed == t + 1
                and actual_offset == expected_offset
            )
            print(
                f"frame={t} cached_frames={crop_state.cached_frames}/{expected_cached_frames} "
                f"total_frames={crop_state.total_frames_processed}/{t + 1} "
                f"offset={actual_offset}/{expected_offset}"
            )
            crop_ok = crop_ok and step_ok

        implicit_state = backbone.init_autoregressive_state(max_cache_frames=crop_frames)
        print("\nImplicit autoregressive forward_step smoke test:")
        for t in range(num_frames):
            img_step = batch.images[:, t : t + 1]
            audio_step = batch.audio[:, :, t * audio_samples_per_frame : (t + 1) * audio_samples_per_frame]
            pred_t, implicit_state = backbone.forward_step(
                img_step,
                audio_step,
                implicit_state,
                max_cache_frames=crop_frames,
            )
            ensure_finite(pred_t, f"implicit_step(frame={t})")
            print(
                f"frame={t} cached_frames={implicit_state.cached_frames} "
                f"total_frames={implicit_state.total_frames_processed}"
            )

    if not exact_ok:
        raise SystemExit("teacher-forced full-vs-step equivalence test failed")
    if not crop_ok:
        raise SystemExit("rolling cache mechanics test failed")

    print("\nAll autoregressive inference checks passed.")


if __name__ == "__main__":
    main()
