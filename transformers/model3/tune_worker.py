import argparse
import gc
import json
from pathlib import Path
from typing import Optional, Dict, Any

import pytorch_lightning as pl
import torch

from config import GlobalConfig
from train import create_global_config, run_training


class BestMetricCallback(pl.Callback):
    def __init__(self, monitor: str):
        super().__init__()
        self.monitor = monitor
        self.best_value: Optional[float] = None
        self.last_step: Optional[int] = None

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return

        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return

        value = float(metric.detach().cpu().item()) if torch.is_tensor(metric) else float(metric)
        self.last_step = int(trainer.global_step)
        if self.best_value is None or value < self.best_value:
            self.best_value = value


class StopAfterGlobalStepCallback(pl.Callback):
    def __init__(self, stop_after_steps: int):
        super().__init__()
        self.stop_after_steps = stop_after_steps

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        # global_step counts optimizer updates (respects grad accumulation)
        if trainer.global_step >= self.stop_after_steps:
            trainer.should_stop = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single HPO trial in a standalone training process.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--project_name", type=str, default="cs2-behavior-cloning-optuna")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--params_json", type=str, required=True)
    parser.add_argument("--result_json", type=str, required=True)
    parser.add_argument("--max_steps", type=int, required=True, help="Early-stop target in optimizer steps.")
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_every_steps", type=int, default=200)
    parser.add_argument("--val_samples_limit", type=int, default=300)
    parser.add_argument("--monitor", type=str, default="val/loss")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_job_type", type=str, default="optuna-trial")
    parser.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated tags.")
    parser.add_argument("--save_checkpoints", action="store_true")
    return parser.parse_args()


def apply_flat_overrides(cfg: GlobalConfig, overrides: Dict[str, Any]) -> None:
    for full_key, value in overrides.items():
        if "." not in full_key:
            raise ValueError(f"Invalid override key '{full_key}'. Expected section.field format.")

        section_name, field_name = full_key.split(".", 1)
        if not hasattr(cfg, section_name):
            raise ValueError(f"Unknown config section '{section_name}' in key '{full_key}'.")

        section = getattr(cfg, section_name)
        if not hasattr(section, field_name):
            raise ValueError(f"Unknown config field '{field_name}' in key '{full_key}'.")

        setattr(section, field_name, value)


def write_result(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    params_path = Path(args.params_json)
    result_path = Path(args.result_json)

    with params_path.open("r", encoding="utf-8") as f:
        params = json.load(f)

    cfg = create_global_config(
        data_root=args.data_root,
        run_name=args.run_name,
        output_dir=args.output_dir,
    )
    cfg.train.project_name = args.project_name
    cfg.train.num_workers = args.num_workers
    cfg.train.val_every_steps = args.val_every_steps
    cfg.train.val_samples_limit = args.val_samples_limit

    apply_flat_overrides(cfg, params)

    metric_cb = BestMetricCallback(monitor=args.monitor)
    stop_cb = StopAfterGlobalStepCallback(stop_after_steps=args.max_steps)
    logger_kwargs: Dict[str, Any] = {}
    if not args.disable_wandb:
        if args.wandb_group:
            logger_kwargs["group"] = args.wandb_group
        if args.wandb_job_type:
            logger_kwargs["job_type"] = args.wandb_job_type
        if args.wandb_tags:
            tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
            if tags:
                logger_kwargs["tags"] = tags

    trainer: Optional[pl.Trainer] = None
    try:
        trainer = run_training(
            global_cfg=cfg,
            resume_from_checkpoint=None,
            debug=False,
            enable_wandb=not args.disable_wandb,
            enable_checkpoints=args.save_checkpoints,
            enable_lr_monitor=True,
            extra_callbacks=[metric_cb, stop_cb],
            logger_kwargs=logger_kwargs,
            trainer_overrides={
                "num_sanity_val_steps": 0,
                "devices": args.devices,
            },
        )

        if trainer.is_global_zero:
            metric = metric_cb.best_value
            if metric is None:
                cb_metric = trainer.callback_metrics.get(args.monitor)
                if cb_metric is not None:
                    metric = float(cb_metric.detach().cpu().item()) if torch.is_tensor(cb_metric) else float(cb_metric)

            if metric is None:
                raise RuntimeError(f"Monitor metric '{args.monitor}' was never produced.")

            write_result(
                result_path,
                {
                    "status": "ok",
                    "best_metric": metric,
                    "last_step": metric_cb.last_step,
                    "target_stop_steps": args.max_steps,
                    "estimated_stepping_batches": int(trainer.estimated_stepping_batches),
                },
            )
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or "cuda error: out of memory" in msg:
            if trainer is None or trainer.is_global_zero:
                write_result(result_path, {"status": "pruned", "reason": "oom", "error": str(exc)})
            raise

        if trainer is None or trainer.is_global_zero:
            write_result(result_path, {"status": "error", "error": str(exc)})
        raise
    except Exception as exc:
        if trainer is None or trainer.is_global_zero:
            write_result(result_path, {"status": "error", "error": str(exc)})
        raise
    finally:
        if trainer is not None:
            del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
