import argparse
import os
import math
import torch
import pytorch_lightning as pl
from typing import Optional, Dict, Any, List

torch.set_float32_matmul_precision('high')

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import FSDPStrategy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from config import GlobalConfig, DatasetConfig, ModelConfig, TrainConfig
from lightning_module import CS2PredictorModule
from lightning_datamodule import CS2DataModule

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="unnamed")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_fsdp")
    parser.add_argument("--compressor_type", choices=["qformer", "perceiver"], default=None)
    parser.add_argument("--perceiver_pos_embedding", choices=["none", "sincos", "learned"], default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def create_global_config(
    data_root: str,
    run_name: str = "unnamed",
    output_dir: str = "./checkpoints_fsdp",
) -> GlobalConfig:
    d_cfg = DatasetConfig(data_root=data_root, run_dir="./runs")
    t_cfg = TrainConfig(
        data_root=data_root,
        run_name=run_name,
        output_dir=output_dir,
    )
    m_cfg = ModelConfig()
    return GlobalConfig(dataset=d_cfg, model=m_cfg, train=t_cfg)


def save_config(global_cfg: GlobalConfig) -> None:
    t_cfg = global_cfg.train
    os.makedirs(t_cfg.output_dir, exist_ok=True)
    global_cfg.to_file(os.path.join(t_cfg.output_dir, "config.json"))


def create_logger(
    global_cfg: GlobalConfig,
    enable_wandb: bool = True,
    logger_kwargs: Optional[Dict[str, Any]] = None,
):
    t_cfg = global_cfg.train
    if not enable_wandb:
        return None

    extra = logger_kwargs or {}
    return WandbLogger(
        project=t_cfg.project_name,
        name=t_cfg.run_name,
        save_dir=t_cfg.output_dir,
        log_model=False,  # Don't upload checkpoints to wandb by default
        **extra,
    )


def create_callbacks(
    global_cfg: GlobalConfig,
    enable_checkpoints: bool = True,
    enable_lr_monitor: bool = True,
) -> List[pl.Callback]:
    t_cfg = global_cfg.train
    callbacks: List[pl.Callback] = []

    if enable_checkpoints:
        # Checkpoint every epoch
        checkpoint_callback_epoch = ModelCheckpoint(
            dirpath=t_cfg.output_dir,
            filename="checkpoint_ep{epoch}",
            save_top_k=-1,
            every_n_epochs=t_cfg.save_every,
            save_on_train_epoch_end=True,
            save_weights_only=False,
        )
        callbacks.append(checkpoint_callback_epoch)

        # Checkpoint every N steps (intra-epoch)
        checkpoint_callback_step = ModelCheckpoint(
            dirpath=t_cfg.output_dir,
            filename="checkpoint_step{step}",
            save_top_k=-1,  # Keep all step checkpoints
            every_n_train_steps=t_cfg.checkpoint_every_n_steps,
            save_weights_only=False,
        )
        callbacks.append(checkpoint_callback_step)

    if enable_lr_monitor:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    return callbacks


def resolve_precision(mixed_precision: str) -> str:
    precision_map = {
        "bf16": "bf16-mixed",
        "fp16": "16-mixed",
        "fp32": "32-true",
        "no": "32-true",
    }
    return precision_map.get(mixed_precision, "bf16-mixed")


def create_strategy() -> FSDPStrategy:
    return FSDPStrategy(
        sharding_strategy="FULL_SHARD",
        auto_wrap_policy={LlamaDecoderLayer},
        cpu_offload=False,  # Can enable if OOM
    )


def compute_val_limit_batches(global_cfg: GlobalConfig):
    t_cfg = global_cfg.train

    # Calculate validation batches limit
    # We estimate based on visible devices. PL might change this internally if devices="auto" selects fewer,
    # but this is a good approximation.
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    val_limit_batches = 1.0  # default 100%
    if t_cfg.val_samples_limit is not None:
        global_batch_size = t_cfg.batch_size * num_devices
        val_limit_batches = math.ceil(t_cfg.val_samples_limit / max(1, global_batch_size))
    else:
        val_limit_batches = 1.0  # Full validation set
    return val_limit_batches


def build_trainer(
    global_cfg: GlobalConfig,
    logger=None,
    callbacks: Optional[List[pl.Callback]] = None,
    debug: bool = False,
    trainer_overrides: Optional[Dict[str, Any]] = None,
) -> pl.Trainer:
    t_cfg = global_cfg.train

    trainer_kwargs: Dict[str, Any] = {
        "default_root_dir": t_cfg.output_dir,
        "max_epochs": t_cfg.max_epochs,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": "auto",
        "strategy": create_strategy(),
        "precision": resolve_precision(t_cfg.mixed_precision),
        "logger": logger,
        "callbacks": callbacks or [],
        "accumulate_grad_batches": t_cfg.grad_accumulation_steps,
        "gradient_clip_val": None,  # Disabled to avoid FSDP MisconfigurationException
        # Validation Scheduling
        "val_check_interval": t_cfg.val_every_steps,
        "check_val_every_n_epoch": None,  # Required when val_check_interval is int (steps)
        "limit_val_batches": compute_val_limit_batches(global_cfg),
        "log_every_n_steps": 1,
        # Debugging
        "fast_dev_run": debug,
    }

    if trainer_overrides:
        trainer_kwargs.update(trainer_overrides)

    return pl.Trainer(**trainer_kwargs)


def run_training(
    global_cfg: GlobalConfig,
    resume_from_checkpoint: Optional[str] = None,
    debug: bool = False,
    enable_wandb: bool = True,
    enable_checkpoints: bool = True,
    enable_lr_monitor: bool = True,
    extra_callbacks: Optional[List[pl.Callback]] = None,
    logger_kwargs: Optional[Dict[str, Any]] = None,
    trainer_overrides: Optional[Dict[str, Any]] = None,
) -> pl.Trainer:
    save_config(global_cfg)

    dm = CS2DataModule(global_cfg)
    model = CS2PredictorModule(global_cfg)
    logger = create_logger(global_cfg, enable_wandb=enable_wandb, logger_kwargs=logger_kwargs)
    callbacks = create_callbacks(
        global_cfg,
        enable_checkpoints=enable_checkpoints,
        enable_lr_monitor=enable_lr_monitor,
    )
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    trainer = build_trainer(
        global_cfg=global_cfg,
        logger=logger,
        callbacks=callbacks,
        debug=debug,
        trainer_overrides=trainer_overrides,
    )
    trainer.fit(model, datamodule=dm, ckpt_path=resume_from_checkpoint)
    return trainer


def main():
    args = parse_args()
    global_cfg = create_global_config(
        data_root=args.data_root,
        run_name=args.run_name,
        output_dir=args.output_dir,
    )
    if args.compressor_type is not None:
        global_cfg.model.compressor_type = args.compressor_type
    if args.perceiver_pos_embedding is not None:
        global_cfg.model.perceiver_pos_embedding = args.perceiver_pos_embedding
    run_training(
        global_cfg=global_cfg,
        resume_from_checkpoint=args.resume_from_checkpoint,
        debug=args.debug,
        enable_wandb=True,
        enable_checkpoints=True,
        enable_lr_monitor=True,
    )

if __name__ == "__main__":
    main()
