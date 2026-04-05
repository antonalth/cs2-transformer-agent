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
    parser.add_argument("--run_name", type=str, default="model4")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--load_weights_only_from", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_fsdp")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accumulation_steps", type=int, default=None)
    parser.add_argument("--perceiver_pos_embedding", choices=["none", "sincos", "learned"], default=None)
    parser.add_argument(
        "--enable_losses",
        type=str,
        default=None,
        help="Comma-separated loss names to enable; all others are disabled.",
    )
    parser.add_argument(
        "--disable_losses",
        type=str,
        default=None,
        help="Comma-separated loss names to disable; all others keep their configured value.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.resume_from_checkpoint and args.load_weights_only_from:
        parser.error("--resume_from_checkpoint and --load_weights_only_from are mutually exclusive")
    return args


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


def apply_loss_overrides(global_cfg: GlobalConfig, enable_losses: Optional[str], disable_losses: Optional[str]) -> None:
    known = set(global_cfg.model.loss_weights.keys())

    if enable_losses is not None:
        enabled = {name.strip() for name in enable_losses.split(",") if name.strip()}
        unknown = enabled.difference(known)
        if unknown:
            raise ValueError(f"Unknown losses in --enable_losses: {sorted(unknown)}")
        global_cfg.model.loss_enabled = {
            name: name in enabled for name in global_cfg.model.loss_weights.keys()
        }

    if disable_losses is not None:
        disabled = {name.strip() for name in disable_losses.split(",") if name.strip()}
        unknown = disabled.difference(known)
        if unknown:
            raise ValueError(f"Unknown losses in --disable_losses: {sorted(unknown)}")
        for name in disabled:
            global_cfg.model.loss_enabled[name] = False


def load_weights_only(model: CS2PredictorModule, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise ValueError(f"Checkpoint at {checkpoint_path} does not contain a state_dict")

    model_state = model.state_dict()
    filtered_state: Dict[str, torch.Tensor] = {}
    skipped_shape: List[str] = []
    for key, value in state_dict.items():
        target = model_state.get(key)
        if target is None:
            continue
        if target.shape != value.shape:
            skipped_shape.append(key)
            continue
        filtered_state[key] = value

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    if skipped_shape:
        print(
            f"Warning: skipped {len(skipped_shape)} shape-mismatched tensors while loading weights-only from "
            f"{checkpoint_path}"
        )
    if missing:
        print(
            f"Warning: initialized {len(missing)} model tensors from defaults while loading weights-only from "
            f"{checkpoint_path}"
        )
    if unexpected:
        print(
            f"Warning: ignored {len(unexpected)} unexpected checkpoint tensors while loading weights-only from "
            f"{checkpoint_path}"
        )


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
    load_weights_only_from: Optional[str] = None,
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
    if load_weights_only_from is not None:
        load_weights_only(model, load_weights_only_from)
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
    if args.max_epochs is not None:
        global_cfg.train.max_epochs = args.max_epochs
    if args.batch_size is not None:
        global_cfg.train.batch_size = args.batch_size
    if args.grad_accumulation_steps is not None:
        global_cfg.train.grad_accumulation_steps = args.grad_accumulation_steps
    if args.perceiver_pos_embedding is not None:
        global_cfg.model.perceiver_pos_embedding = args.perceiver_pos_embedding
    apply_loss_overrides(global_cfg, args.enable_losses, args.disable_losses)
    run_training(
        global_cfg=global_cfg,
        resume_from_checkpoint=args.resume_from_checkpoint,
        load_weights_only_from=args.load_weights_only_from,
        debug=args.debug,
        enable_wandb=True,
        enable_checkpoints=True,
        enable_lr_monitor=True,
    )

if __name__ == "__main__":
    main()
