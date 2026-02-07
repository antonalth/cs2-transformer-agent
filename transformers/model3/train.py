import argparse
import os
import math
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import FSDPStrategy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from config import GlobalConfig, DatasetConfig, ModelConfig, TrainConfig
from lightning_module import CS2PredictorModule
from lightning_datamodule import CS2DataModule

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="unnamed")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_fsdp")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # 1. Config Setup
    d_cfg = DatasetConfig(data_root=args.data_root, run_dir="./runs")
    t_cfg = TrainConfig(
        data_root=args.data_root,
        run_name=args.run_name,
        output_dir=args.output_dir
    )
    m_cfg = ModelConfig()
    global_cfg = GlobalConfig(dataset=d_cfg, model=m_cfg, train=t_cfg)

    # Save config
    os.makedirs(t_cfg.output_dir, exist_ok=True)
    global_cfg.to_file(os.path.join(t_cfg.output_dir, "config.json"))

    # 2. Modules
    dm = CS2DataModule(global_cfg)
    model = CS2PredictorModule(global_cfg)

    # 3. Loggers & Callbacks
    logger = WandbLogger(
        project=t_cfg.project_name,
        name=t_cfg.run_name,
        save_dir=t_cfg.output_dir,
        log_model=False # Don't upload checkpoints to wandb by default
    )
    
    # Checkpoint every epoch
    checkpoint_callback_epoch = ModelCheckpoint(
        dirpath=t_cfg.output_dir,
        filename="checkpoint_ep{epoch}",
        save_top_k=-1, 
        every_n_epochs=t_cfg.save_every,
        save_on_train_epoch_end=True,
        save_weights_only=False
    )

    # Checkpoint every N steps (intra-epoch)
    checkpoint_callback_step = ModelCheckpoint(
        dirpath=t_cfg.output_dir,
        filename="checkpoint_step{step}",
        save_top_k=-1, # Keep all step checkpoints (or use a monitor if you want top K)
        every_n_train_steps=t_cfg.checkpoint_every_n_steps,
        save_weights_only=False
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # 4. Strategy & Precision
    # Map mixed_precision string to PL precision
    precision_map = {
        "bf16": "bf16-mixed",
        "fp16": "16-mixed",
        "fp32": "32-true",
        "no": "32-true"
    }
    precision = precision_map.get(t_cfg.mixed_precision, "bf16-mixed")

    # FSDP Strategy
    strategy = FSDPStrategy(
        sharding_strategy="FULL_SHARD",
        auto_wrap_policy={LlamaDecoderLayer},
        cpu_offload=False, # Can enable if OOM
    )

    # Calculate validation batches limit
    # We estimate based on visible devices. PL might change this internally if devices="auto" selects fewer,
    # but this is a good approximation.
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    val_limit_batches = 1.0 # default 100%
    if t_cfg.val_samples_limit is not None:
        global_batch_size = t_cfg.batch_size * num_devices
        val_limit_batches = math.ceil(t_cfg.val_samples_limit / max(1, global_batch_size))
    else:
        val_limit_batches = 1.0 # Full validation set

    # 5. Trainer
    trainer = pl.Trainer(
        default_root_dir=t_cfg.output_dir,
        max_epochs=t_cfg.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        strategy=strategy,
        precision=precision,
        logger=logger,
        callbacks=[checkpoint_callback_epoch, checkpoint_callback_step, lr_monitor],
        accumulate_grad_batches=t_cfg.grad_accumulation_steps,
        gradient_clip_val=None, # Disabled to avoid FSDP MisconfigurationException
        
        # Validation Scheduling
        val_check_interval=t_cfg.val_every_steps, 
        check_val_every_n_epoch=None, # Required when val_check_interval is int (steps)
        limit_val_batches=val_limit_batches,
        
        log_every_n_steps=1,
        
        # Debugging
        fast_dev_run=args.debug
    )

    # 6. Fit
    trainer.fit(model, datamodule=dm, ckpt_path=args.resume_from_checkpoint)

if __name__ == "__main__":
    main()