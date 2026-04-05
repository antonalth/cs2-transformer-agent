import argparse
import json
import statistics
import time

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pytorch_lightning.callbacks import Callback

from train import create_global_config, create_strategy, resolve_precision
from lightning_datamodule import CS2DataModule
from lightning_module import CS2PredictorModule


class TimingCallback(Callback):
    def __init__(self):
        self.times: list[float] = []
        self._start: float | None = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        torch.cuda.reset_peak_memory_stats()

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch,
        batch_idx: int,
    ) -> None:
        torch.cuda.synchronize()
        self._start = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        torch.cuda.synchronize()
        assert self._start is not None
        self.times.append(time.perf_counter() - self._start)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        peak = torch.tensor(float(torch.cuda.max_memory_allocated()), device=pl_module.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)

        if trainer.is_global_zero:
            steady = self.times[1:] if len(self.times) > 1 else self.times
            mean_s = statistics.mean(steady)
            result = {
                "batch_size_per_gpu": trainer.datamodule.global_cfg.train.batch_size,
                "world_size": trainer.world_size,
                "global_batch_size": trainer.datamodule.global_cfg.train.batch_size * trainer.world_size,
                "batch_times_s": self.times,
                "steady_mean_s": mean_s,
                "samples_per_s": (trainer.datamodule.global_cfg.train.batch_size * trainer.world_size) / mean_s,
                "peak_bytes_max_rank": int(peak.item()),
            }
            print("BENCH_RESULT=" + json.dumps(result))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--limit_train_batches", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = create_global_config(
        data_root=args.data_root,
        run_name="model4_benchmark",
        output_dir="/tmp/model4_benchmark",
    )
    cfg.train.batch_size = args.batch_size
    cfg.train.grad_accumulation_steps = 1
    cfg.train.num_workers = args.num_workers
    cfg.train.max_epochs = 1

    dm = CS2DataModule(cfg)
    model = CS2PredictorModule(cfg)
    timing = TimingCallback()

    trainer = pl.Trainer(
        default_root_dir="/tmp/model4_benchmark",
        max_epochs=1,
        accelerator="gpu",
        devices=2,
        strategy=create_strategy(),
        precision=resolve_precision(cfg.train.mixed_precision),
        logger=False,
        enable_checkpointing=False,
        callbacks=[timing],
        accumulate_grad_batches=1,
        gradient_clip_val=None,
        val_check_interval=1000000,
        check_val_every_n_epoch=None,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        limit_train_batches=args.limit_train_batches,
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
