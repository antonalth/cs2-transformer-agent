from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch


def _model3_root() -> Path:
    return Path(__file__).resolve().parents[2] / "transformers" / "model3"


def ensure_model3_import_path() -> Path:
    model3_root = _model3_root()
    root_str = str(model3_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return model3_root


def _load_checkpoint_config(checkpoint_path: str, data_root: str):
    ensure_model3_import_path()

    from config import DatasetConfig, GlobalConfig, ModelConfig, TrainConfig

    ckpt_path = Path(checkpoint_path).resolve()
    direct = ckpt_path.parent / "config.json"
    parent = ckpt_path.parent.parent / "config.json"

    if direct.exists():
        global_cfg = GlobalConfig.from_file(direct)
    elif parent.exists():
        global_cfg = GlobalConfig.from_file(parent)
    else:
        global_cfg = GlobalConfig(
            dataset=DatasetConfig(data_root=data_root, run_dir="./runs"),
            model=ModelConfig(),
            train=TrainConfig(),
        )

    global_cfg.dataset.data_root = data_root
    return global_cfg


@dataclass(slots=True)
class LoadedModel3:
    checkpoint_path: str
    global_cfg: object
    module: torch.nn.Module
    backbone: torch.nn.Module
    device: torch.device
    dtype: torch.dtype

    def prepare_batch_tensors(self, images: torch.Tensor, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return images.contiguous(), audio.to(device=self.device, dtype=self.dtype, non_blocking=True).contiguous()


def load_model3_checkpoint(
    checkpoint_path: str,
    *,
    device: str = "cuda",
    data_root: str = "./dataset0",
) -> LoadedModel3:
    ensure_model3_import_path()

    from lightning_module import CS2PredictorModule

    resolved_device = torch.device(device)
    global_cfg = _load_checkpoint_config(checkpoint_path, data_root)
    module = CS2PredictorModule.load_from_checkpoint(
        checkpoint_path,
        global_cfg=global_cfg,
        strict=False,
    )
    module.to(resolved_device)
    module.eval()
    backbone = module.model
    backbone.eval()

    return LoadedModel3(
        checkpoint_path=str(Path(checkpoint_path).resolve()),
        global_cfg=global_cfg,
        module=module,
        backbone=backbone,
        device=resolved_device,
        dtype=global_cfg.model.dtype,
    )
