from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch


def _model4_root() -> Path:
    return Path(__file__).resolve().parents[2] / "transformers" / "model4"


def ensure_model4_import_path() -> Path:
    model4_root = _model4_root()
    root_str = str(model4_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return model4_root


def _load_checkpoint_config(checkpoint_path: str, data_root: str):
    ensure_model4_import_path()

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
class LoadedModel4:
    checkpoint_path: str
    global_cfg: object
    module: torch.nn.Module
    backbone: torch.nn.Module
    device: torch.device
    dtype: torch.dtype

    def prepare_batch_tensors(self, images: torch.Tensor, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return images.contiguous(), audio.to(device=self.device, dtype=self.dtype, non_blocking=True).contiguous()


def load_model4_checkpoint(
    checkpoint_path: str,
    *,
    device: str = "cuda",
    data_root: str = "./dataset0",
    enable_fast_vision_preprocess: bool = False,
) -> LoadedModel4:
    ensure_model4_import_path()

    from lightning_module import CS2PredictorModule
    from pytorch_lightning.utilities.migration import pl_legacy_patch

    resolved_device = torch.device(device)
    global_cfg = _load_checkpoint_config(checkpoint_path, data_root)
    module = CS2PredictorModule(global_cfg)

    with pl_legacy_patch():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise ValueError(f"Checkpoint at {checkpoint_path} does not contain a state_dict")

    model_state = module.state_dict()
    filtered_state: dict[str, torch.Tensor] = {}
    skipped_shape = []
    for key, value in state_dict.items():
        target = model_state.get(key)
        if target is None:
            continue
        if target.shape != value.shape:
            skipped_shape.append(key)
            continue
        filtered_state[key] = value

    missing, unexpected = module.load_state_dict(filtered_state, strict=False)
    if skipped_shape:
        print(
            f"Warning: skipped {len(skipped_shape)} checkpoint tensors with shape mismatches "
            f"while loading {checkpoint_path}"
        )
    if unexpected:
        print(
            f"Warning: ignored {len(unexpected)} unexpected checkpoint tensors while loading {checkpoint_path}"
        )
    if missing:
        print(
            f"Warning: initialized {len(missing)} model tensors from defaults while loading {checkpoint_path}"
        )

    module.to(resolved_device)
    module.eval()
    backbone = module.model
    backbone.eval()

    if enable_fast_vision_preprocess:
        backbone.video.set_fast_preprocess(True)

    return LoadedModel4(
        checkpoint_path=str(Path(checkpoint_path).resolve()),
        global_cfg=global_cfg,
        module=module,
        backbone=backbone,
        device=resolved_device,
        dtype=global_cfg.model.dtype,
    )
