from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class DatasetConfig:
    data_root: str
    run_dir: str

    warn_skip: bool = False
    sample_stride: int = 256
    epoch_round_sample_length: int = 512
    epoch_video_decoding_device: str = "cpu"
    audio_sample_rate: int = 24000


@dataclass
class ModelConfig:
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    tokens_per_frame: int = 1

    dtype: torch.dtype = torch.bfloat16

    vision_model_name: str = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
    vision_hidden_size: int = 384
    vision_chunk_size: int = 16

    audio_chunk_size: int = 1

    num_perceiver_queries: int = 58
    perceiver_grid_h: int = 6
    perceiver_grid_w: int = 8
    perceiver_global_count: int = 10
    perceiver_hidden_size: int = 512
    perceiver_heads: int = 8
    patch_compressor_num_blocks: int = 4
    patch_compressor_self_attends_per_block: int = 2
    patch_compressor_mlp_ratio: float = 4.0
    patch_compressor_dropout: float = 0.0
    perceiver_patch_grid_h: int = 30
    perceiver_patch_grid_w: int = 40
    perceiver_pos_embedding: str = "sincos"

    adapter_hidden_dim: int = 1536

    llama_hidden_size: int = 768
    llama_layers: int = 12
    llama_heads: int = 12
    llama_kv_heads: int = 6
    llama_intermediate: int = 2048
    llama_max_pos_embeddings: int = 8192

    keyboard_dim: int = 32
    eco_dim: int = 256

    mouse_bins_count: int = 33
    mouse_mu: int = 255
    mouse_max: float = 30.0
    mouse_center_bin_weight: float = 0.3

    loss_weights: dict | None = None
    loss_enabled: dict | None = None

    def __post_init__(self):
        if self.perceiver_pos_embedding not in {"none", "sincos", "learned"}:
            raise ValueError(f"Unsupported perceiver_pos_embedding: {self.perceiver_pos_embedding}")
        if self.loss_weights is None:
            self.loss_weights = {
                "mouse": 0.4,
                "keyboard": 10.0,
                "eco_buy": 1.2,
                "eco_purchase": 1.2,
            }
        if self.loss_enabled is None:
            self.loss_enabled = {name: True for name in self.loss_weights.keys()}

    @property
    def no_buy_index(self) -> int:
        return self.eco_dim


@dataclass
class TrainConfig:
    project_name: str = "cs2-behavior-cloning"
    run_name: str = "model4"
    output_dir: str = "./checkpoints_fsdp"

    data_root: str = "./dataset0"
    num_workers: int = 2

    batch_size: int = 1
    grad_accumulation_steps: int = 8
    max_epochs: int = 8
    lr: float = 1.7259793100656096e-04
    weight_decay: float = 0.0013399060561509793
    warmup_steps: int = 200
    min_lr_ratio: float = 0.01
    clip_grad_norm: float = 1.0

    save_every: int = 1
    checkpoint_every_n_steps: int = 200
    mixed_precision: str = "bf16"

    val_samples_limit: int = 300
    val_every_steps: int = 1000


@dataclass
class GlobalConfig:
    dataset: DatasetConfig
    model: ModelConfig
    train: TrainConfig

    def to_file(self, path: str | Path) -> None:
        path = Path(path)
        dataset_dict = asdict(self.dataset)
        model_dict = asdict(self.model)
        train_dict = asdict(self.train)
        model_dict["dtype"] = str(self.model.dtype).split(".")[-1]

        data = {
            "dataset": dataset_dict,
            "model": model_dict,
            "train": train_dict,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_file(cls, path: str | Path) -> GlobalConfig:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        dataset_cfg = DatasetConfig(**data["dataset"])

        model_data = dict(data["model"])
        dtype_str = model_data.get("dtype", "bfloat16")
        model_data["dtype"] = getattr(torch, dtype_str)
        model_cfg = ModelConfig(**model_data)

        train_cfg = TrainConfig(**data["train"])
        return cls(dataset=dataset_cfg, model=model_cfg, train=train_cfg)
