from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import torch

# ---------- Sub-configs ----------

@dataclass
class DatasetConfig:
    data_root: str  # root dir of dataset
    run_dir: str    # directory for temp files, logs etc

    warn_skip: bool = False

    epoch_gen_random_seed: int = 42
    epoch_windows_per_round: int = 1  # how many random windows
    epoch_round_sample_length: int = 512  # number of frames per window
    epoch_video_decoding_device: str = "cpu"
    audio_sample_rate: int = 24000


@dataclass
class ModelConfig:
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    tokens_per_frame: int = 6  # 5 player + 1 strategy

    dtype: torch.dtype = torch.bfloat16
    
    vision_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    vision_hidden_size: int = 768
    vision_chunk_size: int = 16  # how many images go through vit+perceiver at a time

    audio_chunk_size: int = 1  # how many audio chunks go through audio encoder at a time

    # --- Fusion (Q-Former) ---
    num_perceiver_queries: int = 50
    perceiver_hidden_size: int = 768
    perceiver_heads: int = 12
    perceiver_layers: int = 4
    patch_compressor_num_blocks: int = 4
    patch_compressor_self_attends_per_block: int = 2
    patch_compressor_mlp_ratio: float = 4.0
    patch_compressor_dropout: float = 0.0

    # Adapter
    adapter_hidden_dim: int = 4096

    # --- Backbone (Llama) ---
    backbone_splits: int = 4
    llama_hidden_size: int = 2048
    llama_layers: int = 24
    llama_heads: int = 32
    llama_kv_heads: int = 8
    llama_intermediate: int = 5632
    llama_max_pos_embeddings: int = 8192

    # --- Output Head Dimensions ---
    keyboard_dim: int = 32
    eco_dim: int = 256
    inventory_dim: int = 128
    weapon_dim: int = 128
    round_state_dim: int = 5
    
    # New/Updated Head Configs
    health_bins: int = 11      # 0-100 in steps of 10
    armor_bins: int = 11       # 0-100 in steps of 10
    money_bins: int = 33       # 0-16000 in steps of 500
    round_num_bins: int = 31   # 0-30
    alive_bins: int = 6        # 0-5

    # bins for position
    bins_x: int = 20
    bins_y: int = 20
    bins_z: int = 20

    # mouse config
    mouse_bins_count: int = 33
    mouse_mu: int = 255
    mouse_max: float = 30.0

    # Loss Configuration
    loss_weights: dict = None
    loss_enabled: dict = None

    def __post_init__(self):
        if self.loss_weights is None:
            self.loss_weights = {
                "health": 1.0, "armor": 1.0, "money": 1.0,
                "round_state": 1.0, "round_num": 1.0,
                "team_alive": 1.0, "enemy_alive": 1.0,
                "active_weapon": 10.0, "eco_buy": 1.0, "eco_purchase": 1.0,
                "player_pos": 0.5, "enemy_pos": 0.5, # grouped x,y,z
                "keyboard": 100.0, "mouse": 10.0
            }
        if self.loss_enabled is None:
            self.loss_enabled = {k: True for k in self.loss_weights.keys()}


@dataclass
class TrainConfig:
    # Experiment
    project_name: str = "cs2-behavior-cloning"
    run_name: str = "llama-dac-fsdp2"
    output_dir: str = "./checkpoints_fsdp"
    
    # Data
    data_root: str = "./dataset0"
    num_workers: int = 4
    
    # Optimization
    batch_size: int = 1          # Per GPU
    grad_accumulation_steps: int = 16
    max_epochs: int = 20
    lr: float = 2e-4
    weight_decay: float = 0.05
    warmup_steps: int = 2000
    clip_grad_norm: float = 1.0
    
    # System
    save_every: int = 1
    mixed_precision: str = "bf16"  # "bf16", "fp16", "fp32", ...


# ---------- GlobalConfig with read/write ----------

@dataclass
class GlobalConfig:
    dataset: DatasetConfig
    model: ModelConfig
    train: TrainConfig

    def to_file(self, path: str | Path) -> None:
        """Serialize the whole config bundle to a JSON file."""
        path = Path(path)

        # Convert sub-configs to plain dicts
        dataset_dict = asdict(self.dataset)
        model_dict = asdict(self.model)
        train_dict = asdict(self.train)

        # Handle non-JSON types (torch.dtype) explicitly
        model_dict["dtype"] = str(self.model.dtype).split('.')[-1]  # e.g. "bfloat16"

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
        """Load JSON from disk and rebuild a GlobalConfig."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct DatasetConfig
        dataset_cfg = DatasetConfig(**data["dataset"])

        # Reconstruct ModelConfig (restore dtype from string)
        model_data = dict(data["model"])
        dtype_str = model_data.get("dtype", "bfloat16")
        model_data["dtype"] = getattr(torch, dtype_str)
        model_cfg = ModelConfig(**model_data)

        # Reconstruct TrainConfig
        train_cfg = TrainConfig(**data["train"])

        return cls(
            dataset=dataset_cfg,
            model=model_cfg,
            train=train_cfg,
        )
