from __future__ import annotations
from dataclasses import dataclass, field
import torch

@dataclass
class DatasetConfig:
    data_root: str #root dir of dataset
    run_dir: str #directory for temp files, logs etc

    epoch_gen_random_seed: int = 42
    epoch_windows_per_round: int = 1 #how many random windows
    epoch_round_sample_length: int = 1024 #number of frames per window
    epoch_video_decoding_device: str = "cpu"
    audio_sample_rate: int = 24000

@dataclass
class ModelConfig:
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    tokens_per_frame: int = 6 # 5 player + 1 strategy

    dtype: torch.dtype = torch.bfloat16
    
    vision_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    vision_hidden_size: int = 768
    vision_chunk_size: int = 16 #how many images go through vit+qformer at a time

    audio_chunk_size: int = 1 # how many audio chunks go through audio encoder at a time

    # --- Fusion (Q-Former) ---
    num_qformer_queries: int = 4  
    qformer_hidden_size: int = 768
    qformer_heads: int = 12
    qformer_layers: int = 4

    # Adapter
    adapter_hidden_dim: int = 4096

    # --- Backbone (Llama) ---
    llama_hidden_size: int = 2048 
    llama_layers: int = 24
    llama_heads: int = 32
    llama_kv_heads: int = 8       
    llama_intermediate: int = 5632 
    llama_max_pos_embeddings: int = 8192

    # --- Output Head Dimensions (Added to prevent AttributeErrors) ---
    keyboard_dim: int = 32
    eco_dim: int = 256
    inventory_dim: int = 128
    weapon_dim: int = 128
    round_state_dim: int = 5
    round_number_dim: int = 1
    bins_x: int = 256
    bins_y: int = 256
    bins_z: int = 32

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
    mixed_precision: str = "bf16"

@dataclass
class GlobalConfig:
    dataset: DatasetConfig
    model: ModelConfig
    train: TrainConfig
