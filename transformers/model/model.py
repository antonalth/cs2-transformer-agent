"""
Erstellt mit Buntstiften. Weil C++ zu schwierig ist für mich.
====================================================================================================
A Causal Autoregressive Multi-Modal Transformer for CS2 Gameplay Generation
====================================================================================================

This script defines the complete architecture for a high-performance, large-scale generative
model designed to produce plausible, second-by-second gameplay for a full 5-player team in
Counter-Strike 2.

----------------------------------------------------------------------------------------------------
1. HIGH-LEVEL TASK & PARADIGM
----------------------------------------------------------------------------------------------------
The model is an AUTOREGRESSIVE 'PLAYER AGENT'. Its fundamental task is to function as a
generative model: given a history of `t` frames of gameplay, it predicts the actions and state
for the next frame, `t+1`.

- Causal Paradigm: The model operates under a strict causal constraint. Its predictions
  for any given moment can only be conditioned on past information. This is enforced
  by a causal attention mask within the transformer backbone, making the model suitable for
  real-time, sequential generation.

- Generative Feedback Loop (Inference): During inference, the model forms a feedback loop.
  1. The model predicts actions for frame `t+1`.
  2. These actions are used to update an external game simulator.
  3. The simulator provides the new visual and audio data for frame `t+1`.
  4. This new frame is encoded, appended to the history cache, and the process repeats.

----------------------------------------------------------------------------------------------------
2. CORE ARCHITECTURAL SPECIFICATIONS (THE THREE STAGES)
----------------------------------------------------------------------------------------------------
The model is a large-scale transformer with a hidden dimension (d_model) of 2048, ~24 layers,
and 32 attention heads. It is composed of three main stages:
  I.   Input Encoding & Token Fusion
  II.  Transformer Backbone
  III. Prediction Heads

====================================================================================================
  I. STAGE 1: INPUT ENCODING & TOKEN FUSION
====================================================================================================
For each of the 5 players, a single [1, 2048] token is created per frame by fusing visual and
audio data streams.

[A] VISUAL STREAM:
  - Input: A SINGLE game screen tensor per player (e.g., 480x640 resolution).
  - Preprocessing: The input image undergoes aspect-ratio-preserving resizing (letterboxing)
    to match the native input size of the vision encoder (e.g., 448x448).
  - Encoder: A SINGLE, SHARED-WEIGHT Vision Transformer (ViT) model, processes the image.
  - Projection: A final linear layer projects the ViT's output feature vector to the model's
    native `[1, 2048]` dimension, creating the final visual embedding.

[B] AUDIO STREAM:
  - Input: A `[128, ~6]` Mel Spectrogram tensor derived from the player's in-game audio.
  - Encoder: A small 2D CNN (`AudioCNN`) extracts features from the spectrogram.
  - Projection: A linear layer projects the flattened CNN features to a `[1, 2048]` audio embedding.

[C] TOKEN FUSION:
  - The final visual and audio embeddings (`[1, 2048]`) are added together element-wise.
  - A LayerNorm is applied to the sum to create the final, fused PLAYER TOKEN.

====================================================================================================
  II. STAGE 2: TRANSFORMER BACKBONE
====================================================================================================
The backbone processes a sequence of token frames over time.

[A] INPUT SEQUENCE ASSEMBLY (PER FRAME):
  A sequence of 7 tokens is created for every frame of gameplay:
  - 5x Player Tokens: One for each player, derived from Stage 1.
  - 2x Special Tokens:
    1. `[GAME_STRATEGY]`: A learned token used to aggregate game-wide state and make
       strategic predictions.
    2. `[SCRATCHSPACE]`: A learned token for the model to use as intermediate memory.

[B] IDENTITY & STATE HANDLING:
  - Player Slot Embedding: To maintain player identity, a unique, learned `player_slot_embedding`
    is ADDED to each of the 5 player tokens.
  - DEAD Token Policy: If a player is dead, their visual/audio input is ignored. Instead,
    their token is replaced by a learned `[DEAD]` embedding, to which their unique
    `player_slot_embedding` is still added.

[C] TRANSFORMER LAYERS:
  - A stack of ~24 custom `CS2TransformerEncoderLayer` modules.
  - Each layer contains:
    1. A Causal Self-Attention block (optimized with GQA and FlashAttention-2).
    2. A Feed-Forward Network (FFN) with a GELU activation.
  - Pre-Layer Normalization is used throughout for training stability.

====================================================================================================
  III. STAGE 3: PREDICTION HEADS
====================================================================================================
The output tokens from the FINAL TIME STEP of the transformer backbone are used to make
predictions for the next frame.

[A] PLAYER PREDICTION HEADS (x5):
  Each of the 5 player output tokens is fed into dedicated heads to predict:
  - Stats (Regression): Health, Armor, Money.
  - Position (Heatmap): A 3D position heatmap `[Z, Y, X]` via a Deconvolutional Network.
  - Mouse (Regression): `[delta_x, delta_y]` mouse movement.
  - Actions (Classification): Logits for Keyboard, Eco, Inventory, and Active Weapon states.

[B] GAME STRATEGY HEAD (x1):
  The `[GAME_STRATEGY]` output token is used to predict:
  - Enemy Positions (Heatmap): A 3D heatmap of predicted enemy locations.
  - Game Phase (Classification): Logits for the round state (e.g., "Bomb Planted").
  - Round Number (Regression): A single scalar value.

----------------------------------------------------------------------------------------------------
3. HIGH-PERFORMANCE OPTIMIZATIONS
----------------------------------------------------------------------------------------------------
- `torch.compile()`: The entire `CS2Transformer` module is JIT-compiled for kernel fusion
  and significant speedups.

- Grouped-Query Attention (GQA): Reduces the size of the KV cache—the primary bottleneck
  for fast autoregressive inference—by using fewer Key/Value heads than Query heads.

- FlashAttention-2: A custom kernel that computes attention without materializing the large
  `N x N` attention matrix, drastically reducing memory usage and increasing speed.

- 2D Rotary Positional Embeddings (RoPE): Positional information is injected by rotating
  queries and keys. This implementation uniquely handles two axes: a `temporal` axis
  (frame index) and a `structural` axis (token index within a frame), providing excellent
  performance on long, structured sequences.

- Mixed Precision & Quantization: Natively supports `BF16` for training and is compatible
  with advanced quantization (e.g., QLoRA) and FP8 inference for maximum throughput.

----------------------------------------------------------------------------------------------------
4. TRAINING & FINE-TUNING STRATEGY
----------------------------------------------------------------------------------------------------
- Objective: The model is trained on a Next-Frame Prediction task using a composite loss
  function to predict the ground-truth data for frame `t+1` given frames `1...t`.

- ViT Fine-Tuning (Two-Stage):
  1. Freeze: Initially, the weights of the single pre-trained ViT encoder are frozen
     (`requires_grad=False`). The rest of the model learns to interpret its powerful,
     off-the-shelf features.
  2. Finetune: After the main model has stabilized, the ViT is unfrozen. The entire model is
     then trained end-to-end using DIFFERENTIAL LEARNING RATES, where the ViT uses a
     learning rate 10-100x smaller than the rest of the model to prevent catastrophic forgetting.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import List, TypedDict, Dict, Optional, Tuple, Literal, Any
import time, argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from contextlib import nullcontext

torch.backends.cuda.enable_flash_sdp(True)
#torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)

# -----------------------------------------------------------------------------
# 0) Public types (Appendix B)
# -----------------------------------------------------------------------------

class PlayerPredictions(TypedDict):
    stats: torch.Tensor                      # [B, 3]
    pos_heatmap_logits: torch.Tensor         # [B, 8, 64, 64]
    mouse_delta_deg: torch.Tensor            # [B, 2]
    keyboard_logits: torch.Tensor            # [B, 31]
    eco_logits: torch.Tensor                 # [B, 224]
    inventory_logits: torch.Tensor           # [B, 128]
    active_weapon_logits: torch.Tensor       # [B, 128]

class GameStrategyPredictions(TypedDict):
    enemy_pos_heatmap_logits: torch.Tensor   # [B, 8, 64, 64]
    round_state_logits: torch.Tensor         # [B, 5]
    round_number: torch.Tensor               # [B, 1]

class Predictions(TypedDict):
    player: List[PlayerPredictions]          # len == 5
    game_strategy: GameStrategyPredictions


# Optional: declare opaque input type if your dataloader provides a struct
class CS2Batch(TypedDict, total=True):
    # Shapes are illustrative; align with your datapipe
    images: torch.Tensor                   # [B, T, 5, 3, 480, 640]
    mel_spectrogram: torch.Tensor          # [B, T, 5, 2, 128, 1]  # stereo, one frame
    alive_mask: torch.Tensor               # [B, T, 5] bool


# -----------------------------------------------------------------------------
# 1) Configuration
# -----------------------------------------------------------------------------

@dataclass
class CS2Config:
    compute_dtype: Literal["fp32", "fp16", "bf16"]  = "bf16"
    amp_autocast: bool = True
    # Model dims
    d_model: int = 2048
    n_layers: int = 24
    n_q_heads: int = 32
    n_kv_heads: int = 8
    ffn_mult: int = 4
    attn_dropout: float = 0.0

    # Sequence
    num_players: int = 5
    tokens_per_frame: int = 7  # 5 players + 2 special

    # Context (training)
    context_frames: int = 1

    max_cache_len_tokens: Optional[int] = None
    
    # NEW: This flag controls inference masking. If True, inference uses a standard
    # (token-by-token) causal mask, ensuring consistency with models trained using
    # `use_fused_causal=True`. Set to False to use the frame-block mask during inference.
    inference_use_standard_causal: bool = True
    training_use_frame_block_causal: bool = False #like above but reverse

    # Vision (HF / DINOv3)
    hf_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    hf_use_processor: bool = True     # True = preprocess inside encoder with HF image processor
    hf_channels_last: bool = True     # same perf
    vit_frozen: bool = True           # will probably stay frozen for entire training since dinov3

    # Audio
    mel_bins: int = 128
    mel_t: int = 1
    audio_cnn_channels: Tuple[int, int, int] = (32, 64, 128)
    # mel_spectrogram: [B, T, 5, 2, 128, 1]  # stereo, one frame

    # Heads
    keyboard_dim: int = 31
    eco_dim: int = 224
    inventory_dim: int = 128
    weapon_dim: int = 128
    round_state_dim: int = 5

    # Heatmaps
    pos_z, pos_y, pos_x = 8, 64, 64

    # RoPE / long-context
    rope_base: int = 10000
    rope_rot_dim: int = None
    rope_scaling: Optional[Dict[str, Any]] = None  # e.g.
    # {"type": "linear", "factor": 2.0}                   # 2× context
    # {"type": "linear_by_len", "orig": 4096, "target": 8192}
    
    # --- START: Cached Training / TBPTT ---
    enable_cached_training: bool = True
    cached_chunk_T: int = 32
    cached_warmup_T: int = 8
    cached_detach: bool = True
    # --- END: Cached Training / TBPTT ---
    
    # --- START: Gradient Checkpointing ---
    enable_grad_checkpoint: bool = True
    grad_ckpt_use_reentrant: bool = False
    # --- END: Gradient Checkpointing ---

@dataclass
class KVCache:
    key: torch.Tensor    # [B, L, H_kv, Hd]
    value: torch.Tensor  # [B, L, H_kv, Hd]
    pos_end: int         # absolute position *after* last cached token

def print_dataclass(dc: Any, indent: int = 0):
    """Recursively print a dataclass and its attributes."""
    if not is_dataclass(dc):
        raise TypeError(f"{type(dc)} is not a dataclass instance")

    pad = " " * indent
    print(f"{pad}{type(dc).__name__}:")
    for f in fields(dc):
        value = getattr(dc, f.name)
        if is_dataclass(value):
            print(f"{pad}  {f.name}:")
            print_dataclass(value, indent + 4)
        else:
            print(f"{pad}  {f.name} = {value}")

def _get_abs_pos_start(kv_cache_list: Optional[List[Optional[KVCache]]]) -> int:
    if not kv_cache_list:
        return 0
    for c in kv_cache_list:
        if c is not None:
            return int(c.pos_end)
    return 0

# --- strengthen cache invariants ---

def _assert_cache_consistency(kv_cache_list: Optional[List[Optional[KVCache]]]) -> None:
    if not kv_cache_list:
        return
    # Either all or none:
    has = [c is not None for c in kv_cache_list]
    if any(has) and not all(has):
        raise RuntimeError("KV cache must be provided for all layers or none.")

    lens = [int(c.key.shape[1]) for c in kv_cache_list if c is not None]
    pos_ends = [int(c.pos_end) for c in kv_cache_list if c is not None]
    if lens:
        L0 = lens[0]
        if not all(L == L0 for L in lens):
            raise RuntimeError(f"Inconsistent cache window lengths across layers: {lens}")
    if pos_ends:
        P0 = pos_ends[0]
        if not all(P == P0 for P in pos_ends):
            raise RuntimeError(f"Inconsistent cache pos_end across layers: {pos_ends}")

def _ensure_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.device != ref.device or x.dtype != ref.dtype:
        x = x.to(device=ref.device, dtype=ref.dtype)
    return x

def _update_cache(old: Optional[KVCache],
                  new_k: torch.Tensor,
                  new_v: torch.Tensor,
                  max_cache_len_tokens: Optional[int]) -> KVCache:
    if old is not None:
        new_k = _ensure_like(new_k, old.key)
        new_v = _ensure_like(new_v, old.value)

    if old is None:
        cat_k, cat_v = new_k, new_v
        pos_end = new_k.shape[1]
    else:
        cat_k = torch.cat([old.key, new_k], dim=1)
        cat_v = torch.cat([old.value, new_v], dim=1)
        pos_end = old.pos_end + new_k.shape[1]

    if max_cache_len_tokens is not None and cat_k.shape[1] > max_cache_len_tokens:
        cat_k = cat_k[:, -max_cache_len_tokens:]
        cat_v = cat_v[:, -max_cache_len_tokens:]

    if not (cat_k.requires_grad or cat_v.requires_grad):
        cat_k = cat_k.detach()
        cat_v = cat_v.detach()
    return KVCache(key=cat_k,
                   value=cat_v,
                   pos_end=pos_end)


def _GN2d(c: int) -> torch.nn.GroupNorm:
    # pick a divisor of c; 8→4→1 fallback mirrors _GN3d
    g = 8 if c % 8 == 0 else 4 if c % 4 == 0 else 1
    return torch.nn.GroupNorm(num_groups=g, num_channels=c)

def _GN3d(c: int) -> torch.nn.GroupNorm:
    # 8 groups is a good default; must divide c
    g = 8 if c % 8 == 0 else 4 if c % 4 == 0 else 1
    return torch.nn.GroupNorm(num_groups=g, num_channels=c)

# -----------------------------------------------------------------------------
# 2) Submodules — Encoders & Token Fusion
# -----------------------------------------------------------------------------
class DINOv3VisualEncoder(nn.Module):
    """
    Hugging Face DINOv3 encoder.

    Contract matches ViTVisualEncoder:
      - Input:  [B, T, P, 3, H, W]  (float in [0,1] or [0,255]); if hf_use_processor=False,
                                     then MUST already be resized & normalized for this model.
      - Output: [B, T, P, d_model]
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_model = int(cfg.d_model)
        self.use_channels_last = bool(getattr(cfg, "hf_channels_last", True))
        self.use_processor = bool(getattr(cfg, "hf_use_processor", True))
        self.model_name = getattr(cfg, "hf_model_name", "facebook/dinov3-vitb16-pretrain-lvd1689m")

        try:
            from transformers import AutoModel, AutoImageProcessor
        except Exception as e:
            raise RuntimeError(
                "The DINOv3VisualEncoder requires 'transformers'. "
                "Install it (pip install transformers accelerate) or set vision_backend='timm'.\n"
                f"Original import error: {e}"
            )

        # Image processor gives us the right resize/normalize for this backbone.
        self.processor = None
        if self.use_processor:
            try:
                self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            except Exception as e:
                raise RuntimeError(f"Failed to load AutoImageProcessor for '{self.model_name}': {e}")

        try:
            self.backbone = AutoModel.from_pretrained(self.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load DINOv3 model '{self.model_name}': {e}")

        # Hidden size → project into your transformer width
        self.hidden = int(getattr(self.backbone.config, "hidden_size", 768))
        self.proj = nn.Linear(self.hidden, self.d_model)

        # Optional freezing (your two-stage plan still applies)
        if bool(getattr(cfg, "vit_frozen", False)):
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Expect [B, T, P, C, H, W] with C==3
        if images.dim() != 6:
            raise ValueError(f"Expected 6D tensor [B,T,P,C,H,W], got {images.shape}")
        B, T, P, C, H, W = images.shape
        if C != 3:
            raise ValueError(f"Expected 3 channels, got {C}")

        N = B * T * P
        x = images.reshape(N, C, H, W)
        if self.use_channels_last and not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)

        # Prepare inputs:
        # - If using the HF processor, it will resize/normalize as the model expects
        #   (DINOv3 uses patch size 16; it accepts sizes multiple of 16; processor handles it).
        # - If NOT using processor, assume upstream already resized & normalized; pass as-is.
        inputs = {}
        if self.processor is not None:
            # AutoImageProcessor accepts torch tensors [C,H,W] in a batch
            inputs = self.processor(images=x, return_tensors="pt")
            inputs = {k: v.to(x.device) for k, v in inputs.items()}
        else:
            inputs = {"pixel_values": x}

        # If frozen, skip grad to save memory/compute
        need_grad = torch.is_grad_enabled() and any(p.requires_grad for p in self.backbone.parameters())
        ctx = nullcontext() if need_grad else torch.no_grad()
        with ctx:
            outputs = self.backbone(**inputs)
            # HF model card demonstrates using pooled_output
            feats = outputs.pooler_output  # [N, hidden]

        vis = self.proj(feats)
        return vis.view(B, T, P, self.d_model)

class AudioCNN(nn.Module):
    """Small 2D-CNN over Mel-spectrograms → [B, T, P, d_model].

    This version accepts a 2-channel (Stereo) spectrogram.
    Input mel: [B, T, P, 2, mel_bins, mel_t]
    """
    def __init__(self, cfg: CS2Config):
        super().__init__()
        c1, c2, c3 = cfg.audio_cnn_channels
        self.conv1 = nn.Conv2d(2, c1, 5, padding=2)
        self.gn1 = _GN2d(c1)
        self.conv2 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.gn2 = _GN2d(c2)
        self.conv3 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.gn3 = _GN2d(c3)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Linear(c3 * 4 * 4, cfg.d_model)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: [B, T, P, 2, mel_bins, mel_t] -> pack to [B*T*P, 2, mel_bins, mel_t]
        B, T, P = mel.shape[:3]
        x = mel.reshape(B * T * P, mel.shape[3], mel.shape[4], mel.shape[5])
        x = F.gelu(self.gn1(self.conv1(x)))
        x = F.gelu(self.gn2(self.conv2(x)))
        x = F.gelu(self.gn3(self.conv3(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)  # [N, d_model]
        x = x.reshape(B, T, P, -1)
        return x #das ist mein roter buntstift, habe ich ganz alleine gemalt. (mithilfe von chef gpt)
    
class PlayerTokenFuser(nn.Module):
    """Fuse visual+audio via elementwise add, add slot identity, then LayerNorm.

    DEAD handling policy: if dead, replace fused token with
      dead_embedding + player_slot_embedding before final norm.
    """
    def __init__(
        self,
        cfg: CS2Config,
        player_slot_embed: nn.Embedding,
        dead_embedding: nn.Parameter,
    ):
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.d_model)
        self.slot_embed = player_slot_embed
        self.dead_embedding = dead_embedding

    def forward(
    self,
    vis: torch.Tensor,
    aud: torch.Tensor,
    alive_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse visual+audio, apply the DEAD policy, add player identity, and finally normalize.
        """
        B, T, P, d = vis.shape

        # 1) Fuse sensors for alive tokens.
        fused_alive_token = vis + aud

        # 2) Build broadcastable dead mask.
        dead_mask = ~alive_mask.bool().unsqueeze(-1)

        # 3) Choose base token: dead -> learned dead token, else fused sensors.
        base_token = torch.where(dead_mask, self.dead_embedding, fused_alive_token)

        # 4) CREATE slot_ids ON THE FLY on the correct device.
        # This is the key change. We use `vis.device` which is guaranteed to be correct.
        slot_ids = torch.arange(self.cfg.num_players, device=vis.device, dtype=torch.long).view(1, 1, -1)
        slots = self.slot_embed(slot_ids)

        # 5) Add player slot identity
        unnormalized = base_token + slots

        # 6) Final LayerNorm.
        return self.norm(unnormalized)
# -----------------------------------------------------------------------------
# 3) Attention core & Transformer layers (stubs for FA2+GQA+RoPE)
# -----------------------------------------------------------------------------
# In model.py


class RoPEPositionalEncoding(nn.Module):
    """Rotary Positional Embedding (RoPE) with 2D position support and optional scaling.

    This patched version handles two distinct positional axes:
    1. Temporal Axis: The frame index in a sequence. Context scaling (e.g., NTK)
       is applied here to extend the context window.
    2. Structural Axis: The token's position within a frame (e.g., P1, P2, ...).
       No scaling is applied to this axis.

    It applies sin/cos rotations to designated slices of the Q/K vectors per head.
    The total rotation dimension (`rot_dim`) is split evenly between the two axes.
    """
    def __init__(self, cfg: CS2Config):
        super().__init__()
        self.cfg = cfg
        self.base = cfg.rope_base
        self.rot_dim = cfg.rope_rot_dim
        # Default scale = 1.0. This scale is ONLY for the temporal axis.
        self.register_buffer("scale", torch.tensor(1.0), persistent=True)
        self._apply_cfg_scaling(getattr(cfg,"rope_scaling", None))

    def _apply_cfg_scaling(self, scaling: Optional[Dict[str, Any]]) -> None:
        """
        Supports:
          - {"type": "linear", "factor": s}          # extend context by s×
          - {"type": "linear_by_len", "orig": L0, "target": L1}  # extend from L0 to L1
        Internally we multiply inv_freq by `1/s` to reduce frequencies.
        This scaling is only applied to the temporal axis.
        """
        if not scaling:
            return
        sc_type = scaling.get("type", "linear")
        if sc_type == "linear":
            s = float(scaling.get("factor", 1.0))
            if s <= 0:
                raise ValueError("rope_scaling.factor must be > 0")
            self.set_scale(1.0 / s)
        elif sc_type == "linear_by_len":
            L0 = int(scaling["orig"])
            L1 = int(scaling["target"])
            if L0 <= 0 or L1 <= 0:
                raise ValueError("rope_scaling.orig/target must be > 0")
            self.set_scale(float(L0) / float(L1))
        else:
            raise ValueError(f"Unsupported rope_scaling.type: {sc_type}")

    def set_scale(self, scale: float) -> None:
        """Sets the scaling factor for the temporal axis."""
        self.scale.fill_(float(scale))

    def _inv_freq(self, dim: int, device: torch.device, use_scaling: bool) -> torch.Tensor:
        """Calculates inverse frequencies, with an option to apply temporal scaling."""
        half = dim // 2
        idx = torch.arange(0, half, device=device, dtype=torch.float32)
        inv = self.base ** (idx / half)
        inv_freq = 1.0 / inv
        
        # Apply NTK-like scaling only if specified (i.e., for the temporal axis)
        if use_scaling:
            inv_freq = inv_freq * self.scale
            
        return inv_freq

    def _build_cos_sin(
        self, positions: torch.Tensor, dim: int, device: torch.device, dtype: torch.dtype, use_scaling: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Builds the cosine and sine matrices for a given position tensor."""
        # Pass the scaling flag down to the frequency calculation
        inv_freq = self._inv_freq(dim, device, use_scaling=use_scaling).to(dtype)
        
        t = positions.to(device=device, dtype=dtype).unsqueeze(-1)      # [L, 1]
        freqs = t * inv_freq.unsqueeze(0)                                # [L, half]
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return cos, sin

    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Applies rotary embeddings to the entire input tensor 'x'."""
        # x: [B, L, H, rot_dim_slice]
        B, L, H, rd_slice = x.shape
        assert rd_slice % 2 == 0, "Rotary dimension must be even."
        
        x_pair = x.reshape(B, L, H, rd_slice // 2, 2)
        half_dim = rd_slice // 2
        cos = cos.reshape(1, L, 1, half_dim)
        sin = sin.reshape(1, L, 1, half_dim)

        x1 = x_pair[..., 0]
        x2 = x_pair[..., 1]
        
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        
        y = torch.stack([y1, y2], dim=-1).reshape(B, L, H, rd_slice)
        return y

    def forward(self, q: torch.Tensor, k: torch.Tensor, temporal_pos: torch.Tensor, structural_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies 2D RoPE to query and key tensors.

        Args:
            q: Query tensor of shape [B, L, Hq, Hd].
            k: Key tensor of shape [B, L, Hkv, Hd].
            temporal_pos: Integer tensor of temporal positions (frame index), shape [L].
            structural_pos: Integer tensor of structural positions (token index in frame), shape [L].

        Returns:
            A tuple of (rotated_q, rotated_k).
        """
        L = q.shape[1]
        if temporal_pos.shape[0] != L or structural_pos.shape[0] != L:
            raise ValueError(f"Length of position tensors must match sequence length {L}.")

        # Determine the total dimension to rotate.
        rd = self.rot_dim or q.shape[-1]
        
        # For 2D RoPE, the rotation dimension must be divisible by 4 to be split evenly
        # into two valid (even) rotation dimensions.
        if rd % 4 != 0:
            raise ValueError(f"Rotation dimension ({rd}) must be divisible by 4 for 2D RoPE.")

        # Split the rotation dimension for the two axes.
        rot_dim_temp = rd // 2
        rot_dim_struc = rd // 2

        # --- Slice the Tensors ---
        # First half of the rotatable dimension is for temporal positions.
        q_t, k_t = q[..., :rot_dim_temp], k[..., :rot_dim_temp]
        # Second half is for structural positions.
        q_s, k_s = q[..., rot_dim_temp:rd], k[..., rot_dim_temp:rd]
        # The remaining part is not rotated.
        q_pass, k_pass = q[..., rd:], k[..., rd:]

        # --- Temporal Rotation (with scaling) ---
        cos_t, sin_t = self._build_cos_sin(temporal_pos, rot_dim_temp, q.device, q.dtype, use_scaling=True)
        q_t_rot = self._apply_rotary(q_t, cos_t, sin_t)
        k_t_rot = self._apply_rotary(k_t, cos_t, sin_t)
        
        # --- Structural Rotation (without scaling) ---
        cos_s, sin_s = self._build_cos_sin(structural_pos, rot_dim_struc, q.device, q.dtype, use_scaling=False)
        q_s_rot = self._apply_rotary(q_s, cos_s, sin_s)
        k_s_rot = self._apply_rotary(k_s, cos_s, sin_s)
        
        # --- Recombine the tensors ---
        rotated_q = torch.cat([q_t_rot, q_s_rot, q_pass], dim=-1)
        rotated_k = torch.cat([k_t_rot, k_s_rot, k_pass], dim=-1)

        return rotated_q, rotated_k
        

class CS2GQAAttention(nn.Module):
    """Grouped-Query Attention with FlashAttention-2.

    Inputs: x [B, L, D]. Applies RoPE to Q/K, then attention.
    If FlashAttention-2 is available (and on CUDA + fp16/bf16), uses it when no external mask is given.
    Otherwise falls back to PyTorch SDPA.
    """
    def __init__(self, cfg: CS2Config, rope: RoPEPositionalEncoding):
        super().__init__()
        self.cfg = cfg
        self.rope = rope
        d, hq, hkv = cfg.d_model, cfg.n_q_heads, cfg.n_kv_heads
        assert d % hq == 0, "d_model must be divisible by n_q_heads"
        # FIX: Assert n_q_heads is divisible by n_kv_heads for GQA
        assert hq % hkv == 0, "n_q_heads must be divisible by n_kv_heads for GQA"
        self.head_dim = d // hq
        # RoPE needs even rotation dim; with full-head RoPE this means even head_dim
        assert self.head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
        self.dropout = float(getattr(cfg, "attn_dropout", 0.0))

        # Projections
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, hkv * self.head_dim, bias=False)
        self.wv = nn.Linear(d, hkv * self.head_dim, bias=False)
        self.wo = nn.Linear(d, d, bias=False)

    def _prep_attn_mask(self, attn_mask: Optional[torch.Tensor],
                        B: int,
                        H: int,
                        L_q: int,
                        L_k: int,
                        device: torch.device,
                        dtype: torch.dtype) -> Optional[torch.Tensor]:
        if attn_mask is None:
            return None

        m = attn_mask.to(device=device)
        if m.dtype == torch.bool:
                pass  # keep bool, SDPA will treat True as masked
        else:
            m = m.to(dtype=dtype)
        # 2D masks
        if m.dim() == 2:
            # Rectangular [L_q, L_k] → leave shape; SDPA will broadcast to [B, H, L_q, L_k]
            if m.shape[0] == L_q and m.shape[1] == L_k:
                return m
            # Square [L, L] → lift to [1, 1, L, L]
            if m.shape[0] == m.shape[1]:
                return m.reshape(1, 1, m.shape[0], m.shape[1])
            raise ValueError(f"attn_mask 2D shape {tuple(m.shape)} incompatible with (L_q={L_q}, L_k={L_k})")

        # 3D [B, L_q, L_k] → lift heads
        if m.dim() == 3 and m.shape[0] == B:
            return m.unsqueeze(1)  # [B,1,L_q,L_k]

        # 4D assumed to be [B,H,L_q,L_k]
        if m.dim() == 4:
            return m

        raise ValueError(f"Unsupported attn_mask dims: {m.dim()}")
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        temporal_pos_ids: torch.Tensor,
        structural_pos_ids: torch.Tensor,
        kv_cache: Optional[KVCache] = None
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, L_new, D = x.shape
        Hq, Hkv, Hd = self.cfg.n_q_heads, self.cfg.n_kv_heads, self.head_dim
        assert Hq % Hkv == 0, "n_q_heads must be a multiple of n_kv_heads for GQA"
        gqa_factor = Hq // Hkv
        
        use_kv_cache = kv_cache is not None

        if kv_cache is not None and kv_cache.key.device != x.device:
            raise RuntimeError("KV cache/device mismatch between steps.")
        
        if L_new == 0:
            # nothing to do; return no-op and preserve cache
            return x[:, :0, :], (kv_cache if use_kv_cache else None)

        # --- ABSOLUTE positions for RoPE ---
        # The caller (CS2Backbone) is responsible for providing correct absolute positions.
        # This module just applies them.

        # Projections for NEW tokens
        q_new = self.wq(x).reshape(B, L_new, Hq, Hd)
        k_new = self.wk(x).reshape(B, L_new, Hkv, Hd)
        v_new = self.wv(x).reshape(B, L_new, Hkv, Hd)

        # Apply RoPE to NEW tokens using ABS positions
        q_new, k_new = self.rope(q_new, k_new, temporal_pos_ids, structural_pos_ids)

        # --- Update cache with NEW (already-rotated) K/V ---
        new_cache = None
        if use_kv_cache:
            maxL = self.cfg.max_cache_len_tokens
            new_cache = _update_cache(kv_cache, k_new, v_new, maxL)

        # Build full K/V for attention
        if use_kv_cache:
            k_full = new_cache.key    # [B, L_tot, H_kv, Hd]
            v_full = new_cache.value  # [B, L_tot, H_kv, Hd]
        else:
            k_full, v_full = k_new, v_new
        # ---------------------------------------------------------------------------

        # GQA expansion for SDPA path
        if gqa_factor != 1:
            k_gqa = k_full.repeat_interleave(gqa_factor, dim=2)
            v_gqa = v_full.repeat_interleave(gqa_factor, dim=2)
        else:
            k_gqa, v_gqa = k_full, v_full

        # --- Mask selection ---
        sdpa_mask = None
        sdpa_is_causal = False
        L_tot = k_full.shape[1]

        if attn_mask is not None:
            sdpa_mask = self._prep_attn_mask(
                attn_mask, B=B, H=Hq, L_q=L_new, L_k=L_tot, device=x.device, dtype=q_new.dtype
            )

        elif (not self.training) and not getattr(self.cfg, "inference_use_standard_causal", True):
            # INFERENCE: frame-block causal mask
            G = self.cfg.tokens_per_frame
            if use_kv_cache:
                abs_end = new_cache.pos_end
                abs_start = abs_end - L_tot
            else:
                abs_start = 0
                abs_end   = L_tot

            pos_k = torch.arange(L_tot, device=x.device)
            frame_id_k = (abs_start + pos_k) // G
            L_past = L_tot - L_new
            q_idx = torch.arange(L_past, L_tot, device=x.device)
            frame_id_q = frame_id_k[q_idx]

            fi_k = frame_id_k.view(1, L_tot)
            fi_q = frame_id_q.view(L_new, 1)
            sdpa_mask = (fi_k > fi_q)
            sdpa_is_causal = False

        elif self.training and getattr(self.cfg, "training_use_frame_block_causal", False):
            # TRAINING: opt-in frame-block causal
            G = self.cfg.tokens_per_frame
            pos = torch.arange(L_new, device=x.device)
            frame_id = (pos // G)
            fi_k = frame_id.view(1, L_new)
            fi_q = frame_id.view(L_new, 1)
            sdpa_mask = (fi_k > fi_q)
            sdpa_is_causal = False
            assert sdpa_mask.shape == (L_new, L_tot), "Unexpected mask shape"
        else:
            # Standard rectangular-causal attention (PyTorch handles L_q != L_k)
            sdpa_is_causal = True
        # ---------------------------------------------------------------------------

        # --- SDPA ---
        q_bHLD = q_new.permute(0, 2, 1, 3)
        k_bHLD = k_gqa.permute(0, 2, 1, 3)
        v_bHLD = v_gqa.permute(0, 2, 1, 3)

        out = F.scaled_dot_product_attention(
            q_bHLD, k_bHLD, v_bHLD,
            attn_mask=sdpa_mask,
            is_causal=sdpa_is_causal,
            dropout_p=(self.dropout if self.training else 0.0),
        )
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, L_new, D)
        out = self.wo(out)
        # ---------------------------------------------------------------------------
        
        return out, new_cache

    
class CS2TransformerEncoderLayer(nn.Module):
    def __init__(self, cfg: CS2Config, rope: RoPEPositionalEncoding):
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CS2GQAAttention(cfg, rope)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * cfg.ffn_mult),
            nn.GELU(),
            nn.Linear(cfg.d_model * cfg.ffn_mult, cfg.d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        temporal_pos_ids: torch.Tensor,
        structural_pos_ids: torch.Tensor,
        kv_cache: Optional[KVCache] = None
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        
        use_checkpoint = self.training and self.cfg.enable_grad_checkpoint and kv_cache is None
        
        if use_checkpoint:
            # --- Gradient Checkpointing Path ---
            
            def attn_block(x_in):
                attn_out, _ = self.attn(self.ln1(x_in), attn_mask, temporal_pos_ids, structural_pos_ids, kv_cache=None)
                return attn_out

            attn_output = checkpoint(attn_block, x, use_reentrant=self.cfg.grad_ckpt_use_reentrant)

            x = x + attn_output
            
            def ff_block(x_in):
                return self.ff(self.ln2(x_in))
                
            ff_output = checkpoint(ff_block, x, use_reentrant=self.cfg.grad_ckpt_use_reentrant)
            x = x + ff_output
            
            return x, None # No cache is updated in this path
            
        else:
            # --- Standard Path ---
            attn_output, updated_cache = self.attn(
                self.ln1(x), attn_mask, temporal_pos_ids, structural_pos_ids, kv_cache=kv_cache
            )
            x = x + attn_output
            x = x + self.ff(self.ln2(x))
            return x, updated_cache


class CS2Backbone(nn.Module):
    def __init__(self, cfg: CS2Config):
        super().__init__()
        self.cfg = cfg
        self.rope = RoPEPositionalEncoding(cfg)
        self.layers = nn.ModuleList([CS2TransformerEncoderLayer(cfg, self.rope) for _ in range(cfg.n_layers)])

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor], kv_cache_list: Optional[List[KVCache]] = None
    ) -> Tuple[torch.Tensor, Optional[List[Optional[KVCache]]]]:
        """
        Threads a list of KV caches through the transformer layers and calculates
        correct positional IDs for autoregressive decoding.
        """

        if kv_cache_list is not None and len(kv_cache_list) != len(self.layers):
            raise RuntimeError(f"Expected {len(self.layers)} caches, got {len(kv_cache_list)}")

        B, L_new, _ = x.shape
        G = self.cfg.tokens_per_frame

        _assert_cache_consistency(kv_cache_list)
        abs_pos_start = _get_abs_pos_start(kv_cache_list)
        pos = torch.arange(abs_pos_start, abs_pos_start + L_new, device=x.device)

        temporal_pos_ids = pos // G
        structural_pos_ids = pos % G

        new_kv_cache_list = []
        use_kv_cache = kv_cache_list is not None
        
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache_list[i] if use_kv_cache else None
            
            x, new_layer_cache = layer(
                x, attn_mask, temporal_pos_ids, structural_pos_ids, kv_cache=layer_cache
            )
            if use_kv_cache:
                new_kv_cache_list.append(new_layer_cache)
            
        return x, (new_kv_cache_list if use_kv_cache else None)

# -----------------------------------------------------------------------------
# 4) Heads
# -----------------------------------------------------------------------------

class PlayerHeads(nn.Module):
    def __init__(self, cfg: CS2Config):
        super().__init__()
        d = cfg.d_model
        
        # --- Standard prediction heads ---
        self.stats = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 3))
        self.mouse = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 2))
        self.keyboard = nn.Linear(d, cfg.keyboard_dim)
        self.eco = nn.Linear(d, cfg.eco_dim)
        self.inventory = nn.Linear(d, cfg.inventory_dim)
        self.active_weapon = nn.Linear(d, cfg.weapon_dim)

        # --- Corrected 3D Heatmap Deconvolutional Head ---
        # New target shape for the position heatmap
        self.pos_shape = (8, 64, 64) # (Z, Y, X)
        
        self.vol_feats = 48  # Latent channels for the deconv stem (tunable)
        
        # 1. New, smaller seed layer. Projects 2048 -> 1024. (~2.1M params)
        # We will create a tiny 2x2x2 seed volume.
        self.pos_seed_projector = nn.Linear(d, self.vol_feats * 2 * 2 * 2)

        # 2. Seed Upsampler block. This grows the seed from 2x2x2 to 8x8x8.
        self.seed_upsampler = nn.Sequential(
            # Input: [B, 128, 2, 2, 2] -> Output: [B, 128, 4, 4, 4]
            nn.ConvTranspose3d(self.vol_feats, self.vol_feats, kernel_size=4, stride=2, padding=1),
            _GN3d(self.vol_feats),
            nn.GELU(),
            
            # Input: [B, 128, 4, 4, 4] -> Output: [B, 128, 8, 8, 8]
            nn.ConvTranspose3d(self.vol_feats, self.vol_feats, kernel_size=4, stride=2, padding=1),
            _GN3d(self.vol_feats),
            nn.GELU(),
        )
        
        class Up2x2(nn.Module):
            def __init__(self, cin, cout):
                super().__init__()
                self.reduce = nn.Conv3d(cin, cout, kernel_size=1, bias=False)
                self.norm   = _GN3d(cout)
                self.act    = nn.GELU()
                self.conv   = nn.Conv3d(cout, cout, kernel_size=3, padding=1, bias=False)
                self.ups    = nn.ConvTranspose3d(cout, cout, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False)
            def forward(self, x):
                x = self.reduce(x)           # lower C before spatial growth
                x = self.ups(x)              # cheap XY upsample without huge temp
                x = self.norm(x); x = self.act(x)
                x = self.conv(x)
                return x


        # Start with fewer channels ASAP and step them down each stage
        self.pos_deconv = nn.Sequential(
            Up2x2(self.vol_feats,       self.vol_feats // 2),   # 128 -> 64
            Up2x2(self.vol_feats // 2,  self.vol_feats // 4),   # 64  -> 32
            Up2x2(self.vol_feats // 4,  self.vol_feats // 8),   # 32  -> 16
        )
        
        # 3. Final Projection: Reduce feature channels to a single logit channel
        self.pos_final_conv = nn.Conv3d(self.vol_feats // 8, 1, kernel_size=1)

    def forward(self, token: torch.Tensor) -> PlayerPredictions:
        # token: [B, d]
        B = token.shape[0]

        # 1. Project token to the initial small seed vector
        tiny_seed_vec = self.pos_seed_projector(token)
        # 2. Reshape into a tiny 3D volume
        tiny_seed_vol = tiny_seed_vec.reshape(B, self.vol_feats, 2, 2, 2)
        # 3. Pass through the learned upsampling network to get the final seed
        seed = self.seed_upsampler(tiny_seed_vol) # Shape: [B, 128, 8, 8, 8]
        # Sanity check to ensure the output shape of our upsampler is correct
        assert seed.shape == (B, self.vol_feats, 8, 8, 8)
        # Pass through the learned upsampling network -> [B, 16, 8, 64, 64]
        upsampled_vol = self.pos_deconv(seed)
        # Project to the final single-channel logit map -> [B, 1, 8, 64, 64]
        heatmap_vol = self.pos_final_conv(upsampled_vol)
        # Remove channel dimension to get final shape [B, 8, 64, 64]
        pos_heatmap_logits = heatmap_vol.squeeze(1)

        # Sanity check to ensure the output shape is always correct
        assert pos_heatmap_logits.shape == (B, *self.pos_shape), \
            f"Shape mismatch! Expected {(B, *self.pos_shape)}, but got {pos_heatmap_logits.shape}"

        return {
            "stats": self.stats(token),
            "pos_heatmap_logits": pos_heatmap_logits,
            "mouse_delta_deg": self.mouse(token),
            "keyboard_logits": self.keyboard(token),
            "eco_logits": self.eco(token),
            "inventory_logits": self.inventory(token),
            "active_weapon_logits": self.active_weapon(token),
        }
class StrategyHead(nn.Module):
    def __init__(self, cfg: CS2Config):
        super().__init__()
        d = cfg.d_model
        self.enemy_shape = (cfg.pos_z, cfg.pos_y, cfg.pos_x)
        
        # --- Corrected 3D Deconvolutional Head for Enemy Positions ---
        self.vol_feats = 48  # Latent channels for the deconv stem (tunable)
        
        self.enemy_seed_projector = nn.Linear(d, self.vol_feats * 2 * 2 * 2)
        self.seed_upsampler = nn.Sequential(
            nn.ConvTranspose3d(self.vol_feats, self.vol_feats, kernel_size=4, stride=2, padding=1),
            _GN3d(self.vol_feats),
            nn.GELU(),
            nn.ConvTranspose3d(self.vol_feats, self.vol_feats, kernel_size=4, stride=2, padding=1),
            _GN3d(self.vol_feats),
            nn.GELU(),
        )
        
        # 2. Deconvolution path: Upsample Y and X dimensions by 8x (2^3) using
        #    anisotropic strides to preserve the Z dimension. This architecture
        #    is modeled directly on the working PlayerHeads implementation.
        class Up2x2(nn.Module):
            def __init__(self, cin, cout):
                super().__init__()
                self.reduce = nn.Conv3d(cin, cout, kernel_size=1, bias=False)
                self.norm   = _GN3d(cout)
                self.act    = nn.GELU()
                self.conv   = nn.Conv3d(cout, cout, kernel_size=3, padding=1, bias=False)
                self.ups    = nn.ConvTranspose3d(cout, cout, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False)
            def forward(self, x):
                x = self.reduce(x)           # lower C before spatial growth
                x = self.ups(x)              # cheap XY upsample without huge temp
                x = self.norm(x); x = self.act(x)
                x = self.conv(x)
                return x

        # Start with fewer channels ASAP and step them down each stage
        self.enemy_deconv = nn.Sequential(
            Up2x2(self.vol_feats,       self.vol_feats // 2),   # 128 -> 64
            Up2x2(self.vol_feats // 2,  self.vol_feats // 4),   # 64  -> 32
            Up2x2(self.vol_feats // 4,  self.vol_feats // 8),   # 32  -> 16
        )
        
        # 3. Final Projection: Reduce feature channels to a single logit channel
        self.enemy_final_conv = nn.Conv3d( self.vol_feats // 8, 1, kernel_size=1)

        # --- Other standard prediction heads ---
        self.round_state = nn.Linear(d, cfg.round_state_dim)
        self.round_number = nn.Linear(d, 1)

    def forward(self, token: torch.Tensor) -> GameStrategyPredictions:
        B = token.shape[0]
        
        tiny_seed_vec = self.enemy_seed_projector(token)
        tiny_seed_vol = tiny_seed_vec.reshape(B, self.vol_feats, 2, 2, 2)
        seed = self.seed_upsampler(tiny_seed_vol)
        # Pass through the learned upsampling network -> [B, 16, 8, 64, 64]
        upsampled_vol = self.enemy_deconv(seed)
        # Project to the final single-channel logit map -> [B, 1, 8, 64, 64]
        heatmap_vol = self.enemy_final_conv(upsampled_vol)
        # Remove channel dimension to get final shape [B, 8, 64, 64]
        enemy_pos_heatmap_logits = heatmap_vol.squeeze(1)
        assert enemy_pos_heatmap_logits.shape == (B, *self.enemy_shape), \
            f"Shape mismatch! Expected {(B, *self.enemy_shape)}, but got {enemy_pos_heatmap_logits.shape}"

        return {
            "enemy_pos_heatmap_logits": enemy_pos_heatmap_logits,
            "round_state_logits": self.round_state(token),
            "round_number": self.round_number(token),
        }
    
# -----------------------------------------------------------------------------
# 5) Top-level model
# -----------------------------------------------------------------------------
class CS2Transformer(nn.Module):
    """Causal autoregressive multi-modal transformer for CS2.

    forward(batch) returns next-step predictions as per Appendix B.
    """
    def __init__(self, cfg: CS2Config, use_dummy_vision: bool = False):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        # Encoders
        if use_dummy_vision:
            # --- MODIFIED: Dummy encoder now has a single-image interface ---
            class DummyVisualEncoder(nn.Module):
                def __init__(self, d_model):
                    super().__init__()
                    self.d_model = d_model
                def forward(self, images): # Takes a single 'images' tensor
                    B, T, P = images.shape[:3]
                    # Instantly return a correctly-shaped random tensor
                    return torch.randn(B, T, P, self.d_model, device=images.device, dtype=torch.bfloat16)
            self.visual_encoder = DummyVisualEncoder(d)
        else:
            self.visual_encoder = DINOv3VisualEncoder(cfg)
        self.audio_encoder = AudioCNN(cfg)
        # Special tokens & embeddings
        self.token_game_strategy = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.token_scratch = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.dead_embedding = nn.Parameter(torch.randn(1, 1, 1, d) * 0.02)
        self.player_slot_embed = nn.Embedding(cfg.num_players, d)
        # Fuser
        self.player_fuser = PlayerTokenFuser(
            cfg,
            player_slot_embed=self.player_slot_embed,
            dead_embedding=self.dead_embedding,
        )
        # Backbone
        self.backbone = CS2Backbone(cfg)
        # Heads
        self.player_head = PlayerHeads(cfg)
        self.strategy_head = StrategyHead(cfg)

        # ---- autocast setup ----
        self.use_amp = (
            getattr(self.cfg,"amp_autocast", True)
            and torch.cuda.is_available()
            and getattr(self.cfg,"compute_dtype", "bf16") != "fp32"
        )
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        self.amp_dtype = dtype_map[getattr(self.cfg,"compute_dtype", "bf16")]

    # --------------------------- utility methods --------------------------- #
    def set_vit_frozen(self, frozen: bool) -> None:
        for p in self.visual_encoder.parameters():
            p.requires_grad = not frozen

    def parameter_groups(self) -> List[Dict[str, object]]:
        vit_params = list(self.visual_encoder.parameters())
        core_modules = [self.audio_encoder, self.player_fuser, self.backbone, self.player_head, self.strategy_head]
        rest = [p for m in core_modules for p in m.parameters()]
        rest += [
            self.token_game_strategy, self.token_scratch, self.dead_embedding,
            *self.player_slot_embed.parameters(),
        ]
        #todo make sure optimizer uses lr_scale
        return [
            {"params": vit_params, "name": "vit", "lr_scale": 0.1},
            {"params": rest, "name": "core", "lr_scale": 1.0},
        ]


    def forward(self, batch: CS2Batch) -> Predictions:
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.amp_dtype) if self.use_amp else nullcontext()
        )
        with autocast_ctx:
            # --- Input processing & tokenization ---
            images = batch["images"]
            mel    = batch["mel_spectrogram"]
            alive  = batch["alive_mask"].bool()
            
            dev = next(self.parameters()).device
            images, mel, alive = (
                images.to(dev, non_blocking=True),
                mel.to(dev, non_blocking=True),
                alive.to(dev, non_blocking=True),
            )
            B, T, P, C, H, W = images.shape
            d = self.cfg.d_model
            Lpf = self.cfg.tokens_per_frame
            param_dtype = next(self.parameters()).dtype
            target_dtype = self.amp_dtype if self.use_amp else param_dtype

            vis = torch.zeros(B, T, P, d, device=dev, dtype=target_dtype)
            if torch.count_nonzero(alive):
                alive_images = images[alive]
                num_alive = alive_images.shape[0]
                packed_vis_out = self.visual_encoder(alive_images.reshape(num_alive, 1, 1, C, H, W))
                packed = packed_vis_out.squeeze(1).squeeze(1)
                if packed.dtype != target_dtype:
                    packed = packed.to(target_dtype)
                vis[alive] = packed

            aud = self.audio_encoder(mel)
            if aud.dtype != target_dtype:
                aud = aud.to(target_dtype)
            player_tokens = self.player_fuser(vis, aud, alive)
            
            tok_gs = self.token_game_strategy.unsqueeze(2).expand(B, T, 1, d).to(target_dtype)
            tok_sc = self.token_scratch .unsqueeze(2).expand(B, T, 1, d).to(target_dtype)
            frame_tokens = torch.cat([player_tokens, tok_gs, tok_sc], dim=2)
            seq = frame_tokens.reshape(B, T * Lpf, d)

            # --- Backbone Pass ---
            h = None
            if self.training and self.cfg.enable_cached_training:
                # --- Cached Training (TBPTT) Path ---
                chunk_T = self.cfg.cached_chunk_T
                # When carrying a running KV cache, do NOT re-feed warmup tokens.
                kv_cache_list = [None] * self.cfg.n_layers
                outputs = []

                for t_start in range(0, T, chunk_T):
                    t_end = min(t_start + chunk_T, T)
                    token_slice_start = t_start * Lpf
                    token_slice_end = t_end * Lpf
                    seq_slice = seq[:, token_slice_start:token_slice_end]

                    if kv_cache_list[0] is not None and self.cfg.cached_detach:
                        for cache in kv_cache_list:
                            cache.key = cache.key.detach()
                            cache.value = cache.value.detach()

                    h_slice, updated_kv_cache = self.backbone(seq_slice, attn_mask=None, kv_cache_list=kv_cache_list)
                    kv_cache_list = updated_kv_cache
                    outputs.append(h_slice)
                
                h = torch.cat(outputs, dim=1)
            else:
                # --- Standard Single-Shot Path ---
                h, _ = self.backbone(seq, attn_mask=None, kv_cache_list=None)

            # --- Prediction Heads ---
            h_frames = h.reshape(B, T, Lpf, d)
            num_players = self.cfg.num_players
            
            player_tok_all_frames = h_frames[:, :, :num_players, :]
            strat_tok_all_frames  = h_frames[:, :, num_players, :]

            player_preds: List[PlayerPredictions] = []
            for i in range(num_players):
                p_tok_flat = player_tok_all_frames[:, :, i, :].reshape(B * T, d)
                # Only checkpoint during training, and when it's enabled in the config
                if self.training and self.cfg.enable_grad_checkpoint:
                    preds_flat = checkpoint(self.player_head, p_tok_flat, use_reentrant=self.cfg.grad_ckpt_use_reentrant)
                else:
                    preds_flat = self.player_head(p_tok_flat)
                preds_seq = {k: v.reshape(B, T, *v.shape[1:]) for k, v in preds_flat.items()}
                player_preds.append(preds_seq)

            strat_preds_flat = self.strategy_head(strat_tok_all_frames.reshape(B * T, d))
            strategy_preds = {k: v.reshape(B, T, *v.shape[1:]) for k, v in strat_preds_flat.items()}

            return {"player": player_preds, "game_strategy": strategy_preds}

    def autoregressive_step(
        self,
        single_frame_batch: CS2Batch,
        past_kv_cache: Optional[List[KVCache]] = None
    ) -> Tuple[Predictions, List[Optional[KVCache]]]:
        """
        Processes a SINGLE frame of data for efficient autoregressive inference.
        """
        assert not self.training, "autoregressive_step should be called in eval mode"
        # Initialize caches on first call so the backbone builds them.
        if past_kv_cache is None:
            past_kv_cache = [None] * self.cfg.n_layers
        # --- MODIFIED: Check the new 'images' key ---
        assert single_frame_batch["images"].shape[1] == 1, "Input for autoregressive_step must have T=1"

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.amp_dtype) if self.use_amp else nullcontext()
        )
        with torch.inference_mode():
            with autocast_ctx:
                # --- MODIFIED: Use the single 'images' tensor ---
                images = single_frame_batch["images"]
                mel = single_frame_batch["mel_spectrogram"]
                alive = single_frame_batch["alive_mask"].bool()

                
                dev = next(self.parameters()).device
                images = images.to(dev, non_blocking=True)
                mel    = mel.to(dev,    non_blocking=True)
                alive  = alive.to(dev,  non_blocking=True)

                B, T, P = images.shape[:3]  # T==1
                C, H, W = images.shape[-3:]
                d = self.cfg.d_model

                param_dtype = next(self.parameters()).dtype
                target_dtype = self.amp_dtype if self.use_amp else param_dtype

                vis = torch.zeros(B, T, P, d, device=dev, dtype=target_dtype)
                if torch.count_nonzero(alive):
                    alive_images = images[alive]  # [num_alive, 3, H, W] (non-contiguous)
                    num_alive = alive_images.shape[0]

                    # Re-wrap to the encoder’s expected 6D input
                    packed_vis_out = self.visual_encoder(
                        alive_images.reshape(num_alive, 1, 1, C, H, W)
                    )  # [num_alive, 1, 1, d]

                    vis[alive] = packed_vis_out.squeeze(1).squeeze(1).to(target_dtype)  # [num_alive, d]

                # Audio: cheap enough to run for all
                aud = self.audio_encoder(mel).to(target_dtype)  # [B, 1, P, d]

                # Fuse
                player_tokens = self.player_fuser(vis, aud, alive)

                tok_gs = self.token_game_strategy.unsqueeze(2).expand(B, T, 1, d).to(target_dtype)
                tok_sc = self.token_scratch .unsqueeze(2).expand(B, T, 1, d).to(target_dtype)
                frame_tokens = torch.cat([player_tokens, tok_gs, tok_sc], dim=2) # Concat on dim 2

                seq = frame_tokens.reshape(B, self.cfg.tokens_per_frame, d)

                # --- Backbone call with cache ---
                h, updated_kv_cache = self.backbone(seq, attn_mask=None, kv_cache_list=past_kv_cache)
                if updated_kv_cache is None:
                    updated_kv_cache = [None] * self.cfg.n_layers

                # --- Prediction Heads ---
                last = h
                num_players = self.cfg.num_players
                player_tok = last[:, :num_players, :]
                strat_tok  = last[:, num_players, :]

                player_preds: List[PlayerPredictions] = [self.player_head(player_tok[:, i, :]) for i in range(num_players)]
                strategy_preds = self.strategy_head(strat_tok)

                predictions = {"player": player_preds, "game_strategy": strategy_preds}

                return predictions, updated_kv_cache
# If this file is imported, users can create and compile as follows:
#   model = CS2Transformer(CS2Config())
#   model = torch.compile(model)  # outside this module


# -----------------------------------------------------------------------------
# 6) Test harness and benchmark
# -----------------------------------------------------------------------------
def main():
    """A comprehensive testing and benchmarking harness for the CS2Transformer."""
    parser = argparse.ArgumentParser(description="Test and benchmark the CS2Transformer model.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for the test.")
    parser.add_argument("--context-frames", type=int, default=1, help="Sequence length (time dimension) for the test.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on.")
    parser.add_argument("--dtype", type=str, choices=["fp32", "fp16", "bf16"], default="bf16", help="Compute data type.")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile() for the model.")
    parser.add_argument("--tensorrt", action="store_true", help="Use torch_tensorrt backend when --compile is enabled.")
    parser.add_argument("--trt-debug", action="store_true",
                        help="Enable verbose TensorRT logging and debug during torch_tensorrt compilation.")
    parser.add_argument("--freeze-vit", action="store_true", help="Freeze the ViT encoder weights.")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark instead of just a shape test.")
    parser.add_argument("--autoregressive", type=int, metavar="N_FRAMES", help="Run autoregressive KV-cached benchmark for N frames.")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Number of warmup steps for benchmark.")
    parser.add_argument("--bench-steps", type=int, default=20, help="Number of benchmark steps.")
    parser.add_argument("--dummy-vit", action="store_true", help="Disable ViT to bench main model.")
    parser.add_argument("--num-layers", type=int, default=24, help="Set number of main tf layers.")
    args = parser.parse_args()

    if args.compile and args.tensorrt and args.dtype == "bf16":
        print("[WARNING] bfloat16 not fully supported by TensorRT; forcing fp16 for TRT backend.")
        args.dtype = "fp16"


    # --- Setup Device and DType ---
    device = torch.device(args.device)
    if args.dtype == "bf16" and not (device.type == "cuda" and torch.cuda.is_bf16_supported()):
        print(f"[WARNING] bf16 not supported on {args.device}. Falling back to fp32.")
        args.dtype = "fp32"
    if args.dtype == "fp16" and device.type == "cpu":
        print(f"[WARNING] fp16 not supported on CPU. Falling back to fp32.")
        args.dtype = "fp32"

    print("-" * 60)
    print("Initializing Test Run")
    print(f"  - Device: {device}")
    print(f"  - DType: {args.dtype}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Context Frames: {args.context_frames}")
    if args.compile:
        backend = "torch_tensorrt" if args.tensorrt else "default torch.compile"
        print(f"  - torch.compile(): True (backend: {backend})")
    else:
        print(f"  - torch.compile(): False")
    print(f"  - Autoregressive benchmark: {'Yes (' + str(args.autoregressive) + ' frames)' if args.autoregressive else 'No'}")
    print(f"  - Dummy Vit: {args.dummy_vit}")
    print("-" * 60)

    # --- Instantiate Model ---
    cfg = CS2Config(
        compute_dtype=args.dtype,
        context_frames=args.context_frames,
        n_layers=args.num_layers,
    )
    #print_dataclass(cfg)
    model = CS2Transformer(cfg, args.dummy_vit).to(device).eval()

    if args.freeze_vit:
        model.set_vit_frozen(True)

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    precision = dtype_map[args.dtype]

    if args.compile:
        if args.tensorrt:
            # --- Compile with torch_tensorrt backend ---
            try:
                import torch_tensorrt
                # Enable verbose TensorRT logging if requested
                if args.trt_debug:
                    try:
                        from torch_tensorrt import logging as trt_logging
                        trt_logging.set_is_colored_output_on(True)
                        trt_logging.set_reportable_log_level(trt_logging.Level.Debug)
                        print("[TRT] Verbose logging enabled (Level=DEBUG).")
                    except Exception as _e:
                        print(f"[TRT] Could not configure torch_tensorrt logging: {_e}")
            except ImportError:
                print("\n[ERROR] torch_tensorrt is not installed.")
                print("Please install it to use the --tensorrt flag, e.g., `pip install torch-tensorrt`")
                exit(1)

            print("[INFO] Compiling model with torch_tensorrt backend... (this may take a long time)")

            trt_options = {
                # Set the precision for the TensorRT engine
                "enabled_precisions": {precision},
                # Workspace for fastest kernels (1GB default)
                "workspace_size": 5 << 30,
                # Avoid issues with unsupported dtypes
                "truncate_long_and_double": True,
            }
            if args.trt_debug:
                trt_options["debug"] = True

            model = torch.compile(
                model,
                backend="torch_tensorrt",
                options=trt_options,
            )
        else:
            print("[INFO] Compiling model with default torch.compile() backend... (this may take a moment)")
            model = torch.compile(model)

        print("[INFO] Compilation complete.")

    B, T, P = args.batch_size, cfg.context_frames, cfg.num_players
    # Using 480x640 as the representative input size
    dummy_batch: CS2Batch = {
        "images": torch.randn(B, T, P, 3, 480, 640, device=device, dtype=precision),
        "mel_spectrogram": torch.randn(B, T, P, 2, cfg.mel_bins, cfg.mel_t, device=device, dtype=precision),
        "alive_mask": torch.ones(B, T, P, device=device, dtype=torch.float32),
    }

    # --- Run Shape Test ---
    print("\n[PHASE 1] Running Shape Test & Initial Compilation...")
    # For shape test, we only need to predict the last frame's outputs
    # so we can slice the predictions.
    with torch.no_grad():
        full_preds = model(dummy_batch)
        
    predictions: Predictions = {
        "player": [{k: v[:, -1] for k, v in p.items()} for p in full_preds["player"]],
        "game_strategy": {k: v[:, -1] for k, v in full_preds["game_strategy"].items()}
    }

    # Player predictions
    assert isinstance(predictions["player"], list) and len(predictions["player"]) == P, "Player predictions must be a list of length num_players"
    for i, p_pred in enumerate(predictions["player"]):
        assert p_pred["stats"].shape == (B, 3), f"Player {i} stats shape is wrong"
        assert p_pred["pos_heatmap_logits"].shape == (B, cfg.pos_z, cfg.pos_y, cfg.pos_x), f"Player {i} pos_heatmap shape is wrong"
        assert p_pred["mouse_delta_deg"].shape == (B, 2), f"Player {i} mouse_delta shape is wrong"
        assert p_pred["keyboard_logits"].shape == (B, cfg.keyboard_dim), f"Player {i} keyboard_logits shape is wrong"
        assert p_pred["eco_logits"].shape == (B, cfg.eco_dim), f"Player {i} eco_logits shape is wrong"
        assert p_pred["inventory_logits"].shape == (B, cfg.inventory_dim), f"Player {i} inventory_logits shape is wrong"
        assert p_pred["active_weapon_logits"].shape == (B, cfg.weapon_dim), f"Player {i} active_weapon_logits shape is wrong"

    # Strategy predictions
    strat_pred = predictions["game_strategy"]
    assert strat_pred["enemy_pos_heatmap_logits"].shape == (B, cfg.pos_z, cfg.pos_y, cfg.pos_x), "Enemy pos_heatmap shape is wrong"
    assert strat_pred["round_state_logits"].shape == (B, cfg.round_state_dim), "Round state_logits shape is wrong"
    assert strat_pred["round_number"].shape == (B, 1), "Round number shape is wrong"
    print("✅ Shape Test Passed!")

    # --- Run Standard Benchmark ---
    if args.benchmark:
        print("\n[PHASE 2] Running Standard Benchmark...")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            start_mem = torch.cuda.max_memory_allocated(device)

        with torch.no_grad():
            # Warmup
            print(f"  - Warming up for {args.warmup_steps} iterations...")
            for _ in range(args.warmup_steps):
                _ = model(dummy_batch)

            # Benchmark
            print(f"  - Benchmarking for {args.bench_steps} iterations...")
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            for _ in range(args.bench_steps):
                _ = model(dummy_batch)

            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()

        total_time = end_time - start_time
        avg_time_ms = (total_time / args.bench_steps) * 1000
        throughput = args.bench_steps / total_time

        print("\n" + "=" * 25 + " BENCHMARK RESULTS " + "=" * 25)
        print(f"  - Average time per iteration: {avg_time_ms:.2f} ms")
        print(f"  - Throughput (iterations/sec): {throughput:.2f}")
        if device.type == "cuda":
            end_mem = torch.cuda.max_memory_allocated(device)
            peak_mem_gb = (end_mem - start_mem) / (1024 ** 3)
            print(f"  - Peak GPU Memory Allocated: {peak_mem_gb:.2f} GB")
        print("=" * 69)

    # --- Run Autoregressive Benchmark ---
    if args.autoregressive:
        print("\n[PHASE 3] Running Autoregressive (KV-Cached) Benchmark...")
        num_frames = args.autoregressive
        model.eval()

        single_frame_batch: CS2Batch = {
            "images": torch.randn(B, 1, P, 3, 480, 640, device=device, dtype=precision),
            "mel_spectrogram": torch.randn(B, 1, P, 2, cfg.mel_bins, cfg.mel_t, device=device, dtype=precision),
            "alive_mask": torch.randint(0, 2, (B, 1, P), device=device, dtype=torch.bool),
        }

        print(f"  - Warming up for {args.warmup_steps} autoregressive steps...")
        with torch.no_grad():
            try:
                kv_cache = [None] * cfg.n_layers
                for _ in range(args.warmup_steps):
                    _, kv_cache = model.autoregressive_step(single_frame_batch, kv_cache)
            except Exception as e:
                print("\n[ERROR] Autoregressive warmup failed. This can happen with compiled stateful models.")
                print("The torch_tensorrt backend may not have been able to handle the dynamic KV cache.")
                print(f"Error details: {e}")
                return

        print(f"  - Benchmarking for {num_frames} autoregressive steps...")
        kv_cache = [None] * cfg.n_layers
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            start_mem = torch.cuda.max_memory_allocated(device)
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_frames):
                _, kv_cache = model.autoregressive_step(single_frame_batch, kv_cache)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        total_time = end_time - start_time
        avg_time_ms = (total_time / num_frames) * 1000
        throughput_fps = num_frames / total_time

        print("\n" + "=" * 20 + " AUTOREGRESSIVE BENCHMARK RESULTS " + "=" * 20)
        print(f"  - Total frames generated: {num_frames}")
        print(f"  - Average time per frame: {avg_time_ms:.2f} ms")
        print(f"  - Throughput (Frames/Sec): {throughput_fps:.2f} FPS")
        if device.type == "cuda":
            end_mem = torch.cuda.max_memory_allocated(device)
            peak_mem_gb = end_mem / (1024 ** 3)
            print(f"  - Peak GPU Memory Allocated (incl. cache): {peak_mem_gb:.2f} GB")
        print("=" * 69)

if __name__ == "__main__":
    # To run standard benchmark: python model.py --benchmark --compile
    # To run autoregressive benchmark: python model.py --autoregressive 100 --compile
    main()

__all__ = [
    "CS2Config",
    "PlayerPredictions",
    "GameStrategyPredictions",
    "Predictions",
    "CS2Transformer",
]