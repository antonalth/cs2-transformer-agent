"""
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

- Causal Paradigm: The model operates under a strict causal constraint, meaning its predictions
  for any given moment can only be conditioned on past and present information. This is enforced
  by a causal attention mask within the transformer backbone, making the model suitable for
  real-time, sequential generation.

- Generative Feedback Loop (Inference): During inference, the model forms a feedback loop with a
  game simulator.
  1. The model predicts actions for frame `t+1`.
  2. These actions are used to update the game simulator.
  3. The simulator generates the new visual and audio data for frame `t+1`.
  4. This new frame is encoded, appended to the history, and the process repeats.

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
  - Input: Two 384x384 pixel tensors per player:
    1. Foveal View: A non-scaled, high-resolution crop around the player's crosshair.
    2. Peripheral View: The full 640x480 game screen, scaled down (with letterboxing).
  - Encoder: A SINGLE, SHARED-WEIGHT ViT-Large model (`google/vit-large-patch16-384`) processes
    both views independently.
  - Fusion: The [CLS] tokens from each view (each `[1, 1024]`) are extracted and concatenated
    to form a `[1, 2048]` intermediate representation.
  - Projection: A final linear layer projects this concatenated tensor to the model's native
    `[1, 2048]` dimension, creating the final visual embedding.

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
  - Player Slot Embedding: To maintain player identity across the sequence, a unique, learned
    `player_slot_embedding` is ADDED to each of the 5 player tokens.
  - DEAD Token Policy: If a player is dead, their entire visual/audio input is ignored. Instead,
    their token is replaced by a learned `[DEAD]` embedding, to which their unique
    `player_slot_embedding` is still added.

[C] TRANSFORMER LAYERS:
  - A stack of 16-24 custom `CS2TransformerEncoderLayer` modules.
  - Each layer contains:
    1. A Causal Self-Attention block (optimized with GQA and FlashAttention-2).
    2. A Feed-Forward Network (FFN) with a GELU activation.
  - Pre-Layer Normalization is used throughout for training stability.

====================================================================================================
  III. STAGE 3: PREDICTION HEADS
====================================================================================================
The output tokens from the FINAL TIME STEP of the transformer backbone are used to make
predictions for the next frame, as specified in Appendix B.

[A] PLAYER PREDICTION HEADS (x5):
  Each of the 5 player output tokens is fed into a set of dedicated heads to predict:
  - Stats (Regression): Health, Armor, Money (via an MLP).
  - Position (Heatmap): A 3D position heatmap `[Z, Y, X]` (via a Deconvolutional Network).
  - Mouse (Regression): `[delta_x, delta_y]` mouse movement (via an MLP).
  - Actions (Classification): Logits for Keyboard, Eco, Inventory, and Active Weapon states
    (via MLPs).

[B] GAME STRATEGY HEAD (x1):
  The `[GAME_STRATEGY]` output token is used to predict:
  - Enemy Positions (Heatmap): A 3D heatmap of predicted enemy locations.
  - Game Phase (Classification): Logits for the round state (e.g., "Bomb Planted").
  - Round Number (Regression): A single scalar value.

----------------------------------------------------------------------------------------------------
3. HIGH-PERFORMANCE OPTIMIZATIONS
----------------------------------------------------------------------------------------------------
The model is designed for extreme performance and scalability to very long context windows.

- `torch.compile()`: The entire `CS2Transformer` module is intended to be wrapped in
  `torch.compile(model)` for JIT compilation, kernel fusion, and significant speedups.

- Grouped-Query Attention (GQA): The core attention mechanism uses fewer Key/Value heads than
  Query heads (e.g., 8 KV heads for 32 Q heads). This dramatically reduces the size of the KV
  cache, which is the primary bottleneck for fast autoregressive inference.

- FlashAttention-2: Where available, this custom kernel is used to compute attention. It
  avoids the explicit creation of the `N x N` attention matrix, drastically reducing memory
  usage and increasing computational speed.

- Rotary Positional Embeddings (RoPE): Positional information is injected via RoPE, which is
  applied directly to queries and keys. It provides excellent performance on long sequences
  and strong extrapolation capabilities.

- Mixed Precision & Quantization:
  - Training: Natively supports `BF16` for faster computation and reduced memory footprint.
  - Inference/Fine-Tuning: Compatible with advanced quantization (e.g., QLoRA with 4-bit
    NormalFloat) and FP8 inference on supported hardware for maximum throughput.

----------------------------------------------------------------------------------------------------
4. TRAINING & FINE-TUNING STRATEGY
----------------------------------------------------------------------------------------------------
- Objective: The model is trained on a Next-Frame Prediction task. Given a sequence of `t`
  ground-truth frames, it is optimized via a COMPOSITE LOSS function to predict the
  ground-truth data for frame `t+1`.

- ViT Fine-Tuning (Two-Stage):
  1. Freeze: Initially, the weights of the pre-trained ViT-Large encoder are frozen
     (`requires_grad=False`). The rest of the model is trained to learn how to interpret
     its powerful, off-the-shelf features.
  2. Finetune: After the main model has stabilized, the ViT is unfrozen. The entire model is
     then trained end-to-end using DIFFERENTIAL LEARNING RATES, where the ViT uses a learning
     rate 10-100x smaller than the rest of the model to prevent catastrophic forgetting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, TypedDict, Dict, Optional, Tuple, Literal, Any
import time, argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

# -----------------------------------------------------------------------------
# 0) Public types (Appendix B)
# -----------------------------------------------------------------------------

class PlayerPredictions(TypedDict):
    stats: torch.Tensor                      # [B, 3]
    pos_heatmap_logits: torch.Tensor         # [B, 8, 64, 64]
    mouse_delta_deg: torch.Tensor            # [B, 2]
    keyboard_logits: torch.Tensor            # [B, 31]
    eco_logits: torch.Tensor                 # [B, 384]
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
    foveal_images: torch.Tensor            # [B, T, 5, 3, 384, 384]
    peripheral_images: torch.Tensor        # [B, T, 5, 3, 384, 384]
    mel_spectrogram: torch.Tensor          # [B, T, 5, 1, 128, ~6]
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
    context_frames: int = 128

    max_cache_len_tokens: int = None


    # --- Masking Behavior ---
    # use_fused_causal: For TRAINING. Uses FlashAttention's internal causal mask. Fastest.
    # enables FA2 for training, 
    use_fused_causal: bool = True #enables FA2 for Training

    # use_frame_block_causal_mask: For INFERENCE/TRAINING. A custom mask that only allows
    # attending to tokens in the current frame or any token in past frames, training fallback to SDPA
    use_frame_block_causal_mask: bool = False
    
    # NEW: This flag controls inference masking. If True, inference uses a standard
    # (token-by-token) causal mask, ensuring consistency with models trained using
    # `use_fused_causal=True`. Set to False to use the frame-block mask during inference.
    inference_use_standard_causal: bool = True

    # Vision
    vit_name_hf: str = "google/vit-large-patch16-384"
    vit_name_timm: str = "vit-large-patch16-384" #preferred if available
    vit_out_dim: int = 1024

    # Audio
    mel_bins: int = 128
    mel_t: int = 6
    audio_cnn_channels: Tuple[int, int, int] = (32, 64, 128)

    # Heads
    keyboard_dim: int = 31
    eco_dim: int = 384
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

@dataclass
class KVCache:
    key: torch.Tensor    # [B, L, H_kv, Hd]
    value: torch.Tensor  # [B, L, H_kv, Hd]
    pos_end: int         # absolute position *after* last cached token

def _get_abs_pos_start(kv_cache_list: Optional[List[Optional[KVCache]]]) -> int:
    if not kv_cache_list:
        return 0
    for c in kv_cache_list:
        if c is not None:
            return int(c.pos_end)
    return 0

def _update_cache(old: Optional[KVCache],
                  new_k: torch.Tensor,  # rotated for ABS positions
                  new_v: torch.Tensor,  # rotated for ABS positions
                  max_cache_len_tokens: Optional[int]) -> KVCache:
    if old is None:
        cat_k, cat_v = new_k, new_v
        pos_end = new_k.shape[1]  # starts at 0, ends at L_new
    else:
        cat_k = torch.cat([old.key, new_k], dim=1)   # [B, L_old+L_new, H_kv, Hd]
        cat_v = torch.cat([old.value, new_v], dim=1)
        pos_end = old.pos_end + new_k.shape[1]

    if max_cache_len_tokens is not None and cat_k.shape[1] > max_cache_len_tokens:
        cat_k = cat_k[:, -max_cache_len_tokens:]
        cat_v = cat_v[:, -max_cache_len_tokens:]
        # NOTE: pos_end stays absolute (no change)

    return KVCache(key=cat_k.detach(), value=cat_v.detach(), pos_end=pos_end)

def _assert_cache_consistency(kv_cache_list: Optional[List[Optional[KVCache]]]) -> None:
    if not kv_cache_list:
        return
    lens = [int(c.key.shape[1]) for c in kv_cache_list if c is not None]
    pos_ends = [int(c.pos_end) for c in kv_cache_list if c is not None]
    if lens:
        L0 = lens[0]
        if not all(L == L0 for L in lens):
            raise RuntimeError(f"Inconsistent cache window lengths across layers: {lens}")
    if pos_ends:
        P0 = pos_ends[0]
        if not all(P == P0 for P in pos_ends):
            raise RuntimeError(f"Inconsistent absolute positions across layers: {pos_ends}")


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

class ViTVisualEncoder(nn.Module):
    """Shared-weight ViT-Large encoder for both views.

    Inputs per player-frame:
      - foveal:     [B, T, P, 3, 384, 384]
      - peripheral: [B, T, P, 3, 384, 384]
    Returns per player-frame visual embedding: [B, T, P, d_model]

    Notes
    -----
    * Prefers timm's `vit_large_patch16_384` (headless) for speed.
    * Falls back to Hugging Face `google/vit-large-patch16-384` if timm
      is unavailable.
    * This module expects images in **ImageNet normalization**
      (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]).
    """
    def __init__(self, cfg: CS2Config):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model

        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3, 1, 1))

        # Optional backends (unchanged)
        try:
            import timm  # type: ignore
            self._has_timm = True
        except Exception:
            self._has_timm = False
            timm = None  # type: ignore
        try:
            from transformers import ViTModel  # type: ignore
            self._has_hf = True
        except Exception:
            self._has_hf = False
            ViTModel = None  # type: ignore

        self.backend = None
        if self._has_timm:
            self.vit = timm.create_model(getattr(self.cfg,"vit_name_timm"), pretrained=True, num_classes=0)
            self.backend = "timm"
            self.vit_out_dim = 1024
        elif self._has_hf:
            self.vit = ViTModel.from_pretrained(getattr(self.cfg,"vit_name_hf"))
            self.backend = "hf"
            self.vit_out_dim = 1024
        else:
            raise ImportError("Neither timm nor transformers is available for ViTVisualEncoder.")

        self.proj = nn.Linear(self.vit_out_dim * 2, cfg.d_model)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Accept uint8 (0..255) or float already in [0,1]
        if not x.is_floating_point():
            x = x.float() / 255.0
        # If callers accidentally pass floats in 0..255, you can add an opt-in flag to rescale.
        return (x - self.img_mean) / self.img_std

    def _vit_requires_grad(self) -> bool:
        # True if any ViT parameter requires grad
        return any(p.requires_grad for p in self.vit.parameters())

    def _forward_one_view_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Backend-agnostic single-view forward. Returns CLS [N, 1024]."""
        if self.backend == "timm":
            return self.vit(x)  # [N, 1024] with num_classes=0
        else:
            out = self.vit(pixel_values=x)
            return out.last_hidden_state[:, 0, :]  # [N, 1024]

    def _forward_one_reshape(self, x: torch.Tensor) -> torch.Tensor:
        # Freeze-aware grad control
        need_grad = torch.is_grad_enabled() and self._vit_requires_grad()
        if need_grad:
            return self._forward_one_view_impl(x)
        else:
            with torch.no_grad():
                return self._forward_one_view_impl(x)

    def forward(self, foveal: torch.Tensor, peripheral: torch.Tensor) -> torch.Tensor:
        B, T, P = foveal.shape[:3]
        N = B * T * P

        # Normalize to ImageNet stats (broadcasted)
        foveal = self._normalize(foveal)
        peripheral = self._normalize(peripheral)

        # Flatten player-frames
        fov = foveal.reshape(N, *foveal.shape[-3:])          # [N, 3, 384, 384]
        per = peripheral.reshape(N, *peripheral.shape[-3:])  # [N, 3, 384, 384]

        # Shared ViT; take CLS token per view
        cls_a = self._forward_one_reshape(fov)   # [N, 1024]
        cls_b = self._forward_one_reshape(per)   # [N, 1024]

        vis = self.proj(torch.cat([cls_a, cls_b], dim=-1)).reshape(B, T, P, self.d_model)
        return vis


class AudioCNN(nn.Module):
    """Small 2D-CNN over Mel-spectrograms → [B, T, P, d_model].

    Robust to variable time dimension via AdaptiveAvgPool2d.
    Input mel: [B, T, P, 1, mel_bins(=128), mel_t(~6..N)]
    """
    def __init__(self, cfg: CS2Config):
        super().__init__()
        c1, c2, c3 = cfg.audio_cnn_channels
        self.conv1 = nn.Conv2d(1, c1, 5, padding=2)
        self.gn1 = _GN2d(c1)
        self.conv2 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.gn2 = _GN2d(c2)
        self.conv3 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.gn3 = _GN2d(c3)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Linear(c3 * 4 * 4, cfg.d_model)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: [B, T, P, 1, 128, ~6] → pack to [B*T*P, 1, 128, ~6]
        B, T, P = mel.shape[:3]
        x = mel.reshape(B * T * P, 1, mel.shape[-2], mel.shape[-1])
        x = F.gelu(self.gn1(self.conv1(x)))
        x = F.gelu(self.gn2(self.conv2(x)))
        x = F.gelu(self.gn3(self.conv3(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)  # [N, d_model]
        x = x.reshape(B, T, P, -1)
        return x
    
class PlayerTokenFuser(nn.Module):
    """Fuse visual+audio via elementwise add, add slot identity, then LayerNorm.

    DEAD handling policy: if dead, replace fused token with
      dead_embedding + player_slot_embedding before final norm.
    """
    def __init__(self, cfg: CS2Config, player_slot_embed: nn.Embedding, dead_embedding: nn.Parameter):
        super().__init__()
        self.cfg = cfg
        # Apply LayerNorm as the final step
        self.norm = nn.LayerNorm(cfg.d_model)
        self.slot_embed = player_slot_embed
        self.dead_embedding = dead_embedding

    def forward(
        self,
        vis: torch.Tensor,   # [B, T, P, d]
        aud: torch.Tensor,   # [B, T, P, d]
        alive_mask: torch.Tensor,  # [B, T, P] bool
    ) -> torch.Tensor:
        """
        Fuse visual+audio, apply the DEAD policy, add player identity, and finally normalize.
        """
        B, T, P, d = vis.shape
        assert P == self.cfg.num_players, f"P={P} must equal cfg.num_players={self.cfg.num_players}"

        # 1. Fuse the sensor data for the "alive" case.
        fused_alive_token = vis + aud

        # 2. Prepare the mask for broadcasting to select the base token.
        dead_mask = ~alive_mask.bool().unsqueeze(-1) # Shape: [B, T, P, 1]

        # 3. Select the base token (fused sensors or the DEAD embedding)
        base_token = torch.where(
            dead_mask,
            self.dead_embedding,
            fused_alive_token,
        )

        # 4. Add the unique player slot identity. This is now done BEFORE normalization.
        slot_ids = torch.arange(P, device=vis.device).reshape(1, 1, P)
        slots = self.slot_embed(slot_ids)
        
        unnormalized_token = base_token + slots

        # 5. Apply LayerNorm as the FINAL step to the fully assembled token.
        final_token = self.norm(unnormalized_token)

        return final_token
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
        self.register_buffer("scale", torch.tensor(1.0), persistent=False)
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
        cos = cos.reshape(1, L, 1, half_dim).to(x.dtype)
        sin = sin.reshape(1, L, 1, half_dim).to(x.dtype)

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

        assert not (cfg.use_fused_causal and cfg.use_frame_block_causal_mask), \
            "use_fused_causal and use_frame_block_causal_mask are mutually exclusive"

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

        # Try to import FlashAttention-2
        self._fa2 = None
        try:
            from flash_attn.flash_attn_interface import flash_attn_func as _fa
            self._fa2 = _fa
        except Exception:
            try:
                from flash_attn import flash_attn_func as _fa  # older entry point
                self._fa2 = _fa
            except Exception:
                self._fa2 = None

    def _use_fa2(self, x: torch.Tensor, q: Optional[torch.Tensor] = None) -> bool:
        """Conservative check before trying FA2; shape/dtype/device guards."""
        if self._fa2 is None:
            return False
        if not x.is_cuda:
            return False
        if x.dtype not in (torch.float16, torch.bfloat16):
            return False
        if q is not None:
            # expect q shaped [B, L, H, Hd]
            if q.dim() != 4 or q.shape[-1] <= 0 or q.shape[-2] <= 0:
                return False
        return True

    def _prep_attn_mask(self, attn_mask: Optional[torch.Tensor], L: int, B: int, device: torch.device) -> Optional[torch.Tensor]:
            """
            Normalize attention masks for PyTorch SDPA:
            - Boolean mask: True = DISALLOW (masked-out), False = allow.
            - Float mask: additive; large negative (e.g., <= -1e4) = disallow, 0 = allow.
            Supports shapes: [L,L], [B,L,L], [B,1,L,L], [1,1,L,L].
            Returns bool mask broadcastable to [B, H, L, L] on the correct device.
            """
            if attn_mask is None:
                return None
            m = attn_mask
            if m.dtype != torch.bool:
                m = m <= -1e4  # treat very negative as masked-out
            
            # FIX: Ensure mask is on the same device as the input tensors
            m = m.to(device)

            # Shape normalize
            if m.dim() == 2:          # [L, L] -> [1,1,L,L]
                m = m.reshape(1, 1, L, L)
            elif m.dim() == 3:        # [B, L, L] -> [B,1,L,L]
                m = m.unsqueeze(1)
            elif m.dim() == 4:
                # assume already [B or 1, H or 1, L, L]
                pass
            else:
                raise ValueError(f"Unsupported attn_mask dims: {m.shape}")
            # ensure batch dimension is either 1 or B
            if m.shape[0] not in (1, B):
                raise ValueError(f"attn_mask batch dim {m.shape[0]} incompatible with B={B}")
            return m.to(torch.bool)

    def _fa2_call(self, q, k, v, dropout_p: float, causal: bool):
        """Handle FA2 signature drift gracefully; return tensor or raise to trigger fallback."""
        try:
            return self._fa2(
                q, k, v,
                dropout_p=dropout_p,
                softmax_scale=None,
                causal=causal,
                return_attn_probs=False,
            )
        except TypeError:
            # older builds use `dropout` kwarg and may lack return_attn_probs
            return self._fa2(
                q, k, v,
                dropout=dropout_p,
                softmax_scale=None,
                causal=causal,
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        temporal_pos_ids: torch.Tensor,
        structural_pos_ids: torch.Tensor,
        kv_cache: Optional[KVCache] = None
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        Dual-path attention optimized for training and autoregressive inference.

        - Training Path: For standard training runs (no KV cache), this uses the highly
          optimized FlashAttention-2 kernel if available.

        - Universal SDPA Path: This path handles all other cases, including:
            1. Autoregressive inference with KV Caching.
            2. Custom masking logic (e.g., frame-block causality).
            3. Fallbacks when FlashAttention-2 is unavailable or fails.

        It correctly uses the `is_causal=True` flag for both square (training) and
        non-square (inference) attention patterns, avoiding manual mask construction
        where possible, per modern PyTorch best practices.
        """
        B, L_new, D = x.shape
        Hq, Hkv, Hd = self.cfg.n_q_heads, self.cfg.n_kv_heads, self.head_dim
        assert Hq % Hkv == 0, "n_q_heads must be a multiple of n_kv_heads for GQA"
        gqa_factor = Hq // Hkv

        q_new = self.wq(x).reshape(B, L_new, Hq, Hd)
        k_new = self.wk(x).reshape(B, L_new, Hkv, Hd)
        v_new = self.wv(x).reshape(B, L_new, Hkv, Hd)
        q_new, k_new = self.rope(q_new, k_new, temporal_pos_ids, structural_pos_ids)

        # Treat eval mode as inference; bootstrap cache if needed
        inference_mode = not self.training

        if kv_cache is not None:
            # (optional) align device/dtype first
            k_prev = kv_cache.key
            v_prev = kv_cache.value
            if k_prev.device != k_new.device or k_prev.dtype != k_new.dtype:
                k_prev = k_prev.to(device=k_new.device, dtype=k_new.dtype)
                v_prev = v_prev.to(device=v_new.device, dtype=v_new.dtype)
            k = torch.cat([k_prev, k_new], dim=1)
            v = torch.cat([v_prev, v_new], dim=1)
        else:
            k = k_new
            v = v_new

        # Sliding window (optional)
        if inference_mode and getattr(self.cfg, "max_cache_len_tokens", None):
            maxL = self.cfg.max_cache_len_tokens
            k = k[:, -maxL:, :, :]
            v = v[:, -maxL:, :, :]

        # GQA expansion for SDPA
        k_gqa = k.repeat_interleave(gqa_factor, dim=2)
        v_gqa = v.repeat_interleave(gqa_factor, dim=2)

        # Mask selection
        sdpa_mask = None
        sdpa_is_causal = False
        if attn_mask is not None:
            sdpa_mask = self._prep_attn_mask(attn_mask, L=k.shape[1], B=B, device=x.device)
        elif inference_mode and not self.cfg.inference_use_standard_causal:
            # frame-block causal
            total_len = k.shape[1]
            G = self.cfg.tokens_per_frame
            pos = torch.arange(total_len, device=x.device)
            frame_id = pos // G
            fi_k = frame_id.reshape(1, total_len)
            L_past = total_len - L_new
            q_idx = torch.arange(L_past, total_len, device=x.device)
            fi_q = frame_id[q_idx].reshape(L_new, 1)
            sdpa_mask = (fi_k > fi_q)
        else:
            sdpa_is_causal = True

        # SDPA call
        q_bHLD = q_new.permute(0, 2, 1, 3)
        k_bHLD = k_gqa.permute(0, 2, 1, 3)
        v_bHLD = v_gqa.permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(
            q_bHLD, k_bHLD, v_bHLD,
            attn_mask=sdpa_mask, is_causal=sdpa_is_causal,
            dropout_p=(self.dropout if self.training else 0.0),
        )
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, L_new, D)

        # Always return a cache in inference
        updated_cache = None
        if inference_mode:
            updated_cache = KVCache(key=k.detach(), value=v.detach())

        return self.wo(out), updated_cache
    
class CS2TransformerEncoderLayer(nn.Module):
    def __init__(self, cfg: CS2Config, rope: RoPEPositionalEncoding):
        super().__init__()
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
        """
        Now accepts and returns a KV cache.
        """
        # Pass the cache to the attention module and receive the updated cache back
        attn_output, updated_cache = self.attn(
            self.ln1(x), attn_mask, temporal_pos_ids, structural_pos_ids, kv_cache=kv_cache
        )
        x = x + attn_output
        x = x + self.ff(self.ln2(x))
        # Return both the output tensor and the updated cache for this layer
        return x, updated_cache


class CS2Backbone(nn.Module):
    def __init__(self, cfg: CS2Config):
        super().__init__()
        self.cfg = cfg
        self.rope = RoPEPositionalEncoding(cfg)
        self.layers = nn.ModuleList([CS2TransformerEncoderLayer(cfg, self.rope) for _ in range(cfg.n_layers)])

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor], kv_cache_list: Optional[List[KVCache]] = None
    ) -> Tuple[torch.Tensor, List[Optional[KVCache]]]:
        """
        Threads a list of KV caches through the transformer layers and calculates
        correct positional IDs for autoregressive decoding.
        """
        B, L_new, _ = x.shape
        G = self.cfg.tokens_per_frame

        # --- CRITICAL: Calculate Positional IDs with offset for caching ---
        # The starting position is 0 if there's no cache, otherwise it's the
        # length of the sequence already in the cache.

        #todo check okay
        _assert_cache_consistency(kv_cache_list)
        abs_pos_start = _get_abs_pos_start(kv_cache_list)  # int
        L_new = x.size(1)  # or the variable you already use for new-token length
        pos = torch.arange(abs_pos_start, abs_pos_start + L_new, device=x.device)  # [L_new]
        # use `pos` for rotating the *new* q/k only (do not re-rotate cached states)

        temporal_pos_ids = pos // G    # Frame index
        structural_pos_ids = pos % G  # Token index within a frame

        new_kv_cache_list = []
        for i, layer in enumerate(self.layers):
            # Get the cache for the current layer, or None if it's the first run
            layer_cache = kv_cache_list[i] if kv_cache_list is not None else None
            
            # Pass the cache and get the updated one back
            x, new_layer_cache = layer(
                x, attn_mask, temporal_pos_ids, structural_pos_ids, kv_cache=layer_cache
            )
            new_kv_cache_list.append(new_layer_cache)
            
        return x, new_kv_cache_list

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
        
        self.vol_feats = 128  # Latent channels for the deconv stem (tunable)
        
        # 1. Seed: Project the flat token to a starting 3D volume
        # The starting volume will have the correct Z dimension already.
        # We start with a small spatial resolution (8x8) that will be upsampled to 64x64.
        self.pos_seed = nn.Linear(d, self.vol_feats * 8 * 8 * 8) 
        
        # 2. Deconvolution path: Upsample the Y and X dimensions by 8x (2^3)
        # We use anisotropic strides (1, 2, 2) to only upsample Y and X.
        self.pos_deconv = nn.Sequential(
            # Input: [B, 128, 8, 8, 8]
            # Block 1 -> [B, 64, 8, 16, 16]
            nn.ConvTranspose3d(self.vol_feats, self.vol_feats // 2, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            _GN3d(self.vol_feats // 2),
            nn.GELU(),
            
            # Block 2 -> [B, 32, 8, 32, 32]
            nn.ConvTranspose3d(self.vol_feats // 2, self.vol_feats // 4, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            _GN3d(self.vol_feats // 4),
            nn.GELU(),

            # Block 3 -> [B, 16, 8, 64, 64]
            nn.ConvTranspose3d(self.vol_feats // 4, self.vol_feats // 8, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            _GN3d(self.vol_feats // 8),
            nn.GELU(),
        )
        
        # 3. Final Projection: Reduce feature channels to a single logit channel
        self.pos_final_conv = nn.Conv3d(self.vol_feats // 8, 1, kernel_size=1)

    def forward(self, token: torch.Tensor) -> PlayerPredictions:
        # token: [B, d]
        B = token.shape[0]

        # --- Generate Heatmap ---
        # Project token to the initial seed volume [B, 128, 8, 8, 8]
        seed = self.pos_seed(token).reshape(B, self.vol_feats, 8, 8, 8)
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
        self.vol_feats = 128  # Latent channels for the deconv stem (tunable)
        
        # 1. Seed: Project the flat token to a starting 3D volume [B, 128, 8, 8, 8]
        self.enemy_seed = nn.Linear(d, self.vol_feats * 8 * 8 * 8)
        
        # 2. Deconvolution path: Upsample Y and X dimensions by 8x (2^3) using
        #    anisotropic strides to preserve the Z dimension. This architecture
        #    is modeled directly on the working PlayerHeads implementation.
        self.enemy_deconv = nn.Sequential(
            # Input: [B, 128, 8, 8, 8] -> [B, 64, 8, 16, 16]
            nn.ConvTranspose3d( self.vol_feats,  self.vol_feats // 2, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            _GN3d( self.vol_feats // 2),
            nn.GELU(),
            
            # Block 2 -> [B, 32, 8, 32, 32]
            nn.ConvTranspose3d( self.vol_feats // 2,  self.vol_feats // 4, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            _GN3d( self.vol_feats // 4),
            nn.GELU(),

            # Block 3 -> [B, 16, 8, 64, 64]
            nn.ConvTranspose3d( self.vol_feats // 4,  self.vol_feats // 8, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            _GN3d( self.vol_feats // 8),
            nn.GELU(),
        )
        
        # 3. Final Projection: Reduce feature channels to a single logit channel
        self.enemy_final_conv = nn.Conv3d( self.vol_feats // 8, 1, kernel_size=1)

        # --- Other standard prediction heads ---
        self.round_state = nn.Linear(d, cfg.round_state_dim)
        self.round_number = nn.Linear(d, 1)

    def forward(self, token: torch.Tensor) -> GameStrategyPredictions:
        B = token.shape[0]
        
        # --- Generate Heatmap (Corrected Path) ---
        # Project token to the initial seed volume -> [B, 128, 8, 8, 8]
        seed = self.enemy_seed(token).reshape(B, 128, 8, 8, 8)
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
            # Replace the real ViT with a dummy module
            class DummyVisualEncoder(nn.Module):
                def __init__(self, d_model):
                    super().__init__()
                    self.d_model = d_model
                def forward(self, foveal, peripheral):
                    B, T, P = foveal.shape[:3]
                    # Instantly return a correctly-shaped random tensor
                    return torch.randn(B, T, P, self.d_model, device=foveal.device, dtype=torch.bfloat16)
            self.visual_encoder = DummyVisualEncoder(d)
        else:
            self.visual_encoder = ViTVisualEncoder(cfg)
        self.audio_encoder = AudioCNN(cfg)
        # Special tokens & embeddings
        self.token_game_strategy = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.token_scratch = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.dead_embedding = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.player_slot_embed = nn.Embedding(cfg.num_players, d)
        # Fuser
        self.player_fuser = PlayerTokenFuser(cfg, self.player_slot_embed, self.dead_embedding)
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
        # add standalone parameters/modules
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
        """Compute next-frame predictions (t+1) given frames 1..t."""
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.amp_dtype) if self.use_amp else nullcontext()
        )
        with autocast_ctx:
            fov   = batch["foveal_images"]        # [B, T, 5, 3, 384, 384]
            periph= batch["peripheral_images"]     # [B, T, 5, 3, 384, 384]
            mel   = batch["mel_spectrogram"]       # [B, T, 5, 1, 128, ~6]
            alive = batch["alive_mask"].bool()     # [B, T, 5]

            B, T, P = fov.shape[:3]
            d = getattr(self.cfg,"d_model")

            # Optional consistency check
            if __debug__:
                assert getattr(self.cfg,"tokens_per_frame") == getattr(self.cfg,"num_players") + 2, \
                    "tokens_per_frame must equal num_players + 2 (players + strategy + scratch)"

            # ---- encoders ----
            vis = self.visual_encoder(fov, periph)   # [B, T, P, d]
            aud = self.audio_encoder(mel)            # [B, T, P, d]

            # ---- fuse to player tokens ----
            player_tokens = self.player_fuser(vis, aud, alive)  # [B, T, P, d]

            # ---- special tokens per frame ----
            # self.token_game_strategy / token_scratch are [d]; make them [1,1,1,d] then expand
            tok_gs = self.token_game_strategy.reshape(1, 1, 1, -1).expand(B, T, 1, d)  # [B, T, 1, d]
            tok_sc = self.token_scratch.reshape(1, 1, 1, -1).expand(B, T, 1, d)        # [B, T, 1, d]
            frame_tokens = torch.cat([player_tokens, tok_gs, tok_sc], dim=2)        # [B, T, 7, d]

            # ---- flatten time to sequence ----
            Lpf = getattr(self.cfg,"tokens_per_frame")
            seq = frame_tokens.reshape(B, T * Lpf, d)     # [B, L, d]

            attn_mask = None  # handled inside CS2GQAAttention

            # ---- backbone ----
            h, _ = self.backbone(seq, attn_mask, kv_cache_list=None)

            # ---- slice last frame ----
            last = h[:, -Lpf:, :]                         # [B, 7, d]
            num_players = getattr(self.cfg,"num_players")
            player_tok = last[:, :num_players, :]         # [B, 5, d]
            strat_tok  = last[:, num_players, :]          # [B, d]
            # scratch_tok = last[:, num_players + 1, :]

            # ---- heads ----
            player_preds: List[PlayerPredictions] = []
            for i in range(num_players):
                token_i = player_tok[:, i, :]             # [B, d]
                player_preds.append(self.player_head(token_i))

            strategy_preds = self.strategy_head(strat_tok)

            return {"player": player_preds, "game_strategy": strategy_preds} 
        
    def autoregressive_step(
        self,
        single_frame_batch: CS2Batch,
        past_kv_cache: Optional[List[KVCache]] = None
    ) -> Tuple[Predictions, List[Optional[KVCache]]]:
        """
        Processes a SINGLE frame of data for efficient autoregressive inference.
        
        Args:
            single_frame_batch: A batch dict containing data for ONE time step (T=1).
            past_kv_cache: The list of KV caches from the previous generation step.
                           Should be None for the very first frame.

        Returns:
            A tuple containing:
            - predictions: The dictionary of predictions for the next frame.
            - updated_kv_cache: The list of updated KV caches for the next step.
        """
        # This method assumes evaluation mode and no gradients
        assert not self.training, "autoregressive_step should be called in eval mode"
        
        # Ensure the input batch is for a single time step
        assert single_frame_batch["foveal_images"].shape[1] == 1, "Input for autoregressive_step must have T=1"

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.amp_dtype) if self.use_amp else nullcontext()
        )
        with torch.inference_mode():
            with autocast_ctx:
                # --- Encoders and Token Fusion (for a single frame) ---
                fov = single_frame_batch["foveal_images"]
                periph = single_frame_batch["peripheral_images"]
                mel = single_frame_batch["mel_spectrogram"]
                alive = single_frame_batch["alive_mask"].bool()
                
                B, T, P = fov.shape[:3] # T is 1
                d = self.cfg.d_model

                vis = self.visual_encoder(fov, periph)
                aud = self.audio_encoder(mel)
                player_tokens = self.player_fuser(vis, aud, alive)

                tok_gs = self.token_game_strategy.expand(B, T, 1, d)
                tok_sc = self.token_scratch.expand(B, T, 1, d)
                frame_tokens = torch.cat([player_tokens, tok_gs, tok_sc], dim=2)

                # --- Prepare sequence of new tokens (L will be 7) ---
                seq = frame_tokens.reshape(B, self.cfg.tokens_per_frame, d)

                # --- Backbone call with cache ---
                # Pass the new tokens and the cache from the previous step
                h, updated_kv_cache = self.backbone(seq, attn_mask=None, kv_cache_list=past_kv_cache)

                # --- Prediction Heads ---
                # `h` contains the output for only the new tokens
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
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for the test.")
    parser.add_argument("--context-frames", type=int, default=16, help="Sequence length (time dimension) for the test.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on.")
    parser.add_argument("--dtype", type=str, choices=["fp32", "fp16", "bf16"], default="bf16", help="Compute data type.")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile() for the model.")
    parser.add_argument("--freeze-vit", action="store_true", help="Freeze the ViT encoder weights.")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark instead of just a shape test.")
    parser.add_argument("--autoregressive", type=int, metavar="N_FRAMES", help="Run autoregressive KV-cached benchmark for N frames.")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Number of warmup steps for benchmark.")
    parser.add_argument("--bench-steps", type=int, default=20, help="Number of benchmark steps.")
    parser.add_argument("--dummy-vit", action="store_true",help="Disable ViT to bench main model.")
    parser.add_argument("--num-layers", type=int, default=24, help="Set number of main tf layers.")
    args = parser.parse_args()

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
    print(f"  - torch.compile(): {args.compile}")
    print(f"  - Autoregressive benchmark: {'Yes (' + str(args.autoregressive) + ' frames)' if args.autoregressive else 'No'}")
    print("-" * 60)


    # --- Instantiate Model ---
    cfg = CS2Config(
        compute_dtype=args.dtype,
        context_frames=args.context_frames,
        n_layers=args.num_layers
    )
    model = CS2Transformer(cfg, args.dummy_vit).to(device).eval()

    if args.freeze_vit:
        model.set_vit_frozen(True)

    if args.compile:
        print("[INFO] Compiling model with torch.compile()... (this may take a moment)")
        model = torch.compile(model)
        print("[INFO] Compilation complete.")

    # --- Create Dummy Input Batch ---
    B, T, P = args.batch_size, cfg.context_frames, cfg.num_players
    dummy_batch: CS2Batch = {
        "foveal_images": torch.randint(0, 256, (B, T, P, 3, 384, 384), device=device, dtype=torch.uint8),
        "peripheral_images": torch.randint(0, 256, (B, T, P, 3, 384, 384), device=device, dtype=torch.uint8),
        "mel_spectrogram": torch.randn(B, T, P, 1, cfg.mel_bins, cfg.mel_t, device=device),
        "alive_mask": torch.randint(0, 2, (B, T, P), device=device, dtype=torch.bool),
    }

    # --- Run Shape Test ---
    print("\n[PHASE 1] Running Shape Test...")
    with torch.no_grad():
        predictions = model(dummy_batch)

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

        # Report results
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
        model.eval() # Ensure model is in eval mode

        # Create a dummy batch for a single frame (T=1)
        single_frame_batch: CS2Batch = {
            "foveal_images": torch.randint(0, 256, (B, 1, P, 3, 384, 384), device=device, dtype=torch.uint8),
            "peripheral_images": torch.randint(0, 256, (B, 1, P, 3, 384, 384), device=device, dtype=torch.uint8),
            "mel_spectrogram": torch.randn(B, 1, P, 1, cfg.mel_bins, cfg.mel_t, device=device),
            "alive_mask": torch.randint(0, 2, (B, 1, P), device=device, dtype=torch.bool),
        }

        # Warmup
        print(f"  - Warming up for {args.warmup_steps} autoregressive steps...")
        with torch.no_grad():
            kv_cache = [None] * cfg.n_layers
            for _ in range(args.warmup_steps):
                _, kv_cache = model.autoregressive_step(single_frame_batch, kv_cache)

        # Benchmark
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

        # Report results
        total_time = end_time - start_time
        avg_time_ms = (total_time / num_frames) * 1000
        throughput_fps = num_frames / total_time

        print("\n" + "=" * 20 + " AUTOREGRESSIVE BENCHMARK RESULTS " + "=" * 20)
        print(f"  - Total frames generated: {num_frames}")
        print(f"  - Average time per frame: {avg_time_ms:.2f} ms")
        print(f"  - Throughput (Frames/Sec): {throughput_fps:.2f} FPS")
        if device.type == "cuda":
            end_mem = torch.cuda.max_memory_allocated(device)
            # The cache is now the dominant memory user
            peak_mem_gb = (end_mem) / (1024 ** 3)
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