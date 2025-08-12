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
class CS2Batch(TypedDict, total=False):
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
    compute_dtype: Literal["fp32", "fp16", "bf16"]  
    amp_autocast: bool    
    # Model dims
    d_model: int = 2048
    n_layers: int = 24
    n_q_heads: int = 32
    n_kv_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.0

    # Sequence
    num_players: int = 5
    tokens_per_frame: int = 7  # 5 players + 2 special

    # Context (training)
    context_frames: int = 128

    # Once fixed, these can never change during training or inference!
    # If fused_causal is used, any passed mask is ignored.
    use_fused_causal: bool = True #use FA2, mutually exclusive with use_frame_block_causal_mask
    use_frame_block_causal_mask: bool = False #cannot be used with FA2 e.g.; use_fused_causal

    # Vision
    vit_name: str = "google/vit-large-patch16-384"
    vit_out_dim: int = 1024
    use_two_views: bool = True

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
    rope_scaling: Optional[Dict[str, Any]]  # e.g.
    # {"type": "linear", "factor": 2.0}                   # 2× context
    # {"type": "linear_by_len", "orig": 4096, "target": 8192}


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
            self.vit = timm.create_model("vit_large_patch16_384", pretrained=True, num_classes=0)
            self.backend = "timm"
            self.vit_out_dim = 1024
        elif self._has_hf:
            self.vit = ViTModel.from_pretrained("google/vit-large-patch16-384")
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

    def _forward_one_view(self, x: torch.Tensor) -> torch.Tensor:
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
        cls_a = self._forward_one_view(fov)   # [N, 1024]
        cls_b = self._forward_one_view(per)   # [N, 1024]

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
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(c3)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Linear(c3 * 4 * 4, cfg.d_model)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: [B, T, P, 1, 128, ~6] → pack to [B*T*P, 1, 128, ~6]
        B, T, P = mel.shape[:3]
        x = mel.reshape(B * T * P, 1, mel.shape[-2], mel.shape[-1])
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)  # [N, d_model]
        x = x.reshape(B, T, P, -1)
        return x
    
class PlayerTokenFuser(nn.Module):
    """Fuse visual+audio via elementwise add, LayerNorm, add slot identity.

    DEAD handling policy: if dead, replace fused token with
      dead_embedding + player_slot_embedding.
    """
    def __init__(self, cfg: CS2Config, player_slot_embed: nn.Embedding, dead_embedding: nn.Parameter):
        super().__init__()
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
        Fuse visual+audio, apply the DEAD policy, and add player identity.
        This corrected version is more efficient and readable by using broadcasting.
        """
        B, T, P, d = vis.shape

        # 1. Fuse the sensor data for the "alive" case. This is only computed once.
        fused_alive_token = self.norm(vis + aud)

        # 2. Prepare the mask for broadcasting. We invert the alive_mask to get a
        #    dead_mask and add a dimension for the feature dimension `d`.
        # The mask will be True where players are DEAD.
        dead_mask = ~alive_mask.bool().unsqueeze(-1) # Shape: [B, T, P, 1]

        # 3. Select the base token using torch.where. This is highly efficient as it
        #    avoids creating a large intermediate tensor for the dead embedding.
        #    PyTorch will broadcast self.dead_embedding to the correct shape only
        #    where the mask is True.
        base_token = torch.where(
            dead_mask,
            self.dead_embedding, # The single [1, 1, d] DEAD embedding
            fused_alive_token,   # The [B, T, P, d] fused sensor token
        )

        # 4. Add the unique player slot identity to the selected base token.
        #    This happens in a single, non-redundant step, regardless of whether
        #    the player is alive or dead. Broadcasting handles the expansion.
        slot_ids = torch.arange(P, device=vis.device).reshape(1, 1, P)
        slots = self.slot_embed(slot_ids) # Broadcasts from [1, 1, P, d] to [B, T, P, d]
        
        final_token = base_token + slots

        return final_token

# -----------------------------------------------------------------------------
# 3) Attention core & Transformer layers (stubs for FA2+GQA+RoPE)
# -----------------------------------------------------------------------------
# In model.py

class RoPEPositionalEncoding(nn.Module):
    """Rotary Positional Embedding (RoPE) with optional NTK-like scaling.

    Applies sin/cos rotations to the first `rot_dim` dimensions of Q/K per head.
    Standard usage sets `rot_dim = head_dim`.
    """
    def __init__(self, cfg: CS2Config, rot_dim: Optional[int] = None):
        super().__init__()
        self.base = cfg.rope_base
        self.rot_dim = rot_dim
        # Default scale = 1.0 (no scaling). Expose setter for long-context tuning.
        self.register_buffer("scale", torch.tensor(1.0), persistent=False)
        self._apply_cfg_scaling(cfg.get("rope_scaling"))

    def _apply_cfg_scaling(self, scaling: Optional[Dict[str, Any]]) -> None:
        """
        Supports:
          - {"type": "linear", "factor": s}          # extend context by s×
          - {"type": "linear_by_len", "orig": L0, "target": L1}  # extend from L0 to L1
        Internally we multiply inv_freq by `1/s` to reduce frequencies.
        """
        if not scaling:
            return
        sc_type = scaling.get("type", "linear")
        if sc_type == "linear":
            s = float(scaling.get("factor", 1.0))
            if s <= 0:
                raise ValueError("rope_scaling.factor must be > 0")
            self.set_scale(1.0 / s)  # smaller frequencies
        elif sc_type == "linear_by_len":
            L0 = int(scaling["orig"])
            L1 = int(scaling["target"])
            if L0 <= 0 or L1 <= 0:
                raise ValueError("rope_scaling.orig/target must be > 0")
            self.set_scale(float(L0) / float(L1))  # = 1/s where s = L1/L0
        else:
            raise ValueError(f"Unsupported rope_scaling.type: {sc_type}")

    def set_scale(self, scale: float) -> None:
        self.scale.fill_(float(scale))

    def _inv_freq(self, dim: int, device: torch.device) -> torch.Tensor:
        half = dim // 2
        idx = torch.arange(0, half, device=device, dtype=torch.float32)
        inv = self.base ** (idx / half)  # base^(i/(dim/2))
        inv_freq = 1.0 / inv
        # NTK-like scaling: reduce frequencies to extend context
        inv_freq = inv_freq * self.scale
        return inv_freq  # [half]

    def _build_cos_sin(
        self, positions: torch.Tensor, dim: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self._inv_freq(dim, device).to(dtype)
        t = positions.to(device=device, dtype=dtype).unsqueeze(-1)      # [L, 1]
        freqs = t * inv_freq.unsqueeze(0)                                # [L, half]
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return cos, sin

    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rot_dim: int) -> torch.Tensor:
        """x: [B, L, H, Hd]. Apply rotary to first rot_dim dims of Hd."""
        assert rot_dim % 2 == 0, "RoPE dimension must be even"
        B, L, H, Hd = x.shape
        rd = rot_dim
        x_rot = x[..., :rd]
        x_pass = x[..., rd:]
        # pair dims (rd/2, 2)
        x_pair = x_rot.reshape(B, L, H, rd // 2, 2)
        cos = cos.reshape(1, L, 1, rd // 2, 1).to(x.dtype)
        sin = sin.reshape(1, L, 1, rd // 2, 1).to(x.dtype)
        x1 = x_pair[..., 0]
        x2 = x_pair[..., 1]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        y = torch.stack([y1, y2], dim=-1).reshape(B, L, H, rd)
        return torch.cat([y, x_pass], dim=-1)


    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q: [B, L, Hq, Hd], k: [B, L, Hkv, Hd]; positions: [L]
        L = q.shape[1]
        if positions.shape[0] != L:
            raise ValueError(f"Length of positions tensor ({positions.shape[0]}) does not match sequence length ({L}).")

        rd = self.rot_dim or q.shape[-1]
        
        # This function call is now correct because _build_cos_sin will respect the `positions` tensor.
        cos, sin = self._build_cos_sin(positions, rd, q.device, q.dtype)
        
        q = self._apply_rotary(q, cos, sin, rd)
        k = self._apply_rotary(k, cos, sin, rd)
        return q, k
        

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
        self.head_dim = d // hq
        # RoPE needs even rotation dim; with full-head RoPE this means even head_dim
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
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

    def _prep_attn_mask(self, attn_mask: Optional[torch.Tensor], L: int, B: int) -> Optional[torch.Tensor]:
        """
        Normalize attention masks for PyTorch SDPA:
          - Boolean mask: True = DISALLOW (masked-out), False = allow.
          - Float mask: additive; large negative (e.g., <= -1e4) = disallow, 0 = allow.
        Supports shapes: [L,L], [B,L,L], [B,1,L,L], [1,1,L,L].
        Returns bool mask broadcastable to [B, H, L, L] (we pass H explicitly in SDPA).
        """
        if attn_mask is None:
            return None
        m = attn_mask
        if m.dtype != torch.bool:
            m = m <= -1e4  # treat very negative as masked-out
        # Shape normalize
        if m.dim() == 2:          # [L, L] -> [1,1,L,L]
            m = m.view(1, 1, L, L)
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

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor], pos_ids: torch.Tensor) -> torch.Tensor:
        """
        x:         [B, L, D]
        attn_mask: Boolean or additive mask. Boolean True = DISALLOW. Additive: large negative = disallow.
                Shapes supported: [L,L], [B,L,L], [B,1,L,L], [1,1,L,L].
        pos_ids:   [L] integer positions for RoPE
        """
        B, L, D = x.shape
        Hq, Hkv, Hd = self.cfg.n_q_heads, self.cfg.n_kv_heads, self.head_dim

        # Projections
        q = self.wq(x).reshape(B, L, Hq, Hd)
        k = self.wk(x).reshape(B, L, Hkv, Hd)
        v = self.wv(x).reshape(B, L, Hkv, Hd)

        # RoPE on q/k
        q, k = self.rope(q, k, pos_ids)

        # GQA expand KV to match Q heads
        if Hq % Hkv != 0:
            raise ValueError(f"n_q_heads ({Hq}) must be divisible by n_kv_heads ({Hkv}) for GQA.")
        group = Hq // Hkv
        k = k.unsqueeze(2).expand(B, L, Hkv, group, Hd).reshape(B, L, Hq, Hd)
        v = v.unsqueeze(2).expand(B, L, Hkv, group, Hd).reshape(B, L, Hq, Hd)

        q_bLHD = q.contiguous()
        k_bLHD = k.contiguous()
        v_bLHD = v.contiguous()

        dropout_p = self.dropout if self.training else 0.0

        # Fast path only if no external mask and no frame-block mask required
        can_use_fused_path = (
            self._use_fa2(x, q_bLHD)
            and self.cfg.use_fused_causal
            and not self.cfg.use_frame_block_causal_mask
            and attn_mask is None
        )

        if can_use_fused_path:
            try:
                out = self._fa2_call(
                    q_bLHD, k_bLHD, v_bLHD,
                    dropout_p=dropout_p,
                    causal=True,
                )
            except Exception:
                can_use_fused_path = False  # fall back to SDPA on any FA2 error

        if not can_use_fused_path:
            # --- Build/normalize masks and combine ---
            ext_mask = self._prep_attn_mask(attn_mask, L=L, B=B)  # bool or None

            built_mask = None
            if self.cfg.use_frame_block_causal_mask:
                # Per-frame causal: attend only within frame and to the past (<=)
                G = self.cfg.tokens_per_frame
                pos = torch.arange(L, device=x.device)
                frame_id = (pos // G)
                fi_q = frame_id.view(1, 1, L, 1)
                fi_k = frame_id.view(1, 1, 1, L)
                built_mask = (fi_k > fi_q)  # True = disallow
            elif self.cfg.use_fused_causal:
                i = torch.arange(L, device=x.device)
                built_mask = (i[None, :] > i[:, None]).view(1, 1, L, L)  # True = disallow

            if ext_mask is None:
                disallow_mask = built_mask
            elif built_mask is None:
                disallow_mask = ext_mask
            else:
                disallow_mask = (ext_mask.to(x.device) | built_mask)

            # SDPA expects [B,H,L,D]
            q_bHLD = q_bLHD.permute(0, 2, 1, 3).contiguous()
            k_bHLD = k_bLHD.permute(0, 2, 1, 3).contiguous()
            v_bHLD = v_bLHD.permute(0, 2, 1, 3).contiguous()

            out = torch.nn.functional.scaled_dot_product_attention(
                q_bHLD, k_bHLD, v_bHLD,
                attn_mask=disallow_mask,  # bool mask, True = disallow; can be None
                is_causal=False,          # explicit mask handles causality
                dropout_p=dropout_p,
            )
            out = out.permute(0, 2, 1, 3).contiguous()

        out = out.reshape(B, L, D)
        return self.wo(out)


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

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor], pos_ids: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask, pos_ids)
        x = x + self.ff(self.ln2(x))
        return x


class CS2Backbone(nn.Module):
    def __init__(self, cfg: CS2Config):
        super().__init__()
        self.rope = RoPEPositionalEncoding(cfg)
        self.layers = nn.ModuleList([CS2TransformerEncoderLayer(cfg, self.rope) for _ in range(cfg.n_layers)])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # x: [B, L, d]; attn_mask: [1, 1, L, L] boolean where True=allowed
        B, L, _ = x.shape
        pos_ids = torch.arange(L, device=x.device)
        for layer in self.layers:
            x = layer(x, attn_mask, pos_ids)
        return x


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
        
        vol_feats = 128  # Latent channels for the deconv stem (tunable)
        
        # 1. Seed: Project the flat token to a starting 3D volume
        # The starting volume will have the correct Z dimension already.
        # We start with a small spatial resolution (8x8) that will be upsampled to 64x64.
        self.pos_seed = nn.Linear(d, vol_feats * 8 * 8 * 8) 
        
        # 2. Deconvolution path: Upsample the Y and X dimensions by 8x (2^3)
        # We use anisotropic strides (1, 2, 2) to only upsample Y and X.
        self.pos_deconv = nn.Sequential(
            # Input: [B, 128, 8, 8, 8]
            # Block 1 -> [B, 64, 8, 16, 16]
            nn.ConvTranspose3d(vol_feats, vol_feats // 2, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            _GN3d(vol_feats // 2),
            nn.GELU(),
            
            # Block 2 -> [B, 32, 8, 32, 32]
            nn.ConvTranspose3d(vol_feats // 2, vol_feats // 4, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            _GN3d(vol_feats // 4),
            nn.GELU(),

            # Block 3 -> [B, 16, 8, 64, 64]
            nn.ConvTranspose3d(vol_feats // 4, vol_feats // 8, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            _GN3d(vol_feats // 8),
            nn.GELU(),
        )
        
        # 3. Final Projection: Reduce feature channels to a single logit channel
        self.pos_final_conv = nn.Conv3d(vol_feats // 8, 1, kernel_size=1)

    def forward(self, token: torch.Tensor) -> PlayerPredictions:
        # token: [B, d]
        B = token.shape[0]

        # --- Generate Heatmap ---
        # Project token to the initial seed volume [B, 128, 8, 8, 8]
        seed = self.pos_seed(token).reshape(B, 128, 8, 8, 8)
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
        vol_feats = 128  # Latent channels for the deconv stem (tunable)
        
        # 1. Seed: Project the flat token to a starting 3D volume [B, 128, 8, 8, 8]
        self.enemy_seed = nn.Linear(d, vol_feats * 8 * 8 * 8)
        
        # 2. Deconvolution path: Upsample Y and X dimensions by 8x (2^3) using
        #    anisotropic strides to preserve the Z dimension. This architecture
        #    is modeled directly on the working PlayerHeads implementation.
        self.enemy_deconv = nn.Sequential(
            # Input: [B, 128, 8, 8, 8] -> [B, 64, 8, 16, 16]
            nn.ConvTranspose3d(vol_feats, vol_feats // 2, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            _GN3d(vol_feats // 2),
            nn.GELU(),
            
            # Block 2 -> [B, 32, 8, 32, 32]
            nn.ConvTranspose3d(vol_feats // 2, vol_feats // 4, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            _GN3d(vol_feats // 4),
            nn.GELU(),

            # Block 3 -> [B, 16, 8, 64, 64]
            nn.ConvTranspose3d(vol_feats // 4, vol_feats // 8, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            _GN3d(vol_feats // 8),
            nn.GELU(),
        )
        
        # 3. Final Projection: Reduce feature channels to a single logit channel
        self.enemy_final_conv = nn.Conv3d(vol_feats // 8, 1, kernel_size=1)

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
    def __init__(self, cfg: CS2Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        # Encoders
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
        return [
            {"params": vit_params, "name": "vit", "lr_scale": 0.1},
            {"params": rest, "name": "core", "lr_scale": 1.0},
        ]


    def forward(self, batch: CS2Batch) -> Predictions:
        """Compute next-frame predictions (t+1) given frames 1..t."""
        # ---- autocast setup ----
        use_amp = (
            self.cfg.get("amp_autocast", True)
            and torch.cuda.is_available()
            and self.cfg.get("compute_dtype", "bf16") != "fp32"
        )
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        amp_dtype = dtype_map[self.cfg.get("compute_dtype", "bf16")]
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
        )

        with autocast_ctx:
            fov   = batch["foveal_images"]        # [B, T, 5, 3, 384, 384]
            periph= batch["peripheral_images"]     # [B, T, 5, 3, 384, 384]
            mel   = batch["mel_spectrogram"]       # [B, T, 5, 1, 128, ~6]
            alive = batch["alive_mask"].bool()     # [B, T, 5]

            B, T, P = fov.shape[:3]
            d = self.cfg.get("d_model")

            # Optional consistency check
            if __debug__:
                assert self.cfg.get("tokens_per_frame") == self.cfg.get("num_players") + 2, \
                    "tokens_per_frame must equal num_players + 2 (players + strategy + scratch)"

            # ---- encoders ----
            vis = self.visual_encoder(fov, periph)   # [B, T, P, d]
            aud = self.audio_encoder(mel)            # [B, T, P, d]

            # ---- fuse to player tokens ----
            player_tokens = self.player_fuser(vis, aud, alive)  # [B, T, P, d]

            # ---- special tokens per frame ----
            # self.token_game_strategy / token_scratch are [d]; make them [1,1,1,d] then expand
            tok_gs = self.token_game_strategy.view(1, 1, 1, -1).expand(B, T, 1, d)  # [B, T, 1, d]
            tok_sc = self.token_scratch.view(1, 1, 1, -1).expand(B, T, 1, d)        # [B, T, 1, d]
            frame_tokens = torch.cat([player_tokens, tok_gs, tok_sc], dim=2)        # [B, T, 7, d]

            # ---- flatten time to sequence ----
            Lpf = self.cfg.get("tokens_per_frame")
            seq = frame_tokens.reshape(B, T * Lpf, d)     # [B, L, d]

            attn_mask = None  # handled inside CS2GQAAttention

            # ---- backbone ----
            h = self.backbone(seq, attn_mask)             # [B, L, d]

            # ---- slice last frame ----
            last = h[:, -Lpf:, :]                         # [B, 7, d]
            num_players = self.cfg.get("num_players")
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
        
# If this file is imported, users can create and compile as follows:
#   model = CS2Transformer(CS2Config())
#   model = torch.compile(model)  # outside this module

__all__ = [
    "CS2Config",
    "PlayerPredictions",
    "GameStrategyPredictions",
    "Predictions",
    "CS2Transformer",
]
