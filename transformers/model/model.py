"""
A Causal Autoregressive Multi-Modal Transformer for CS2 Gameplay Generation
model.py — basic structure & scaffolding

This file defines the top-level CS2Transformer and its submodules as stubs with
clear interfaces, docstrings, and shapes. Fill in TODOs to complete the model.

Spec highlights implemented here:
- 7 tokens per frame (5×players + [GAME_STRATEGY] + [SCRATCHSPACE])
- Shared ViT-Large encoder for foveal & peripheral views (CLS concat → 2048)
- Audio CNN → 2048, fusion via elementwise add + LayerNorm, slot embeddings
- DEAD token support
- Transformer backbone (Pre-LN) with GQA + RoPE + FlashAttention-2 (hooks/stubs)
- Player heads & strategy head with Appendix B output schema

Note: The attention kernel and some internals are left as TODOs so you can
choose your preferred FlashAttention-2 integration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, TypedDict, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 0) Public types (Appendix B)
# -----------------------------------------------------------------------------

class PlayerPredictions(TypedDict):
    stats: torch.Tensor                      # [B, 3]
    pos_heatmap_logits: torch.Tensor         # [B, 64, 256, 256]
    mouse_delta_deg: torch.Tensor            # [B, 2]
    keyboard_logits: torch.Tensor            # [B, 31]
    eco_logits: torch.Tensor                 # [B, 384]
    inventory_logits: torch.Tensor           # [B, 128]
    active_weapon_logits: torch.Tensor       # [B, 128]

class GameStrategyPredictions(TypedDict):
    enemy_pos_heatmap_logits: torch.Tensor   # [B, 64, 256, 256]
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
    pos_z: int = 64
    pos_y: int = 256
    pos_x: int = 256

    # RoPE / long-context
    rope_base: int = 10000
    rope_scaling: Optional[str] = "ntk-yarn"  # placeholder selector


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

        # Register normalization buffers (broadcastable)
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3, 1, 1))

        # Optional backends
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
            # Headless model; forward returns [N, 1024]
            self.vit = timm.create_model(
                "vit_large_patch16_384", pretrained=True, num_classes=0
            )
            self.backend = "timm"
            self.vit_out_dim = 1024
        elif self._has_hf:
            # HF returns hidden states; we'll grab CLS at index 0
            self.vit = ViTModel.from_pretrained("google/vit-large-patch16-384")
            self.backend = "hf"
            self.vit_out_dim = 1024
        else:
            raise ImportError(
                "Neither timm nor transformers is available for ViTVisualEncoder."
            )

        # Project concat(CLS_a, CLS_b) → d_model
        self.proj = nn.Linear(self.vit_out_dim * 2, cfg.d_model)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., 3, H, W] in [0,1]. If input appears in [0,255], scale down.
        if x.dtype in (torch.uint8, torch.int8, torch.int16):
            x = x.float() / 255.0
        elif x.max() > 1.5:
            x = x / 255.0
        # Move channel dim to match buffers for broadcasting
        return (x - self.img_mean) / self.img_std

    @torch.no_grad()
    def _forward_one_view_eval(self, x: torch.Tensor) -> torch.Tensor:
        """Fast path for eval; returns CLS [N, 1024]."""
        if self.backend == "timm":
            y = self.vit(x)  # [N, 1024] when num_classes=0
            return y
        else:
            outputs = self.vit(pixel_values=x)
            return outputs.last_hidden_state[:, 0, :]  # [N, 1024]

    def _forward_one_view_train(self, x: torch.Tensor) -> torch.Tensor:
        """Train path with grads; returns CLS [N, 1024]."""
        if self.backend == "timm":
            y = self.vit(x)
            return y
        else:
            outputs = self.vit(pixel_values=x)
            return outputs.last_hidden_state[:, 0, :]

    def _forward_one_view(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_one_view_eval(x) if not self.training else self._forward_one_view_train(x)

    def forward(
        self,
        foveal: torch.Tensor,        # [B, T, P, 3, 384, 384]
        peripheral: torch.Tensor,    # [B, T, P, 3, 384, 384]
    ) -> torch.Tensor:
        B, T, P = foveal.shape[:3]
        N = B * T * P

        # normalize to ImageNet stats
        foveal = self._normalize(foveal)
        peripheral = self._normalize(peripheral)

        fov = foveal.view(N, *foveal.shape[-3:])      # [N, 3, 384, 384]
        per = peripheral.view(N, *peripheral.shape[-3:])

        # Run both views through shared ViT, take CLS per view
        cls_a = self._forward_one_view(fov)  # [N, 1024]
        cls_b = self._forward_one_view(per)  # [N, 1024]

        concat = torch.cat([cls_a, cls_b], dim=-1)    # [N, 2048]
        vis = self.proj(concat)                       # [N, d_model]
        vis = vis.view(B, T, P, self.d_model)
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
        x = mel.view(B * T * P, 1, mel.shape[-2], mel.shape[-1])
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)  # [N, d_model]
        x = x.view(B, T, P, -1)
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
        B, T, P, d = vis.shape
        fused = self.norm(vis + aud)
        # add slot embeddings
        slot_ids = torch.arange(P, device=vis.device).view(1, 1, P).expand(B, T, P)
        slots = self.slot_embed(slot_ids)  # [B, T, P, d]
        fused = fused + slots
        # DEAD swap
        dead = ~alive_mask.bool()
        if dead.any():
            fused = torch.where(
                dead[..., None],
                self.dead_embedding + slots,
                fused,
            )
        return fused  # [B, T, P, d]


# -----------------------------------------------------------------------------
# 3) Attention core & Transformer layers (stubs for FA2+GQA+RoPE)
# -----------------------------------------------------------------------------

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

    def _build_cos_sin(self, L: int, dim: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self._inv_freq(dim, device).to(dtype)
        t = torch.arange(L, device=device, dtype=dtype).unsqueeze(-1)  # [L,1]
        freqs = t * inv_freq.unsqueeze(0)                               # [L,half]
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return cos, sin

    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rot_dim: int) -> torch.Tensor:
        """x: [B, L, H, Hd]. Apply rotary to first rot_dim dims of Hd."""
        B, L, H, Hd = x.shape
        rd = rot_dim
        x_rot = x[..., :rd]
        x_pass = x[..., rd:]
        # pair dims (rd/2, 2)
        x_pair = x_rot.view(B, L, H, rd // 2, 2)
        cos = cos.view(1, L, 1, rd // 2, 1).to(x.dtype)
        sin = sin.view(1, L, 1, rd // 2, 1).to(x.dtype)
        x1 = x_pair[..., 0]
        x2 = x_pair[..., 1]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        y = torch.stack([y1, y2], dim=-1).view(B, L, H, rd)
        return torch.cat([y, x_pass], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q: [B, L, Hq, Hd], k: [B, L, Hkv, Hd]; positions: [L]
        L = q.shape[1]
        rd = self.rot_dim or q.shape[-1]
        cos, sin = self._build_cos_sin(L, rd, q.device, q.dtype)
        q = self._apply_rotary(q, cos, sin, rd)
        k = self._apply_rotary(k, cos, sin, rd)
        return q, k


class CS2GQAAttention(nn.Module):
    """Grouped-Query Attention with FlashAttention-2.

    Inputs: x [B, L, d]. Applies RoPE to Q/K, then attention.
    If FlashAttention-2 is available (and on CUDA + half/bfloat16), uses it.
    Otherwise falls back to a naive attention for small-debug runs.
    """
    def __init__(self, cfg: CS2Config, rope: RoPEPositionalEncoding):
        super().__init__()
        self.cfg = cfg
        self.rope = rope
        d, hq, hkv = cfg.d_model, cfg.n_q_heads, cfg.n_kv_heads
        assert d % hq == 0, "d_model must be divisible by n_q_heads"
        self.head_dim = d // hq
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

    def _use_fa2(self, x: torch.Tensor) -> bool:
        return (
            self._fa2 is not None
            and x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16)
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor], pos_ids: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        Hq = self.cfg.n_q_heads
        Hkv = self.cfg.n_kv_heads
        Hd = self.head_dim
        q = self.wq(x).view(B, L, Hq, Hd)
        k = self.wk(x).view(B, L, Hkv, Hd)
        v = self.wv(x).view(B, L, Hkv, Hd)
        # Apply RoPE
        q, k = self.rope(q, k, pos_ids)

        # Map K/V heads to Q heads by repeating (GQA grouping)
        if Hq % Hkv != 0:
            raise ValueError(f"n_q_heads ({Hq}) must be a multiple of n_kv_heads ({Hkv}) for GQA repeat.")
        repeat = Hq // Hkv
        k_t = k.repeat_interleave(repeat, dim=2)  # [B, L, Hq, Hd]
        v_t = v.repeat_interleave(repeat, dim=2)  # [B, L, Hq, Hd]

        if self._use_fa2(x) and attn_mask is None:
            # FlashAttention expects [B, L, H, Hd]
            out = self._fa2(q, k_t, v_t, dropout_p=0.0, causal=True)  # [B, L, Hq, Hd]
            out = out.view(B, L, Hq, Hd)
            out = out.permute(0, 1, 2, 3).contiguous().view(B, L, D)
            return self.wo(out)
        else:
            # ... inside CS2GQAAttention.forward else block
            # Fallback: scaled dot-product with causal mask
            # Reshape for torch's native attention function [B, H, L, Hd]
            q = q.permute(0, 2, 1, 3)
            k_t = k_t.permute(0, 2, 1, 3)
            v_t = v_t.permute(0, 2, 1, 3)

            # Use the built-in, optimized function which handles the causal mask internally.
            # The 'attn_mask' passed in from the backbone is already a causal mask.
            # Note: F.scaled_dot_product_attention expects a boolean mask where True means "attend".
            # The _build_causal_mask function already creates it in this format.
            # For purely causal attention, you can also just pass is_causal=True.
            out = F.scaled_dot_product_attention(q, k_t, v_t, attn_mask=attn_mask, is_causal=attn_mask is None)

            out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
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
        self.stats = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 3))
        self.mouse = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 2))
        self.keyboard = nn.Linear(d, cfg.keyboard_dim)
        self.eco = nn.Linear(d, cfg.eco_dim)
        self.inventory = nn.Linear(d, cfg.inventory_dim)
        self.active_weapon = nn.Linear(d, cfg.weapon_dim)
        # Simple upsampling stub for 3D heatmap from token
        vol_feats = 64  # latent channels for deconv stem (tunable)
        self.pos_seed = nn.Linear(d, vol_feats * 8 * 8 * 8)
        self.pos_deconv = nn.Sequential(
            nn.ConvTranspose3d(vol_feats, vol_feats // 2, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose3d(vol_feats // 2, 1, 4, stride=2, padding=1),
        )
        self.pos_shape = (cfg.pos_z, cfg.pos_y, cfg.pos_x)

    def forward(self, token: torch.Tensor) -> PlayerPredictions:
        # token: [B, d]
        B = token.shape[0]
        stats = self.stats(token)
        mouse_delta_deg = self.mouse(token)
        keyboard_logits = self.keyboard(token)
        eco_logits = self.eco(token)
        inventory_logits = self.inventory(token)
        active_weapon_logits = self.active_weapon(token)
        # 3D heatmap (coarse → upsample twice → later resize to (64,256,256) if needed)
        seed = self.pos_seed(token).view(B, 64, 8, 8, 8)
        vol = self.pos_deconv(seed)  # [B, 1, 32, 32, 32]
        # For skeleton, pad/resize to target shape lazily
        pos_heatmap_logits = F.interpolate(vol, size=self.pos_shape, mode="trilinear", align_corners=False).squeeze(1)
        return {
            "stats": stats,
            "pos_heatmap_logits": pos_heatmap_logits,
            "mouse_delta_deg": mouse_delta_deg,
            "keyboard_logits": keyboard_logits,
            "eco_logits": eco_logits,
            "inventory_logits": inventory_logits,
            "active_weapon_logits": active_weapon_logits,
        }


class StrategyHead(nn.Module):
    def __init__(self, cfg: CS2Config):
        super().__init__()
        d = cfg.d_model
        vol_feats = 64
        self.enemy_seed = nn.Linear(d, vol_feats * 8 * 8 * 8)
        self.enemy_deconv = nn.Sequential(
            nn.ConvTranspose3d(vol_feats, vol_feats // 2, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose3d(vol_feats // 2, 1, 4, stride=2, padding=1),
        )
        self.enemy_shape = (cfg.pos_z, cfg.pos_y, cfg.pos_x)
        self.round_state = nn.Linear(d, cfg.round_state_dim)
        self.round_number = nn.Linear(d, 1)

    def forward(self, token: torch.Tensor) -> GameStrategyPredictions:
        B = token.shape[0]
        seed = self.enemy_seed(token).view(B, 64, 8, 8, 8)
        vol = self.enemy_deconv(seed)  # [B, 1, 32, 32, 32]
        enemy_pos_heatmap_logits = F.interpolate(vol, size=self.enemy_shape, mode="trilinear", align_corners=False).squeeze(1)
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
        self.player_heads = nn.ModuleList([PlayerHeads(cfg) for _ in range(cfg.num_players)])
        self.strategy_head = StrategyHead(cfg)

    # --------------------------- utility methods --------------------------- #
    def set_vit_frozen(self, frozen: bool) -> None:
        for p in self.visual_encoder.parameters():
            p.requires_grad = not frozen

    def parameter_groups(self) -> List[Dict[str, object]]:
        vit_params = list(self.visual_encoder.parameters())
        rest = [p for m in [self.audio_encoder, self.player_fuser, self.backbone, self.player_heads, self.strategy_head]
                for p in m.parameters()]
        return [
            {"params": vit_params, "name": "vit", "lr_scale": 0.1},
            {"params": rest, "name": "core", "lr_scale": 1.0},
        ]

    # ------------------------------ helpers ------------------------------- #
    @staticmethod
    def _build_causal_mask(L: int, device: torch.device) -> torch.Tensor:
        # Returns boolean mask of shape [1, 1, L, L]: True = allowed
        i = torch.arange(L, device=device)
        mask = (i[:, None] >= i[None, :])
        return mask.view(1, 1, L, L)

    # ------------------------------- forward ------------------------------ #
    def forward(self, batch: CS2Batch) -> Predictions:
        """Compute next-frame predictions (t+1) given a sequence of frames 1..t.

        Expects fields in `batch` matching CS2Batch. Returns Predictions dict.
        """
        fov = batch["foveal_images"]         # [B, T, 5, 3, 384, 384]
        periph = batch["peripheral_images"]  # [B, T, 5, 3, 384, 384]
        mel = batch["mel_spectrogram"]       # [B, T, 5, 1, 128, ~6]
        alive = batch["alive_mask"].bool()    # [B, T, 5]

        B, T, P = fov.shape[:3]
        d = self.cfg.d_model

        # Encode modalities
        vis = self.visual_encoder(fov, periph)   # [B, T, P, d]
        aud = self.audio_encoder(mel)            # [B, T, P, d]

        # Fuse to player tokens (handles DEAD swap and slot id)
        player_tokens = self.player_fuser(vis, aud, alive)   # [B, T, P, d]

        # Append special tokens per frame
        tok_gs = self.token_game_strategy.expand(B, T, -1, -1)  # [B, T, 1, d]
        tok_sc = self.token_scratch.expand(B, T, -1, -1)        # [B, T, 1, d]
        frame_tokens = torch.cat([player_tokens, tok_gs, tok_sc], dim=2)  # [B, T, 7, d]

        # Flatten time to sequence and build causal mask
        seq = frame_tokens.view(B, T * self.cfg.tokens_per_frame, d)       # [B, L, d]
        L = seq.shape[1]
        attn_mask = self._build_causal_mask(L, seq.device)                 # [1,1,L,L]

        # Backbone
        h = self.backbone(seq, attn_mask)  # [B, L, d]

        # Select final frame tokens (last 7 positions)
        last = h[:, -self.cfg.tokens_per_frame:, :]              # [B, 7, d]
        player_tok = last[:, :self.cfg.num_players, :]           # [B, 5, d]
        strat_tok = last[:, self.cfg.num_players, :]             # [B, d]
        # scratch_tok = last[:, self.cfg.num_players + 1, :]     # optional

        # Heads
        player_preds: List[PlayerPredictions] = []
        for i in range(self.cfg.num_players):
            token_i = player_tok[:, i, :]  # [B, d]
            player_preds.append(self.player_heads[i](token_i))

        strategy_preds = self.strategy_head(strat_tok)  # GameStrategyPredictions

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
