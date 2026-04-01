#!/usr/bin/env python3
"""
Copyright 2025 Anton Althoff

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
------------------------------------------------------------------------
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Any, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import (
    AutoModel, AutoImageProcessor, LlamaConfig, LlamaModel, 
    DacModel, Blip2QFormerConfig, Blip2QFormerModel
)
from transformers.cache_utils import Cache, DynamicCache

from config import ModelConfig


def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False, device=None, dtype=torch.float32):
    # Compute the trigonometric basis in fp32 for numerical stability, then cast at the end.
    base_dtype = torch.float32
    grid_h = torch.arange(grid_size_h, dtype=base_dtype, device=device)
    grid_w = torch.arange(grid_size_w, dtype=base_dtype, device=device)
    grid = torch.meshgrid(grid_w, grid_h, indexing='xy')
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    expected = grid_size_h * grid_size_w
    if pos_embed.shape != (expected, embed_dim):
        raise AssertionError(
            f"2D sincos shape mismatch: got {tuple(pos_embed.shape)}, expected {(expected, embed_dim)}"
        )
    if cls_token:
        pos_embed = torch.cat([torch.zeros([1, embed_dim], dtype=base_dtype, device=device), pos_embed], dim=0)
    return pos_embed.to(dtype=dtype)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    # With indexing='xy', grid[0] is width/x and grid[1] is height/y.
    emb_x = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_y = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return torch.cat([emb_x, emb_y], dim=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=pos.dtype, device=pos.device)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = torch.einsum('m,d->md', pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    return torch.cat([emb_sin, emb_cos], dim=1)


def validate_2d_sincos_axis_convention(device=None):
    # With indexing='xy', width changes fastest in the flattened sequence while height stays fixed.
    pe = get_2d_sincos_pos_embed(8, 2, 3, device=device, dtype=torch.float32)
    if torch.allclose(pe[0, :4], pe[1, :4]):
        raise AssertionError("2D sincos axis convention mismatch: x component did not change across columns")
    if not torch.allclose(pe[0, 4:], pe[1, 4:]):
        raise AssertionError("2D sincos axis convention mismatch: y component changed across columns")
    if not torch.allclose(pe[0, :4], pe[3, :4]):
        raise AssertionError("2D sincos axis convention mismatch: x component changed across rows")
    if torch.allclose(pe[0, 4:], pe[3, 4:]):
        raise AssertionError("2D sincos axis convention mismatch: y component did not change across rows")


validate_2d_sincos_axis_convention()

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dtype):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(in_dim, out_dim, dtype=dtype)
        )
    def forward(self, x):
        return self.net(x)

@dataclass
class ModelPrediction:
    """
    Container for all output heads of the model.
    Shape Convention: [Batch, Time, 5 (Players), ...Dimensions...]
    Global heads (from strategy) are broadcasted or kept as [B, T, 1/5, ...] depending on usage.
    """
    # --- Action Heads ---
    mouse_x: torch.Tensor       # [B, T, 5, 33] (Logits)
    mouse_y: torch.Tensor       # [B, T, 5, 33] (Logits)
    keyboard_logits: torch.Tensor   # [B, T, 5, 32] (Logits)
    
    # --- Economy/Item Heads ---
    eco_buy_logits: torch.Tensor    # [B, T, 5, 256] (Logits) - What to buy
    eco_purchase_logits: torch.Tensor # [B, T, 5, 1] (Logits) - Whether to buy
    active_weapon_logits: torch.Tensor # [B, T, 5, 128] (Logits)
    
    # --- Stats Heads ---
    health_logits: torch.Tensor     # [B, T, 5, 11] (Logits)
    armor_logits: torch.Tensor      # [B, T, 5, 11] (Logits)
    money_logits: torch.Tensor      # [B, T, 5, 33] (Logits)
    
    # --- Spatial Heads (Player) ---
    player_pos_x: torch.Tensor      # [B, T, 5, bins_x] (Logits)
    player_pos_y: torch.Tensor      # [B, T, 5, bins_y] (Logits)
    player_pos_z: torch.Tensor      # [B, T, 5, bins_z] (Logits)

    # --- Global/Enemy Heads (Strategy Token) ---
    # Global state
    round_state_logits: torch.Tensor # [B, T, 1, 5] (Logits)
    round_num_logits: torch.Tensor   # [B, T, 1, 31] (Logits)
    team_alive_logits: torch.Tensor  # [B, T, 1, 6] (Logits)
    enemy_alive_logits: torch.Tensor # [B, T, 1, 6] (Logits)
    
    # Enemy Spatial (Expanded to 5 enemies)
    enemy_pos_x: torch.Tensor       # [B, T, 5, bins_x] (Logits)
    enemy_pos_y: torch.Tensor       # [B, T, 5, bins_y] (Logits)
    enemy_pos_z: torch.Tensor       # [B, T, 5, bins_z] (Logits)


class RollingDynamicCache(DynamicCache):
    """
    Dynamic cache that keeps track of the absolute token offset of the first
    retained key/value. This allows left-cropping while preserving causal mask
    positions for subsequent autoregressive decoding.
    """

    def __init__(self, *args, position_offset: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.position_offset = int(position_offset)

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        if layer_idx >= len(self.layers):
            return cache_position.shape[0], self.position_offset
        return self.layers[layer_idx].get_seq_length() + cache_position.shape[0], self.position_offset

    def crop(self, max_length: int):
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        seq_length = self.get_seq_length()
        if seq_length <= max_length:
            return

        trim = seq_length - max_length
        for layer in self.layers:
            max_cache_shape = layer.get_max_cache_shape()
            if max_cache_shape not in (-1, None):
                raise ValueError("RollingDynamicCache only supports non-sliding cache layers")
            if not getattr(layer, "is_initialized", False) or layer.keys.numel() == 0:
                continue
            layer.keys = layer.keys[..., trim:, :]
            layer.values = layer.values[..., trim:, :]
        self.position_offset += trim


@dataclass
class AutoregressiveState:
    split_caches: list[Optional[Cache]]
    total_tokens_processed: int = 0
    max_cache_frames: Optional[int] = None
    tokens_per_frame: int = 6

    @property
    def cached_tokens(self) -> int:
        for cache in self.split_caches:
            if cache is not None:
                return cache.get_seq_length()
        return 0

    @property
    def total_frames_processed(self) -> int:
        return self.total_tokens_processed // self.tokens_per_frame

    @property
    def cached_frames(self) -> int:
        return self.cached_tokens // self.tokens_per_frame


class _CrossAttentionBlock(nn.Module):
    def __init__(self, latent_dim: int, input_dim: int, num_heads: int, dropout: float, dtype):
        super().__init__()
        self.q_norm = nn.LayerNorm(latent_dim, dtype=dtype)
        self.kv_norm = nn.LayerNorm(input_dim, dtype=dtype)
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=input_dim,
            vdim=input_dim,
            dtype=dtype,
        )
        self.out_norm = nn.LayerNorm(latent_dim, dtype=dtype)

    def forward(self, latents, inputs, inputs_key_padding_mask=None, attn_mask=None):
        q = self.q_norm(latents)
        kv = self.kv_norm(inputs)
        out, _ = self.attn(q, kv, kv, key_padding_mask=inputs_key_padding_mask, attn_mask=attn_mask, need_weights=False)
        latents = latents + out
        return self.out_norm(latents)


class _SelfAttentionFFNBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float, dtype):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, dtype=dtype)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True, dtype=dtype)
        self.norm2 = nn.LayerNorm(dim, dtype=dtype)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden, dim, dtype=dtype),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + out
        x = x + self.mlp(self.norm2(x))
        return x


class QFormerPatchCompressor(nn.Module):
    """
    Blip2QFormer-based compressor.
    Input: [B, N_patches, D_vision]
    Output: [B, N_queries, D_qformer]
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        
        vision_dim = cfg.vision_hidden_size
        qformer_dim = cfg.qformer_hidden_size
        num_queries = cfg.qformer_num_queries
        num_patches = cfg.vision_num_patches
        
        # 1. Instantiate the Configuration
        config = Blip2QFormerConfig(
            hidden_size=qformer_dim,
            encoder_hidden_size=vision_dim,
            num_hidden_layers=cfg.qformer_num_hidden_layers,
            num_attention_heads=cfg.qformer_num_attention_heads,
            intermediate_size=cfg.qformer_intermediate_size,
        )
        
        # 2. Instantiate the Model
        self.qformer = Blip2QFormerModel(config)
        self.qformer.to(dtype=cfg.dtype)
        
        # 3. Create the Learned Query Tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, qformer_dim, dtype=cfg.dtype))
        
        # 4. Learned Position Embeddings for Patches
        # Simple learned embeddings added to input patches
        self.patch_pos_embed = nn.Parameter(torch.randn(1, num_patches, vision_dim, dtype=cfg.dtype) * 0.02)

    def forward(self, patch_tokens: torch.Tensor):
        # patch_tokens: [B, N, D]
        B, N, D = patch_tokens.shape
        
        # Add position embeddings to tokens (patches + extra tokens like CLS/Registers)
        # Slice or broadcast position embeddings to match input token count N
        if N <= self.patch_pos_embed.shape[1]:
            patch_tokens = patch_tokens + self.patch_pos_embed[:, :N, :]
        else:
            # Fallback if N is somehow larger than our pre-allocated embeddings
            patch_tokens[:, :self.patch_pos_embed.shape[1], :] = \
                patch_tokens[:, :self.patch_pos_embed.shape[1], :] + self.patch_pos_embed
        
        # Expand learned queries
        query_embeds = self.query_tokens.expand(B, -1, -1)
        
        # Forward Pass
        outputs = self.qformer(
            query_embeds=query_embeds,
            encoder_hidden_states=patch_tokens,
        )
        
        return outputs.last_hidden_state


class PerceiverPatchCompressor(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        dt = cfg.dtype

        self.num_latents = cfg.num_perceiver_queries
        self.latent_dim = cfg.perceiver_hidden_size
        self.input_dim = cfg.vision_hidden_size
        self.num_blocks = cfg.patch_compressor_num_blocks
        self.num_self_attends_per_block = cfg.patch_compressor_self_attends_per_block
        self.mlp_ratio = cfg.patch_compressor_mlp_ratio
        self.dropout = cfg.patch_compressor_dropout
        self.patch_grid_h = cfg.perceiver_patch_grid_h
        self.patch_grid_w = cfg.perceiver_patch_grid_w
        self.num_patch_positions = self.patch_grid_h * self.patch_grid_w
        self.pos_embedding = cfg.perceiver_pos_embedding

        self.latents = nn.Parameter(torch.randn(1, self.num_latents, self.latent_dim, dtype=dt) * 0.02)
        self.grid_h = cfg.perceiver_grid_h
        self.grid_w = cfg.perceiver_grid_w
        self.global_count = cfg.perceiver_global_count
        self.use_spatial_mask = False
        if self.grid_h > 0 and self.grid_w > 0:
            expected_total = self.grid_h * self.grid_w + self.global_count
            if expected_total == self.num_latents:
                self.use_spatial_mask = True
                mask = self._generate_spatial_mask(self.patch_grid_h, self.patch_grid_w)
                self.register_buffer("spatial_mask", mask, persistent=False)

        if self.pos_embedding == "learned":
            self.patch_pos_embed = nn.Parameter(
                torch.randn(1, self.num_patch_positions, self.input_dim, dtype=dt) * 0.02
            )
            if self.use_spatial_mask:
                num_spatial = self.grid_h * self.grid_w
                self.latent_pos_embed = nn.Parameter(
                    torch.randn(1, num_spatial, self.latent_dim, dtype=dt) * 0.02
                )

        self._validate_config()

        self.cross_blocks = nn.ModuleList([
            _CrossAttentionBlock(self.latent_dim, self.input_dim, cfg.perceiver_heads, self.dropout, dt)
            for _ in range(self.num_blocks)
        ])
        self.self_tower_per_block = nn.ModuleList([
            nn.ModuleList([
                _SelfAttentionFFNBlock(self.latent_dim, cfg.perceiver_heads, self.mlp_ratio, self.dropout, dt)
                for _ in range(self.num_self_attends_per_block)
            ])
            for _ in range(self.num_blocks)
        ])

    def _validate_config(self):
        if self.patch_grid_h <= 0 or self.patch_grid_w <= 0:
            raise ValueError(
                f"Perceiver patch grid must be positive, got {self.patch_grid_h}x{self.patch_grid_w}"
            )
        if self.num_patch_positions <= 0:
            raise ValueError("Perceiver patch grid must contain at least one position")
        if self.use_spatial_mask:
            if self.patch_grid_h % self.grid_h != 0 or self.patch_grid_w % self.grid_w != 0:
                raise ValueError(
                    "Perceiver spatial mask requires patch grid to divide evenly by latent grid: "
                    f"patch={self.patch_grid_h}x{self.patch_grid_w}, latent={self.grid_h}x{self.grid_w}"
                )

    def _generate_spatial_mask(self, height: int, width: int):
        num_patches = height * width
        num_local = self.grid_h * self.grid_w
        total_latents = num_local + self.global_count
        mask = torch.full((total_latents, num_patches), float('-inf'))
        block_h = height // self.grid_h
        block_w = width // self.grid_w

        for i in range(self.grid_h):
            for j in range(self.grid_w):
                latent_idx = i * self.grid_w + j
                row_start = i * block_h
                row_end = (i + 1) * block_h
                col_start = j * block_w
                col_end = (j + 1) * block_w
                for r in range(row_start, row_end):
                    for c in range(col_start, col_end):
                        patch_idx = r * width + c
                        if patch_idx < num_patches:
                            mask[latent_idx, patch_idx] = 0.0

        for k in range(self.global_count):
            mask[num_local + k, :] = 0.0
        return mask

    def _apply_latent_positional_embedding(self, latents: torch.Tensor) -> torch.Tensor:
        if not self.use_spatial_mask or self.pos_embedding == "none":
            return latents

        num_spatial = self.grid_h * self.grid_w
        if self.pos_embedding == "learned":
            spatial_latents = latents[:, :num_spatial, :] + self.latent_pos_embed
        else:
            pe_latents = get_2d_sincos_pos_embed(
                self.latent_dim,
                self.grid_h,
                self.grid_w,
                cls_token=False,
                device=latents.device,
                dtype=latents.dtype,
            )
            spatial_latents = latents[:, :num_spatial, :] + pe_latents.unsqueeze(0)
        global_latents = latents[:, num_spatial:, :]
        return torch.cat([spatial_latents, global_latents], dim=1)

    def _apply_patch_positional_embedding(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if self.pos_embedding == "none":
            return patch_tokens

        _, num_tokens, _ = patch_tokens.shape
        if num_tokens < self.num_patch_positions:
            raise AssertionError(
                f"Patch token count {num_tokens} is smaller than configured patch grid "
                f"{self.num_patch_positions} ({self.patch_grid_h}x{self.patch_grid_w})"
            )

        prefix_len = num_tokens - self.num_patch_positions
        if self.pos_embedding == "learned":
            patch_pos = self.patch_pos_embed
        else:
            patch_pos = get_2d_sincos_pos_embed(
                self.input_dim,
                self.patch_grid_h,
                self.patch_grid_w,
                cls_token=False,
                device=patch_tokens.device,
                dtype=patch_tokens.dtype,
            ).unsqueeze(0)

        if prefix_len > 0:
            patches = patch_tokens[:, prefix_len:, :] + patch_pos
            return torch.cat([patch_tokens[:, :prefix_len, :], patches], dim=1)
        return patch_tokens + patch_pos

    def _get_attention_mask(self, num_tokens: int, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if not self.use_spatial_mask:
            return None
        if num_tokens < self.num_patch_positions:
            raise AssertionError(
                f"Spatial mask requires at least {self.num_patch_positions} tokens, got {num_tokens}"
            )

        mask = self.spatial_mask
        prefix_len = num_tokens - self.num_patch_positions
        if prefix_len > 0:
            prefix = torch.zeros((mask.shape[0], prefix_len), device=mask.device, dtype=mask.dtype)
            mask = torch.cat([prefix, mask], dim=1)
        if mask.shape != (self.num_latents, num_tokens):
            raise AssertionError(
                f"Attention mask shape mismatch: got {tuple(mask.shape)}, expected {(self.num_latents, num_tokens)}"
            )
        return mask.to(dtype=dtype)

    def forward(self, patch_tokens: torch.Tensor):
        latents = self.latents.expand(patch_tokens.shape[0], -1, -1)
        latents = self._apply_latent_positional_embedding(latents)
        patch_tokens = self._apply_patch_positional_embedding(patch_tokens)
        attn_mask = self._get_attention_mask(patch_tokens.shape[1], latents.dtype)

        for block_idx, cross in enumerate(self.cross_blocks):
            latents = cross(latents, patch_tokens, attn_mask=attn_mask)
            for layer in self.self_tower_per_block[block_idx]:
                latents = layer(latents)
        return latents


class PatchCompressor(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.compressor_type == "perceiver":
            self.impl = PerceiverPatchCompressor(cfg)
        else:
            self.impl = QFormerPatchCompressor(cfg)

    def forward(self, patch_tokens: torch.Tensor):
        return self.impl(patch_tokens)


class GameVideoEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.vision_processor = AutoImageProcessor.from_pretrained(
            cfg.vision_model_name,
            trust_remote_code=True,
        )
        self.vision = AutoModel.from_pretrained(
            cfg.vision_model_name,
            trust_remote_code=True,
            dtype=cfg.dtype,
        )
        self.vision.eval()
        for p in self.vision.parameters():
            p.requires_grad = False

        self.compressor = PatchCompressor(cfg)
        
        
    def _forward_vision(self, chunk_cpu: torch.Tensor) -> torch.Tensor:
        # chunk_cpu: [N, C, H, W] (likely uint8 or float32 on CPU)
        
        device = self.vision.device
        dtype = self.cfg.dtype
        
        with torch.no_grad():
            # Run processor on the chunk (it handles moving to tensor/normalization)
            # We assume chunk_cpu is proper input for the processor.
            proc = self.vision_processor(
                images=chunk_cpu,
                return_tensors="pt",
                data_format="channels_first",
                do_resize=False, 
                do_center_crop=False
            )
            pixel_values = proc["pixel_values"].to(device=device, dtype=dtype)
            
            # Run ViT (Frozen)
            vis_out = self.vision(pixel_values=pixel_values).last_hidden_state
        
        return vis_out

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, T, P=5, C, H, W] (Preferred on CPU)
        returns: [B, T, P=5, N_q, D_q]
        """
        B, T, P, C, H, W = images.shape
        flat_imgs = images.reshape(-1, C, H, W)  # [B*T*P, C, H, W]
        N = flat_imgs.shape[0]
        chunk_size = self.cfg.vision_chunk_size

        q_chunks = []
        
        # Checkpoint if training and enabled
        use_ckpt = self.training and self.cfg.gradient_checkpointing

        for i in range(0, N, chunk_size):
            chunk = flat_imgs[i : i + chunk_size]  # [n, C, H, W]
            
            # 1. Run ViT (Frozen, No Checkpoint needed as we want to cache output)
            vis_out = self._forward_vision(chunk)

            # 2. Run Compressor (Checkpointed if needed)
            if use_ckpt:
                # vis_out is already on device.
                # checkpoint requires inputs to have requires_grad=True OR use_reentrant=False
                q_out = checkpoint(self.compressor, vis_out, use_reentrant=False)
            else:
                q_out = self.compressor(vis_out)
                
            q_chunks.append(q_out)

        q_all = torch.cat(q_chunks, dim=0)  # [B*T*P, N_q, D_q]
        q_all = q_all.view(B, T, P, self.cfg.compressor_num_queries, self.cfg.compressor_hidden_size)
        return q_all

class GameAudioEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.model = DacModel.from_pretrained("descript/dac_24khz")
        self.model.to(dtype=cfg.dtype)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.output_dim = 256 #trust me bro

    def forward(self, audio: torch.Tensor, target_frames: int) -> torch.Tensor:
        """
        audio: [B, P, C=2, S]
        """
        B, P, C, S = audio.shape
        flat_audio = audio.reshape(B * P * C, 1, S)
        N = flat_audio.shape[0]
        chunk_size = self.cfg.audio_chunk_size
        feat_chunks = []
        with torch.no_grad():
            for i in range(0, N, chunk_size):
                chunk = flat_audio[i : i + chunk_size]  # [n, 1, S]
                feats = self.model.encode(chunk).projected_latents  # [n, 1024, Time]
                feat_chunks.append(feats)
                del feats, chunk

        features = torch.cat(feat_chunks, dim=0)  # [B*P*C, 1024, Time]
        del feat_chunks

        aligned = F.adaptive_avg_pool1d(features, target_frames)  # [B*P*C, 1024, T]
        aligned = aligned.permute(0, 2, 1)                        # [B*P*C, T, 1024]
        out = aligned.view(B, P, C, target_frames, -1)            # [B, P, C, T, H]
        out = out.permute(0, 3, 1, 2, 4)                          # [B, T, P, C, H]
        out = out.to(dtype=self.cfg.dtype) # to bf16
        return out


class ModelOutputHeads(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.llama_hidden_size
        dt = cfg.dtype
        
        # Player Heads
        self.mouse_x = MLP(d, cfg.mouse_bins_count, dtype=dt)
        self.mouse_y = MLP(d, cfg.mouse_bins_count, dtype=dt)
        self.keyboard = MLP(d, cfg.keyboard_dim, dtype=dt)
        
        self.eco_buy = MLP(d, cfg.eco_dim, dtype=dt)
        self.eco_purchase = MLP(d, 1, dtype=dt)
        self.active_weapon = MLP(d, cfg.weapon_dim, dtype=dt)
        
        self.health = MLP(d, cfg.health_bins, dtype=dt)
        self.armor = MLP(d, cfg.armor_bins, dtype=dt)
        self.money = MLP(d, cfg.money_bins, dtype=dt)
        
        self.player_pos_x = MLP(d, cfg.bins_x, dtype=dt)
        self.player_pos_y = MLP(d, cfg.bins_y, dtype=dt)
        self.player_pos_z = MLP(d, cfg.bins_z, dtype=dt)

        # Global/Strategy Heads
        self.round_state = MLP(d, cfg.round_state_dim, dtype=dt)
        self.round_num = MLP(d, cfg.round_num_bins, dtype=dt)
        self.team_alive = MLP(d, cfg.alive_bins, dtype=dt) 
        self.enemy_alive = MLP(d, cfg.alive_bins, dtype=dt)

        self.enemy_expander = MLP(d, 5 * d, dtype=dt)
        self.enemy_pos_x = MLP(d, cfg.bins_x, dtype=dt)
        self.enemy_pos_y = MLP(d, cfg.bins_y, dtype=dt)
        self.enemy_pos_z = MLP(d, cfg.bins_z, dtype=dt)

    def forward_player(self, x: torch.Tensor):
        # x: [N, D]
        return {
            "mouse_x": self.mouse_x(x),
            "mouse_y": self.mouse_y(x),
            "keyboard_logits": self.keyboard(x),
            "eco_buy_logits": self.eco_buy(x),
            "eco_purchase_logits": self.eco_purchase(x),
            "active_weapon_logits": self.active_weapon(x),
            "health_logits": self.health(x),
            "armor_logits": self.armor(x),
            "money_logits": self.money(x),
            "player_pos_x": self.player_pos_x(x),
            "player_pos_y": self.player_pos_y(x),
            "player_pos_z": self.player_pos_z(x),
        }

    def forward_global(self, x: torch.Tensor):
        # x: [N, D] where N = B*T
        out = {
            "round_state_logits": self.round_state(x),
            "round_num_logits": self.round_num(x),
            "team_alive_logits": self.team_alive(x),
            "enemy_alive_logits": self.enemy_alive(x),
        }
        B = x.shape[0] # Actually N
        # Expand 1 strategy token -> 5 enemy tokens per sequence
        # Here x is [B*T, D]
        # expander -> [B*T, 5*D] -> [B*T, 5, D]
        enemy_feats = self.enemy_expander(x).view(B, 5, -1)
        # Flatten: [B*T*5, D]
        flat_enemy = enemy_feats.view(-1, x.shape[-1])
        
        out["enemy_pos_x"] = self.enemy_pos_x(flat_enemy)
        out["enemy_pos_y"] = self.enemy_pos_y(flat_enemy)
        out["enemy_pos_z"] = self.enemy_pos_z(flat_enemy)
        return out

class GamePredictorBackbone(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.video = GameVideoEncoder(cfg)
        self.audio = GameAudioEncoder(cfg)
        
        # Audio Projection
        # Project audio to match vision dimension for concatenation
        self.audio_proj = nn.Linear(self.audio.output_dim, cfg.compressor_hidden_size, dtype=cfg.dtype)
        
        # Initial Queries
        # Use Embedding(1, D) to avoid FSDP issues with standalone parameters
        self.player_query = nn.Embedding(1, cfg.llama_hidden_size, dtype=cfg.dtype)
        self.strat_query = nn.Embedding(1, cfg.llama_hidden_size, dtype=cfg.dtype)
        
        # Split Llama
        assert cfg.llama_layers % cfg.backbone_splits == 0, "llama_layers must be divisible by backbone_splits"
        layers_per_split = cfg.llama_layers // cfg.backbone_splits
        
        self.blocks = nn.ModuleList()
        
        llama_conf = LlamaConfig(
            vocab_size=1, 
            hidden_size=cfg.llama_hidden_size,
            intermediate_size=cfg.llama_intermediate, 
            num_hidden_layers=layers_per_split,
            num_attention_heads=cfg.llama_heads, 
            num_key_value_heads=cfg.llama_kv_heads,
            max_position_embeddings=cfg.llama_max_pos_embeddings,
            use_cache=False, 
            dtype=cfg.dtype,
            attn_implementation="flash_attention_2" if cfg.use_flash_attention else "eager"
        )

        for i in range(cfg.backbone_splits):
            cross = _CrossAttentionBlock(
                latent_dim=cfg.llama_hidden_size,
                input_dim=cfg.compressor_hidden_size,
                num_heads=cfg.llama_heads, 
                dropout=0.0,
                dtype=cfg.dtype
            )
            
            part = LlamaModel(llama_conf).to(dtype=cfg.dtype)
            if cfg.gradient_checkpointing:
                part.gradient_checkpointing_enable()
            
            # Remove norm for all but last
            if i < cfg.backbone_splits - 1:
                part.norm = nn.Identity()
                
            self.blocks.append(nn.ModuleDict({"cross": cross, "llama": part}))
            
        self.heads = ModelOutputHeads(cfg)

    def _prepare_context(self, images: torch.Tensor, audio: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        B, T, P = images.shape[:3]
        vid_feats = self.video(images)  # [B, T, P, N_q, D]
        aud_feats = self.audio(audio, T)  # [B, T, P, C, H]
        aud_feats = self.audio_proj(aud_feats)  # [B, T, P, C, D]
        context = torch.cat([vid_feats, aud_feats], dim=3)
        context_flat = context.view(B * T * P, -1, self.cfg.compressor_hidden_size)
        return context_flat, B, T

    def _init_hidden_queries(self, B: int, T: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        dummy_idx = torch.zeros(1, dtype=torch.long, device=device)
        p_vec = self.player_query(dummy_idx).view(1, 1, 1, -1)
        s_vec = self.strat_query(dummy_idx).view(1, 1, 1, -1)
        p_q = p_vec.expand(B, T, 5, -1)
        s_q = s_vec.expand(B, T, 1, -1)
        return (
            p_q.reshape(B * T * 5, 1, self.cfg.llama_hidden_size),
            s_q.reshape(B * T, 1, self.cfg.llama_hidden_size),
        )

    def _assemble_sequence(
        self,
        current_hidden_p: torch.Tensor,
        current_hidden_s: torch.Tensor,
        B: int,
        T: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p_view = current_hidden_p.view(B, T, 5, -1)
        s_view = current_hidden_s.view(B, T, 1, -1)
        seq = torch.cat([p_view, s_view], dim=2)
        return p_view, s_view, seq.view(B, T * self.cfg.tokens_per_frame, -1)

    def _build_prediction(
        self,
        p_view: torch.Tensor,
        s_view: torch.Tensor,
        B: int,
        T: int,
    ) -> dict[str, torch.Tensor]:
        p_flat = p_view.reshape(-1, self.cfg.llama_hidden_size)
        s_flat = s_view.reshape(-1, self.cfg.llama_hidden_size)

        p_preds = self.heads.forward_player(p_flat)
        s_preds = self.heads.forward_global(s_flat)

        def rs(x, is_player=True):
            if is_player:
                return x.view(B, T, 5, *x.shape[1:])
            if x.shape[0] == B * T * 5:
                return x.view(B, T, 5, *x.shape[1:])
            return x.view(B, T, *x.shape[1:])

        mp = ModelPrediction(
            mouse_x=rs(p_preds["mouse_x"]),
            mouse_y=rs(p_preds["mouse_y"]),
            keyboard_logits=rs(p_preds["keyboard_logits"]),
            eco_buy_logits=rs(p_preds["eco_buy_logits"]),
            eco_purchase_logits=rs(p_preds["eco_purchase_logits"]),
            active_weapon_logits=rs(p_preds["active_weapon_logits"]),
            health_logits=rs(p_preds["health_logits"]),
            armor_logits=rs(p_preds["armor_logits"]),
            money_logits=rs(p_preds["money_logits"]),
            player_pos_x=rs(p_preds["player_pos_x"]),
            player_pos_y=rs(p_preds["player_pos_y"]),
            player_pos_z=rs(p_preds["player_pos_z"]),
            enemy_pos_x=rs(s_preds["enemy_pos_x"], is_player=False),
            enemy_pos_y=rs(s_preds["enemy_pos_y"], is_player=False),
            enemy_pos_z=rs(s_preds["enemy_pos_z"], is_player=False),
            round_state_logits=rs(s_preds["round_state_logits"], is_player=False),
            round_num_logits=rs(s_preds["round_num_logits"], is_player=False),
            team_alive_logits=rs(s_preds["team_alive_logits"], is_player=False),
            enemy_alive_logits=rs(s_preds["enemy_alive_logits"], is_player=False),
        )
        return {k: getattr(mp, k) for k in mp.__dataclass_fields__}

    def init_autoregressive_state(self, *, max_cache_frames: Optional[int] = None) -> AutoregressiveState:
        if max_cache_frames is not None and max_cache_frames <= 0:
            raise ValueError("max_cache_frames must be positive when provided")
        return AutoregressiveState(
            split_caches=[None for _ in range(len(self.blocks))],
            total_tokens_processed=0,
            max_cache_frames=max_cache_frames,
            tokens_per_frame=self.cfg.tokens_per_frame,
        )

    def reset_autoregressive_state(self, *, max_cache_frames: Optional[int] = None) -> AutoregressiveState:
        return self.init_autoregressive_state(max_cache_frames=max_cache_frames)

    def crop_autoregressive_state(
        self,
        state: AutoregressiveState,
        *,
        max_cache_frames: Optional[int] = None,
    ) -> AutoregressiveState:
        resolved_max_frames = state.max_cache_frames if max_cache_frames is None else max_cache_frames
        if resolved_max_frames is None:
            return state
        if resolved_max_frames <= 0:
            raise ValueError("max_cache_frames must be positive when provided")

        max_tokens = resolved_max_frames * self.cfg.tokens_per_frame
        for cache in state.split_caches:
            if cache is None:
                continue
            if not isinstance(cache, RollingDynamicCache):
                raise TypeError("autoregressive cropping expects RollingDynamicCache instances")
            cache.crop(max_tokens)
        state.max_cache_frames = resolved_max_frames
        return state

    def _forward_from_context_flat(self, context_flat: torch.Tensor, B: int, T: int) -> dict[str, torch.Tensor]:
        current_hidden_p, current_hidden_s = self._init_hidden_queries(B, T, context_flat.device)

        for block in self.blocks:
            p_out = block["cross"](current_hidden_p, context_flat)
            current_hidden_p = p_out

            _, _, seq_flat = self._assemble_sequence(current_hidden_p, current_hidden_s, B, T)
            llama_out = block["llama"](inputs_embeds=seq_flat).last_hidden_state

            out_frames = llama_out.view(B, T, self.cfg.tokens_per_frame, -1)
            p_view = out_frames[:, :, :5, :]
            s_view = out_frames[:, :, 5:, :]

            current_hidden_p = p_view.reshape(B * T * 5, 1, -1)
            current_hidden_s = s_view.reshape(B * T, 1, -1)

        return self._build_prediction(p_view, s_view, B, T)

    def _forward_step_from_context_flat(
        self,
        context_flat: torch.Tensor,
        B: int,
        T: int,
        state: Optional[AutoregressiveState] = None,
        *,
        max_cache_frames: Optional[int] = None,
    ) -> tuple[dict[str, torch.Tensor], AutoregressiveState]:
        current_hidden_p, current_hidden_s = self._init_hidden_queries(B, T, context_flat.device)

        if state is None:
            state = self.init_autoregressive_state(max_cache_frames=max_cache_frames)
        elif max_cache_frames is not None:
            state.max_cache_frames = max_cache_frames

        step_tokens = T * self.cfg.tokens_per_frame
        if state.max_cache_frames is not None:
            max_tokens = state.max_cache_frames * self.cfg.tokens_per_frame
            keep_previous_tokens = max(0, max_tokens - step_tokens)
            for cache in state.split_caches:
                if cache is None:
                    continue
                if not isinstance(cache, RollingDynamicCache):
                    raise TypeError("autoregressive cropping expects RollingDynamicCache instances")
                cache.crop(keep_previous_tokens)

        cache_position = torch.arange(
            state.total_tokens_processed,
            state.total_tokens_processed + step_tokens,
            device=context_flat.device,
            dtype=torch.long,
        )

        for block_idx, block in enumerate(self.blocks):
            current_hidden_p = block["cross"](current_hidden_p, context_flat)

            _, _, seq_flat = self._assemble_sequence(current_hidden_p, current_hidden_s, B, T)
            cache = state.split_caches[block_idx]
            if cache is None:
                cache = RollingDynamicCache(config=block["llama"].config)
            llama_outputs = block["llama"](
                inputs_embeds=seq_flat,
                past_key_values=cache,
                cache_position=cache_position,
                use_cache=True,
            )
            state.split_caches[block_idx] = llama_outputs.past_key_values

            out_frames = llama_outputs.last_hidden_state.view(B, T, self.cfg.tokens_per_frame, -1)
            p_view = out_frames[:, :, :5, :]
            s_view = out_frames[:, :, 5:, :]
            current_hidden_p = p_view.reshape(B * T * 5, 1, -1)
            current_hidden_s = s_view.reshape(B * T, 1, -1)

        state.total_tokens_processed += step_tokens
        self.crop_autoregressive_state(state)
        return self._build_prediction(p_view, s_view, B, T), state

    def forward_step(
        self,
        images: torch.Tensor,
        audio: torch.Tensor,
        state: Optional[AutoregressiveState] = None,
        *,
        max_cache_frames: Optional[int] = None,
    ) -> tuple[dict[str, torch.Tensor], AutoregressiveState]:
        if images.shape[1] != 1:
            raise ValueError(f"forward_step expects a single-frame step (T=1), got T={images.shape[1]}")

        context_flat, B, T = self._prepare_context(images, audio)
        return self._forward_step_from_context_flat(
            context_flat,
            B,
            T,
            state,
            max_cache_frames=max_cache_frames,
        )

    def forward(self, images: torch.Tensor, audio: torch.Tensor):
        context_flat, B, T = self._prepare_context(images, audio)
        return self._forward_from_context_flat(context_flat, B, T)
