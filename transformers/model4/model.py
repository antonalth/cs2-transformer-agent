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
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoImageProcessor, AutoModel, EncodecModel, LlamaConfig, LlamaModel
from transformers.cache_utils import DynamicCache

from config import ModelConfig


def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False, device=None, dtype=torch.float32):
    base_dtype = torch.float32
    grid_h = torch.arange(grid_size_h, dtype=base_dtype, device=device)
    grid_w = torch.arange(grid_size_w, dtype=base_dtype, device=device)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = torch.cat([torch.zeros([1, embed_dim], dtype=base_dtype, device=device), pos_embed], dim=0)
    return pos_embed.to(dtype=dtype)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_x = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_y = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return torch.cat([emb_x, emb_y], dim=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=pos.dtype, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    return torch.cat([emb_sin, emb_cos], dim=1)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dtype: torch.dtype):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(in_dim, out_dim, dtype=dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ModelPrediction:
    mouse_x: torch.Tensor            # [B, T, mouse_bins]
    mouse_y: torch.Tensor            # [B, T, mouse_bins]
    keyboard_logits: torch.Tensor    # [B, T, keyboard_dim]
    eco_buy_logits: torch.Tensor     # [B, T, eco_dim]
    eco_purchase_logits: torch.Tensor # [B, T, 1]


class RollingDynamicCache(DynamicCache):
    def __init__(self, *args, position_offset: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.position_offset = int(position_offset)

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        if layer_idx >= len(self.layers):
            return cache_position.shape[0], self.position_offset
        return self.layers[layer_idx].get_seq_length() + cache_position.shape[0], self.position_offset

    def crop(self, max_length: int):
        seq_length = self.get_seq_length()
        if seq_length <= max_length:
            return
        trim = seq_length - max_length
        for layer in self.layers:
            if not getattr(layer, "is_initialized", False) or layer.keys.numel() == 0:
                continue
            layer.keys = layer.keys[..., trim:, :]
            layer.values = layer.values[..., trim:, :]
        self.position_offset += trim


@dataclass
class AutoregressiveState:
    cache: Optional[RollingDynamicCache] = None
    total_frames_processed: int = 0
    max_cache_frames: Optional[int] = None
    prev_keyboard_mask: Optional[torch.Tensor] = None  # [B]
    prev_mouse_x_bin: Optional[torch.Tensor] = None    # [B]
    prev_mouse_y_bin: Optional[torch.Tensor] = None    # [B]
    prev_eco_buy_idx: Optional[torch.Tensor] = None    # [B], eco_dim means no-buy
    batch_size: Optional[int] = None

    @property
    def cached_frames(self) -> int:
        if self.cache is None:
            return 0
        return self.cache.get_seq_length()


class CrossAttentionBlock(nn.Module):
    def __init__(self, latent_dim: int, input_dim: int, num_heads: int, dtype: torch.dtype):
        super().__init__()
        self.q_norm = nn.LayerNorm(latent_dim, dtype=dtype)
        self.kv_norm = nn.LayerNorm(input_dim, dtype=dtype)
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            batch_first=True,
            kdim=input_dim,
            vdim=input_dim,
            dtype=dtype,
        )
        self.out_norm = nn.LayerNorm(latent_dim, dtype=dtype)

    def forward(self, latents: torch.Tensor, inputs: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.q_norm(latents)
        kv = self.kv_norm(inputs)
        out, _ = self.attn(q, kv, kv, attn_mask=attn_mask, need_weights=False)
        return self.out_norm(latents + out)


class SelfAttentionFFNBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float, dtype: torch.dtype):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + out
        x = x + self.mlp(self.norm2(x))
        return x


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

        self.cross_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(self.latent_dim, self.input_dim, cfg.perceiver_heads, dt)
                for _ in range(self.num_blocks)
            ]
        )
        self.self_tower_per_block = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        SelfAttentionFFNBlock(self.latent_dim, cfg.perceiver_heads, self.mlp_ratio, self.dropout, dt)
                        for _ in range(self.num_self_attends_per_block)
                    ]
                )
                for _ in range(self.num_blocks)
            ]
        )

    def _generate_spatial_mask(self, height: int, width: int):
        num_patches = height * width
        num_local = self.grid_h * self.grid_w
        total_latents = num_local + self.global_count
        mask = torch.full((total_latents, num_patches), float("-inf"))
        block_h = height // self.grid_h
        block_w = width // self.grid_w

        for i in range(self.grid_h):
            for j in range(self.grid_w):
                latent_idx = i * self.grid_w + j
                row_start = i * block_h
                row_end = (i + 1) * block_h
                col_start = j * block_w
                col_end = (j + 1) * block_w
                for row in range(row_start, row_end):
                    for col in range(col_start, col_end):
                        patch_idx = row * width + col
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
        mask = self.spatial_mask
        prefix_len = num_tokens - self.num_patch_positions
        if prefix_len > 0:
            prefix = torch.zeros((mask.shape[0], prefix_len), device=mask.device, dtype=mask.dtype)
            mask = torch.cat([prefix, mask], dim=1)
        return mask.to(dtype=dtype)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        latents = self.latents.expand(patch_tokens.shape[0], -1, -1)
        latents = self._apply_latent_positional_embedding(latents)
        patch_tokens = self._apply_patch_positional_embedding(patch_tokens)
        attn_mask = self._get_attention_mask(patch_tokens.shape[1], latents.dtype)
        for block_idx, cross in enumerate(self.cross_blocks):
            latents = cross(latents, patch_tokens, attn_mask=attn_mask)
            for layer in self.self_tower_per_block[block_idx]:
                latents = layer(latents)
        return latents


class GameVideoEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.output_dim = cfg.perceiver_hidden_size
        self.num_queries = cfg.num_perceiver_queries
        self.vision_processor = AutoImageProcessor.from_pretrained(cfg.vision_model_name, trust_remote_code=True)
        self.vision = AutoModel.from_pretrained(cfg.vision_model_name, trust_remote_code=True, dtype=cfg.dtype)
        self.vision.eval()
        for param in self.vision.parameters():
            param.requires_grad = False

        self.compressor = PerceiverPatchCompressor(cfg)
        self.fast_preprocess_enabled = False

        do_rescale = bool(getattr(self.vision_processor, "do_rescale", False))
        do_normalize = bool(getattr(self.vision_processor, "do_normalize", False))
        image_mean = getattr(self.vision_processor, "image_mean", [0.0, 0.0, 0.0])
        image_std = getattr(self.vision_processor, "image_std", [1.0, 1.0, 1.0])
        rescale_factor = float(getattr(self.vision_processor, "rescale_factor", 1.0)) if do_rescale else 1.0

        self.register_buffer("_vision_rescale_factor", torch.tensor(rescale_factor, dtype=torch.float32), persistent=False)
        self.register_buffer(
            "_vision_image_mean",
            torch.tensor(image_mean, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_vision_image_std",
            torch.tensor(image_std, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=False,
        )
        self._vision_do_rescale = do_rescale
        self._vision_do_normalize = do_normalize

    def set_fast_preprocess(self, enabled: bool) -> None:
        self.fast_preprocess_enabled = bool(enabled)

    def _vision_device_and_dtype(self) -> tuple[torch.device, torch.dtype]:
        param = next(self.vision.parameters())
        return param.device, self.cfg.dtype

    def _fast_preprocess_pixel_values(self, chunk: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pixel_values = chunk.to(device=device, dtype=torch.float32, non_blocking=True)
        if self._vision_do_rescale:
            pixel_values = pixel_values * self._vision_rescale_factor
        if self._vision_do_normalize:
            pixel_values = (pixel_values - self._vision_image_mean) / self._vision_image_std
        return pixel_values.to(dtype=dtype)

    def _forward_vision(self, chunk_cpu: torch.Tensor) -> torch.Tensor:
        device, dtype = self._vision_device_and_dtype()
        with torch.no_grad():
            if self.fast_preprocess_enabled:
                pixel_values = self._fast_preprocess_pixel_values(chunk_cpu, device=device, dtype=dtype)
            else:
                proc = self.vision_processor(
                    images=chunk_cpu,
                    return_tensors="pt",
                    data_format="channels_first",
                    do_resize=False,
                    do_center_crop=False,
                )
                pixel_values = proc["pixel_values"].to(device=device, dtype=dtype)
            return self.vision(pixel_values=pixel_values).last_hidden_state

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, T, C, H, W]
        returns: [B, T, N_q, D_q]
        """
        B, T, C, H, W = images.shape
        flat_images = images.reshape(B * T, C, H, W)
        chunk_size = self.cfg.vision_chunk_size
        outputs: list[torch.Tensor] = []
        use_ckpt = self.training and self.cfg.gradient_checkpointing

        for start in range(0, flat_images.shape[0], chunk_size):
            chunk = flat_images[start : start + chunk_size]
            vision_out = self._forward_vision(chunk)
            if use_ckpt:
                compressed = checkpoint(self.compressor, vision_out, use_reentrant=False)
            else:
                compressed = self.compressor(vision_out)
            outputs.append(compressed)

        features = torch.cat(outputs, dim=0)
        return features.view(B, T, self.num_queries, self.output_dim)


class GameAudioEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.model = EncodecModel.from_pretrained("facebook/encodec_24khz")
        self.model.to(dtype=cfg.dtype)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.encoder_hidden_size = int(self.model.config.hidden_size)
        self.output_dim = 256
        self.output_proj = nn.Linear(self.encoder_hidden_size, self.output_dim, dtype=cfg.dtype)

    def _normalize_audio(self, chunk: torch.Tensor) -> torch.Tensor:
        if not getattr(self.model.config, "normalize", False):
            return chunk
        scale = chunk.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8
        return chunk / scale

    def forward(self, audio: torch.Tensor, target_frames: int) -> torch.Tensor:
        """
        audio: [B, 2, S]
        returns: [B, T, 2, H]
        """
        B, C, S = audio.shape
        dtype = self.output_proj.weight.dtype
        device = self.output_proj.weight.device
        flat_audio = audio.reshape(B * C, 1, S).to(device=device, dtype=dtype)
        with torch.no_grad():
            normalized = self._normalize_audio(flat_audio)
            features = self.model.encoder(normalized)
        aligned = F.adaptive_avg_pool1d(features, target_frames)
        aligned = aligned.permute(0, 2, 1)
        projected = self.output_proj(aligned)
        return projected.view(B, C, target_frames, -1).permute(0, 2, 1, 3).to(dtype=self.cfg.dtype)


class ModelOutputHeads(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.llama_hidden_size
        dt = cfg.dtype
        self.mouse_x = MLP(d, cfg.mouse_bins_count, dtype=dt)
        self.mouse_y = MLP(d, cfg.mouse_bins_count, dtype=dt)
        self.keyboard = MLP(d, cfg.keyboard_dim, dtype=dt)
        self.eco_buy = MLP(d, cfg.eco_dim, dtype=dt)
        self.eco_purchase = MLP(d, 1, dtype=dt)

    def forward(self, x: torch.Tensor) -> ModelPrediction:
        return ModelPrediction(
            mouse_x=self.mouse_x(x),
            mouse_y=self.mouse_y(x),
            keyboard_logits=self.keyboard(x),
            eco_buy_logits=self.eco_buy(x),
            eco_purchase_logits=self.eco_purchase(x),
        )


class GamePredictorBackbone(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tokens_per_frame = 1
        self.video = GameVideoEncoder(cfg)
        self.audio = GameAudioEncoder(cfg)
        self.context_dim = self.video.output_dim
        self.audio_proj = nn.Linear(self.audio.output_dim, self.context_dim, dtype=cfg.dtype)

        self.keyboard_action_embed = nn.Embedding(cfg.keyboard_dim, cfg.llama_hidden_size, dtype=cfg.dtype)
        self.mouse_x_prev_embed = nn.Embedding(cfg.mouse_bins_count, cfg.llama_hidden_size, dtype=cfg.dtype)
        self.mouse_y_prev_embed = nn.Embedding(cfg.mouse_bins_count, cfg.llama_hidden_size, dtype=cfg.dtype)
        self.eco_buy_prev_embed = nn.Embedding(cfg.eco_dim + 1, cfg.llama_hidden_size, dtype=cfg.dtype)
        self.action_sos = nn.Embedding(1, cfg.llama_hidden_size, dtype=cfg.dtype)
        self.action_seed_mlp = nn.Sequential(
            nn.Linear(2 * cfg.llama_hidden_size, cfg.adapter_hidden_dim, dtype=cfg.dtype),
            nn.GELU(),
            nn.Linear(cfg.adapter_hidden_dim, cfg.llama_hidden_size, dtype=cfg.dtype),
        )

        self.cross = CrossAttentionBlock(
            latent_dim=cfg.llama_hidden_size,
            input_dim=self.context_dim,
            num_heads=cfg.llama_heads,
            dtype=cfg.dtype,
        )

        llama_conf = LlamaConfig(
            vocab_size=1,
            hidden_size=cfg.llama_hidden_size,
            intermediate_size=cfg.llama_intermediate,
            num_hidden_layers=cfg.llama_layers,
            num_attention_heads=cfg.llama_heads,
            num_key_value_heads=cfg.llama_kv_heads,
            max_position_embeddings=cfg.llama_max_pos_embeddings,
            use_cache=False,
            dtype=cfg.dtype,
            attn_implementation="flash_attention_2" if cfg.use_flash_attention else "eager",
        )
        self.backbone = LlamaModel(llama_conf).to(dtype=cfg.dtype)
        if cfg.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        self.heads = ModelOutputHeads(cfg)

    def _prepare_context(self, images: torch.Tensor, audio: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        B, T = images.shape[:2]
        video_features = self.video(images)
        audio_features = self.audio(audio, T)
        audio_features = self.audio_proj(audio_features)
        context = torch.cat([video_features, audio_features], dim=2)
        return context, B, T

    def _keyboard_embedding(self, keyboard_mask: torch.Tensor) -> torch.Tensor:
        bits = torch.arange(self.cfg.keyboard_dim, device=keyboard_mask.device, dtype=torch.long)
        active = ((keyboard_mask.long().unsqueeze(-1) >> bits) & 1).to(dtype=self.cfg.dtype)
        return active @ self.keyboard_action_embed.weight

    def _action_seed(
        self,
        keyboard_mask: torch.Tensor,
        mouse_x_bin: torch.Tensor,
        mouse_y_bin: torch.Tensor,
        eco_buy_idx: torch.Tensor,
    ) -> torch.Tensor:
        keyboard_vec = self._keyboard_embedding(keyboard_mask)
        mouse_vec = self.mouse_x_prev_embed(mouse_x_bin.long()) + self.mouse_y_prev_embed(mouse_y_bin.long())
        eco_idx = eco_buy_idx.long().clamp(0, self.cfg.eco_dim)
        eco_vec = self.eco_buy_prev_embed(eco_idx)
        return self.action_seed_mlp(torch.cat([keyboard_vec, mouse_vec], dim=-1)) + eco_vec

    def _sos_seed(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        dummy = torch.zeros(shape, dtype=torch.long, device=device)
        return self.action_sos(dummy)

    def _build_training_seeds(
        self,
        B: int,
        T: int,
        device: torch.device,
        prev_keyboard_mask: Optional[torch.Tensor],
        prev_mouse_x_bin: Optional[torch.Tensor],
        prev_mouse_y_bin: Optional[torch.Tensor],
        prev_eco_buy_idx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        sos = self._sos_seed((B, T), device)
        if (
            prev_keyboard_mask is None
            or prev_mouse_x_bin is None
            or prev_mouse_y_bin is None
            or prev_eco_buy_idx is None
        ):
            return sos
        seeds = self._action_seed(prev_keyboard_mask, prev_mouse_x_bin, prev_mouse_y_bin, prev_eco_buy_idx)
        seeds = seeds.to(dtype=self.cfg.dtype)
        seeds[:, 0, :] = sos[:, 0, :]
        return seeds

    def _predict_from_hidden(self, hidden: torch.Tensor, B: int, T: int) -> dict[str, torch.Tensor]:
        pred = self.heads(hidden.reshape(B * T, -1))
        return {
            "mouse_x": pred.mouse_x.view(B, T, -1),
            "mouse_y": pred.mouse_y.view(B, T, -1),
            "keyboard_logits": pred.keyboard_logits.view(B, T, -1),
            "eco_buy_logits": pred.eco_buy_logits.view(B, T, -1),
            "eco_purchase_logits": pred.eco_purchase_logits.view(B, T, -1),
        }

    def _crop_cache(self, state: AutoregressiveState) -> None:
        if state.max_cache_frames is None or state.cache is None:
            return
        if state.max_cache_frames <= 0:
            raise ValueError("max_cache_frames must be positive when provided")
        state.cache.crop(state.max_cache_frames)

    def init_autoregressive_state(self, *, max_cache_frames: Optional[int] = None) -> AutoregressiveState:
        return AutoregressiveState(max_cache_frames=max_cache_frames)

    def reset_autoregressive_state(self, *, max_cache_frames: Optional[int] = None) -> AutoregressiveState:
        return self.init_autoregressive_state(max_cache_frames=max_cache_frames)

    def _ensure_state_batch(self, state: AutoregressiveState, batch_size: int) -> None:
        if state.batch_size is None:
            state.batch_size = batch_size
            return
        if state.batch_size != batch_size:
            raise ValueError(f"autoregressive state batch size mismatch: expected {state.batch_size}, got {batch_size}")

    def _state_seed(self, state: AutoregressiveState, batch_size: int, device: torch.device) -> torch.Tensor:
        if (
            state.prev_keyboard_mask is None
            or state.prev_mouse_x_bin is None
            or state.prev_mouse_y_bin is None
            or state.prev_eco_buy_idx is None
        ):
            return self._sos_seed((batch_size,), device)
        return self._action_seed(
            state.prev_keyboard_mask.to(device=device),
            state.prev_mouse_x_bin.to(device=device),
            state.prev_mouse_y_bin.to(device=device),
            state.prev_eco_buy_idx.to(device=device),
        )

    def _update_state_actions(self, prediction: dict[str, torch.Tensor], state: AutoregressiveState) -> None:
        keyboard_logits = prediction["keyboard_logits"][:, -1, :]
        mouse_x = prediction["mouse_x"][:, -1, :]
        mouse_y = prediction["mouse_y"][:, -1, :]
        eco_buy = prediction["eco_buy_logits"][:, -1, :]
        eco_purchase = prediction["eco_purchase_logits"][:, -1, 0]
        keyboard_mask = (torch.sigmoid(keyboard_logits) >= 0.5).to(torch.int64)
        bits = (2 ** torch.arange(self.cfg.keyboard_dim, device=keyboard_mask.device, dtype=torch.int64)).view(1, -1)
        state.prev_keyboard_mask = (keyboard_mask * bits).sum(dim=-1)
        state.prev_mouse_x_bin = torch.argmax(mouse_x, dim=-1)
        state.prev_mouse_y_bin = torch.argmax(mouse_y, dim=-1)
        no_buy = torch.full_like(torch.argmax(eco_buy, dim=-1), self.cfg.eco_dim)
        predicted_buy = torch.argmax(eco_buy, dim=-1)
        did_buy = torch.sigmoid(eco_purchase) >= 0.5
        state.prev_eco_buy_idx = torch.where(did_buy, predicted_buy, no_buy)

    def forward(
        self,
        images: torch.Tensor,
        audio: torch.Tensor,
        prev_keyboard_mask: Optional[torch.Tensor] = None,
        prev_mouse_x_bin: Optional[torch.Tensor] = None,
        prev_mouse_y_bin: Optional[torch.Tensor] = None,
        prev_eco_buy_idx: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        context, B, T = self._prepare_context(images, audio)
        seeds = self._build_training_seeds(
            B,
            T,
            context.device,
            prev_keyboard_mask=prev_keyboard_mask,
            prev_mouse_x_bin=prev_mouse_x_bin,
            prev_mouse_y_bin=prev_mouse_y_bin,
            prev_eco_buy_idx=prev_eco_buy_idx,
        )
        latent = self.cross(seeds.view(B * T, 1, -1), context.view(B * T, context.shape[2], context.shape[3]))
        token_sequence = latent.view(B, T, self.cfg.llama_hidden_size)
        hidden = self.backbone(inputs_embeds=token_sequence).last_hidden_state
        return self._predict_from_hidden(hidden, B, T)

    def forward_step(
        self,
        images: torch.Tensor,
        audio: torch.Tensor,
        state: Optional[AutoregressiveState] = None,
        *,
        max_cache_frames: Optional[int] = None,
        prev_keyboard_mask: Optional[torch.Tensor] = None,
        prev_mouse_x_bin: Optional[torch.Tensor] = None,
        prev_mouse_y_bin: Optional[torch.Tensor] = None,
        prev_eco_buy_idx: Optional[torch.Tensor] = None,
    ) -> tuple[dict[str, torch.Tensor], AutoregressiveState]:
        if images.shape[1] != 1:
            raise ValueError(f"forward_step expects T=1, got T={images.shape[1]}")

        context, B, T = self._prepare_context(images, audio)
        if state is None:
            state = self.init_autoregressive_state(max_cache_frames=max_cache_frames)
        elif max_cache_frames is not None:
            state.max_cache_frames = max_cache_frames

        self._ensure_state_batch(state, B)
        self._crop_cache(state)

        if (
            prev_keyboard_mask is not None
            and prev_mouse_x_bin is not None
            and prev_mouse_y_bin is not None
            and prev_eco_buy_idx is not None
        ):
            seed = self._action_seed(prev_keyboard_mask, prev_mouse_x_bin, prev_mouse_y_bin, prev_eco_buy_idx)
        else:
            seed = self._state_seed(state, B, context.device)

        latent = self.cross(seed.view(B, 1, -1), context[:, 0])

        cache_position = torch.arange(
            state.total_frames_processed,
            state.total_frames_processed + T,
            device=context.device,
            dtype=torch.long,
        )
        if state.cache is None:
            state.cache = RollingDynamicCache(config=self.backbone.config)
        llama_out = self.backbone(
            inputs_embeds=latent,
            past_key_values=state.cache,
            cache_position=cache_position,
            use_cache=True,
        )
        state.cache = llama_out.past_key_values
        state.total_frames_processed += T
        self._crop_cache(state)

        prediction = self._predict_from_hidden(llama_out.last_hidden_state, B, T)
        self._update_state_actions(prediction, state)
        return prediction, state
