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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import (
    AutoModel, AutoImageProcessor, LlamaConfig, LlamaModel, 
    DacModel
)

from config import ModelConfig

@dataclass
class ModelPrediction:
    """
    Container for all output heads of the model.
    Shape Convention: [Batch, Time, 5 (Players), ...Dimensions...]
    """
    # --- Action Heads ---
    mouse_x: torch.Tensor       # [B, T, 5, 256]   (Linear)
    mouse_y: torch.Tensor
    keyboard_logits: torch.Tensor   # [B, T, 5, 32]  (Logits)
    
    # --- Economy/Item Heads ---
    eco_logits: torch.Tensor        # [B, T, 5, 256] (Logits)
    inventory_logits: torch.Tensor  # [B, T, 5, 128] (Logits)
    weapon_logits: torch.Tensor     # [B, T, 5, 128] (Logits)
    
    # --- Stats Heads ---
    stats_logits: torch.Tensor      # [B, T, 5, 3]   (Logits -> Sigmoid in loss)
    
    # --- Spatial Heads (Player) ---
    player_pos_x: torch.Tensor      # [B, T, 5, 256] (Logits)
    player_pos_y: torch.Tensor      # [B, T, 5, 256] (Logits)
    player_pos_z: torch.Tensor      # [B, T, 5, 32]  (Logits)

    # --- Spatial Heads (Enemy - Sorted/Canonical) ---
    enemy_pos_x: torch.Tensor       # [B, T, 5, 256] (Logits)
    enemy_pos_y: torch.Tensor       # [B, T, 5, 256] (Logits)
    enemy_pos_z: torch.Tensor       # [B, T, 5, 32]  (Logits)

    # --- Global Game State Heads ---
    round_state_logits: torch.Tensor # [B, T, 5]     (Logits)
    round_num_logit: torch.Tensor    # [B, T, 1]     (Logit -> Sigmoid)
    team_alive_logits: torch.Tensor  # [B, T, 6]     (Logits, Classes 0-5)
    enemy_alive_logits: torch.Tensor # [B, T, 6]     (Logits, Classes 0-5)
    
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
        )
        self.out_norm = nn.LayerNorm(latent_dim, dtype=dtype)

    def forward(self, latents, inputs, inputs_key_padding_mask=None):
        q = self.q_norm(latents)
        kv = self.kv_norm(inputs)
        out, _ = self.attn(q, kv, kv, key_padding_mask=inputs_key_padding_mask, need_weights=False)
        latents = latents + out
        return self.out_norm(latents)


class _SelfAttentionFFNBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float, dtype):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, dtype=dtype)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
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


class PatchCompressor(nn.Module):
    """
    Perceiver-style iterative latent attention:
      repeat num_blocks times:
        1) cross-attn: latents -> patch tokens
        2) latent self-attn tower (num_self_attends_per_block layers)

    Output: [B, Nq, Dq]
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        dt = cfg.dtype

        # Keep your naming so adapter doesn't change:
        self.num_latents = cfg.num_perceiver_queries
        self.latent_dim = cfg.perceiver_hidden_size
        self.input_dim = cfg.vision_hidden_size

        # Perceiver-like knobs (add these to cfg if you want; safe defaults):
        self.num_blocks = int(getattr(cfg, "patch_compressor_num_blocks", 4))  # how many times to re-query inputs
        self.num_self_attends_per_block = int(getattr(cfg, "patch_compressor_self_attends_per_block", 2))
        self.mlp_ratio = float(getattr(cfg, "patch_compressor_mlp_ratio", 4.0))
        self.dropout = float(getattr(cfg, "patch_compressor_dropout", 0.0))

        # Weight sharing (paper discusses optional sharing) :contentReference[oaicite:3]{index=3}
        self.share_cross = bool(getattr(cfg, "patch_compressor_share_cross", True))
        self.share_self = bool(getattr(cfg, "patch_compressor_share_self", True))

        self.latents = nn.Parameter(
            torch.randn(1, self.num_latents, self.latent_dim, dtype=dt) * 0.02
        )

        if self.share_cross:
            self.cross_blocks = nn.ModuleList([
                _CrossAttentionBlock(self.latent_dim, self.input_dim, cfg.perceiver_heads, self.dropout, dt)
            ])
        else:
            self.cross_blocks = nn.ModuleList([
                _CrossAttentionBlock(self.latent_dim, self.input_dim, cfg.perceiver_heads, self.dropout, dt)
                for _ in range(self.num_blocks)
            ])

        if self.share_self:
            self.self_tower = nn.ModuleList([
                _SelfAttentionFFNBlock(self.latent_dim, cfg.perceiver_heads, self.mlp_ratio, self.dropout, dt)
                for _ in range(self.num_self_attends_per_block)
            ])
        else:
            self.self_tower_per_block = nn.ModuleList([
                nn.ModuleList([
                    _SelfAttentionFFNBlock(self.latent_dim, cfg.perceiver_heads, self.mlp_ratio, self.dropout, dt)
                    for _ in range(self.num_self_attends_per_block)
                ])
                for _ in range(self.num_blocks)
            ])

        self.use_checkpointing = bool(getattr(cfg, "gradient_checkpointing", False))

    def _ckpt(self, fn, *args):
        if self.training and self.use_checkpointing:
            return checkpoint(lambda *a: fn(*a), *args, use_reentrant=False)
        return fn(*args)

    def forward(self, patch_tokens: torch.Tensor, patch_key_padding_mask: Optional[torch.Tensor] = None):
        B = patch_tokens.shape[0]
        latents = self.latents.expand(B, -1, -1)  # [B, Nq, Dq]

        for b in range(self.num_blocks):
            cross = self.cross_blocks[0] if self.share_cross else self.cross_blocks[b]
            latents = self._ckpt(cross, latents, patch_tokens, patch_key_padding_mask)

            if self.share_self:
                for layer in self.self_tower:
                    latents = self._ckpt(layer, latents)
            else:
                for layer in self.self_tower_per_block[b]:
                    latents = self._ckpt(layer, latents)

        return latents

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


    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, T, P=5, C, H, W]
        returns: [B, T, P=5, N_q, D_q]
        """
        B, T, P, C, H, W = images.shape
        flat_imgs = images.reshape(-1, C, H, W)  # [B*T*P, C, H, W]
        N = flat_imgs.shape[0]
        chunk_size = self.cfg.vision_chunk_size

        q_chunks = []

        for i in range(0, N, chunk_size):
            chunk = flat_imgs[i : i + chunk_size]  # [n, C, H, W]
            with torch.no_grad():
                proc = self.vision_processor(
                    images=chunk,
                    return_tensors="pt",
                    data_format="channels_first",
                    do_resize=False, # Assuming data is already resized in dataset
                    do_center_crop=False
                )
                pixel_values = proc["pixel_values"].to(
                    device=chunk.device,
                    dtype=self.cfg.dtype,
                )
                del proc
                vis_out = self.vision(pixel_values=pixel_values).last_hidden_state #[n, L_v, D_v]
                del pixel_values

            n = vis_out.size(0)
            queries = self.query_tokens.expand(n, -1, -1)  # [n, N_q, D_q]

            q_out = self.compressor(vis_out)  # [n, N_q, D_q]

            q_chunks.append(q_out)
            del vis_out, q_out, queries

        q_all = torch.cat(q_chunks, dim=0)  # [B*T*P, N_q, D_q]
        q_all = q_all.view(B, T, P, self.cfg.num_perceiver_queries, self.cfg.perceiver_hidden_size)
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
        
        self.mouse_x = nn.Linear(d, cfg.mouse_bins_count, dtype=dt)
        self.mouse_y = nn.Linear(d, cfg.mouse_bins_count, dtype=dt)
        self.keyboard = nn.Linear(d, cfg.keyboard_dim, dtype=dt)
        self.eco = nn.Linear(d, cfg.eco_dim, dtype=dt)
        self.inventory = nn.Linear(d, cfg.inventory_dim, dtype=dt)
        self.weapon = nn.Linear(d, cfg.weapon_dim, dtype=dt)
        self.stats = nn.Sequential(nn.Linear(d, d, dtype=dt), nn.GELU(), nn.Linear(d, 3, dtype=dt))
        
        self.player_pos_x = nn.Linear(d, cfg.bins_x, dtype=dt)
        self.player_pos_y = nn.Linear(d, cfg.bins_y, dtype=dt)
        self.player_pos_z = nn.Linear(d, cfg.bins_z, dtype=dt)

        self.round_state = nn.Linear(d, cfg.round_state_dim, dtype=dt)
        self.round_number = nn.Linear(d, cfg.round_number_dim, dtype=dt)
        self.team_alive = nn.Linear(d, 6, dtype=dt) 
        self.enemy_alive = nn.Linear(d, 6, dtype=dt)

        self.enemy_expander = nn.Linear(d, 5 * d, dtype=dt)
        self.enemy_pos_x = nn.Linear(d, cfg.bins_x, dtype=dt)
        self.enemy_pos_y = nn.Linear(d, cfg.bins_y, dtype=dt)
        self.enemy_pos_z = nn.Linear(d, cfg.bins_z, dtype=dt)

    def forward_player(self, x: torch.Tensor):
        # x: [N, D]
        return {
            "mouse_x": self.mouse_x(x),
            "mouse_y": self.mouse_y(x),
            "keyboard_logits": self.keyboard(x),
            "eco_logits": self.eco(x),
            "inventory_logits": self.inventory(x),
            "weapon_logits": self.weapon(x),
            "stats_logits": self.stats(x),
            "player_pos_x": self.player_pos_x(x),
            "player_pos_y": self.player_pos_y(x),
            "player_pos_z": self.player_pos_z(x),
        }

    def forward_global(self, x: torch.Tensor):
        # x: [N, D]
        out = {
            "round_state_logits": self.round_state(x),
            "round_num_logit": self.round_number(x),
            "team_alive_logits": self.team_alive(x),
            "enemy_alive_logits": self.enemy_alive(x),
        }
        B = x.shape[0]
        # Expand 1 strategy token -> 5 enemy tokens per sequence
        enemy_feats = self.enemy_expander(x).view(B, 5, -1)
        # We need to flatten enemy features again for the linear heads to work
        # [B, 5, D] -> [B*5, D]
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
        
        adapter_dim_in = (cfg.perceiver_hidden_size * cfg.num_perceiver_queries) + (self.audio.output_dim * 2)
        
        self.adapter = nn.Sequential(
            nn.Linear(adapter_dim_in, cfg.adapter_hidden_dim, dtype=cfg.dtype),
            nn.GELU(),
            nn.Linear(cfg.adapter_hidden_dim, cfg.llama_hidden_size, dtype=cfg.dtype),
            nn.LayerNorm(cfg.llama_hidden_size, dtype=cfg.dtype) 
        )
    
        self.strat_node = nn.Embedding(1, cfg.llama_hidden_size, dtype=cfg.dtype)
        nn.init.normal_(self.strat_node.weight, std=0.02)


        use_cache = not cfg.gradient_checkpointing

        llama_conf = LlamaConfig(
            vocab_size=1, 
            hidden_size=cfg.llama_hidden_size,
            intermediate_size=cfg.llama_intermediate, 
            num_hidden_layers=cfg.llama_layers,
            num_attention_heads=cfg.llama_heads, 
            num_key_value_heads=cfg.llama_kv_heads,
            max_position_embeddings=cfg.llama_max_pos_embeddings,
            use_cache=use_cache,
            dtype=cfg.dtype,
            attn_implementation="flash_attention_2" if cfg.use_flash_attention else "eager"
        )
        self.backbone = LlamaModel(llama_conf)
        if cfg.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        
        self.heads = ModelOutputHeads(cfg)
        

    def forward(self, images: torch.Tensor, audio: torch.Tensor):
        B, T, P = images.shape[:3] # images: [B, T, P, C, H, W]
        
        video_features = self.video(images) # [B, T, P, N_q, D_q]
        audio_features = self.audio(audio, T) # [B, T, P, C, Aud_Hidden]
        del images, audio 

        # Flatten modalities
        vid_flat = video_features.reshape(B, T, P, -1) # [B, T, P, N_q * D_q]
        aud_flat = audio_features.reshape(B, T, P, -1) # [B, T, P, C * Aud_Hidden]
        
        # Fuse
        fused = torch.cat([vid_flat, aud_flat], dim=-1) # [B, T, P, (N_q * D_q) + (2 * Aud_Hidden)]
        del video_features, audio_features
        
        # Adapter projection
        # Checkpoint requires input to require grad, usually safe in training
        if self.training and self.cfg.gradient_checkpointing:
             fused.requires_grad_(True)
             adapted = checkpoint(self.adapter, fused, use_reentrant=False) # [B, T, P, D_llama]
        else:
             adapted = self.adapter(fused)

        adapted = adapted.to(dtype=self.cfg.dtype) #undo LN upcast?
        # Append Strategy Token
        ids = torch.zeros(B * T, dtype=torch.long, device=adapted.device)
        strat = self.strat_node(ids).view(B, T, 1, -1)   # [B, T, 1, D]
        frame_seq = torch.cat([adapted, strat], dim=2)

        
        # Flatten for Llama Backbone [Batch, Sequence Length, Hidden]
        # Sequence Length = Time * (Players + Strategy)
        frame_seq_flat = frame_seq.view(B, T * self.cfg.tokens_per_frame, -1)
        del adapted, strat

        outputs = self.backbone(inputs_embeds=frame_seq_flat)
        hidden_states = outputs.last_hidden_state # [B, T * (P+1), D]

        # Reshape back to separate players and strategy
        hidden_frames = hidden_states.view(B, T, self.cfg.tokens_per_frame, -1) # [B, T, P+1, D]
        p_hidden = hidden_frames[..., :5, :] # [B, T, 5, D]
        s_hidden = hidden_frames[..., 5, :]  # [B, T, D]

        p_flat = p_hidden.reshape(-1, self.cfg.llama_hidden_size)
        s_flat = s_hidden.reshape(-1, self.cfg.llama_hidden_size)

        p_preds = self.heads.forward_player(p_flat)
        s_preds = self.heads.forward_global(s_flat)
        
        def rs(x, is_player=True):
            if is_player: 
                # x is [B*T*5, Dim] -> [B, T, 5, Dim]
                return x.view(B, T, 5, *x.shape[1:])
            else: 
                # x is [B*T, Dim] -> [B, T, Dim]
                # Special case for enemy pos which is [B*T*5, Dim] but coming from strategy token
                if x.shape[0] == B * T * 5:
                    return x.view(B, T, 5, *x.shape[1:])
                return x.view(B, T, *x.shape[1:])
        
        mp = ModelPrediction(
            mouse_x=rs(p_preds["mouse_x"]),
            mouse_y=rs(p_preds["mouse_y"]),
            keyboard_logits=rs(p_preds["keyboard_logits"]),
            eco_logits=rs(p_preds["eco_logits"]),
            inventory_logits=rs(p_preds["inventory_logits"]),
            weapon_logits=rs(p_preds["weapon_logits"]),
            stats_logits=rs(p_preds["stats_logits"]),
            player_pos_x=rs(p_preds["player_pos_x"]),
            player_pos_y=rs(p_preds["player_pos_y"]),
            player_pos_z=rs(p_preds["player_pos_z"]),
            
            # Enemy preds come from s_preds but were expanded to 5
            enemy_pos_x=rs(s_preds["enemy_pos_x"], is_player=False), 
            enemy_pos_y=rs(s_preds["enemy_pos_y"], is_player=False),
            enemy_pos_z=rs(s_preds["enemy_pos_z"], is_player=False),
            
            round_state_logits=rs(s_preds["round_state_logits"], is_player=False),
            round_num_logit=rs(s_preds["round_num_logit"], is_player=False),
            team_alive_logits=rs(s_preds["team_alive_logits"], is_player=False),
            enemy_alive_logits=rs(s_preds["enemy_alive_logits"], is_player=False),
        )
        return {k: getattr(mp, k) for k in mp.__dataclass_fields__}