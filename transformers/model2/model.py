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

    def forward(self, patch_tokens: torch.Tensor, patch_key_padding_mask: Optional[torch.Tensor] = None):
        B = patch_tokens.shape[0]
        latents = self.latents.expand(B, -1, -1)  # [B, Nq, Dq]

        for b in range(self.num_blocks):
            cross = self.cross_blocks[0] if self.share_cross else self.cross_blocks[b]
            latents = cross(latents, patch_tokens, patch_key_padding_mask)

            if self.share_self:
                for layer in self.self_tower:
                    latents = layer(latents)
            else:
                for layer in self.self_tower_per_block[b]:
                    latents = layer(latents)

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
        
        # We need to know the device of the vision model for the chunk processor
        self.dummy_param = nn.Parameter(torch.empty(0))

    def _forward_chunk(self, chunk_cpu: torch.Tensor) -> torch.Tensor:
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
        
        # Run Compressor (Requires Grad)
        return self.compressor(vis_out)

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
            
            if use_ckpt:
                # checkpoint requires inputs to have requires_grad=True OR use_reentrant=False
                q_out = checkpoint(self._forward_chunk, chunk, use_reentrant=False)
            else:
                q_out = self._forward_chunk(chunk)
                
            q_chunks.append(q_out)

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
        self.audio_proj = nn.Linear(self.audio.output_dim, cfg.perceiver_hidden_size, dtype=cfg.dtype)
        
        # Initial Queries
        # 1 learned vector for players (will be expanded to 5)
        # "Start off our hidden state with 5x the same learned player query vector"
        self.player_query = nn.Parameter(torch.randn(1, 1, cfg.llama_hidden_size, dtype=cfg.dtype))
        # 1 learned vector for strategy
        self.strat_query = nn.Parameter(torch.randn(1, 1, cfg.llama_hidden_size, dtype=cfg.dtype))
        
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
                input_dim=cfg.perceiver_hidden_size,
                num_heads=cfg.llama_heads, 
                dropout=0.0,
                dtype=cfg.dtype
            )
            
            part = LlamaModel(llama_conf)
            if cfg.gradient_checkpointing:
                part.gradient_checkpointing_enable()
            
            # Remove norm for all but last
            if i < cfg.backbone_splits - 1:
                part.norm = nn.Identity()
                
            self.blocks.append(nn.ModuleDict({"cross": cross, "llama": part}))
            
        self.heads = ModelOutputHeads(cfg)

    def forward(self, images: torch.Tensor, audio: torch.Tensor):
        B, T, P = images.shape[:3]
        
        # --- Perception ---
        # Checkpointing is handled inside GameVideoEncoder's compressor
        vid_feats = self.video(images) # [B, T, P, 50, 768]
        aud_feats = self.audio(audio, T) # [B, T, P, C, 256]
        del images, audio
        
        # --- Prepare Context (Keys/Values) ---
        aud_feats = self.audio_proj(aud_feats) # [B, T, P, C, 768]
        
        # Concat per player
        # vid: [B, T, P, 50, 768]
        # aud: [B, T, P, C, 768]
        context = torch.cat([vid_feats, aud_feats], dim=3) # [B, T, P, 50+C, 768]
        del vid_feats, aud_feats
        
        # Reshape context for cross attention: [B*T*P, Tokens, Dim]
        context_flat = context.view(B*T*P, -1, self.cfg.perceiver_hidden_size)
        
        # --- Prepare Initial Hidden State ---
        # Players: [B, T, 5, D]
        # Expand 1 query -> 5 players
        p_q = self.player_query.expand(B, T*5, -1).view(B, T, 5, -1)
        # Strat: [B, T, 1, D]
        s_q = self.strat_query.expand(B, T, -1).view(B, T, 1, -1)
        
        current_hidden_p = p_q.reshape(B*T*5, 1, self.cfg.llama_hidden_size)
        current_hidden_s = s_q.reshape(B*T, 1, self.cfg.llama_hidden_size)
        
        for block in self.blocks:
            # 1. Cross Attention (Players only)
            # Query: current_hidden_p [N_players, 1, D]
            # Key: context_flat [N_players, K, D_v]
            
            p_out = block["cross"](current_hidden_p, context_flat) # [N_players, 1, D]
            current_hidden_p = p_out
            
            # 2. Combine for Llama
            # Reassemble [B, T, 6, D]
            p_view = current_hidden_p.view(B, T, 5, -1)
            s_view = current_hidden_s.view(B, T, 1, -1)
            
            seq = torch.cat([p_view, s_view], dim=2) # [B, T, 6, D]
            seq_flat = seq.view(B, T * 6, -1) # [B, L, D]
            
            # 3. Llama
            # Llama expects [Batch, Seq, Dim]
            llama_out = block["llama"](inputs_embeds=seq_flat).last_hidden_state # [B, L, D]
            
            # 4. Split back
            out_frames = llama_out.view(B, T, 6, -1)
            p_view = out_frames[:, :, :5, :]
            s_view = out_frames[:, :, 5:, :]
            
            current_hidden_p = p_view.reshape(B*T*5, 1, -1)
            current_hidden_s = s_view.reshape(B*T, 1, -1)
            
        # --- Final Heads ---
        # p_view: [B, T, 5, D]
        # s_view: [B, T, 1, D]
        
        p_flat = p_view.reshape(-1, self.cfg.llama_hidden_size)
        s_flat = s_view.reshape(-1, self.cfg.llama_hidden_size)
        
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
            
            eco_buy_logits=rs(p_preds["eco_buy_logits"]),
            eco_purchase_logits=rs(p_preds["eco_purchase_logits"]),
            active_weapon_logits=rs(p_preds["active_weapon_logits"]),
            
            health_logits=rs(p_preds["health_logits"]),
            armor_logits=rs(p_preds["armor_logits"]),
            money_logits=rs(p_preds["money_logits"]),
            
            player_pos_x=rs(p_preds["player_pos_x"]),
            player_pos_y=rs(p_preds["player_pos_y"]),
            player_pos_z=rs(p_preds["player_pos_z"]),
            
            # Enemy preds come from s_preds but were expanded to 5
            enemy_pos_x=rs(s_preds["enemy_pos_x"], is_player=False), 
            enemy_pos_y=rs(s_preds["enemy_pos_y"], is_player=False),
            enemy_pos_z=rs(s_preds["enemy_pos_z"], is_player=False),
            
            round_state_logits=rs(s_preds["round_state_logits"], is_player=False),
            round_num_logits=rs(s_preds["round_num_logits"], is_player=False),
            team_alive_logits=rs(s_preds["team_alive_logits"], is_player=False),
            enemy_alive_logits=rs(s_preds["enemy_alive_logits"], is_player=False),
        )
        return {k: getattr(mp, k) for k in mp.__dataclass_fields__}