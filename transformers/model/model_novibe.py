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
    Blip2QFormerConfig, Blip2QFormerModel, DacModel
)

@dataclass
class ModelConfig:
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    tokens_per_frame: int = 6 # 5 player + 1 strategy

    dtype: torch.dtype = torch.bfloat16
    
    vision_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    vision_hidden_size: int = 768
    vision_chunk_size: int = 16

    audio_chunk_size: int = 1

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
class ModelPrediction:
    """
    Container for all output heads of the model.
    Shape Convention: [Batch, Time, 5 (Players), ...Dimensions...]
    """
    # --- Action Heads ---
    mouse_delta: torch.Tensor       # [B, T, 5, 2]   (Linear)
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

        q_config = Blip2QFormerConfig(
            hidden_size=cfg.qformer_hidden_size,
            num_hidden_layers=cfg.qformer_layers,
            num_attention_heads=cfg.qformer_heads,
            encoder_hidden_size=cfg.vision_hidden_size,
            vocab_size=1,
            dtype=cfg.dtype,
        )
        self.qformer = Blip2QFormerModel(q_config)
        if cfg.gradient_checkpointing:
            self.qformer.gradient_checkpointing_enable()

        self.query_tokens = nn.Parameter(
            torch.randn(
                1,
                cfg.num_qformer_queries,
                cfg.qformer_hidden_size,
                dtype=cfg.dtype,
            )
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, T, P=5, C, H, W]
        returns: [B, T, P=5, N_q, D_q]
        """
        B, T, P, C, H, W = images.shape
        flat_imgs = images.view(-1, C, H, W)  # [B*T*P, C, H, W]
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

            q_out = self.qformer(
                query_embeds=queries,
                encoder_hidden_states=vis_out,
            ).last_hidden_state  # [n, N_q, D_q]
            q_chunks.append(q_out)
            del vis_out, q_out, queries

        q_all = torch.cat(q_chunks, dim=0)  # [B*T*P, N_q, D_q]
        q_all = q_all.view(B, T, P, self.cfg.num_qformer_queries, self.cfg.qformer_hidden_size)
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
        flat_audio = audio.view(B * P * C, 1, S)
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
        
        self.mouse = nn.Linear(d, 2, dtype=dt)
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
            "mouse_delta": self.mouse(x),
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
        
        # FIX: Audio output is [C, H]. Since dataset is stereo, C=2.
        # We concat these in the forward pass, so input dim must account for both channels.
        adapter_dim_in = (cfg.qformer_hidden_size * cfg.num_qformer_queries) + (self.audio.output_dim * 2)
        
        self.adapter = nn.Sequential(
            nn.Linear(adapter_dim_in, cfg.adapter_hidden_dim, dtype=cfg.dtype),
            nn.GELU(),
            nn.Linear(cfg.adapter_hidden_dim, cfg.llama_hidden_size, dtype=cfg.dtype),
            nn.LayerNorm(cfg.llama_hidden_size, dtype=cfg.dtype) 
        )
        
        self.strat_token = nn.Parameter(torch.randn(1, 1, cfg.llama_hidden_size, dtype=cfg.dtype))

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
             adapted = checkpoint(self.adapter, fused, use_reentrant=False) # [B, T, P, D_llama]
        else:
             adapted = self.adapter(fused)

        adapted = adapted.to(dtype=self.cfg.dtype) #undo LN upcast?
        # Append Strategy Token
        strat = self.strat_token.expand(B, T, -1, -1) # [B, T, 1, D_llama]
        frame_seq = torch.cat([adapted, strat], dim=2) # [B, T, P+1, D_llama]
        
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

        # FIX: Heads expect [N, D], not [B, T, P, D]
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
        
        return ModelPrediction(
            mouse_delta=rs(p_preds["mouse_delta"]),
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