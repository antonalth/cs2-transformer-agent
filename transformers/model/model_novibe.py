
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import (
    AutoModel, LlamaConfig, LlamaModel, 
    Blip2QFormerConfig, Blip2QFormerModel
)

from model_loss import ModelPrediction

@dataclass
class ModelConfig:
    dtype: torch.dtype = torch.bfloat16
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    tokens_per_frame: int = 6 #5 player + 1 strategy
    
    vision_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    vision_hidden_size: int = 768
    vision_chunk_size: int = 16

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
            torch_dtype=cfg.dtype,
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
            torch_dtype=cfg.dtype,
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
                )
                pixel_values = proc["pixel_values"].to(
                    device=chunk.device,
                    dtype=self.cfg.dtype,
                )
                del proc
                vis_out = self.vision(pixel_values=pixel_values).last_hidden_state #[n, L_v, D_v]
                del pixel_values  # optional

            n = vis_out.size(0)
            queries = self.query_tokens.expand(n, -1, -1)  # [n, N_q, D_q]

            q_out = self.qformer(
                query_embeds=queries,
                encoder_hidden_states=vis_out,
            ).last_hidden_state  # [n, N_q, D_q]
            q_chunks.append(q_out)
            del vis_out, q_out, queries

        q_all = torch.cat(q_chunks, dim=0)  # [B*T*P, N_q, D_q]
        q_all = q_all.view(B, T, P, self.cfg.num_qformer_queries, self.qformer_hidden_size)
        return q_all

class GameAudioEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.model = DacModel.from_pretrained("descript/dac_24khz")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.output_dim = self.model.config.hidden_size 

    def forward(self, audio: torch.Tensor, target_frames: int) -> torch.Tensor:
        B, P, C, S = audio.shape
        flat_audio = audio.view(B * P * C, 1, S)
        with torch.no_grad():
            features = self.model.encode(flat_audio).latents
        aligned = F.adaptive_avg_pool1d(features, target_frames)  # [B*P*C, 1024?, T]
        aligned = aligned.permute(0, 2, 1) # [B*P*C, T, 1024?]
        out = aligned.view(B, P, C, target_frames, -1) # [B, P, C, T, H]
        out = out.permute(0, 3, 1, 2, 4) # [B, T, P, C, H]
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
        out = {
            "round_state_logits": self.round_state(x),
            "round_num_logit": self.round_number(x),
            "team_alive_logits": self.team_alive(x),
            "enemy_alive_logits": self.enemy_alive(x),
        }
        B = x.shape[0]
        enemy_feats = self.enemy_expander(x).view(B, 5, -1)
        out["enemy_pos_x"] = self.enemy_pos_x(enemy_feats)
        out["enemy_pos_y"] = self.enemy_pos_y(enemy_feats)
        out["enemy_pos_z"] = self.enemy_pos_z(enemy_feats)
        return out

class GamePredictorBackbone(nn.Module):
    def __init__(self, cfg: ModelConfig):
        self.video = GameVideoEncoder(cfg)
        self.audio = GameAudioEncoder(cfg)
        
        adapter_dim_in = (cfg.qformer_hidden_size * cfg.num_qformer_queries) + self.audio.output_dim
        self.adapter = nn.Sequential(
            nn.Linear(adapter_dim_in, cfg.adapter_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.adapter_dim_in, cfg.llama_hidden_size),
            nn.LayerNorm(cfg.llama_hidden_size) #i guess? 
        )
        
        self.strat_token = nn.Parameter(torch.randn(1, 1, cfg.llama_hidden_size, dtype=cfg.dtype))

        llama_conf = LlamaConfig(
            vocab_size=1, 
            hidden_size=cfg.llama_hidden_size,
            intermediate_size=cfg.llama_intermediate, 
            num_hidden_layers=cfg.llama_layers,
            num_attention_heads=cfg.llama_heads, 
            num_key_value_heads=cfg.llama_kv_heads,
            max_position_embeddings=cfg.llama_max_pos_embeddings,
            use_cache=use_cache,
            torch_dtype=cfg.torch_dtype,
            attn_implementation="flash_attention_2" if cfg.use_flash_attention else "eager"
        )
        self.backbone = LlamaModel(llama_conf)
        if cfg.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        
        self.heads = ModelOutputHeads(cfg)
        

    def forward(self, images: torch.Tensor, audio: torch.Tensor):
        B, T, P = images.shape[:3] #images: [B, T, P, C, H, W]
        video_features = self.video(images) # [B, T, P, N_q, D_q]
        audio_features = self.audio(audio, T) #[B, T, P, C, Aud_Hidden]
        del images, audio # does something?

        vid_flat = video_features.reshape(B, T, P, -1) # [B, T, P, N_q * D_q]
        aud_flat = audio_features.reshape(B, T, P, -1) # [B, T, P, C*Aud_Hidden]
        fused = torch.cat([vid_flat, aud_flat], dim=-1) # [B, T, P, (N_q * D_q) + (C * Aud_Hidden)]
        del video_features, audio_features #probably does nothing?
        
        adapted = checkpoint(self.adapter,fused) # [B, T, P, D_llama]

        strat = self.strat_token.expand(B, T, -1, -1) # [B, T, 1, D_llama]
        frame_seq = torch.cat([adapted, strat], dim=2) # [B, T, P+1, D_llama]?
        frame_seq_flat = frame_seq.view(B, T * self.cfg.tokens_per_frame, -1)
        del adapted, strat #does this do anything? 

        outputs = self.backbone(inputs_embeds=frame_seq_flat)
        hidden_states = outputs.last_hidden_state

        hidden_frames = hidden_stats.view(B, T, self.cfg.tokens_per_frame, -1) # [B, T, P+1, D_llama]
        p_hidden = hidden_frames[..., :5, :]
        s_hidden = hidden_frammes[..., 5, :]

        p_preds = self.heads.forward_player(p_flat)
        s_preds = self.heads.forward_global(s_flat)
        
        def rs(x, is_player=True):
            if is_player: return x.view(B, T, 5, *x.shape[1:])
            else: return x.view(B, T, *x.shape[1:])
        
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
            enemy_pos_x=rs(s_preds["enemy_pos_x"], is_player=False),
            enemy_pos_y=rs(s_preds["enemy_pos_y"], is_player=False),
            enemy_pos_z=rs(s_preds["enemy_pos_z"], is_player=False),
            round_state_logits=rs(s_preds["round_state_logits"], is_player=False),
            round_num_logit=rs(s_preds["round_num_logit"], is_player=False),
            team_alive_logits=rs(s_preds["team_alive_logits"], is_player=False),
            enemy_alive_logits=rs(s_preds["enemy_alive_logits"], is_player=False),
        )
