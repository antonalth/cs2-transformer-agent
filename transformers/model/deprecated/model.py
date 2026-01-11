"""
model.py
------------------------------------------------------------------------
A standard-library heavy implementation of the CS2 Generative Agent.
Relies on HuggingFace Transformers for the heavy lifting (RoPE, GQA, FlashAttn).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Any

import torch
import torch.nn as nn
from transformers import (
    AutoModel, LlamaConfig, LlamaModel, 
    Blip2QFormerConfig, Blip2QFormerModel
)

from model_loss import ModelPrediction
import debug

# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------

@dataclass
class CS2Config:
    # --- Data & Dimensions ---
    num_players: int = 5
    tokens_per_frame: int = 7  
    context_frames: int = 64
    
    # --- Vision (Frozen) ---
    vision_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    vision_hidden_size: int = 768
    vision_chunk_size: int = 8 
    
    # --- Audio ---
    audio_mel_bins: int = 128
    audio_time_steps: int = 32
    audio_channels: Tuple[int, int, int] = (32, 64, 128)
    
    # --- Fusion (Q-Former) ---
    num_qformer_queries: int = 4  
    qformer_hidden_size: int = 768
    qformer_heads: int = 12
    qformer_layers: int = 4
    
    # --- Backbone (Llama) ---
    llama_hidden_size: int = 2048 
    llama_layers: int = 24
    llama_heads: int = 32
    llama_kv_heads: int = 8       
    llama_intermediate: int = 5632 
    
    # --- Prediction Heads ---
    keyboard_dim: int = 32
    eco_dim: int = 256
    inventory_dim: int = 128
    weapon_dim: int = 128
    round_state_dim: int = 5
    round_number_dim: int = 1
    
    bins_x: int = 256
    bins_y: int = 256
    bins_z: int = 32

    # --- System ---
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    torch_dtype: torch.dtype = torch.bfloat16 

# -----------------------------------------------------------------------------
# 2. Submodules
# -----------------------------------------------------------------------------

class AudioEncoder(nn.Module):
    def __init__(self, cfg: CS2Config):
        super().__init__()
        c1, c2, c3 = cfg.audio_channels
        
        self.net = nn.Sequential(
            nn.Conv2d(2, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast input to match weights if needed (e.g. bf16)
        if x.dtype != self.net[0].weight.dtype:
            x = x.to(self.net[0].weight.dtype)
        x = self.net(x)
        return x.flatten(1).unsqueeze(1) 


class MultimodalProjector(nn.Module):
    def __init__(self, cfg: CS2Config):
        super().__init__()
        self.num_queries = cfg.num_qformer_queries
        
        q_config = Blip2QFormerConfig(
            hidden_size=cfg.qformer_hidden_size,
            num_hidden_layers=cfg.qformer_layers,
            num_attention_heads=cfg.qformer_heads,
            encoder_hidden_size=cfg.vision_hidden_size, 
            vocab_size=1, 
            torch_dtype=cfg.torch_dtype
        )
        self.qformer = Blip2QFormerModel(q_config)
        
        if cfg.gradient_checkpointing:
            self.qformer.gradient_checkpointing_enable()
        
        self.query_tokens = nn.Parameter(torch.randn(1, self.num_queries, cfg.qformer_hidden_size, dtype=cfg.torch_dtype))
        self.audio_proj = nn.Linear(cfg.audio_channels[2] * 16, cfg.vision_hidden_size, dtype=cfg.torch_dtype)
        
        in_dim = self.num_queries * cfg.qformer_hidden_size
        self.llama_proj = nn.Sequential(
            nn.Linear(in_dim, cfg.llama_hidden_size * 2, dtype=cfg.torch_dtype),
            nn.GELU(),
            nn.Linear(cfg.llama_hidden_size * 2, cfg.llama_hidden_size, dtype=cfg.torch_dtype)
        )
        self.norm = nn.LayerNorm(cfg.llama_hidden_size, dtype=cfg.torch_dtype)

    def forward(self, vision_feats: torch.Tensor, audio_feats: torch.Tensor) -> torch.Tensor:
        B = vision_feats.shape[0]
        audio_emb = self.audio_proj(audio_feats)
        encoder_hidden_states = torch.cat([vision_feats, audio_emb], dim=1)
        query_embeds = self.query_tokens.expand(B, -1, -1)
        
        outputs = self.qformer(
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_hidden_states,
        )
        q_out = outputs.last_hidden_state
        
        fused = q_out.reshape(B, -1)
        token = self.llama_proj(fused)
        return self.norm(token).unsqueeze(1) 


class CS2Heads(nn.Module):
    def __init__(self, cfg: CS2Config):
        super().__init__()
        d = cfg.llama_hidden_size
        dt = cfg.torch_dtype
        
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

# -----------------------------------------------------------------------------
# 3. Main Model
# -----------------------------------------------------------------------------

class CS2BehaviorModel(nn.Module):
    def __init__(self, cfg: CS2Config = CS2Config()):
        super().__init__()
        self.cfg = cfg
        
        print(f"Loading Frozen DINOv3: {cfg.vision_model_name}...")
        self.vision = AutoModel.from_pretrained(
            cfg.vision_model_name, 
            trust_remote_code=True,
            torch_dtype=cfg.torch_dtype
        )
        self.vision.eval()
        for p in self.vision.parameters(): p.requires_grad = False
        
        self.audio = AudioEncoder(cfg)
        # FIX: Explicitly cast AudioEncoder to BF16 (or whatever cfg.torch_dtype is)
        # This prevents the FSDP error: "ValueError: Must flatten tensors with uniform dtype"
        self.audio.to(cfg.torch_dtype) 
        
        self.projector = MultimodalProjector(cfg)
        self.strat_token = nn.Parameter(torch.randn(1, 1, cfg.llama_hidden_size, dtype=cfg.torch_dtype))
        self.scratch_token = nn.Parameter(torch.randn(1, 1, cfg.llama_hidden_size, dtype=cfg.torch_dtype))
        
        use_cache = True
        if cfg.gradient_checkpointing:
            use_cache = False
            
        llama_conf = LlamaConfig(
            vocab_size=1, 
            hidden_size=cfg.llama_hidden_size,
            intermediate_size=cfg.llama_intermediate, 
            num_hidden_layers=cfg.llama_layers,
            num_attention_heads=cfg.llama_heads, 
            num_key_value_heads=cfg.llama_kv_heads,
            max_position_embeddings=4096,
            use_cache=use_cache,
            torch_dtype=cfg.torch_dtype,
            attn_implementation="flash_attention_2" if cfg.use_flash_attention else "eager"
        )
        self.backbone = LlamaModel(llama_conf)
        if cfg.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        self.heads = CS2Heads(cfg)

    def forward(self, images: torch.Tensor, audio: torch.Tensor, ground_truth_struct: Any = None) -> ModelPrediction:
        debug.log("Start Forward")
        
        B, T, P, C, H, W = images.shape
        flat_imgs = images.view(-1, C, H, W)
        flat_audio = audio.view(-1, 2, self.cfg.audio_mel_bins, self.cfg.audio_time_steps)
        
        # --- 1. Chunked Vision Forward ---
        vis_chunks = []
        total_images = flat_imgs.shape[0]
        chunk_size = self.cfg.vision_chunk_size
        
        with torch.no_grad():
            vis_in_full = flat_imgs if flat_imgs.dtype == self.cfg.torch_dtype else flat_imgs.to(self.cfg.torch_dtype)
            for i in range(0, total_images, chunk_size):
                chunk = vis_in_full[i : i + chunk_size]
                chunk_out = self.vision(pixel_values=chunk).last_hidden_state
                vis_chunks.append(chunk_out)
        
        vis_out = torch.cat(vis_chunks, dim=0) 
        
        # Cleanup to reduce fragmentation
        del vis_chunks
        del vis_in_full
        
        debug.log("Vision Done")
        
        aud_out = self.audio(flat_audio)
        
        # --- 2. Chunked Fusion (Q-Former) ---
        projector_chunks = []
        for i in range(0, total_images, chunk_size):
            v_chunk = vis_out[i : i + chunk_size]
            a_chunk = aud_out[i : i + chunk_size]
            p_chunk = self.projector(v_chunk, a_chunk)
            projector_chunks.append(p_chunk)
            
        player_tokens_flat = torch.cat(projector_chunks, dim=0)
        
        del projector_chunks
        del vis_out
        del aud_out
        
        debug.log("Q-Former Done")
        
        # --- 3. Sequence Assembly ---
        player_tokens = player_tokens_flat.view(B, T, P, -1)
        strat = self.strat_token.expand(B, T, -1, -1)
        scratch = self.scratch_token.expand(B, T, -1, -1)
        frame_seq = torch.cat([player_tokens, strat, scratch], dim=2)
        llama_input = frame_seq.view(B, T * self.cfg.tokens_per_frame, -1)
        
        # --- 4. Backbone Pass ---
        outputs = self.backbone(inputs_embeds=llama_input)
        hidden_states = outputs.last_hidden_state
        debug.log("Llama Done")
        
        # --- 5. Unpack & Heads ---
        hidden_frames = hidden_states.view(B, T, self.cfg.tokens_per_frame, -1)
        p_hidden = hidden_frames[:, :, :5, :]
        s_hidden = hidden_frames[:, :, 5, :]
        
        p_flat = p_hidden.reshape(-1, self.cfg.llama_hidden_size)
        s_flat = s_hidden.reshape(-1, self.cfg.llama_hidden_size)
        
        p_preds = self.heads.forward_player(p_flat)
        s_preds = self.heads.forward_global(s_flat)
        
        # --- 6. Pack ---
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

if __name__ == "__main__":
    cfg = CS2Config(llama_layers=2, context_frames=4, torch_dtype=torch.bfloat16)
    model = CS2BehaviorModel(cfg).cuda()
    imgs = torch.randn(2, 4, 5, 3, 224, 224, dtype=torch.bfloat16).cuda()
    audio = torch.randn(2, 4, 5, 2, 128, 32, dtype=torch.bfloat16).cuda()
    print("Running forward pass...")
    with torch.amp.autocast("cuda"):
        out = model(imgs, audio)
    print("Success.")