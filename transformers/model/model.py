"""
model.py
------------------------------------------------------------------------
A standard-library heavy implementation of the CS2 Generative Agent.
Relies on HuggingFace Transformers for the heavy lifting (RoPE, GQA, FlashAttn).

Architecture:
1. Vision: Frozen DINOv3 (Standard AutoModel)
2. Audio:  Simple CNN
3. Fusion: Q-Former (BLIP-2 style) -> Extracts K tokens -> MLP -> 1 Token
4. Brain:  LlamaModel (Standard Decoder-only Transformer)
5. Heads:  Prediction Heads (Linear Classifiers for Binned Positions)
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

from model_loss import ModelPrediction # The output dataclass

# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------

@dataclass
class CS2Config:
    # --- Data & Dimensions ---
    num_players: int = 5
    tokens_per_frame: int = 7  # 5 Players + 1 Strategy + 1 Scratch
    context_frames: int = 64   # How far back we look
    
    # --- Vision (Frozen) ---
    # DINOv3-Base (768 dim). We load it via AutoModel.
    vision_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    vision_hidden_size: int = 768
    
    # --- Audio ---
    audio_mel_bins: int = 128
    audio_time_steps: int = 128 # Depends on dataset hop length
    audio_channels: Tuple[int, int, int] = (32, 64, 128)
    
    # --- Fusion (Q-Former) ---
    # How many queries to use to probe the image/audio features
    num_qformer_queries: int = 4  
    qformer_hidden_size: int = 768
    qformer_heads: int = 12
    qformer_layers: int = 4
    
    # --- Backbone (Llama) ---
    llama_hidden_size: int = 2048 # d_model
    llama_layers: int = 24
    llama_heads: int = 32
    llama_kv_heads: int = 8       # GQA
    llama_intermediate: int = 5632 # SwiGLU dimension
    
    # --- Prediction Heads ---
    keyboard_dim: int = 32
    eco_dim: int = 256
    inventory_dim: int = 128
    weapon_dim: int = 128
    round_state_dim: int = 5
    round_number_dim: int = 1
    
    # Spatial Bins (Must match CS2Loss in model_loss.py)
    bins_x: int = 256
    bins_y: int = 256
    bins_z: int = 32

    # --- System ---
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    # Precision: "bfloat16" is recommended for FlashAttn
    dtype: torch.dtype = torch.bfloat16 

# -----------------------------------------------------------------------------
# 2. Submodules
# -----------------------------------------------------------------------------

class AudioEncoder(nn.Module):
    """
    Standard CNN to process Mel Spectrograms into feature vectors.
    Input: [B, 2 (Stereo), Freq, Time]
    Output: [B, T_out, C_out] sequence of audio features.
    """
    def __init__(self, cfg: CS2Config):
        super().__init__()
        c1, c2, c3 = cfg.audio_channels
        
        self.net = nn.Sequential(
            nn.Conv2d(2, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.MaxPool2d(2), # 64

            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.GELU(),
            nn.MaxPool2d(2), # 32

            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)) # Force fixed spatial output
        )
        self.out_dim = c3 * 4 * 4 # Flattened

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, 2, Freq, Time]
        # Ensure input type matches weights (e.g., bf16)
        if x.dtype != self.net[0].weight.dtype:
            x = x.to(self.net[0].weight.dtype)
            
        x = self.net(x)
        # Flatten spatial dims: [B, C, 4, 4] -> [B, C * 16]
        return x.flatten(1).unsqueeze(1) # Return as sequence length 1: [B, 1, Dim]


class MultimodalProjector(nn.Module):
    """
    The Q-Former module. 
    It uses learnable queries to attend to the frozen Visual + Audio features.
    Then projects the result down to a single token for the Llama backbone.
    """
    def __init__(self, cfg: CS2Config):
        super().__init__()
        self.num_queries = cfg.num_qformer_queries
        
        # 1. Config Standard BLIP-2 Q-Former
        q_config = Blip2QFormerConfig(
            hidden_size=cfg.qformer_hidden_size,
            num_hidden_layers=cfg.qformer_layers,
            num_attention_heads=cfg.qformer_heads,
            encoder_hidden_size=cfg.vision_hidden_size, # Size of DINO features
            vocab_size=1, # Not used but required by config
            dtype=cfg.dtype
        )
        self.qformer = Blip2QFormerModel(q_config)
        
        # 2. Learnable Queries: [1, Num_Queries, Hidden]
        self.query_tokens = nn.Parameter(torch.randn(1, self.num_queries, cfg.qformer_hidden_size, dtype=cfg.dtype))
        
        # 3. Audio Adapter
        # Project audio to match DINO feature dimension so they can be concatenated
        self.audio_proj = nn.Linear(cfg.audio_channels[2] * 16, cfg.vision_hidden_size, dtype=cfg.dtype)
        
        # 4. Final Projection (MLP)
        # Flattens all Q-Former output tokens -> Single Llama Token
        in_dim = self.num_queries * cfg.qformer_hidden_size
        self.llama_proj = nn.Sequential(
            nn.Linear(in_dim, cfg.llama_hidden_size * 2, dtype=cfg.dtype),
            nn.GELU(),
            nn.Linear(cfg.llama_hidden_size * 2, cfg.llama_hidden_size, dtype=cfg.dtype)
        )
        
        self.norm = nn.LayerNorm(cfg.llama_hidden_size, dtype=cfg.dtype)

    def forward(self, vision_feats: torch.Tensor, audio_feats: torch.Tensor) -> torch.Tensor:
        """
        vision_feats: [B, Num_Patches, Vis_Dim]
        audio_feats:  [B, 1, Audio_Dim]
        Returns:      [B, 1, Llama_Dim] (The Player Token)
        """
        B = vision_feats.shape[0]
        
        # 1. Project audio and concat with vision
        # Result: [B, Patches + 1, Vis_Dim]
        audio_emb = self.audio_proj(audio_feats)
        encoder_hidden_states = torch.cat([vision_feats, audio_emb], dim=1)
        
        # 2. Expand queries for batch
        query_embeds = self.query_tokens.expand(B, -1, -1)
        
        # 3. Run Q-Former
        # We assume the vision feats are the "encoder_hidden_states"
        outputs = self.qformer(
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_hidden_states,
        )
        # last_hidden_state: [B, Num_Queries, Q_Dim]
        q_out = outputs.last_hidden_state
        
        # 4. Fuse to single token via MLP
        # Flatten queries: [B, Num_Queries * Q_Dim]
        fused = q_out.reshape(B, -1)
        token = self.llama_proj(fused)
        
        return self.norm(token).unsqueeze(1) # [B, 1, Llama_Dim]


class CS2Heads(nn.Module):
    """
    All prediction heads.
    Uses Linear layers to predict bins for spatial coordinates (Classification)
    instead of 3D heatmaps.
    """
    def __init__(self, cfg: CS2Config):
        super().__init__()
        d = cfg.llama_hidden_size
        dt = cfg.dtype
        
        # --- Player Action Heads ---
        self.mouse = nn.Linear(d, 2, dtype=dt)
        self.keyboard = nn.Linear(d, cfg.keyboard_dim, dtype=dt)
        self.eco = nn.Linear(d, cfg.eco_dim, dtype=dt)
        self.inventory = nn.Linear(d, cfg.inventory_dim, dtype=dt)
        self.weapon = nn.Linear(d, cfg.weapon_dim, dtype=dt)
        self.stats = nn.Sequential(nn.Linear(d, d, dtype=dt), nn.GELU(), nn.Linear(d, 3, dtype=dt))
        
        # --- Player Spatial Heads (Bin Classification) ---
        self.player_pos_x = nn.Linear(d, cfg.bins_x, dtype=dt)
        self.player_pos_y = nn.Linear(d, cfg.bins_y, dtype=dt)
        self.player_pos_z = nn.Linear(d, cfg.bins_z, dtype=dt)

        # --- Global/Strategy Heads ---
        self.round_state = nn.Linear(d, cfg.round_state_dim, dtype=dt)
        self.round_number = nn.Linear(d, cfg.round_number_dim, dtype=dt)
        self.team_alive = nn.Linear(d, 6, dtype=dt) # 0-5 count
        self.enemy_alive = nn.Linear(d, 6, dtype=dt)

        # --- Enemy Spatial Heads ---
        # The Strategy token needs to predict 5 discrete enemies.
        # We assume the Strategy token is a compressed representation of the whole map.
        # We expand it into 5 "Enemy Query" vectors.
        self.enemy_expander = nn.Linear(d, 5 * d, dtype=dt)
        
        # Independent classifiers for enemy positions (conceptually distinct from self-pos)
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
        # x: [Batch, Dim] (Strategy Token)
        
        # 1. Global Scalars
        out = {
            "round_state_logits": self.round_state(x),
            "round_num_logit": self.round_number(x),
            "team_alive_logits": self.team_alive(x),
            "enemy_alive_logits": self.enemy_alive(x),
        }

        # 2. Enemy Expansion [Batch, Dim] -> [Batch, 5, Dim]
        B = x.shape[0]
        enemy_feats = self.enemy_expander(x).view(B, 5, -1)
        
        # 3. Enemy Spatial
        # Result: [Batch, 5, Bins]
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
        
        # 1. Vision Backbone (Frozen)
        print(f"Loading Frozen DINOv3: {cfg.vision_model_name}...")
        self.vision = AutoModel.from_pretrained(
            cfg.vision_model_name, 
            trust_remote_code=True,
            dtype=cfg.dtype
        )
        self.vision.eval()
        for p in self.vision.parameters(): p.requires_grad = False
        
        # 2. Audio Encoder (Trainable)
        self.audio = AudioEncoder(cfg)
        
        # 3. Multimodal Projector (Trainable Q-Former)
        self.projector = MultimodalProjector(cfg)
        
        # 4. Global Tokens (Strategy + Scratchpad)
        # Added to the sequence for every frame
        self.strat_token = nn.Parameter(torch.randn(1, 1, cfg.llama_hidden_size, dtype=cfg.dtype))
        self.scratch_token = nn.Parameter(torch.randn(1, 1, cfg.llama_hidden_size, dtype=cfg.dtype))
        
        # 5. Transformer Backbone (Llama)
        # Standard Llama configuration (RoPE, GQA, etc. are automatic)
        
        # Logic to handle cache vs checkpointing conflict
        use_cache = True
        if cfg.gradient_checkpointing:
            use_cache = False
            
        llama_conf = LlamaConfig(
            vocab_size=1, # Not used
            hidden_size=cfg.llama_hidden_size,
            intermediate_size=cfg.llama_intermediate,
            num_hidden_layers=cfg.llama_layers,
            num_attention_heads=cfg.llama_heads,
            num_key_value_heads=cfg.llama_kv_heads,
            max_position_embeddings=4096,
            use_cache=use_cache,
            dtype=cfg.dtype,
            attn_implementation="flash_attention_2" if cfg.use_flash_attention else "eager"
        )
        self.backbone = LlamaModel(llama_conf)
        
        if cfg.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        # 6. Heads
        self.heads = CS2Heads(cfg)

    def forward(self, images: torch.Tensor, audio: torch.Tensor, ground_truth_struct: Any = None) -> ModelPrediction:
        """
        images: [B, T, 5, 3, H, W] - Visual History
        audio:  [B, T, 5, 2, F, Time] - Audio History
        Returns: ModelPrediction populated with logits
        """
        B, T, P, C, H, W = images.shape
        
        # --- 1. Encoder Pass (Vision + Audio) ---
        # Flatten Batch, Time, Players for parallel encoding
        flat_imgs = images.view(-1, C, H, W)
        flat_audio = audio.view(-1, 2, self.cfg.audio_mel_bins, self.cfg.audio_time_steps)
        
        with torch.no_grad():
            # DINOv3 Forward
            # output.last_hidden_state: [N, Patches, 768]
            # Ensure pixel values are correct dtype
            vis_in = flat_imgs if flat_imgs.dtype == self.cfg.dtype else flat_imgs.to(self.cfg.dtype)
            vis_out = self.vision(pixel_values=vis_in).last_hidden_state
        
        # Audio Forward: [N, 1, Audio_Dim]
        aud_out = self.audio(flat_audio)
        
        # --- 2. Fusion Pass (Q-Former) ---
        # Project DINO+Audio -> Single Llama Token: [N, 1, Llama_Dim]
        player_tokens_flat = self.projector(vis_out, aud_out)
        
        # Reshape back to Sequence: [B, T, 5, Dim]
        player_tokens = player_tokens_flat.view(B, T, P, -1)
        
        # --- 3. Sequence Assembly ---
        # Construct frame: [P1, P2, P3, P4, P5, Strategy, Scratch]
        strat = self.strat_token.expand(B, T, -1, -1)   # [B, T, 1, Dim]
        scratch = self.scratch_token.expand(B, T, -1, -1) # [B, T, 1, Dim]
        
        # Concatenate: [B, T, 7, Dim]
        frame_seq = torch.cat([player_tokens, strat, scratch], dim=2)
        
        # Flatten Time for Llama: [B, T*7, Dim]
        llama_input = frame_seq.view(B, T * self.cfg.tokens_per_frame, -1)
        
        # --- 4. Backbone Pass ---
        outputs = self.backbone(inputs_embeds=llama_input)
        hidden_states = outputs.last_hidden_state # [B, T*7, Dim]
        
        # --- 5. Unpack & Heads ---
        # Reshape to [B, T, 7, Dim]
        hidden_frames = hidden_states.view(B, T, self.cfg.tokens_per_frame, -1)
        
        # Extract relevant tokens
        p_hidden = hidden_frames[:, :, :5, :] # [B, T, 5, Dim]
        s_hidden = hidden_frames[:, :, 5, :]  # [B, T, Dim]
        
        # Run Heads (Flattening for efficiency)
        p_flat = p_hidden.reshape(-1, self.cfg.llama_hidden_size)
        s_flat = s_hidden.reshape(-1, self.cfg.llama_hidden_size)
        
        p_preds = self.heads.forward_player(p_flat)
        s_preds = self.heads.forward_global(s_flat)
        
        # --- 6. Pack into ModelPrediction Dataclass ---
        # Helper to reshape back to [B, T, ...]
        def rs(x, is_player=True):
            if is_player: return x.view(B, T, 5, *x.shape[1:])
            else: return x.view(B, T, *x.shape[1:])

        # Note: Enemy Pos is [B*T, 5, Bins], so rs(x, is_player=False) works 
        # because the internal 5 expands to [B, T, 5, Bins] naturally.
        
        return ModelPrediction(
            # Player Actions
            mouse_delta=rs(p_preds["mouse_delta"]),
            keyboard_logits=rs(p_preds["keyboard_logits"]),
            eco_logits=rs(p_preds["eco_logits"]),
            inventory_logits=rs(p_preds["inventory_logits"]),
            weapon_logits=rs(p_preds["weapon_logits"]),
            stats_logits=rs(p_preds["stats_logits"]),
            
            # Player Position (Bins)
            player_pos_x=rs(p_preds["player_pos_x"]),
            player_pos_y=rs(p_preds["player_pos_y"]),
            player_pos_z=rs(p_preds["player_pos_z"]),
            
            # Enemy Position (Bins)
            # s_preds output is [B*T, 5, Bins]. 
            # rs(..., False) -> [B, T, 5, Bins]
            enemy_pos_x=rs(s_preds["enemy_pos_x"], is_player=False),
            enemy_pos_y=rs(s_preds["enemy_pos_y"], is_player=False),
            enemy_pos_z=rs(s_preds["enemy_pos_z"], is_player=False),
            
            # Global State (Single Token)
            # Loss expects [B, T, Dim]. 
            # s_preds['round_state'] is [B*T, 5]. 
            # rs(..., False) -> [B, T, 5]. Matches ModelPrediction.
            round_state_logits=rs(s_preds["round_state_logits"], is_player=False),
            round_num_logit=rs(s_preds["round_num_logit"], is_player=False),
            team_alive_logits=rs(s_preds["team_alive_logits"], is_player=False),
            enemy_alive_logits=rs(s_preds["enemy_alive_logits"], is_player=False),
        )

if __name__ == "__main__":
    # Smoke Test
    cfg = CS2Config(llama_layers=2, context_frames=4, dtype=torch.bfloat16)
    model = CS2BehaviorModel(cfg).cuda()
    
    # Fake Batch
    # [B, T, 5, 3, 224, 224] (DINO Input Size)
    imgs = torch.randn(2, 4, 5, 3, 224, 224, dtype=torch.bfloat16).cuda()
    audio = torch.randn(2, 4, 5, 2, 128, 128, dtype=torch.bfloat16).cuda()
    
    print("Running forward pass...")
    with torch.amp.autocast("cuda"):
        out = model(imgs, audio)
    
    print("Output shapes:")
    print(f"Mouse: {out.mouse_delta.shape}")
    print(f"Player X Bins: {out.player_pos_x.shape}")
    print(f"Enemy X Bins: {out.enemy_pos_x.shape}")
    print("Success.")