
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Any

@dataclass
class ModelConfig:
    dtype: torch.dtype = torch.bfloat16
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    
    vision_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    vision_hidden_size: int = 768
    vision_chunk_size: int = 16

    # --- Fusion (Q-Former) ---
    num_qformer_queries: int = 4  
    qformer_hidden_size: int = 768
    qformer_heads: int = 12
    qformer_layers: int = 4

    # Audio
    audio_hidden_size: int = 512

    # --- Backbone (Llama) ---
    llama_hidden_size: int = 2048 
    llama_layers: int = 24
    llama_heads: int = 32
    llama_kv_heads: int = 8       
    llama_intermediate: int = 5632 
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
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:

class GamePredictorBackbone(nn.Module):
    def __init__(self, cfg: ModelConfig):
        self.video = GameVideoEncoder(cfg)
        self.audio = GameAudioEncoder(cfg)
        self.adapter = nn.Sequential()
    def forward(self, images: torch.Tensor, audio: torch.Tensor)