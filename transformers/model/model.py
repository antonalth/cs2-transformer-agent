# model.py

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict
import argparse
import time

# We will use the `transformers` library from Hugging Face.
from transformers import ViTConfig, ViTModel

# --- Data structure constants ---
NUM_KEYBOARD_KEYS = 31
NUM_ECO_ACTIONS = 384
NUM_INVENTORY_ITEMS = 128
NUM_ROUND_STATE_BITS = 5
MAP_RESOLUTION = (64, 256, 256)
MOUSE_MAP_RESOLUTION = (64, 64)

# ======================================================================================
# STAGE 1 & 2 (Now fully device-aware)
# ======================================================================================

class AudioEncoder(nn.Module):
    """Encodes a Mel Spectrogram into a fixed-size embedding vector."""
    def __init__(self, hidden_dim: int = 2048, spectrogram_shape: tuple = (128, 6)):
        super().__init__()
        self.hidden_dim = hidden_dim; self.spectrogram_shape = spectrogram_shape
        self.cnn = nn.Sequential(nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2))
        self.projection = nn.Linear(self._get_cnn_output_shape(), hidden_dim); self.no_audio_embedding = nn.Parameter(torch.randn(1, hidden_dim))
    def _get_cnn_output_shape(self) -> int:
        # FIX: Ensure dummy tensor is created on the correct device for calculation
        device = next(self.parameters()).device
        dummy_input = torch.randn(1, 1, *self.spectrogram_shape, device=device)
        return self.cnn(dummy_input).flatten().shape[0]
    def forward(self, spectrogram: Optional[torch.Tensor]) -> torch.Tensor:
        if spectrogram is None: return self.no_audio_embedding.expand(1, -1) # Default to batch size 1 if no other info
        return self.projection(self.cnn(spectrogram.unsqueeze(1)).flatten(1))

class VisionEncoder(nn.Module):
    """Encodes two 384x384 player POV views into a fixed-size embedding."""
    def __init__(self, hidden_dim: int = 2048, freeze_vit: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim; vit_model_name = "google/vit-large-patch16-384"
        self.vit_config = ViTConfig.from_pretrained(vit_model_name); self.vit = ViTModel.from_pretrained(vit_model_name, config=self.vit_config, add_pooling_layer=False)
        self.projection = nn.Linear(self.vit_config.hidden_size * 2, hidden_dim)
        if freeze_vit:
            for param in self.vit.parameters(): param.requires_grad = False
    def unfreeze_vit(self):
        print("Unfreezing Vision Transformer for fine-tuning."); [p.requires_grad_(True) for p in self.vit.parameters()]
    def forward(self, foveal_view: torch.Tensor, peripheral_view: torch.Tensor) -> torch.Tensor:
        # The .to(device) on the model will ensure inputs are processed on the correct device
        f_cls = self.vit(pixel_values=foveal_view).last_hidden_state[:, 0, :]
        p_cls = self.vit(pixel_values=peripheral_view).last_hidden_state[:, 0, :]
        return self.projection(torch.cat([f_cls, p_cls], dim=1))

class PlayerEncoder(nn.Module):
    """Wrapper module combining Vision and Audio encoders for a single player."""
    def __init__(self, hidden_dim: int = 2048, freeze_vit: bool = True):
        super().__init__()
        self.vision_encoder = VisionEncoder(hidden_dim, freeze_vit); self.audio_encoder = AudioEncoder(hidden_dim)
        self.final_norm = nn.LayerNorm(hidden_dim)
    def forward(self, foveal_view: torch.Tensor, peripheral_view: torch.Tensor, spectrogram: Optional[torch.Tensor]) -> torch.Tensor:
        fused = self.vision_encoder(foveal_view, peripheral_view) + self.audio_encoder(spectrogram)
        return self.final_norm(fused)

class RotaryEmbedding(nn.Module):
    """Implements Rotary Positional Embeddings (RoPE)."""
    def __init__(self, dim: int):
        super().__init__(); inv_freq = 1.0 / (10000**(torch.arange(0, dim, 2).float() / dim)); self.register_buffer("inv_freq", inv_freq); self.cached_cos = self.cached_sin = None
    def forward(self, x: torch.Tensor):
        seq_len = x.shape[0]
        if self.cached_cos is None or self.cached_cos.shape[0] < seq_len:
            # FIX: Ensure `t` is created on the correct device
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1); self.cached_cos = emb.cos()[:, None, None, :]; self.cached_sin = emb.sin()[:, None, None, :]
        return self.cached_cos[:seq_len, ...], self.cached_sin[:seq_len, ...]

def rotate_half(x: torch.Tensor): return torch.cat((-x[..., 1::2], x[..., ::2]), dim=-1)
def apply_rotary_pos_emb(q, k, cos, sin): return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class CS2TransformerEncoderLayer(nn.Module):
    """A Transformer Encoder layer modified to be CAUSAL for autoregression."""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.d_model, self.nhead = d_model, nhead
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.linear1, self.linear2 = nn.Linear(d_model, dim_feedforward), nn.Linear(dim_feedforward, d_model)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout, self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.attn_mask = None
    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)
    def forward(self, src: torch.Tensor, rotary_cos: torch.Tensor, rotary_sin: torch.Tensor) -> torch.Tensor:
        seq_len = src.shape[0]
        if self.attn_mask is None or self.attn_mask.size(0) != seq_len:
            self.attn_mask = self._generate_causal_mask(seq_len, src.device)
        x = self.norm1(src)
        q_s, k_s = x.reshape(seq_len, -1, self.nhead, self.d_model // self.nhead), x.reshape(seq_len, -1, self.nhead, self.d_model // self.nhead)
        q_r, k_r = apply_rotary_pos_emb(q_s, k_s, rotary_cos, rotary_sin)
        q_r, k_r = q_r.reshape(x.shape), k_r.reshape(x.shape)
        attn_output, _ = self.self_attn(q_r, k_r, x, attn_mask=self.attn_mask, need_weights=False)
        src = src + self.dropout1(attn_output)
        x = self.norm2(src)
        x = self.linear2(self.dropout(self.activation(self.linear1(x)))); src = src + self.dropout2(x)
        return src

# ======================================================================================
# STAGE 3: PREDICTION HEADS
# ======================================================================================

class MLPHead(nn.Sequential):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(nn.Linear(input_dim, input_dim // 2), nn.GELU(), nn.LayerNorm(input_dim // 2), nn.Linear(input_dim // 2, output_dim))

class HeatmapHead(nn.Module):
    def __init__(self, input_dim: int, output_shape: Tuple[int, ...]):
        super().__init__()
        self.decoder = MLPHead(input_dim, int(torch.prod(torch.tensor(output_shape)))); self.output_shape = output_shape
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x).view(-1, *self.output_shape)

class PlayerPredictionHeads(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.stats_head = MLPHead(hidden_dim, 3); self.pos_head = HeatmapHead(hidden_dim, MAP_RESOLUTION); self.mouse_head = HeatmapHead(hidden_dim, MOUSE_MAP_RESOLUTION)
        self.keyboard_head = MLPHead(hidden_dim, NUM_KEYBOARD_KEYS); self.eco_head = MLPHead(hidden_dim, NUM_ECO_ACTIONS)
        self.inventory_head = MLPHead(hidden_dim, NUM_INVENTORY_ITEMS); self.active_weapon_head = MLPHead(hidden_dim, NUM_INVENTORY_ITEMS)
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"stats": self.stats_head(x), "pos_heatmap_logits": self.pos_head(x), "mouse_heatmap_logits": self.mouse_head(x), "keyboard_logits": self.keyboard_head(x),
                "eco_logits": self.eco_head(x), "inventory_logits": self.inventory_head(x), "active_weapon_logits": self.active_weapon_head(x)}

class GameStrategyPredictionHeads(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.enemy_pos_head = HeatmapHead(hidden_dim, MAP_RESOLUTION); self.round_state_head = MLPHead(hidden_dim, NUM_ROUND_STATE_BITS); self.round_number_head = MLPHead(hidden_dim, 1)
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"enemy_pos_heatmap_logits": self.enemy_pos_head(x), "round_state_logits": self.round_state_head(x), "round_number": self.round_number_head(x)}

# ======================================================================================
# FINAL MODEL ASSEMBLY
# ======================================================================================

class CS2Transformer(nn.Module):
    """The main AUTOREGRESSIVE model that can "play the game"."""
    def __init__(self, hidden_dim: int = 2048, num_layers: int = 16, num_heads: int = 32,
                 dropout: float = 0.1, freeze_vit: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim; self.num_players = 5; self.num_special_tokens = 2
        self.player_encoder = PlayerEncoder(hidden_dim, freeze_vit)
        self.game_strategy_token = nn.Parameter(torch.randn(1, 1, hidden_dim)); self.scratchspace_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.dead_player_token = nn.Parameter(torch.randn(1, hidden_dim)); self.player_slot_embeddings = nn.Embedding(self.num_players, hidden_dim)
        self.rotary_embeddings = RotaryEmbedding(dim=hidden_dim // num_heads)
        encoder_layers = [CS2TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*4, dropout) for _ in range(num_layers)]
        self.transformer_encoder = nn.ModuleList(encoder_layers)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.player_prediction_heads = nn.ModuleList([PlayerPredictionHeads(hidden_dim) for _ in range(self.num_players)])
        self.game_strategy_prediction_heads = GameStrategyPredictionHeads(hidden_dim)

    def _prepare_input_frame(self, frame_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        B, P, _, _, _ = frame_data['foveal_views'].shape
        device = self.game_strategy_token.device
        foveal = frame_data['foveal_views'].view(B * P, 3, 384, 384)
        peripheral = frame_data['peripheral_views'].view(B * P, 3, 384, 384)
        audio = frame_data['spectrograms'].view(B * P, 128, 6)
        player_tokens = self.player_encoder(foveal, peripheral, audio).view(B, P, self.hidden_dim)
        slot_ids = torch.arange(P, device=device).view(1, P); slot_embeddings = self.player_slot_embeddings(slot_ids)
        player_tokens_with_slots = player_tokens + slot_embeddings
        input_frame = torch.zeros(B, P + self.num_special_tokens, self.hidden_dim, device=device)
        alive_mask = frame_data['is_alive_mask'].unsqueeze(-1).expand_as(player_tokens_with_slots)
        input_frame[:, :P, :] = torch.where(alive_mask, player_tokens_with_slots, self.dead_player_token)
        input_frame[:, P, :] = self.game_strategy_token; input_frame[:, P + 1, :] = self.scratchspace_token
        return input_frame

    def forward(self, history_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, S, T, D = history_sequence.shape
        x = history_sequence.permute(1, 2, 0, 3).reshape(S * T, B, D)
        rope_cos, rope_sin = self.rotary_embeddings(x)
        for layer in self.transformer_encoder:
            x = layer(x, rope_cos, rope_sin)
        output_sequence = self.final_norm(x)
        last_step_tokens = output_sequence.view(S, T, B, D)[-1].permute(1, 0, 2)
        predictions = {"player": [{} for _ in range(self.num_players)], "game_strategy": {}}
        for i in range(self.num_players):
            predictions["player"][i] = self.player_prediction_heads[i](last_step_tokens[:, i, :])
        predictions["game_strategy"] = self.game_strategy_prediction_heads(last_step_tokens[:, self.num_players, :])
        return predictions
        
    def generate(self, initial_frames: Dict[str, torch.Tensor], steps: int, max_context_frames: int) -> float:
        self.eval(); device = self.game_strategy_token.device
        B, S, P, _, _, _ = initial_frames['foveal_views'].shape
        history = [self._prepare_input_frame({k: v[:, i] for k, v in initial_frames.items()}) for i in range(S)]
        history_sequence = torch.stack(history, dim=1)
        t_start = time.perf_counter()
        for _ in range(steps):
            with torch.no_grad():
                next_frame_predictions = self.forward(history_sequence)
                next_frame_data = {
                    'foveal_views': torch.randn(B, P, 3, 384, 384, device=device),
                    'peripheral_views': torch.randn(B, P, 3, 384, 384, device=device),
                    'spectrograms': torch.randn(B, P, 128, 6, device=device),
                    'is_alive_mask': torch.randint(0, 2, (B, P), dtype=torch.bool, device=device),
                }
                next_encoded_frame = self._prepare_input_frame(next_frame_data).unsqueeze(1)
                history_sequence = torch.cat([history_sequence, next_encoded_frame], dim=1)
                # FIX: Enforce sliding window to keep context bounded
                if history_sequence.shape[1] > max_context_frames:
                    history_sequence = history_sequence[:, 1:, :, :]
        if device.type == 'cuda': torch.cuda.synchronize()
        t_end = time.perf_counter()
        return t_end - t_start

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run sanity checks for the CS2Transformer model.")
    parser.add_argument('--cuda-only', action='store_true', help="Run checks on a CUDA device if available.")
    parser.add_argument('--benchmark-seconds', type=int, default=30, help="Number of seconds to simulate for the autoregressive benchmark.")
    args = parser.parse_args()

    device = torch.device("cpu")
    if args.cuda_only:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # FIX: Set the default device for safety
            torch.set_default_device("cuda")
            print("--- Running all checks on CUDA device ---")
        else:
            print("CUDA not available. Aborting CUDA-only check."); exit()
    else:
        print("--- Running all checks on CPU device ---")

    print("\n--- Running Autoregressive Benchmark ---")
    # FIX: Move the model to the target device
    model = CS2Transformer(hidden_dim=2048, num_layers=4, num_heads=32, freeze_vit=True).to(device)
    
    B, P = 1, 5; PROMPT_FRAMES = 10; FPS = 32; SIM_SECONDS = args.benchmark_seconds
    STEPS_TO_GENERATE = (SIM_SECONDS * FPS) - PROMPT_FRAMES
    # Use a realistic context window size for the benchmark
    MAX_CONTEXT = 256
    
    print(f"Benchmark parameters: Simulating {SIM_SECONDS}s at {FPS} FPS.")
    print(f"Prompt: {PROMPT_FRAMES} frames, Generating: {STEPS_TO_GENERATE} frames, Max Context: {MAX_CONTEXT} frames")
    
    # FIX: Create all dummy data directly on the target device
    initial_frames_data = {
        'foveal_views': torch.randn(B, PROMPT_FRAMES, P, 3, 384, 384, device=device),
        'peripheral_views': torch.randn(B, PROMPT_FRAMES, P, 3, 384, 384, device=device),
        'spectrograms': torch.randn(B, PROMPT_FRAMES, P, 128, 6, device=device),
        'is_alive_mask': torch.ones((B, PROMPT_FRAMES, P), dtype=torch.bool, device=device),
    }

    try:
        if args.cuda_only:
            print("Warming up GPU...")
            model.generate(initial_frames_data, steps=10, max_context_frames=MAX_CONTEXT)
        
        print("Starting benchmark...")
        duration = model.generate(initial_frames_data, steps=STEPS_TO_GENERATE, max_context_frames=MAX_CONTEXT)
        
        print(f"\nSuccessfully generated {STEPS_TO_GENERATE} frames in {duration:.4f} seconds.")
        print(f"Average generation speed: {STEPS_TO_GENERATE / duration:.2f} frames/sec.")
        
    except Exception as e:
        print("\nAn error occurred during the benchmark:")
        raise e