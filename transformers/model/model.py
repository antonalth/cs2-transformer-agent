# model.py

import torch
import torch.nn as nn
from typing import Optional, List, Tuple

# We will use the `transformers` library from Hugging Face to easily load
# a pre-trained ViT-Large model. Make sure to install it:
# pip install transformers
from transformers import ViTConfig, ViTModel

# ======================================================================================
# STAGE 1: INPUT ENCODING (Completed)
# ======================================================================================

class AudioEncoder(nn.Module):
    """Encodes a Mel Spectrogram into a fixed-size embedding vector."""
    def __init__(self, hidden_dim: int = 2048, spectrogram_shape: tuple = (128, 6)):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spectrogram_shape = spectrogram_shape
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        cnn_output_shape = self._get_cnn_output_shape()
        self.projection = nn.Linear(cnn_output_shape, hidden_dim)
        self.no_audio_embedding = nn.Parameter(torch.randn(1, hidden_dim))

    def _get_cnn_output_shape(self) -> int:
        dummy_input = torch.randn(1, 1, *self.spectrogram_shape)
        with torch.no_grad():
            output = self.cnn(dummy_input)
        return output.flatten().shape[0]

    def forward(self, spectrogram: Optional[torch.Tensor]) -> torch.Tensor:
        if spectrogram is None:
            return self.no_audio_embedding
        x = spectrogram.unsqueeze(1)
        x = self.cnn(x)
        x = x.flatten(start_dim=1)
        return self.projection(x)

class VisionEncoder(nn.Module):
    """Encodes two 384x384 player POV views into a fixed-size embedding."""
    def __init__(self, hidden_dim: int = 2048, freeze_vit: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        vit_model_name = "google/vit-large-patch16-384"
        self.vit_config = ViTConfig.from_pretrained(vit_model_name)
        self.vit = ViTModel.from_pretrained(vit_model_name, config=self.vit_config)
        vit_output_dim = self.vit_config.hidden_size
        self.projection = nn.Linear(vit_output_dim * 2, hidden_dim)
        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False

    def unfreeze_vit(self):
        print("Unfreezing Vision Transformer for fine-tuning.")
        for param in self.vit.parameters():
            param.requires_grad = True

    def forward(self, foveal_view: torch.Tensor, peripheral_view: torch.Tensor) -> torch.Tensor:
        foveal_outputs = self.vit(pixel_values=foveal_view)
        foveal_cls_token = foveal_outputs.last_hidden_state[:, 0, :]
        peripheral_outputs = self.vit(pixel_values=peripheral_view)
        peripheral_cls_token = peripheral_outputs.last_hidden_state[:, 0, :]
        concatenated_tokens = torch.cat([foveal_cls_token, peripheral_cls_token], dim=1)
        return self.projection(concatenated_tokens)

class PlayerEncoder(nn.Module):
    """Wrapper module combining Vision and Audio encoders for a single player."""
    def __init__(self, hidden_dim: int = 2048, freeze_vit: bool = True):
        super().__init__()
        self.vision_encoder = VisionEncoder(hidden_dim=hidden_dim, freeze_vit=freeze_vit)
        self.audio_encoder = AudioEncoder(hidden_dim=hidden_dim)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, foveal_view: torch.Tensor, peripheral_view: torch.Tensor,
                spectrogram: Optional[torch.Tensor]) -> torch.Tensor:
        visual_embedding = self.vision_encoder(foveal_view, peripheral_view)
        audio_embedding = self.audio_encoder(spectrogram)
        fused_embedding = visual_embedding + audio_embedding
        return self.final_norm(fused_embedding)

# ======================================================================================
# STAGE 2: CORE TRANSFORMER BACKBONE
# This section defines the main transformer architecture, including the RoPE-enabled
# encoder layers and the overall model structure.
# ======================================================================================

class RotaryEmbedding(nn.Module):
    """
    Implements Rotary Positional Embeddings (RoPE).
    
    This module pre-computes the sinusoidal frequencies for RoPE and applies
    the rotation to query and key tensors.
    """
    def __init__(self, dim: int):
        super().__init__()
        # Pre-compute theta values for the sinusoidal embeddings
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_cos = None
        self.cached_sin = None

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [seq_len, batch_size, ...].
        """
        seq_len = x.shape[0]
        
        # Check if we have cached sin/cos values for this sequence length
        if self.cached_cos is None or self.cached_cos.shape[0] < seq_len:
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cached_cos = emb.cos()[:, None, None, :]
            self.cached_sin = emb.sin()[:, None, None, :]
        
        return self.cached_cos[:seq_len, ...], self.cached_sin[:seq_len, ...]

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Helper function to rotate half the features for RoPE application."""
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary embeddings to query and key tensors."""
    q_emb = (q * cos) + (rotate_half(q) * sin)
    k_emb = (k * cos) + (rotate_half(k) * sin)
    return q_emb, k_emb

class CS2TransformerEncoderLayer(nn.Module):
    """
    A Transformer Encoder layer that incorporates Rotary Positional Embeddings.
    It follows the standard Pre-LN (LayerNorm first) architecture.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, src: torch.Tensor, rotary_cos: torch.Tensor, rotary_sin: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom encoder layer.
        
        Args:
            src (torch.Tensor): Input sequence. Shape: [seq_len, batch_size, embed_dim].
            rotary_cos (torch.Tensor): Cosine part of RoPE.
            rotary_sin (torch.Tensor): Sine part of RoPE.
        """
        # --- Self-Attention Block (with RoPE) ---
        x = self.norm1(src)
        
        # Reshape for MHA and apply RoPE
        # x shape: [seq_len, batch_size, embed_dim]
        query, key, value = x, x, x
        
        # Reshape to [seq_len, batch_size, nhead, head_dim] to apply rotation
        q_s = query.reshape(query.shape[0], query.shape[1], self.nhead, self.d_model // self.nhead)
        k_s = key.reshape(key.shape[0], key.shape[1], self.nhead, self.d_model // self.nhead)
        
        q_rotated, k_rotated = apply_rotary_pos_emb(q_s, k_s, rotary_cos, rotary_sin)

        # Reshape back to [seq_len, batch_size, embed_dim] for MHA module
        q_rotated = q_rotated.reshape(query.shape)
        k_rotated = k_rotated.reshape(key.shape)

        # Pass rotated Q/K and original V to the attention mechanism
        attn_output, _ = self.self_attn(q_rotated, k_rotated, value, need_weights=False)
        
        src = src + self.dropout1(attn_output)

        # --- Feed-Forward Block ---
        x = self.norm2(src)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        src = src + self.dropout2(x)

        return src


class CS2Transformer(nn.Module):
    """
    The main model that processes a full round of CS2 data.
    """
    def __init__(self, hidden_dim: int = 2048, num_layers: int = 16, num_heads: int = 32,
                 dropout: float = 0.1, freeze_vit: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_players = 5
        self.num_special_tokens = 2 # GAME_STRATEGY and SCRATCHSPACE

        # --- Stage 1 Encoder ---
        self.player_encoder = PlayerEncoder(hidden_dim, freeze_vit)

        # --- Special Learnable Tokens ---
        self.game_strategy_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.scratchspace_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.dead_player_token = nn.Parameter(torch.randn(1, hidden_dim))
        self.mask_frame_token = nn.Parameter(torch.randn(1, hidden_dim))
        
        # --- Player Slot Embeddings ---
        # A learned embedding for each of the 5 player slots to ensure the model
        # can distinguish between Player 1, Player 2, etc., over time.
        self.player_slot_embeddings = nn.Embedding(self.num_players, hidden_dim)

        # --- RoPE and Transformer Encoder Layers ---
        self.rotary_embeddings = RotaryEmbedding(dim=hidden_dim // num_heads)
        
        encoder_layers = [
            CS2TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4, # Standard 4x expansion
                dropout=dropout
            ) for _ in range(num_layers)
        ]
        self.transformer_encoder = nn.ModuleList(encoder_layers)
        self.final_norm = nn.LayerNorm(hidden_dim)


    def forward(self, round_data: dict):
        """
        The main forward pass. This is a placeholder for the complex data handling.
        A real implementation requires a detailed data structure definition.
        
        Args:
            round_data (dict): A dictionary containing all data for a batch of rounds.
                Expected to contain keys like 'foveal_views', 'peripheral_views',
                'spectrograms', 'is_alive_mask', 'is_masked_frame_mask'.
        
        Returns:
            torch.Tensor: The output sequence from the transformer.
        """
        # For this script, we'll assume the input is pre-processed into a
        # sequence of player tokens and pass it to the encoder.
        # A full implementation would do the player encoding here.
        # Let's assume `x` is the assembled sequence of shape:
        # [seq_len * num_total_tokens, batch_size, hidden_dim]
        # For simplicity, we'll use a dummy input in the __main__ block.
        pass

if __name__ == '__main__':
    # ==================================
    # STAGE 1 SANITY CHECKS (Completed)
    # ==================================
    print("--- Running Stage 1 Sanity Checks ---")
    player_encoder = PlayerEncoder(hidden_dim=2048, freeze_vit=True)
    vit_params = sum(p.numel() for p in player_encoder.vision_encoder.vit.parameters() if p.requires_grad)
    print(f"Trainable ViT params (frozen): {vit_params}")
    assert vit_params == 0
    player_encoder.vision_encoder.unfreeze_vit()
    vit_params_unfrozen = sum(p.numel() for p in player_encoder.vision_encoder.vit.parameters() if p.requires_grad)
    print(f"Trainable ViT params (unfrozen): {vit_params_unfrozen}")
    assert vit_params_unfrozen > 300_000_000
    print("Stage 1 Checks Passed.\n")
    
    # ==================================
    # STAGE 2 SANITY CHECKS
    # ==================================
    print("--- Running Stage 2 Sanity Checks ---")
    
    # --- Configuration ---
    BATCH_SIZE = 2
    SEQ_LEN = 100 # Number of frames in the sequence
    HIDDEN_DIM = 2048
    NUM_HEADS = 32
    NUM_LAYERS = 4 # Use a smaller number for a quick test
    NUM_PLAYERS = 5
    NUM_SPECIAL_TOKENS = 2
    TOTAL_TOKENS_PER_FRAME = NUM_PLAYERS + NUM_SPECIAL_TOKENS # 7
    
    # --- Create Model ---
    model = CS2Transformer(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        freeze_vit=True
    )
    
    # --- Create Dummy Input Sequence ---
    # This simulates the sequence AFTER it has been assembled from the player encoders.
    # The shape is [seq_len, batch_size, total_tokens_per_frame, hidden_dim]
    # In a real training loop, this assembly is a major step.
    dummy_sequence = torch.randn(SEQ_LEN, BATCH_SIZE, TOTAL_TOKENS_PER_FRAME, HIDDEN_DIM)
    
    # In the transformer, we treat the sequence of frames and tokens within frames as one long sequence.
    # Reshape to: [SEQ_LEN * TOTAL_TOKENS_PER_FRAME, BATCH_SIZE, HIDDEN_DIM]
    x = dummy_sequence.reshape(SEQ_LEN * TOTAL_TOKENS_PER_FRAME, BATCH_SIZE, HIDDEN_DIM)
    
    print(f"Input shape to transformer encoder: {x.shape}")
    
    # --- Test RoPE and Encoder Forward Pass ---
    # 1. Get rotary embeddings
    rope_cos, rope_sin = model.rotary_embeddings(x)
    print(f"RoPE cos shape: {rope_cos.shape}")
    
    # 2. Pass through the encoder layers
    output = x
    for layer in model.transformer_encoder:
        output = layer(output, rope_cos, rope_sin)
    
    # 3. Final normalization
    output = model.final_norm(output)
    
    print(f"Output shape from transformer encoder: {output.shape}")
    
    # --- Assertions ---
    assert output.shape == x.shape, "Output shape must match input shape"
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters (ViT frozen): {total_params / 1e6:.2f}M")
    
    model.player_encoder.vision_encoder.unfreeze_vit()
    total_params_unfrozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters (ViT unfrozen): {total_params_unfrozen / 1e6:.2f}M")
    
    print("\nStage 2 module built and tested successfully!")