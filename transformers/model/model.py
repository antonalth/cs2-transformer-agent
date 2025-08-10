# model.py

import torch
import torch.nn as nn
from typing import Optional, List

# We will use the `transformers` library from Hugging Face to easily load
# a pre-trained ViT-Large model. Make sure to install it:
# pip install transformers
from transformers import ViTConfig, ViTModel

# ======================================================================================
# STAGE 1: INPUT ENCODING
# This section defines the modules that process raw player data (video and audio)
# into a unified, fixed-size token for the main transformer.
# ======================================================================================


class AudioEncoder(nn.Module):
    """
    Encodes a Mel Spectrogram into a fixed-size embedding vector.
    
    This module takes the pre-computed spectrogram, processes it through a small
    2D CNN to extract meaningful features, and then projects it to the main
    transformer's hidden dimension (D). It also handles the case where no audio
    is provided by using a learnable "no_audio" token.
    """
    def __init__(self, hidden_dim: int = 2048, spectrogram_shape: tuple = (128, 6)):
        """
        Initializes the AudioEncoder.
        
        Args:
            hidden_dim (int): The target dimension (D) of the main transformer.
            spectrogram_shape (tuple): The expected (n_mels, time_frames) shape
                                       of the input Mel spectrogram.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spectrogram_shape = spectrogram_shape

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        cnn_output_shape = self._get_cnn_output_shape()
        self.projection = nn.Linear(cnn_output_shape, hidden_dim)
        self.no_audio_embedding = nn.Parameter(torch.randn(1, hidden_dim))

    def _get_cnn_output_shape(self) -> int:
        """Helper function to dynamically calculate the output size of the CNN."""
        dummy_input = torch.randn(1, 1, *self.spectrogram_shape)
        with torch.no_grad():
            output = self.cnn(dummy_input)
        return output.flatten().shape[0]

    def forward(self, spectrogram: Optional[torch.Tensor]) -> torch.Tensor:
        """Forward pass for the AudioEncoder."""
        if spectrogram is None:
            return self.no_audio_embedding

        x = spectrogram.unsqueeze(1)
        x = self.cnn(x)
        x = x.flatten(start_dim=1)
        audio_embedding = self.projection(x)
        return audio_embedding


class VisionEncoder(nn.Module):
    """
    Encodes a player's Point-of-View (POV) into a fixed-size embedding.
    
    This module uses two views of the input frame (a 384x384 foveal/center crop
    and a 384x384 scaled peripheral view) and processes them through a SHARED, 
    pre-trained ViT-Large model.
    """
    def __init__(self, hidden_dim: int = 2048, freeze_vit: bool = True):
        """
        Initializes the VisionEncoder.
        
        Args:
            hidden_dim (int): The target dimension (D) of the main transformer.
            freeze_vit (bool): If True, the weights of the pre-trained ViT will be
                               frozen during initialization.
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Use the ViT model pre-trained on 384x384 images. This is now the
        # native resolution for BOTH of our visual inputs.
        vit_model_name = "google/vit-large-patch16-384"
        
        self.vit_config = ViTConfig.from_pretrained(vit_model_name)
        self.vit = ViTModel.from_pretrained(vit_model_name, config=self.vit_config)
        
        vit_output_dim = self.vit_config.hidden_size # 1024
        self.projection = nn.Linear(vit_output_dim * 2, hidden_dim)
        
        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False

    def unfreeze_vit(self):
        """Public method to unfreeze the ViT weights for fine-tuning."""
        print("Unfreezing Vision Transformer for fine-tuning.")
        for param in self.vit.parameters():
            param.requires_grad = True

    def forward(self, foveal_view: torch.Tensor, peripheral_view: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the VisionEncoder.
        
        Args:
            foveal_view (torch.Tensor): The 384x384 center-cropped (non-scaled) view.
                                        Shape: [Batch, 3, 384, 384].
            peripheral_view (torch.Tensor): The 384x384 scaled full view.
                                            Shape: [Batch, 3, 384, 384].
        
        Returns:
            torch.Tensor: The final visual embedding of shape [Batch, hidden_dim].
        """
        # Process the foveal (center) view through the shared ViT
        foveal_outputs = self.vit(pixel_values=foveal_view)
        foveal_cls_token = foveal_outputs.last_hidden_state[:, 0, :]

        # Process the peripheral view through the same shared ViT
        peripheral_outputs = self.vit(pixel_values=peripheral_view)
        peripheral_cls_token = peripheral_outputs.last_hidden_state[:, 0, :]

        # Concatenate the two [CLS] tokens
        concatenated_tokens = torch.cat([foveal_cls_token, peripheral_cls_token], dim=1)

        # Project to the final dimension
        visual_embedding = self.projection(concatenated_tokens)
        return visual_embedding


class PlayerEncoder(nn.Module):
    """
    A wrapper module that combines the Vision and Audio encoders for a single player.
    """
    def __init__(self, hidden_dim: int = 2048, freeze_vit: bool = True):
        super().__init__()
        
        self.vision_encoder = VisionEncoder(hidden_dim=hidden_dim, freeze_vit=freeze_vit)
        self.audio_encoder = AudioEncoder(hidden_dim=hidden_dim)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self,
                foveal_view: torch.Tensor,
                peripheral_view: torch.Tensor,
                spectrogram: Optional[torch.Tensor]) -> torch.Tensor:
        """Forward pass for the full player encoder."""
        visual_embedding = self.vision_encoder(foveal_view, peripheral_view)
        audio_embedding = self.audio_encoder(spectrogram)
        
        fused_embedding = visual_embedding + audio_embedding
        final_token = self.final_norm(fused_embedding)
        
        return final_token


if __name__ == '__main__':
    # ==================================
    # Example Usage and Sanity Checks
    # ==================================
    
    BATCH_SIZE = 4
    HIDDEN_DIM = 2048
    
    # --- Create Dummy Inputs ---
    # Both visual inputs are now 384x384
    dummy_foveal_view = torch.randn(BATCH_SIZE, 3, 384, 384) # <-- CHANGE HERE
    dummy_peripheral_view = torch.randn(BATCH_SIZE, 3, 384, 384)
    dummy_spectrogram = torch.randn(BATCH_SIZE, 128, 6)
    
    print("--- Initializing PlayerEncoder (ViT Frozen) ---")
    player_encoder = PlayerEncoder(hidden_dim=HIDDEN_DIM, freeze_vit=True)
    
    vit_params = sum(p.numel() for p in player_encoder.vision_encoder.vit.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in ViT: {vit_params}")
    assert vit_params == 0
    
    print("\n--- Testing forward pass ---")
    output_token = player_encoder(dummy_foveal_view, dummy_peripheral_view, dummy_spectrogram)
    print(f"Shape of the final player token: {output_token.shape}")
    assert output_token.shape == (BATCH_SIZE, HIDDEN_DIM)
    
    print("\n--- Unfreezing ViT for fine-tuning ---")
    player_encoder.vision_encoder.unfreeze_vit()
    vit_params_unfrozen = sum(p.numel() for p in player_encoder.vision_encoder.vit.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in ViT after unfreezing: {vit_params_unfrozen}")
    assert vit_params_unfrozen > 300_000_000
    
    print("\nStage 1 module (with 384x384 foveal view) built successfully!")