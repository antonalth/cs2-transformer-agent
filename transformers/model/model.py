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

        # --- CNN Feature Extractor ---
        # A simple stack of 2D convolutional layers to process the spectrogram image.
        self.cnn = nn.Sequential(
            # Input shape: [Batch, 1, 128, 6] (Channels, Height, Width)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # Shape -> [Batch, 16, 64, 3]
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # Shape -> [Batch, 32, 32, 1]
        )

        # --- Projection Layer ---
        # After the CNN, we flatten the output and project it to the hidden_dim.
        # We need to calculate the flattened size first.
        cnn_output_shape = self._get_cnn_output_shape()
        self.projection = nn.Linear(cnn_output_shape, hidden_dim)

        # --- Learnable "No Audio" Token ---
        # A parameter that will be used if the input audio is None.
        # This allows the model to learn a representation for missing audio.
        self.no_audio_embedding = nn.Parameter(torch.randn(1, hidden_dim))

    def _get_cnn_output_shape(self) -> int:
        """Helper function to dynamically calculate the output size of the CNN."""
        # Create a dummy tensor with the expected input shape and pass it through the CNN
        dummy_input = torch.randn(1, 1, *self.spectrogram_shape)
        with torch.no_grad():
            output = self.cnn(dummy_input)
        return output.flatten().shape[0]

    def forward(self, spectrogram: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the AudioEncoder.
        
        Args:
            spectrogram (Optional[torch.Tensor]): A tensor of shape 
                [Batch, n_mels, time_frames] or None.
        
        Returns:
            torch.Tensor: The audio embedding of shape [Batch, hidden_dim].
        """
        if spectrogram is None:
            # If input is None, broadcast the learned "no_audio" embedding to the batch size.
            # We assume the batch size can be inferred from a different input later,
            # but for a single-item forward pass, we can use expand.
            # In a real batch, we'd do: self.no_audio_embedding.expand(batch_size, -1)
            return self.no_audio_embedding

        # The input spectrogram is [Batch, H, W]. Add a channel dimension for the CNN.
        # Shape -> [Batch, 1, 128, 6]
        x = spectrogram.unsqueeze(1)

        # Pass through the CNN and flatten the result
        x = self.cnn(x)
        x = x.flatten(start_dim=1) # Shape -> [Batch, cnn_output_shape]

        # Project to the final hidden dimension
        audio_embedding = self.projection(x) # Shape -> [Batch, hidden_dim]
        return audio_embedding


class VisionEncoder(nn.Module):
    """
    Encodes a player's Point-of-View (POV) into a fixed-size embedding.
    
    This module uses two views of the input frame (a foveal/center crop and a
    full peripheral view) and processes them through a shared, pre-trained
    ViT-Large model. The final class ([CLS]) tokens from both views are
    concatenated and projected to the main transformer's hidden dimension (D).
    
    The ViT model itself is a submodule, which allows us to easily isolate its
    parameters for differential learning rates during fine-tuning.
    """
    def __init__(self, hidden_dim: int = 2048, freeze_vit: bool = True):
        """
        Initializes the VisionEncoder.
        
        Args:
            hidden_dim (int): The target dimension (D) of the main transformer.
            freeze_vit (bool): If True, the weights of the pre-trained ViT will be
                               frozen during initialization. This is the standard
                               first step for fine-tuning.
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # --- Load Pre-trained ViT-Large ---
        # We load the configuration and the model from Hugging Face.
        # "google/vit-large-patch16-224-in21k" is a good general-purpose choice.
        vit_model_name = "google/vit-large-patch16-224-in21k"
        self.vit_config = ViTConfig.from_pretrained(vit_model_name)
        
        # This is the core ViT model. By making it a direct submodule,
        # we can easily access its parameters later (e.g., `model.vit.parameters()`).
        self.vit = ViTModel.from_pretrained(vit_model_name, config=self.vit_config)
        
        # --- Projection Layer ---
        # The ViT-Large model has a hidden size of 1024. We process two views,
        # so we concatenate their [CLS] tokens (1024 + 1024 = 2048).
        # This projection layer fuses them into the final target dimension.
        vit_output_dim = self.vit_config.hidden_size # This will be 1024
        self.projection = nn.Linear(vit_output_dim * 2, hidden_dim)
        
        # --- Freezing Logic ---
        # If `freeze_vit` is True, we iterate through all parameters of the ViT
        # submodule and set `requires_grad` to False. This prevents them from
        # being updated by the optimizer. This is the key to the first stage
        # of fine-tuning.
        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False

    def unfreeze_vit(self):
        """
        Public method to unfreeze the ViT weights for the second stage of training.
        This allows the main training script to control when fine-tuning begins.
        """
        print("Unfreezing Vision Transformer for fine-tuning.")
        for param in self.vit.parameters():
            param.requires_grad = True

    def forward(self, foveal_view: torch.Tensor, peripheral_view: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the VisionEncoder.
        
        Args:
            foveal_view (torch.Tensor): The center-cropped view of the player's
                                        POV. Shape: [Batch, 3, 224, 224].
            peripheral_view (torch.Tensor): The resized full view of the player's
                                            POV. Shape: [Batch, 3, 384, 384].
                                            (Note: ViT automatically handles different
                                             input sizes by adjusting its positional embeddings)
        
        Returns:
            torch.Tensor: The final visual embedding of shape [Batch, hidden_dim].
        """
        # Process the foveal (center) view through the ViT
        # We only need the `last_hidden_state` which contains all output tokens.
        foveal_outputs = self.vit(pixel_values=foveal_view)
        # The [CLS] token is always the first token in the sequence.
        foveal_cls_token = foveal_outputs.last_hidden_state[:, 0, :] # Shape: [Batch, 1024]

        # Process the peripheral view through the same shared ViT
        peripheral_outputs = self.vit(pixel_values=peripheral_view)
        peripheral_cls_token = peripheral_outputs.last_hidden_state[:, 0, :] # Shape: [Batch, 1024]

        # Concatenate the two [CLS] tokens along the feature dimension
        concatenated_tokens = torch.cat([foveal_cls_token, peripheral_cls_token], dim=1) # Shape: [Batch, 2048]

        # Project the concatenated tokens to the final desired dimension
        visual_embedding = self.projection(concatenated_tokens) # Shape: [Batch, hidden_dim]
        return visual_embedding


class PlayerEncoder(nn.Module):
    """
    A wrapper module that combines the Vision and Audio encoders for a single player.
    
    This module takes all of a single player's inputs for a frame and produces
    the final fused token that will be fed into the main transformer.
    """
    def __init__(self, hidden_dim: int = 2048, freeze_vit: bool = True):
        super().__init__()
        
        self.vision_encoder = VisionEncoder(hidden_dim=hidden_dim, freeze_vit=freeze_vit)
        self.audio_encoder = AudioEncoder(hidden_dim=hidden_dim)
        
        # Layer Normalization can help stabilize the final fused embedding.
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self,
                foveal_view: torch.Tensor,
                peripheral_view: torch.Tensor,
                spectrogram: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the full player encoder.
        
        Args:
            foveal_view (torch.Tensor): Center crop view.
            peripheral_view (torch.Tensor): Full resized view.
            spectrogram (Optional[torch.Tensor]): Mel spectrogram or None.
            
        Returns:
            torch.Tensor: The final fused player token of shape [Batch, hidden_dim].
        """
        # Get the individual embeddings from the sub-modules
        visual_embedding = self.vision_encoder(foveal_view, peripheral_view)
        audio_embedding = self.audio_encoder(spectrogram)
        
        # The fusion strategy: simple addition. This is a common and effective baseline.
        # It assumes both embeddings now live in the same semantic space.
        fused_embedding = visual_embedding + audio_embedding
        
        # Apply layer normalization
        final_token = self.final_norm(fused_embedding)
        
        return final_token


if __name__ == '__main__':
    # ==================================
    # Example Usage and Sanity Checks
    # ==================================
    
    # --- Configuration ---
    BATCH_SIZE = 4
    HIDDEN_DIM = 2048 # Our D
    
    # --- Create Dummy Inputs ---
    dummy_foveal_view = torch.randn(BATCH_SIZE, 3, 224, 224)
    dummy_peripheral_view = torch.randn(BATCH_SIZE, 3, 384, 384)
    dummy_spectrogram = torch.randn(BATCH_SIZE, 128, 6) # With audio
    dummy_spectrogram_none = None # Case with no audio
    
    print("--- Initializing PlayerEncoder (ViT Frozen) ---")
    player_encoder = PlayerEncoder(hidden_dim=HIDDEN_DIM, freeze_vit=True)
    
    # --- Check ViT Freezing ---
    vit_params = sum(p.numel() for p in player_encoder.vision_encoder.vit.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in ViT: {vit_params}") # Should be 0
    assert vit_params == 0
    
    # --- Test Forward Pass (with audio) ---
    print("\n--- Testing forward pass with audio ---")
    output_token = player_encoder(dummy_foveal_view, dummy_peripheral_view, dummy_spectrogram)
    print(f"Shape of the final player token: {output_token.shape}")
    assert output_token.shape == (BATCH_SIZE, HIDDEN_DIM)
    
    # --- Test Forward Pass (without audio) ---
    print("\n--- Testing forward pass without audio ---")
    output_token_no_audio = player_encoder(dummy_foveal_view, dummy_peripheral_view, dummy_spectrogram_none)
    print(f"Shape of the final token (no audio): {output_token_no_audio.shape}")
    assert output_token_no_audio.shape == (BATCH_SIZE, HIDDEN_DIM)
    
    # --- Check ViT Unfreezing ---
    print("\n--- Unfreezing ViT for fine-tuning ---")
    player_encoder.vision_encoder.unfreeze_vit()
    vit_params_unfrozen = sum(p.numel() for p in player_encoder.vision_encoder.vit.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in ViT after unfreezing: {vit_params_unfrozen}")
    assert vit_params_unfrozen > 300_000_000 # ViT-Large has ~307M params
    
    print("\nStage 1 module built successfully!")