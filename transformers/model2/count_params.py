import torch
import logging
from config import GlobalConfig
from model import GamePredictorBackbone

# Mock logging
logging.basicConfig(level=logging.INFO)

def count_parameters(model):
    table = []
    total_params = 0
    trainable_params = 0
    
    # Iterate through named modules to group by major components
    # We'll do a simple recursive walk or just check top-level children
    
    print(f"{'Module':<40} | {'Total':<15} | {'Trainable':<15}")
    print("-" * 76)
    
    for name, module in model.named_children():
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        print(f"{name:<40} | {total:<15,} | {trainable:<15,}")
        
        total_params += total
        trainable_params += trainable
        
    print("-" * 76)
    print(f"{'Total':<40} | {total_params:<15,} | {trainable_params:<15,}")

def main():
    # Load default config
    # We need to point to a valid config file or let it use defaults
    # The GlobalConfig defaults might need adjustment if they rely on existing files
    # But usually dataclasses have defaults.
    
    # However, ModelConfig needs to match what is being trained.
    # I'll create a dummy config with the defaults from the code.
    
    # We can try to load from the file on disk if it exists, or just instantiate defaults.
    # The defaults in config.py are what we just edited.
    
    # Let's instantiate from scratch using the classes in config.py
    # We need to import them.
    
    cfg = GlobalConfig(
        dataset=None, # Not needed for model
        model=None,   # Will use defaults
        train=None    # Not needed
    )
    # Re-trigger post_init to set defaults if they were None
    from config import ModelConfig
    cfg.model = ModelConfig()
    cfg.model.__post_init__()
    
    print("Instantiating model...")
    model = GamePredictorBackbone(cfg.model)
    
    print("\nParameter Count Breakdown:")
    count_parameters(model)

if __name__ == "__main__":
    main()
