# benchmark.py

import torch
import time
import argparse
import humanize  # For formatting numbers nicely (pip install humanize)

# Import the model definition from the adjacent model.py file
from model import CS2Transformer
from torch.cuda.amp import autocast

def run_benchmark_for_config(config: dict, args: argparse.Namespace):
    """
    Instantiates and benchmarks a single model configuration.
    
    Args:
        config (dict): A dictionary describing the model parameters.
        args (argparse.Namespace): Command-line arguments.
    
    Returns:
        A tuple containing (parameter_count, average_time_ms).
    """
    device = torch.device(args.device)
    
    print(f"\n--- Benchmarking Config: {config['name']} ---")
    print(f"  Layers: {config['layers']}, Dim: {config['dim']}, Heads: {config['heads']}")

    try:
        # --- 1. Model Initialization ---
        # We freeze the ViT because we only want to measure the performance
        # of the main transformer backbone that we are scaling.
        model = CS2Transformer(
            hidden_dim=config['dim'],
            num_layers=config['layers'],
            num_heads=config['heads'],
            freeze_vit=True
        ).to(device)
        model.eval() # Set to evaluation mode

        # Calculate the number of trainable parameters (ViT is frozen)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable Parameters: {humanize.intword(params)} ({params:,})")

        # --- 2. Dummy Data Creation ---
        B, S, P = args.batch_size, args.frames, 5 # Batch, Sequence, Players
        dummy_round_data = {
            'foveal_views': torch.randn(B, S, P, 3, 384, 384, device=device),
            'peripheral_views': torch.randn(B, S, P, 3, 384, 384, device=device),
            'spectrograms': torch.randn(B, S, P, 128, 6, device=device),
            'is_alive_mask': torch.ones((B, S, P), dtype=torch.bool, device=device),
            'is_masked_frame_mask': torch.zeros((B, S), dtype=torch.bool, device=device),
        }
        # For inference speed, we don't need to mask, but the model expects the key.

        # --- 3. Benchmarking Loop ---
        with torch.no_grad(), autocast():
            # Warmup loop for GPU to handle kernel launch overhead
            if device.type == 'cuda':
                print("  Warming up GPU...")
                for _ in range(5):
                    _ = model(dummy_round_data)
                torch.cuda.synchronize()

            # Timed loop
            print("  Running benchmark...")
            start_time = time.perf_counter()
            for _ in range(args.iterations):
                _ = model(dummy_round_data)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()

        total_time = end_time - start_time
        avg_time_ms = (total_time / args.iterations) * 1000
        print(f"  Done. Average forward pass: {avg_time_ms:.2f} ms")
        
        return params, avg_time_ms

    except torch.cuda.OutOfMemoryError:
        print("  ERROR: Ran out of memory on GPU for this configuration.")
        return params, "OOM"
    except Exception as e:
        print(f"  An unexpected error occurred: {e}")
        return 0, "Error"


def main():
    parser = argparse.ArgumentParser(description="Benchmark different sizes of the CS2Transformer model.")
    parser.add_argument('--frames', type=int, default=100, help="Number of frames in the benchmark sequence.")
    parser.add_argument('--batch-size', type=int, default=2, help="Batch size for the benchmark.")
    parser.add_argument('--iterations', type=int, default=10, help="Number of timed iterations to average.")
    
    # Auto-select device
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument('--device', type=str, default=default_device, help="Device to run on (e.g., 'cuda', 'cpu').")
    
    args = parser.parse_args()

    print("="*60)
    print("CS2Transformer Performance Benchmark")
    print("="*60)
    print(f"Device: {args.device.upper()}")
    print(f"Sequence Length: {args.frames} frames")
    print(f"Batch Size: {args.batch_size}")
    print(f"Timed Iterations: {args.iterations}")
    print("-"*60)
    
    # Define model configurations to test, from ~100M to ~1B
    model_configs = [
        {
            'name': 'Small (~100M)',
            'layers': 8,
            'dim': 1024,
            'heads': 16, # 1024 / 16 = 64
        },
        {
            'name': 'Medium (~200M)',
            'layers': 4,
            'dim': 2048,
            'heads': 32, # 2048 / 32 = 64
        },
        {
            'name': 'Large (~500M)',
            'layers': 10,
            'dim': 2048,
            'heads': 32,
        },
        {
            'name': 'XL (~1B)',
            'layers': 20,
            'dim': 2048,
            'heads': 32,
        },
    ]

    results = []
    for config in model_configs:
        params, avg_time = run_benchmark_for_config(config, args)
        results.append((config['name'], params, avg_time))
        # Clean up memory between runs
        if args.device == 'cuda':
            torch.cuda.empty_cache()

    # --- Print Final Summary Table ---
    print("\n" + "="*60)
    print("Benchmark Summary")
    print("="*60)
    print(f"{'Configuration':<20} | {'Parameters':<20} | {'Avg. Time (ms/pass)':<25}")
    print("-"*60)
    for name, params, avg_time in results:
        param_str = f"{humanize.intword(params)} ({params:,})" if params > 0 else "N/A"
        time_str = f"{avg_time:.2f}" if isinstance(avg_time, float) else avg_time
        print(f"{name:<20} | {param_str:<20} | {time_str:<25}")
    print("-"*60)


if __name__ == '__main__':
    main()