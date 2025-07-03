# ==============================================================================
#
#  Production-Grade Transformer Benchmarking Tool
#
#  This script provides a comprehensive framework for benchmarking the latency
#  of modern transformer architectures in a real-time, recurrent setting.
#
#  Features:
#  - Command-line control over model size, simulation parameters, and attention mechanism.
#  - Head-to-head comparison of:
#      1. FlashAttention-2 ('flash'): The state-of-the-art, fastest implementation.
#      2. PyTorch SDPA ('sdpa'): The modern, robust default that dispatches to the
#         best available backend (often FlashAttention).
#      3. Eager Attention ('eager'): The original, legacy implementation for baseline comparison.
#  - Simulates a real-time loop by processing small batches of new tokens against
#    a growing Key-Value (KV) Cache, measuring the per-update latency.
#  - Dynamically creates a modern, Llama-style model with Grouped-Query Attention
#    to match the specified parameter count.
#
#  Usage:
#  # Test the default (300M, SDPA attention)
#  python benchmark_script.py
#
#  # Test a 700M model with explicit FlashAttention-2
#  python benchmark_script.py --size 700 --attn flash
#
#  # Compare against the slow, legacy eager attention
#  python benchmark_script.py --size 300 --attn eager
#
# ==============================================================================

import torch
import time
import argparse
import matplotlib.pyplot as plt
from transformers import LlamaConfig, AutoModelForCausalLM

def configure_llama_model(target_m_params):
    """
    Approximates a LlamaConfig for a given target parameter size.
    This is a heuristic that creates a reasonable modern architecture (GQA, etc.).
    """
    # Define architectural presets for different size classes
    if target_m_params <= 150:  # e.g., ~125M
        n_layers, n_heads, n_kv_heads = 12, 12, 4
    elif target_m_params <= 400: # e.g., ~300M
        n_layers, n_heads, n_kv_heads = 24, 16, 4
    elif target_m_params <= 800: # e.g., ~700M
        n_layers, n_heads, n_kv_heads = 32, 32, 8
    else:  # e.g., ~1.3B+
        n_layers, n_heads, n_kv_heads = 40, 40, 8

    # Heuristic formula to find hidden size (d_model) based on params and layers.
    # A simplified model parameter count is roughly: P ~= 2 * L * D^2 (for embeddings/projections)
    # + L * (4 * D^2) for FFN layers, but let's use a simpler heuristic for configuration.
    # An empirical scaling factor often used is ~12-14.
    d_model_approx = int((target_m_params * 1_000_000 / (12 * n_layers))**0.5)
    # Round to the nearest multiple of 64 for GPU efficiency
    d_model = round(d_model_approx / 64) * 64
    
    # Standard Llama 2 feed-forward network size ratio
    intermediate_size = int(d_model * 2.6)

    print(f"Targeting ~{target_m_params}M params. Calculated config: "
          f"d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, GQA_groups={n_kv_heads}")

    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=d_model,
        intermediate_size=intermediate_size,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads, # Grouped-Query Attention
        max_position_embeddings=8192,  # Sufficiently long context
        rms_norm_eps=1e-5,
    )
    return config

def benchmark_recurrent(args):
    """Main function to run the benchmark based on provided arguments."""
    config = configure_llama_model(args.size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        return None, None
        
    print(f"Using device: {torch.cuda.get_device_name(0)}")

    # Map the CLI argument to the correct attn_implementation string
    if args.attn == 'flash':
        attn_impl = "flash_attention_2"
    elif args.attn == 'sdpa':
        attn_impl = "sdpa"
    else: # 'eager'
        attn_impl = "eager"

    print(f"INFO: Setting attention implementation to '{attn_impl}'.")

    try:
        # Use the .from_config() factory method to correctly apply attn_implementation
        # to a randomly initialized model.
        model = AutoModelForCausalLM.from_config(
            config,
            attn_implementation=attn_impl,
            torch_dtype=torch.float16,
        ).to(device)
        print(f"Model successfully created with '{attn_impl}' backend.")
    except Exception as e:
        print(f"\n!!! FAILED TO CREATE MODEL with '{attn_impl}' backend: {e} !!!")
        print("Ensure your environment supports the selected attention implementation.")
        return None, None

    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameter count: {num_params / 1_000_000:.2f}M")

    print("\n--- Starting Benchmark ---")
    
    # --- GPU Warmup ---
    print("Warming up GPU...")
    past_key_values = None
    with torch.no_grad():
        for _ in range(50):
            input_ids = torch.randint(0, config.vocab_size, (1, 1), device=device)
            model_output = model(input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = model_output.past_key_values
    torch.cuda.synchronize()
    print("Warmup complete.")

    # --- Benchmark Loop ---
    past_key_values = None
    all_results = []
    total_ticks = 0
    
    print(f"Simulating game for {args.duration} seconds at {args.tickrate}Hz...")
    print(f"Adding {args.tokens_per_tick} token(s) per tick.")
    print("\nFormat: Second | Avg Latency (ms) | Cache Length")
    
    for second in range(1, args.duration + 1):
        latencies_this_second = []
        for tick in range(args.tickrate):
            total_ticks += 1
            # Create a batch of new tokens to add per tick
            input_ids = torch.randint(0, config.vocab_size, (1, args.tokens_per_tick), device=device)
            
            # Precise timing using CUDA synchronization
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                model_output = model(input_ids, past_key_values=past_key_values, use_cache=True)
                past_key_values = model_output.past_key_values
                
            torch.cuda.synchronize()
            end_time = time.time()
            latencies_this_second.append((end_time - start_time) * 1000)

        avg_latency = sum(latencies_this_second) / len(latencies_this_second)
        
        # Safely get sequence length from the KV cache tuple
        cache_seq_len = past_key_values[0][0].shape[2] if past_key_values else 0
        all_results.append((total_ticks, avg_latency))

        # Log progress periodically
        if second % 10 == 0 or second == 1:
            print(f"{second:6d} | {avg_latency:16.2f} | {cache_seq_len:12d}")

    print("\n--- Benchmark Complete ---")
    return all_results, args

def plot_results(results, args):
    """Generates a plot from the benchmark results."""
    ticks = [r[0] for r in results]
    latencies = [r[1] for r in results]
    
    plt.figure(figsize=(12, 7))
    plt.plot(ticks, latencies, marker='.', linestyle='-')
    
    # Set a reasonable Y-axis limit to show the performance clearly
    y_limit = max(latencies) * 1.2 + 10
    plt.ylim(0, y_limit)
    
    # Calculate and draw the real-time budget line
    real_time_budget_ms = 1000 / args.tickrate
    plt.axhline(y=real_time_budget_ms, color='r', linestyle='--', 
                label=f'{real_time_budget_ms:.2f}ms Real-time Budget ({args.tickrate}Hz)')
    
    # Create a descriptive title based on the arguments
    if args.attn == 'flash':
        attn_title = "FlashAttention-2"
    elif args.attn == 'sdpa':
        attn_title = "PyTorch SDPA (Default)"
    else:
        attn_title = "Standard Eager Attention"
        
    plt.title(f'Recurrent Transformer Latency ({args.size}M params, {attn_title})', fontsize=16)
    plt.xlabel('Total Ticks Processed (Cache Size)', fontsize=12)
    plt.ylabel('Average Latency per Update (ms)', fontsize=12)
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Recurrent Transformer Architectures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--size', type=int, default=300, 
                        help='Target model size in millions of parameters.')
    parser.add_argument('--duration', type=int, default=180, 
                        help='Duration of the simulation in seconds.')
    parser.add_argument('--tokens_per_tick', type=int, default=1, 
                        help='Number of new tokens to add at each tick.')
    parser.add_argument('--tickrate', type=int, default=30, 
                        help='Number of ticks (updates) per second.')
    parser.add_argument(
        '--attn', 
        type=str, 
        default='sdpa',
        choices=['flash', 'sdpa', 'eager'],
        help="The attention implementation to use: 'flash' for FlashAttention-2, "
             "'sdpa' for PyTorch's default, or 'eager' for the legacy baseline."
    )
    
    cli_args = parser.parse_args()
    
    results, args = benchmark_recurrent(cli_args)
    
    if results:
        try:
            plot_results(results, args)
        except ImportError:
            print("\nMatplotlib not found, skipping plot. To install: pip install matplotlib")
        except Exception as e:
            print(f"\nAn error occurred during plotting: {e}")