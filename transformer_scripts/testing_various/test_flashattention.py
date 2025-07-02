import torch
import time
import argparse
from transformers import LlamaConfig, AutoModelForCausalLM

def configure_llama_model(target_m_params):
    """
    Approximates a LlamaConfig for a given target parameter size.
    This is a heuristic and may not be exact.
    """
    # A simple scaling relationship found through experimentation
    # L(d, l) = 12*l*d^2 where L is params, l is layers, d is hidden size
    # We fix the number of layers and find the hidden size.
    # Let's use a standard layer count for different sizes.
    if target_m_params <= 150: # ~125M
        n_layers = 12
        n_heads = 12
        n_kv_heads = 4
        intermediate_size_ratio = 2.6
    elif target_m_params <= 400: # ~350M
        n_layers = 24
        n_heads = 16
        n_kv_heads = 4
        intermediate_size_ratio = 2.6
    elif target_m_params <= 800: # ~700M
        n_layers = 32
        n_heads = 32
        n_kv_heads = 8
        intermediate_size_ratio = 2.6
    else: # ~1.3B+
        n_layers = 40
        n_heads = 40
        n_kv_heads = 8
        intermediate_size_ratio = 2.6

    # Heuristic formula to find d_model for a given parameter count and num_layers
    # Params ≈ 2 * n_layers * d_model^2 + intermediate_size_ratio * n_layers * d_model
    # This is complex to solve directly, so we use a simpler approx: Params ~ 12 * L * D^2
    # D = sqrt(Params / (12 * L))
    d_model_approx = int((target_m_params * 1_000_000 / (12 * n_layers))**0.5)
    # Round to the nearest multiple of 64 for efficiency
    d_model = round(d_model_approx / 64) * 64
    
    intermediate_size = int(d_model * intermediate_size_ratio)

    print(f"Targeting ~{target_m_params}M params. Calculated config: "
          f"d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}")

    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=d_model,
        intermediate_size=intermediate_size,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        max_position_embeddings=8192,
        rms_norm_eps=1e-5,
    )
    return config

def benchmark_recurrent(args):
    # --- Configuration ---
    config = configure_llama_model(args.size)
    
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("CUDA not available. This benchmark requires a GPU.")
        return
        
    print(f"Using device: {torch.cuda.get_device_name(0)}")

    try:
        print(f"Initializing custom Llama model from config with Flash Attention 2...")
        model = AutoModelForCausalLM.from_config(
            config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
        ).to(device)
        print("Model successfully created with Flash Attention 2 backend.")
    except Exception as e:
        print(f"\n!!! FAILED TO CREATE MODEL: {e} !!!")
        return

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
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                model_output = model(input_ids, past_key_values=past_key_values, use_cache=True)
                past_key_values = model_output.past_key_values
                
            torch.cuda.synchronize()
            end_time = time.time()
            latencies_this_second.append((end_time - start_time) * 1000)

        avg_latency = sum(latencies_this_second) / len(latencies_this_second)
        
        cache_seq_len = past_key_values[0][0].shape[2] if past_key_values else 0
        all_results.append((total_ticks, avg_latency))

        if second % 10 == 0 or second == 1:
            print(f"{second:6d} | {avg_latency:16.2f} | {cache_seq_len:12d}")

    print("\n--- Benchmark Complete ---")
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Recurrent Transformer with FlashAttention-2.")
    parser.add_argument(
        '--size', 
        type=int, 
        default=300, 
        help='Target model size in millions of parameters (e.g., 300).'
    )
    parser.add_argument(
        '--duration', 
        type=int, 
        default=180, 
        help='Duration of the simulation in seconds.'
    )
    parser.add_argument(
        '--tokens_per_tick', 
        type=int, 
        default=1, 
        help='Number of new tokens to add to the context at each tick.'
    )
    parser.add_argument(
        '--tickrate', 
        type=int, 
        default=30, 
        help='Number of ticks (updates) per second.'
    )
    
    args = parser.parse_args()
    
    results = benchmark_recurrent(args)
    
    if results:
        try:
            import matplotlib.pyplot as plt
            ticks = [r[0] for r in results]
            latencies = [r[1] for r in results]
            
            plt.figure(figsize=(12, 7))
            plt.plot(ticks, latencies, marker='.', linestyle='-')
            plt.ylim(0, 40)
            
            real_time_budget_ms = 1000 / args.tickrate
            plt.axhline(y=real_time_budget_ms, color='r', linestyle='--', 
                        label=f'{real_time_budget_ms:.2f}ms Real-time Budget ({args.tickrate}Hz)')
            
            plt.title(f'Recurrent Transformer Latency ({args.size}M params)')
            plt.xlabel('Total Ticks Processed')
            plt.ylabel('Average Latency per Update (ms)')
            plt.grid(True)
            plt.legend()
            plt.show()
        except ImportError:
            print("\nMatplotlib not found. Skipping plot.")