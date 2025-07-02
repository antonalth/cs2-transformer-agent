import torch
import torch.nn as nn
import time
from transformers import LlamaConfig, LlamaForCausalLM

def benchmark_long_context_recurrent():
    # --- Configuration ---
    # 1. Define a custom Llama-style configuration for ~300M params and long context
    # We will verify the final parameter count after initialization.
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=1024,
        intermediate_size=2816, # A common ratio for Llama-style models
        num_hidden_layers=24,   # gpt2-medium has 24 layers with this hidden size
        num_attention_heads=16,
        num_key_value_heads=4,  # Using Grouped-Query Attention (GQA) for efficiency
        max_position_embeddings=8192, # CRUCIAL: Allows for >5400 token context
        rms_norm_eps=1e-5,
    )
    
    # Game Simulation Parameters
    TICKS_PER_SECOND = 30
    # We will only run up to the max supported length to save time, 
    # but could go to 180 seconds.
    # 5400 tokens / 30 ticks/sec = 180 seconds
    MAX_GAME_SECONDS = 180

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("CUDA not available. This benchmark requires a GPU.")
        return
        
    print(f"Using device: {torch.cuda.get_device_name(0)}")

    try:
        print("Initializing custom ~300M Llama model from config with Flash Attention 2...")
        # Create the model directly from the config object
        # The weights will be random, which is fine for a latency benchmark.
        model = LlamaForCausalLM(config, attn_implementation="flash_attention_2")
        model = model.to(device).to(torch.float16) # Move to GPU and set precision
        
        print("Model successfully created with Flash Attention 2 backend.")
    except Exception as e:
        print(f"\n!!! FAILED TO CREATE MODEL: {e} !!!")
        print("Ensure your environment supports Flash Attention 2.")
        return

    model.eval()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameter count: {num_params / 1_000_000:.2f}M") # Verify the size

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
    
    print("Simulating game up to 5400 token context...")
    print("Format: Second | Avg Latency (ms) | Cache Length")
    
    for second in range(1, MAX_GAME_SECONDS + 1):
        latencies_this_second = []
        
        for tick in range(TICKS_PER_SECOND):
            input_ids = torch.randint(0, config.vocab_size, (1, 1), device=device)
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                model_output = model(input_ids, past_key_values=past_key_values, use_cache=True)
                past_key_values = model_output.past_key_values
                
            torch.cuda.synchronize()
            end_time = time.time()
            latencies_this_second.append((end_time - start_time) * 1000)

        avg_latency = sum(latencies_this_second) / len(latencies_this_second)
        
        if past_key_values:
            cache_seq_len = past_key_values[0][0].shape[2]
        else:
            cache_seq_len = 0
        
        all_results.append((cache_seq_len, avg_latency))

        if second % 10 == 0 or second == 1:
            print(f"{second:6d} | {avg_latency:16.2f} | {cache_seq_len:12d}")

    print("\n--- Benchmark Complete ---")
    return all_results

if __name__ == "__main__":
    results = benchmark_long_context_recurrent()
    if results:
        try:
            import matplotlib.pyplot as plt
            seq_lens = [r[0] for r in results]
            latencies = [r[1] for r in results]
            
            plt.figure(figsize=(12, 7))
            plt.plot(seq_lens, latencies, marker='.', linestyle='-')
            plt.ylim(0, 40) # Keep Y-axis fixed to see stability
            plt.axhline(y=33.33, color='r', linestyle='--', label='33.3ms Real-time Budget')
            plt.title('Recurrent FlashAttention-2 Latency on Custom 300M Model')
            plt.xlabel('KV Cache Sequence Length (tokens)')
            plt.ylabel('Average Latency per Update (ms)')
            plt.grid(True)
            plt.legend()
            plt.show()
        except ImportError:
            print("\nMatplotlib not found. Skipping plot.")