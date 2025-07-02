import torch
import time
from transformers import AutoConfig, AutoModelForCausalLM

def benchmark_recurrent_flashattention():
    # --- Configuration ---
    # We'll use a standard, well-supported model. gpt2-medium is ~355M params.
    MODEL_NAME = "gpt2-medium" 
    
    # Game Simulation Parameters
    TICKS_PER_SECOND = 30
    MAX_GAME_SECONDS = 180
    
    # --- Setup ---
    if not torch.cuda.is_available():
        print("CUDA not available. This benchmark requires a GPU.")
        return

    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(0)}")

    try:
        print(f"Attempting to load '{MODEL_NAME}' with Flash Attention 2...")
        # This is the key step. If Flash Attention 2 is not available in the
        # environment, this line will raise an error.
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        ).to(device)
        print("Model successfully loaded with Flash Attention 2 backend.")
    except Exception as e:
        print("\n-------------------------------------------------------------")
        print("!!! FAILED TO LOAD MODEL WITH FLASH ATTENTION 2 !!!")
        print("Your environment may not be set up correctly.")
        print("Please ensure you have a compatible GPU, PyTorch version, and CUDA version.")
        print(f"Error: {e}")
        print("-------------------------------------------------------------")
        return

    model.eval()
    config = model.config
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameter count: {num_params / 1_000_000:.2f}M")

    print("\n--- Starting Benchmark ---")
    
    # --- GPU Warmup ---
    print("Warming up GPU with recurrent steps...")
    kv_cache = None
    with torch.no_grad():
        for _ in range(50):
            input_ids = torch.randint(0, config.vocab_size, (1, 1), device=device)
            _, kv_cache = model(input_ids, past_key_values=kv_cache)
    torch.cuda.synchronize()
    print("Warmup complete.")

    # --- Benchmark Loop ---
    kv_cache = None
    all_results = []
    
    print("Simulating game...")
    print("Format: Second | Avg Latency (ms) | Cache Length")
    
    for second in range(1, MAX_GAME_SECONDS + 1):
        latencies_this_second = []
        
        # This inner loop simulates the 30Hz updates
        for tick in range(TICKS_PER_SECOND):
            # The input is always just ONE new token
            input_ids = torch.randint(0, config.vocab_size, (1, 1), device=device)
            
            # --- Time the SINGLE recurrent step ---
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                # The model processes one token and uses/updates the KV cache
                _, kv_cache = model(input_ids, past_key_values=kv_cache)
                
            torch.cuda.synchronize()
            end_time = time.time()
            # --- End timing ---

            latencies_this_second.append((end_time - start_time) * 1000)

        # Calculate stats for the second
        avg_latency = sum(latencies_this_second) / len(latencies_this_second)
        cache_seq_len = kv_cache[0][0].shape[2] # Get sequence length from the cache tensor
        all_results.append((cache_seq_len, avg_latency))

        if second % 10 == 0 or second == 1:
            print(f"{second:6d} | {avg_latency:16.2f} | {cache_seq_len:12d}")

    print("\n--- Benchmark Complete ---")
    return all_results

if __name__ == "__main__":
    results = benchmark_recurrent_flashattention()
    if results:
        try:
            import matplotlib.pyplot as plt
            seq_lens = [r[0] for r in results]
            latencies = [r[1] for r in results]
            
            plt.figure(figsize=(10, 6))
            plt.plot(seq_lens, latencies, marker='o', linestyle='-')
            # Set a more appropriate Y-axis limit to see the stability
            plt.ylim(0, max(latencies) * 1.5) 
            plt.axhline(y=33.33, color='r', linestyle='--', label='33.3ms Real-time Budget')
            plt.title('Recurrent FlashAttention-2 Latency vs. KV Cache Size')
            plt.xlabel('KV Cache Sequence Length (tokens)')
            plt.ylabel('Average Latency per Update (ms)')
            plt.grid(True)
            plt.legend()
            plt.show()
        except ImportError:
            print("\nMatplotlib not found. Skipping plot.")
            print("To install: pip install matplotlib")