import torch
import time
from transformers import AutoModelForCausalLM

def benchmark_recurrent_flashattention():
    # --- Configuration ---
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
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        ).to(device)
        print("Model successfully loaded with Flash Attention 2 backend.")
    except Exception as e:
        print("\n-------------------------------------------------------------")
        print("!!! FAILED TO LOAD MODEL WITH FLASH ATTENTION 2 !!!")
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
    
    print("Simulating game...")
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
        
        # --- THIS IS THE FIX ---
        # Get sequence length from the legacy tuple format
        # `past_key_values` is a tuple of layers
        # `past_key_values[0]` is the tuple for the first layer (key, value)
        # `past_key_values[0][0]` is the key tensor for the first layer
        # Its shape is [batch, heads, seq_len, head_dim], so shape[2] is the length
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
    results = benchmark_recurrent_flashattention()
    if results:
        try:
            import matplotlib.pyplot as plt
            seq_lens = [r[0] for r in results]
            latencies = [r[1] for r in results]
            
            plt.figure(figsize=(10, 6))
            plt.plot(seq_lens, latencies, marker='o', linestyle='-')
            plt.ylim(0, 35) # Set a fixed Y-axis limit to see stability clearly
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