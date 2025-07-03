import torch
import time
from transformers import PerformerConfig, PerformerForCausalLM

def benchmark_performer_latency():
    # --- Configuration ---
    # A Performer model configured to be roughly ~200M parameters
    # Note: Performer's parameter count is slightly different due to its structure.
    config = PerformerConfig(
        vocab_size=50257,
        hidden_size=768,
        num_hidden_layers=16, # Increased layers to get closer to 200M
        num_attention_heads=12,
        max_position_embeddings=4096 * 2,
        attention_type="kernelized", # This is the key for Performer
        kernelized_attention_config={
            "feature_map_type": "favor",
            "num_random_features": 256,
        }
    )
    
    # Game Simulation Parameters (same as before)
    TICKS_PER_SECOND = 30
    MAX_GAME_SECONDS = 180
    MAX_SEQUENCE_LENGTH = TICKS_PER_SECOND * MAX_GAME_SECONDS

    # --- Setup ---
    if not torch.cuda.is_available():
        print("CUDA not available. This benchmark requires a GPU.")
        return

    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(0)}")

    # Initialize the model and move to GPU in FP16
    print("Initializing Performer model...")
    model = PerformerForCausalLM(config).to(device).half()
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameter count: {num_params / 1_000_000:.2f}M")

    print("\n--- Starting Benchmark ---")
    
    # --- GPU Warmup ---
    print("Warming up GPU...")
    with torch.no_grad():
        for _ in range(10):
            dummy_input = torch.randint(0, config.vocab_size, (1, 1024), device=device)
            _ = model(dummy_input)
    torch.cuda.synchronize()
    print("Warmup complete.")

    # --- Benchmark Loop ---
    current_sequence_length = 0
    all_latencies = []

    for second in range(1, MAX_GAME_SECONDS + 1):
        current_sequence_length += TICKS_PER_SECOND
        if current_sequence_length > MAX_SEQUENCE_LENGTH:
             current_sequence_length = MAX_SEQUENCE_LENGTH

        input_ids = torch.randint(
            0, config.vocab_size, (1, current_sequence_length), device=device
        )
        
        # --- Time the forward pass ---
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            _ = model(input_ids)
            
        torch.cuda.synchronize()
        end_time = time.time()
        # --- End timing ---

        latency_ms = (end_time - start_time) * 1000
        all_latencies.append((current_sequence_length, latency_ms))
        
        if second % 10 == 0 or second == 1:
            print(f"Second: {second:3d} | Seq Len: {current_sequence_length:4d} | Latency: {latency_ms:8.2f} ms")

        if latency_ms > 33.33:
            print("\n------------------------------------------------------")
            print(f"!!! REAL-TIME CONSTRAINT FAILED at {second} seconds !!!")
            print(f"Latency ({latency_ms:.2f} ms) exceeded the 33.3 ms budget.")
            print("------------------------------------------------------")
            break

    print("\n--- Benchmark Complete ---")
    return all_latencies


if __name__ == "__main__":
    results = benchmark_performer_latency()
    if results:
        try:
            import matplotlib.pyplot as plt
            seq_lens = [r[0] for r in results]
            latencies = [r[1] for r in results]
            
            plt.figure(figsize=(10, 6))
            plt.plot(seq_lens, latencies, marker='o', linestyle='-')
            plt.axhline(y=33.33, color='r', linestyle='--', label='33.3ms Real-time Budget')
            plt.title('Performer Single-Shot Latency vs. Sequence Length')
            plt.xlabel('Sequence Length (tokens)')
            plt.ylabel('Latency (ms)')
            plt.grid(True)
            plt.legend()
            plt.show()
        except ImportError:
            print("\nMatplotlib not found. Skipping plot.")
            print("To install: pip install matplotlib")