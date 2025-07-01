import torch
import torch.nn as nn
import time
import math

class PositionalEncoding(nn.Module):
    # Same as before, no changes needed here.
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # Note the shape for batch_first=True will be [batch_size, seq_len, embedding_dim]
        # so we adjust pe shape to [1, max_len, d_model] for broadcasting
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def setup_benchmark_model(num_params_target_millions, sequence_length, batch_size=1):
    print("--- Setting up benchmark ---")
    
    if not torch.cuda.is_available() or not hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        raise EnvironmentError("CUDA and PyTorch 2.0+ are required.")
    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}")

    # --- FIX: Set batch_first=True as recommended by the warning ---
    d_model = 1024
    nhead = 16
    d_hid = 4096
    nlayers = 12
    
    encoder_layer = nn.TransformerEncoderLayer(
        d_model, nhead, d_hid, batch_first=True, activation=nn.GELU()
    )
    pos_encoder = PositionalEncoding(d_model, max_len=sequence_length)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)
    
    model = nn.Sequential(
        nn.Embedding(30000, d_model),
        pos_encoder,
        transformer_encoder
    ).to(device)
    
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model configured with ~{num_params / 1_000_000:.2f}M parameters.")

    # Input shape is now (batch_size, sequence_length) as before
    dummy_input_ids = torch.randint(0, 30000, (batch_size, sequence_length), device=device)
    
    print(f"Created dummy input tensor with shape: {dummy_input_ids.shape}")
    print("--- Setup complete ---")
    
    return model, dummy_input_ids

def run_benchmark(model, inputs, warmup_runs, benchmark_runs):
    print("\n--- Starting benchmark with Automatic Kernel Dispatch ---")

    # --- NEW: Let's check which optimized backends are available ---
    print("\nVerifying available backends:")
    has_flash = torch.backends.cuda.flash_sdp_enabled()
    has_mem_efficient = torch.backends.cuda.mem_efficient_sdp_enabled()
    print(f"  FlashAttention backend enabled: {has_flash}")
    print(f"  Memory-Efficient backend enabled: {has_mem_efficient}")
    if not has_flash and not has_mem_efficient:
        print("  Warning: No optimized attention backend detected. Performance will be slower.")
    else:
        print("  PyTorch will automatically use the best available backend.")
    
    # --- SIMPLIFIED: No context manager needed! PyTorch does it automatically. ---
    with torch.no_grad():
        print(f"\nPerforming {warmup_runs} warm-up runs...")
        for _ in range(warmup_runs):
            _ = model(inputs)
        torch.cuda.synchronize()
        print("Warm-up complete.")

        print(f"Running {benchmark_runs} benchmark runs...")
        start_time = time.perf_counter()
        for _ in range(benchmark_runs):
            _ = model(inputs)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        print("Benchmark runs complete.")

    total_time = end_time - start_time
    avg_time_per_run = total_time / benchmark_runs
    fps = 1 / avg_time_per_run
    
    print("\n--- Benchmark Results ---")
    print(f"Total time for {benchmark_runs} runs: {total_time:.4f} seconds")
    print(f"Average latency per run: {avg_time_per_run * 1000:.4f} ms")
    print(f"Inference FPS (Frames Per Second): {fps:.2f}")
    
    return fps

if __name__ == "__main__":
    TARGET_PARAMS_MILLIONS = 250
    SEQUENCE_LENGTH = 4096
    BATCH_SIZE = 1
    WARMUP_RUNS = 10
    BENCHMARK_RUNS = 100
    
    try:
        model, dummy_inputs = setup_benchmark_model(TARGET_PARAMS_MILLIONS, SEQUENCE_LENGTH, BATCH_SIZE)
        run_benchmark(model, dummy_inputs, WARMUP_RUNS, BENCHMARK_RUNS)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()