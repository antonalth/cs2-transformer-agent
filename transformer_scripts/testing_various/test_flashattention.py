import torch
import torch.nn as nn
import time
import math

# A simple Positional Encoding layer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def setup_benchmark_model(num_params_target_millions, sequence_length, batch_size=1):
    """
    Builds a vanilla Transformer Encoder to approximate the target parameter count.
    """
    print("--- Setting up benchmark ---")
    
    # 1. Check for GPU and PyTorch version
    if not torch.cuda.is_available() or not hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        raise EnvironmentError(
            "CUDA and PyTorch 2.0+ are required for this benchmark. "
            "FlashAttention is not available."
        )
    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}")

    # 2. Configure a vanilla Transformer Encoder
    # We'll adjust dimensions to get close to the 200-300M parameter target.
    # d_model^2 * num_layers * 12 is a rough approximation for encoder params.
    # Let's try d_model=1024, num_layers=12, d_ff=4096 (standard ratio)
    d_model = 1024 # Embedding dimension
    nhead = 16     # Number of attention heads
    d_hid = 4096   # Dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 12   # Number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    
    encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_hid, batch_first=False)
    pos_encoder = PositionalEncoding(d_model, max_len=sequence_length)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)
    
    model = nn.Sequential(
        nn.Embedding(30000, d_model), # Dummy embedding layer
        pos_encoder,
        transformer_encoder
    ).to(device)
    
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model configured with ~{num_params / 1_000_000:.2f}M parameters.")

    # 3. Create dummy input data
    dummy_input_ids = torch.randint(0, 30000, (batch_size, sequence_length), device=device)
    
    print(f"Created dummy input tensor with shape: {dummy_input_ids.shape}")
    print("--- Setup complete ---")
    
    return model, dummy_input_ids

def run_benchmark(model, inputs, warmup_runs, benchmark_runs):
    """
    Runs the benchmark, explicitly enabling FlashAttention.
    """
    print("\n--- Starting benchmark with FlashAttention enabled ---")
    
    # Use the SDPA kernel context manager to explicitly enable FlashAttention.
    # This will use the FlashAttention kernel if available, otherwise it will error.
    with torch.no_grad(), torch.nn.functional.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        # --- WARM-UP PHASE ---
        print(f"Performing {warmup_runs} warm-up runs...")
        for _ in range(warmup_runs):
            _ = model(inputs)
        torch.cuda.synchronize()
        print("Warm-up complete.")

        # --- MEASUREMENT PHASE ---
        print(f"Running {benchmark_runs} benchmark runs...")
        start_time = time.perf_counter()
        for _ in range(benchmark_runs):
            _ = model(inputs)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        print("Benchmark runs complete.")

    # --- CALCULATE RESULTS ---
    total_time = end_time - start_time
    avg_time_per_run = total_time / benchmark_runs
    fps = 1 / avg_time_per_run
    
    print("\n--- Benchmark Results ---")
    print(f"Total time for {benchmark_runs} runs: {total_time:.4f} seconds")
    print(f"Average latency per run: {avg_time_per_run * 1000:.4f} ms")
    print(f"Inference FPS (Frames Per Second): {fps:.2f}")
    
    return fps

if __name__ == "__main__":
    # --- Configuration ---
    TARGET_PARAMS_MILLIONS = 250
    SEQUENCE_LENGTH = 4096 # Maximum length
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