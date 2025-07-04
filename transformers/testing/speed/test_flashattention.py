# ==============================================================================
#
#  Production-Grade Transformer Benchmarking Tool (with TensorRT Support)
#
#  Adds a '--tensorrt' flag to compile the model for maximum performance.
#
#  Prerequisite:
#  - You MUST install TensorRT-LLM from the official NVIDIA GitHub repository first.
#    https://github.com/NVIDIA/TensorRT-LLM
#
#  Usage:
#  # 1. First run with --tensorrt will be SLOW as it builds the engine.
#  python benchmark_script.py --size 300 --attn tensorrt
#
#  # 2. Subsequent runs will be FAST as they load the pre-built engine.
#  python benchmark_script.py --size 300 --attn tensorrt
#
#  # 3. Compare with other backends.
#  python benchmark_script.py --size 300 --attn flash
#
# ==============================================================================

import torch
import time
import argparse
import matplotlib.pyplot as plt
from transformers import LlamaConfig, AutoModelForCausalLM
import os

# --- Helper Functions (same as before) ---
def configure_llama_model(target_m_params):
    # ... (This function is identical to the previous version)
    if target_m_params <= 150: n_layers, n_heads, n_kv_heads = 12, 12, 4
    elif target_m_params <= 400: n_layers, n_heads, n_kv_heads = 24, 16, 4
    elif target_m_params <= 800: n_layers, n_heads, n_kv_heads = 32, 32, 8
    else: n_layers, n_heads, n_kv_heads = 40, 40, 8
    d_model_approx = int((target_m_params * 1_000_000 / (12 * n_layers))**0.5)
    d_model = round(d_model_approx / 64) * 64
    intermediate_size = int(d_model * 2.6)
    print(f"Targeting ~{target_m_params}M params. Calculated config: "
          f"d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, GQA_groups={n_kv_heads}")
    return LlamaConfig(
        vocab_size=32000, hidden_size=d_model, intermediate_size=intermediate_size,
        num_hidden_layers=n_layers, num_attention_heads=n_heads, num_key_value_heads=n_kv_heads,
        max_position_embeddings=8192, rms_norm_eps=1e-5
    )

# --- TensorRT-LLM Specific Functions ---
def build_tensorrt_engine(config, args):
    """Builds and saves a TensorRT-LLM engine from a LlamaConfig."""
    from tensorrt_llm.builder import Builder
    from tensorrt_llm.network import net_guard
    from tensorrt_llm.plugin.plugin import ContextFMHAType
    from tensorrt_llm.mapping import Mapping
    
    engine_name = f'llama_{args.size}M_fp16.engine'
    engine_dir = 'trt_engines'
    os.makedirs(engine_dir, exist_ok=True)
    engine_path = os.path.join(engine_dir, engine_name)

    if os.path.exists(engine_path):
        print(f"Found existing TensorRT engine: {engine_path}. Skipping build.")
        return engine_path

    print(f"No engine found. Building new TensorRT engine at {engine_path}...")
    print("This will be slow and may take several minutes...")

    # Create a temporary PyTorch model to get random weights
    pytorch_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
    
    builder = Builder()
    builder_config = builder.create_builder_config(
        name='llama',
        precision='float16',
        timing_cache=None,
        tensor_parallel=1,
        parallel_build=True,
    )

    # Initialize TensorRT-LLM network
    trt_network = builder.create_network()
    trt_network.trt_network.strongly_typed = True
    
    # Enable plugins for high-performance features like GQA
    trt_network.add_plugin_config(
        builder_config.plugin_config
    )
    
    with net_guard(trt_network):
        # The from_huggingface_llama function can take a model object directly
        from tensorrt_llm.models import LlamaForCausalLM as TrtLlama
        TrtLlama.from_huggingface_model(pytorch_model, 'llama', 'float16', mapping=Mapping())

    # Build and save the engine
    engine_buffer = builder.build_engine(trt_network, builder_config)
    with open(engine_path, 'wb') as f:
        f.write(engine_buffer)
    
    print(f"TensorRT engine built and saved successfully.")
    return engine_path

def benchmark_tensorrt(engine_path, config, args):
    """Runs the benchmark using a pre-compiled TensorRT-LLM engine."""
    from tensorrt_llm.runtime import ModelRunner, GenerationSession
    
    print("Initializing TensorRT-LLM runtime...")
    # The ModelRunner class handles all the low-level session and buffer management
    runner = ModelRunner.from_dir(os.path.dirname(engine_path), rank=0, stream=torch.cuda.current_stream())
    
    print("\n--- Starting TensorRT Benchmark ---")
    print("Warming up GPU...")
    # Warmup is handled internally by the runtime, but we can do a few runs.
    for _ in range(10):
        dummy_input = torch.randint(0, config.vocab_size, (1, 10), device="cuda")
        runner.generate(dummy_input, max_new_tokens=1)

    all_results = []
    total_ticks = 0
    
    # For TensorRT, we manage the KV cache state via the object itself
    # For recurrent generation, we use `generate` with `max_new_tokens=1` repeatedly
    # This is slightly different from the PyTorch loop but tests the same concept.
    
    print(f"Simulating game for {args.duration} seconds at {args.tickrate}Hz...")
    print("\nFormat: Second | Avg Latency (ms) | Cache Length")
    
    # We create one long input sequence and feed it to the runner chunk by chunk
    full_input_sequence = torch.randint(0, config.vocab_size, (1, args.duration * args.tickrate * args.tokens_per_tick), device="cuda")
    
    current_pos = 0
    for second in range(1, args.duration + 1):
        latencies_this_second = []
        for tick in range(args.tickrate):
            input_ids = full_input_sequence[:, current_pos : current_pos + args.tokens_per_tick]
            current_pos += args.tokens_per_tick
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            # The generate function with max_new_tokens=1 performs one recurrent step
            runner.generate(
                input_ids,
                max_new_tokens=1, # We only care about processing the input and updating state
                eos_token_id=config.vocab_size, # Prevent early stopping
                pad_token_id=config.vocab_size
            )
            
            torch.cuda.synchronize()
            end_time = time.time()
            latencies_this_second.append((end_time - start_time) * 1000)

        avg_latency = sum(latencies_this_second) / len(latencies_this_second)
        cache_seq_len = runner.runtime.session.kv_cache_manager.sequence_length.item()
        all_results.append((current_pos, avg_latency))
        
        if second % 10 == 0 or second == 1:
            print(f"{second:6d} | {avg_latency:16.2f} | {cache_seq_len:12d}")

    print("\n--- Benchmark Complete ---")
    return all_results, args

# --- PyTorch Benchmark Function (for flash, sdpa, eager) ---
def benchmark_pytorch(args):
    # ... (This function is identical to the previous version)
    config = configure_llama_model(args.size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        return None, None
    print(f"Using device: {torch.cuda.get_device_name(0)}")

    if args.attn == 'flash': attn_impl = "flash_attention_2"
    elif args.attn == 'sdpa': attn_impl = "sdpa"
    else: attn_impl = "eager"
    print(f"INFO: Setting attention implementation to '{attn_impl}'.")
    
    try:
        model = AutoModelForCausalLM.from_config(config, attn_implementation=attn_impl, torch_dtype=torch.float16).to(device)
        print(f"Model successfully created with '{attn_impl}' backend.")
    except Exception as e:
        print(f"\n!!! FAILED TO CREATE MODEL with '{attn_impl}' backend: {e} !!!")
        return None, None

    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameter count: {num_params / 1_000_000:.2f}M")
    
    # ... (Rest of the benchmark and warmup loop is identical)
    print("\n--- Starting PyTorch Benchmark ---")
    past_key_values = None
    all_results = []
    total_ticks = 0
    print(f"Simulating game for {args.duration} seconds at {args.tickrate}Hz...")
    print("\nFormat: Second | Avg Latency (ms) | Cache Length")
    for second in range(1, args.duration + 1):
        latencies_this_second = []
        for tick in range(args.tickrate):
            total_ticks += 1
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
    return all_results, args


def plot_results(results, args):
    # ... (This function is identical to the previous version)
    ticks = [r[0] for r in results]
    latencies = [r[1] for r in results]
    plt.figure(figsize=(12, 7))
    plt.plot(ticks, latencies, marker='.', linestyle='-')
    y_limit = max(latencies) * 1.2 + 10
    plt.ylim(0, y_limit)
    real_time_budget_ms = 1000 / args.tickrate
    plt.axhline(y=real_time_budget_ms, color='r', linestyle='--', label=f'{real_time_budget_ms:.2f}ms Real-time Budget ({args.tickrate}Hz)')
    if args.attn == 'flash': attn_title = "FlashAttention-2"
    elif args.attn == 'sdpa': attn_title = "PyTorch SDPA"
    elif args.attn == 'tensorrt': attn_title = "TensorRT-LLM"
    else: attn_title = "Standard Eager Attention"
    plt.title(f'Recurrent Transformer Latency ({args.size}M params, {attn_title})', fontsize=16)
    plt.xlabel('Total Ticks Processed (Cache Size)', fontsize=12)
    plt.ylabel('Average Latency per Update (ms)', fontsize=12)
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Recurrent Transformer Architectures.")
    parser.add_argument('--size', type=int, default=300, help='Target model size in millions of parameters.')
    parser.add_argument('--duration', type=int, default=180, help='Duration of the simulation in seconds.')
    parser.add_argument('--tokens_per_tick', type=int, default=1, help='Number of new tokens to add at each tick.')
    parser.add_argument('--tickrate', type=int, default=30, help='Number of ticks (updates) per second.')
    parser.add_argument(
        '--attn', 
        type=str, 
        default='sdpa',
        choices=['flash', 'sdpa', 'eager', 'tensorrt'],
        help="The attention implementation to use."
    )
    
    cli_args = parser.parse_args()
    
    # --- Main Logic Branch ---
    if cli_args.attn == 'tensorrt':
        try:
            from tensorrt_llm.builder import Builder
            config = configure_llama_model(cli_args.size)
            engine_path = build_tensorrt_engine(config, cli_args)
            results, args = benchmark_tensorrt(engine_path, config, cli_args)
        except ImportError:
            print("\nERROR: tensorrt_llm is not installed. Please follow the official NVIDIA guide.")
            print("https://github.com/NVIDIA/TensorRT-LLM")
            results, args = None, None
    else:
        results, args = benchmark_pytorch(cli_args)
    
    if results:
        try:
            plot_results(results, args)
        except ImportError:
            print("\nMatplotlib not found, skipping plot. To install: pip install matplotlib")
        except Exception as e:
            print(f"\nAn error occurred during plotting: {e}")