# ==============================================================================
#
#  Production-Grade Transformer Benchmarking Tool (with Base TensorRT)
#
#  This script provides a comprehensive framework for benchmarking different
#  transformer execution paradigms and backends for real-time latency.
#
#  Features:
#  - Command-line control over model size, simulation params, paradigm, and attention.
#  - Two Paradigms:
#      1. 'recurrent': The state-of-the-art approach using a KV Cache for
#         constant, low latency. Ideal for real-time applications.
#      2. 'single-shot': The naive approach of processing the entire growing
#         sequence at each step. Useful for comparing raw throughput.
#  - Four Backends:
#      - 'flash', 'sdpa', 'eager': For PyTorch-based execution.
#      - 'tensorrt': Compiles the model to a highly optimized TensorRT engine
#        for single-shot execution.
#
#  Prerequisites for '--attn tensorrt':
#  - The base 'tensorrt' Python package must be installed.
#  - PyTorch with ONNX export support.
#
#  Usage Examples:
#  # 1. Benchmark the recommended recurrent architecture with FlashAttention
#  python benchmark.py --paradigm recurrent --attn flash --size 300
#
#  # 2. Benchmark the single-shot paradigm with a compiled TensorRT engine
#  # (First run will be slow as it builds the engine)
#  python benchmark.py --paradigm single-shot --attn tensorrt --size 300
#
#  # 3. Compare single-shot TensorRT against single-shot PyTorch (Eager)
#  python benchmark.py --paradigm single-shot --attn eager --size 300
#
# ==============================================================================

import torch
import time
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from transformers import LlamaConfig, LlamaForCausalLM

# --- Helper Function for Model Configuration ---
def configure_llama_model(target_m_params, use_cache):
    """Approximates a LlamaConfig for a given target parameter size."""
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
        max_position_embeddings=8192, rms_norm_eps=1e-5, use_cache=use_cache
    )

# --- Base TensorRT Specific Functions ---
def build_base_trt_engine(config, args):
    """Builds and saves a base TensorRT engine from an ONNX model."""
    import tensorrt as trt
    engine_dir = 'trt_engines'
    os.makedirs(engine_dir, exist_ok=True)
    engine_path = os.path.join(engine_dir, f'llama_base_{args.size}M_fp16.engine')
    onnx_path = os.path.join(engine_dir, f'llama_base_{args.size}M.onnx')
    if os.path.exists(engine_path):
        print(f"Found existing base TensorRT engine: {engine_path}. Skipping build.")
        return engine_path
    
    print(f"No engine found. Creating ONNX model at {onnx_path}...")
    pytorch_model = LlamaForCausalLM(config).to("cuda").half().eval()
    dummy_input = torch.randint(0, config.vocab_size, (1, 128), device="cuda")
    torch.onnx.export(
        pytorch_model, dummy_input, onnx_path,
        input_names=['input_ids'], output_names=['logits'],
        dynamic_axes={'input_ids': {1: 'sequence_length'}},
        opset_version=17
    )
    print("ONNX model created.")
    
    print("Building TensorRT engine from ONNX. This may take several minutes...")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            for error in range(parser.num_errors): print(parser.get_error(error))
            return None
    builder_config = builder.create_builder_config()
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)
    builder_config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    profile.set_shape('input_ids', (1, 1), (1, 2048), (1, config.max_position_embeddings))
    builder_config.add_optimization_profile(profile)
    serialized_engine = builder.build_serialized_network(network, builder_config)
    with open(engine_path, "wb") as f: f.write(serialized_engine)
    print("Base TensorRT engine built successfully.")
    return engine_path

# --- Benchmark Functions ---
def benchmark_single_shot_trt(engine_path, config, args):
    """Runs single-shot benchmark using a base TensorRT engine."""
    import tensorrt as trt
    print("Initializing base TensorRT runtime...")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    input_binding_idx, output_binding_idx = engine.get_binding_index("input_ids"), engine.get_binding_index("logits")
    d_input = torch.empty(size=(1, config.max_position_embeddings), dtype=torch.int64, device="cuda")
    d_output = torch.empty(size=(1, config.max_position_embeddings, config.vocab_size), dtype=torch.float16, device="cuda")
    bindings = [None] * 2
    bindings[input_binding_idx], bindings[output_binding_idx] = d_input.data_ptr(), d_output.data_ptr()
    
    print("\n--- Starting Base TensorRT (Single-Shot) Benchmark ---")
    all_results, current_sequence_length = [], 0
    for second in range(1, args.duration + 1):
        current_sequence_length += args.tickrate * args.tokens_per_tick
        input_ids = torch.randint(0, config.vocab_size, (1, current_sequence_length), device="cuda")
        context.set_input_shape("input_ids", input_ids.shape)
        d_input[0, :current_sequence_length] = input_ids[0]
        torch.cuda.synchronize()
        start_time = time.time()
        context.execute_v2(bindings=bindings)
        torch.cuda.synchronize()
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        all_results.append((current_sequence_length, latency_ms))
        if second % 10 == 0 or second == 1: print(f"Second: {second:3d} | Seq Len: {current_sequence_length:4d} | Latency: {latency_ms:8.2f} ms")
        if latency_ms > 1000 / args.tickrate:
            print(f"\n!!! REAL-TIME CONSTRAINT FAILED at {second} seconds !!!")
            break
    return all_results, args

def benchmark_pytorch(model, args):
    """Runs benchmark for PyTorch models (both recurrent and single-shot)."""
    print("\n--- Starting PyTorch Benchmark ---")
    print(f"Paradigm: {args.paradigm.upper()}, Attention: {args.attn.upper()}")
    
    past_key_values, all_results, total_ticks = None, [], 0
    print("\nFormat: Second | Avg Latency (ms) | Sequence Length")
    
    for second in range(1, args.duration + 1):
        latencies_this_second = []
        for tick in range(args.tickrate):
            total_ticks += 1
            if args.paradigm == 'recurrent':
                input_ids = torch.randint(0, config.vocab_size, (1, args.tokens_per_tick), device="cuda")
                seq_len_for_log = past_key_values[0][0].shape[2] if past_key_values else 0
            else: # single-shot
                seq_len_for_log = total_ticks * args.tokens_per_tick
                input_ids = torch.randint(0, config.vocab_size, (1, seq_len_for_log), device="cuda")
            
            torch.cuda.synchronize()
            start_time = time.time()
            with torch.no_grad():
                model_output = model(input_ids, past_key_values=past_key_values, use_cache=(args.paradigm == 'recurrent'))
                if args.paradigm == 'recurrent':
                    past_key_values = model_output.past_key_values
            torch.cuda.synchronize()
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            latencies_this_second.append(latency_ms)

            if args.paradigm == 'single-shot' and latency_ms > 1000 / args.tickrate:
                break # No need to finish the second if one tick already failed
        
        avg_latency = np.mean(latencies_this_second)
        all_results.append((seq_len_for_log, avg_latency))
        if second % 10 == 0 or second == 1: print(f"{second:6d} | {avg_latency:16.2f} | {seq_len_for_log:12d}")
        
        if avg_latency > 1000 / args.tickrate:
            print(f"\n!!! REAL-TIME CONSTRAINT FAILED at {second} seconds !!!")
            break
            
    return all_results, args

def plot_results(results, args):
    """Generates a plot from the benchmark results."""
    ticks, latencies = [r[0] for r in results], [r[1] for r in results]
    plt.figure(figsize=(12, 7))
    plt.plot(ticks, latencies, marker='.', linestyle='-')
    y_limit = max(latencies) * 1.2 + 10
    plt.ylim(0, y_limit)
    real_time_budget_ms = 1000 / args.tickrate
    plt.axhline(y=real_time_budget_ms, color='r', linestyle='--', label=f'{real_time_budget_ms:.2f}ms Budget ({args.tickrate}Hz)')
    
    attn_map = {'flash': "FlashAttention-2", 'sdpa': "PyTorch SDPA", 'eager': "Eager Attention", 'tensorrt': "TensorRT"}
    title = f'{args.paradigm.capitalize()} Paradigm Latency ({args.size}M params, {attn_map.get(args.attn)})'
    xlabel = 'Sequence Length' if args.paradigm == 'single-shot' else 'KV Cache Size'
    
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Latency per Update (ms)', fontsize=12)
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Transformer Architectures.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--size', type=int, default=300, help='Target model size in millions of parameters.')
    parser.add_argument('--duration', type=int, default=60, help='Duration of the simulation in seconds.')
    parser.add_argument('--tokens_per_tick', type=int, default=1, help='Number of new tokens to add at each tick.')
    parser.add_argument('--tickrate', type=int, default=30, help='Number of ticks (updates) per second.')
    parser.add_argument('--paradigm', type=str, default='recurrent', choices=['recurrent', 'single-shot'], help="Execution paradigm.")
    parser.add_argument('--attn', type=str, default='sdpa', choices=['flash', 'sdpa', 'eager', 'tensorrt'], help="Attention implementation/backend.")
    
    cli_args = parser.parse_args()
    
    # --- Input Validation ---
    if cli_args.paradigm == 'recurrent' and cli_args.attn == 'tensorrt':
        print("ERROR: Recurrent paradigm with base TensorRT is not supported in this script. Exiting.")
        exit()
    if cli_args.paradigm == 'single-shot' and cli_args.attn in ['flash', 'sdpa']:
        print("WARNING: Using 'flash' or 'sdpa' in single-shot mode is inefficient. "
              "Comparing 'eager' vs 'tensorrt' is more informative for this paradigm.")

    # --- Main Logic Branch ---
    results, args = None, None
    if cli_args.attn == 'tensorrt':
        try:
            import tensorrt as trt
            config = configure_llama_model(cli_args.size, use_cache=False)
            engine_path = build_base_trt_engine(config, cli_args)
            if engine_path: results, args = benchmark_base_trt(engine_path, config, cli_args)
        except ImportError:
            print("\nERROR: The 'tensorrt' package is not installed.")
    else:
        use_cache_for_model = (cli_args.paradigm == 'recurrent')
        config = configure_llama_model(cli_args.size, use_cache=use_cache_for_model)
        attn_impl = {'flash': 'flash_attention_2', 'sdpa': 'sdpa', 'eager': 'eager'}.get(cli_args.attn)
        try:
            model = AutoModelForCausalLM.from_config(config, attn_implementation=attn_impl, torch_dtype=torch.float16).to("cuda")
            results, args = benchmark_pytorch(model, cli_args)
        except Exception as e:
            print(f"\n!!! FAILED TO CREATE PYTORCH MODEL with '{attn_impl}' backend: {e} !!!")

    if results:
        try:
            plot_results(results, args)
        except ImportError:
            print("\nMatplotlib not found, skipping plot. To install: pip install matplotlib")
        except Exception as e:
            print(f"\nAn error occurred during plotting: {e}")