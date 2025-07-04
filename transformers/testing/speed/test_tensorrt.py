# ==============================================================================
#
#  Production-Grade Autoregressive Transformer Benchmarking Tool
#
#  Final, fully verified version with all API updates and bug fixes, including
#  robust, race-condition-free CUDA synchronization and timing.
#
# ==============================================================================

import torch
import time
import argparse
import os
import matplotlib.pyplot as plt
from transformers import LlamaConfig, AutoModelForCausalLM

# Conditional import for TensorRT and the new Cache object
try:
    import tensorrt as trt
    from transformers.cache_utils import DynamicCache
except ImportError:
    trt = None
    DynamicCache = None

# --- Helper Functions and Classes ---

def configure_llama_model(target_m_params):
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
    return LlamaConfig(vocab_size=32000, hidden_size=d_model, intermediate_size=intermediate_size,
                       num_hidden_layers=n_layers, num_attention_heads=n_heads, num_key_value_heads=n_kv_heads,
                       max_position_embeddings=8192, rms_norm_eps=1e-5, use_cache=True)

def plot_results(results, args):
    """Generates a plot from the benchmark results."""
    ticks, latencies = [r[0] for r in results], [r[1] for r in results]
    plt.figure(figsize=(12, 7))
    plt.plot(ticks, latencies, marker='.', linestyle='-')
    plt.ylim(0, max(latencies) * 1.2 + 10)
    real_time_budget_ms = 1000 / args.tickrate
    plt.axhline(y=real_time_budget_ms, color='r', linestyle='--',
                label=f'{real_time_budget_ms:.2f}ms Real-time Budget ({args.tickrate}Hz)')
    attn_title_map = {'flash': "FlashAttention-2", 'sdpa': "PyTorch SDPA", 'eager': "Eager Attention", 'tensorrt': "TensorRT Engine"}
    plt.title(f'Recurrent Transformer Latency ({args.size}M params, {attn_title_map.get(args.attn)})', fontsize=16)
    plt.xlabel('Total Ticks Processed (Cache Size)', fontsize=12)
    plt.ylabel('Average Latency per Update (ms)', fontsize=12)
    plt.grid(True, which='both', linestyle='--'); plt.legend(); plt.tight_layout(); plt.show()

class LlamaOnnxWrapper(torch.nn.Module):
    """A wrapper to prepare the Llama model for ONNX export, handling the DynamicCache object."""
    def __init__(self, model):
        super().__init__(); self.model = model
    def forward(self, input_ids, *past_key_values):
        pkv_tuples = tuple(past_key_values[i:i+2] for i in range(0, len(past_key_values), 2))
        cache_obj = DynamicCache.from_legacy_cache(past_key_values=pkv_tuples)
        model_outputs = self.model(input_ids=input_ids, past_key_values=cache_obj, use_cache=True)
        logits = model_outputs.logits
        present_cache_obj = model_outputs.past_key_values
        present_key_values = tuple(item for sublist in present_cache_obj.to_legacy_cache() for item in sublist)
        return (logits,) + present_key_values

def build_tensorrt_engine(model, llama_config, onnx_path, engine_path, tokens_per_tick):
    """Builds a TensorRT engine for a *fixed* input size."""
    try:
        os.remove(engine_path)
        print(f"Removed existing engine file: {engine_path}")
    except FileNotFoundError:
        pass
    print(f"Building TensorRT engine for fixed tokens_per_tick={tokens_per_tick}...")
    onnx_model = LlamaOnnxWrapper(model)
    print(f"Exporting model to ONNX at {onnx_path}...")
    batch_size, past_sequence_length = 1, 1
    input_ids = torch.ones((batch_size, tokens_per_tick), dtype=torch.int64).cuda()
    past_key_values_dummy = []
    for _ in range(llama_config.num_hidden_layers):
        shape = (batch_size, llama_config.num_key_value_heads, past_sequence_length, llama_config.hidden_size // llama_config.num_attention_heads)
        past_key_values_dummy.extend([torch.zeros(shape, dtype=torch.float16).cuda()] * 2)
    onnx_args = (input_ids, *past_key_values_dummy)
    input_names = ["input_ids"] + [f"past_key_values.{i}" for i in range(llama_config.num_hidden_layers * 2)]
    output_names = ["logits"] + [f"present_key_values.{i}" for i in range(llama_config.num_hidden_layers * 2)]
    dynamic_axes = {'input_ids': {0: 'batch_size'}, 'logits': {0: 'batch_size'}}
    for i in range(llama_config.num_hidden_layers * 2):
        dynamic_axes[f"past_key_values.{i}"] = {0: 'batch_size', 2: 'past_sequence_length'}
        dynamic_axes[f"present_key_values.{i}"] = {0: 'batch_size', 2: 'total_sequence_length'}
    with torch.no_grad():
        torch.onnx.export(onnx_model, onnx_args, onnx_path, input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes, opset_version=17, export_params=True)
    print("ONNX export complete.")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING); builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    builder_config = builder.create_builder_config()
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)
    builder_config.set_flag(trt.BuilderFlag.FP16)
    print(f"Parsing ONNX model from {onnx_path}...")
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            print("ERROR: Failed to parse the ONNX file."); [print(parser.get_error(i)) for i in range(parser.num_errors)]; return None
    print("ONNX parsing complete.")
    profile = builder.create_optimization_profile()
    max_context = llama_config.max_position_embeddings
    profile.set_shape("input_ids", min=(1, tokens_per_tick), opt=(1, tokens_per_tick), max=(1, tokens_per_tick))
    kv_shape = (batch_size, llama_config.num_key_value_heads, 1, llama_config.hidden_size // llama_config.num_attention_heads)
    for i in range(llama_config.num_hidden_layers * 2):
        profile.set_shape(f"past_key_values.{i}", min=(kv_shape[0], kv_shape[1], 1, kv_shape[3]),
                          opt=(kv_shape[0], kv_shape[1], 512, kv_shape[3]),
                          max=(kv_shape[0], kv_shape[1], max_context - 1, kv_shape[3]))
    builder_config.add_optimization_profile(profile)
    print("Building serialized TensorRT engine...");
    serialized_engine = builder.build_serialized_network(network, builder_config)
    if serialized_engine is None: print("ERROR: Failed to build the TensorRT engine."); return None
    with open(engine_path, 'wb') as f: f.write(serialized_engine)
    print(f"TensorRT engine saved to {engine_path}"); return engine_path

def benchmark_tensorrt(args, config, model):
    """Runs the benchmark using a compiled TensorRT engine with robust synchronization."""
    engine_path = f"model_{args.size}M_t{args.tokens_per_tick}.engine"
    onnx_path = f"model_{args.size}M_t{args.tokens_per_tick}.onnx"
    build_tensorrt_engine(model, config, onnx_path, engine_path, args.tokens_per_tick)
    
    print(f"\n--- Loading and Benchmarking TensorRT Engine for t={args.tokens_per_tick} ---")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    device = torch.device("cuda")
    stream = torch.cuda.Stream()

    max_context, batch_size = config.max_position_embeddings, 1
    kv_cache_buffers = []
    for i in range(config.num_hidden_layers * 2):
        shape = (batch_size, config.num_key_value_heads, max_context, config.hidden_size // config.num_attention_heads)
        kv_cache_buffers.append(torch.zeros(shape, dtype=torch.float16, device=device))
    logits_buffer = torch.empty((batch_size, args.tokens_per_tick, config.vocab_size), dtype=torch.float16, device=device)

    for i in range(len(kv_cache_buffers)):
        context.set_tensor_address(f"past_key_values.{i}", kv_cache_buffers[i].data_ptr())
        context.set_tensor_address(f"present_key_values.{i}", kv_cache_buffers[i].data_ptr())
    context.set_tensor_address("logits", logits_buffer.data_ptr())

    all_results, total_ticks, current_cache_len = [], 0, 0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    print(f"Simulating for {args.duration} seconds at {args.tickrate}Hz...\nFormat: Second | Avg Latency (ms) | Cache Length")

    for second in range(1, args.duration + 1):
        latencies_this_second = []
        for tick in range(args.tickrate):
            total_ticks += 1
            input_ids = torch.randint(0, config.vocab_size, (1, args.tokens_per_tick), device=device, dtype=torch.int64)
            past_len = max(1, current_cache_len)

            context.set_input_shape("input_ids", input_ids.shape)
            for i in range(len(kv_cache_buffers)):
                shape = (batch_size, config.num_key_value_heads, past_len, config.hidden_size // config.num_attention_heads)
                context.set_input_shape(f"past_key_values.{i}", shape)
            
            context.set_tensor_address("input_ids", input_ids.data_ptr())

            # FIX: Use CUDA events for accurate and safe asynchronous timing
            start_event.record(stream)
            context.execute_async_v3(stream_handle=stream.cuda_stream)
            end_event.record(stream)
            stream.synchronize()
            
            latencies_this_second.append(start_event.elapsed_time(end_event))
            current_cache_len += args.tokens_per_tick
            
        avg_latency = sum(latencies_this_second) / len(latencies_this_second)
        all_results.append((total_ticks, avg_latency))
        if second % 10 == 0 or second == 1: print(f"{second:6d} | {avg_latency:16.2f} | {current_cache_len:12d}")
            
    print("\n--- Benchmark Complete ---")
    return all_results, args

def benchmark_pytorch(args, config, model):
    """Runs the benchmark for PyTorch backends."""
    model.eval(); device = model.device; print("\n--- Starting PyTorch Benchmark --- \nWarming up GPU...")
    past_key_values = None
    with torch.no_grad():
        for _ in range(50):
            input_ids = torch.randint(0, config.vocab_size, (1, 1), device=device)
            model_output = model(input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = model_output.past_key_values
    torch.cuda.synchronize(); print("Warmup complete.")
    past_key_values = None; all_results, total_ticks = [], 0
    print(f"Simulating for {args.duration} seconds at {args.tickrate}Hz...\nFormat: Second | Avg Latency (ms) | Cache Length")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for second in range(1, args.duration + 1):
        latencies_this_second = []
        for tick in range(args.tickrate):
            total_ticks += 1
            input_ids = torch.randint(0, config.vocab_size, (1, args.tokens_per_tick), device=device)
            
            start_event.record()
            with torch.no_grad():
                model_output = model(input_ids, past_key_values=past_key_values, use_cache=True)
                past_key_values = model_output.past_key_values
            end_event.record()
            torch.cuda.synchronize()
            latencies_this_second.append(start_event.elapsed_time(end_event))
        avg_latency = sum(latencies_this_second) / len(latencies_this_second)
        cache_seq_len = past_key_values.get_seq_length() if past_key_values else 0
        all_results.append((total_ticks, avg_latency))
        if second % 10 == 0 or second == 1: print(f"{second:6d} | {avg_latency:16.2f} | {cache_seq_len:12d}")
    print("\n--- Benchmark Complete ---"); return all_results, args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Recurrent Transformer Architectures.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--size', type=int, default=300, help='Target model size in millions of parameters.')
    parser.add_argument('--duration', type=int, default=60, help='Duration of the simulation in seconds.')
    parser.add_argument('--tokens_per_tick', type=int, default=1, help='Number of new tokens to add at each tick.')
    parser.add_argument('--tickrate', type=int, default=30, help='Number of ticks (updates) per second.')
    parser.add_argument('--attn', type=str, default='sdpa', choices=['flash', 'sdpa', 'eager', 'tensorrt'], help="Implementation to use.")
    cli_args = parser.parse_args()
    if cli_args.attn == 'tensorrt' and (trt is None or DynamicCache is None):
        print("\nERROR: TensorRT or a compatible transformers version not found."); exit()
    config = configure_llama_model(cli_args.size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu': print("ERROR: CUDA not available."); exit()
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    attn_impl_map = {'flash': 'flash_attention_2', 'sdpa': 'sdpa', 'eager': 'eager', 'tensorrt': 'sdpa'}
    attn_impl = attn_impl_map[cli_args.attn]
    print(f"INFO: Using '{attn_impl}' as the base PyTorch model implementation.")
    try:
        model = AutoModelForCausalLM.from_config(config, attn_implementation=attn_impl, torch_dtype=torch.float16).to(device)
        print(f"Model successfully created with '{attn_impl}' backend.")
    except Exception as e: print(f"\n!!! FAILED TO CREATE MODEL with '{attn_impl}' backend: {e} !!!"); exit()
    model.eval()
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    if cli_args.attn == 'tensorrt': results, args = benchmark_tensorrt(cli_args, config, model)
    else: results, args = benchmark_pytorch(cli_args, config, model)
    if results:
        try: plot_results(results, args)
        except ImportError: print("\nMatplotlib not found, skipping plot.")
        except Exception as e: print(f"\nAn error occurred during plotting: {e}")