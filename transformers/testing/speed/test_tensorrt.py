# ==============================================================================
#
#  Production-Grade Transformer Benchmarking Tool (Extended for TensorRT)
#
#  This script provides a framework for benchmarking latency, comparing:
#
#  1. PyTorch Eager/SDPA/FlashAttention-2: Native PyTorch implementations.
#  2. TensorRT: A fully compiled and optimized inference engine.
#
#  Usage:
#  # Benchmark FlashAttention-2
#  python benchmark_poc.py --size 300 --attn flash
#
#  # Build and benchmark a TensorRT engine for the same model
#  python benchmark_poc.py --size 300 --attn tensorrt
#
# ==============================================================================

import torch
import time
import argparse
import os
import matplotlib.pyplot as plt
from transformers import LlamaConfig, AutoModelForCausalLM

# Conditional import for TensorRT
try:
    import tensorrt as trt
except ImportError:
    trt = None

# --- Original Helper Functions (configure_llama_model, plot_results) ---
# These are unchanged from your original script.

def configure_llama_model(target_m_params):
    """
    Approximates a LlamaConfig for a given target parameter size.
    This is a heuristic that creates a reasonable modern architecture (GQA, etc.).
    """
    # Define architectural presets for different size classes
    if target_m_params <= 150:  # e.g., ~125M
        n_layers, n_heads, n_kv_heads = 12, 12, 4
    elif target_m_params <= 400: # e.g., ~300M
        n_layers, n_heads, n_kv_heads = 24, 16, 4
    elif target_m_params <= 800: # e.g., ~700M
        n_layers, n_heads, n_kv_heads = 32, 32, 8
    else:  # e.g., ~1.3B+
        n_layers, n_heads, n_kv_heads = 40, 40, 8

    d_model_approx = int((target_m_params * 1_000_000 / (12 * n_layers))**0.5)
    d_model = round(d_model_approx / 64) * 64
    intermediate_size = int(d_model * 2.6)

    print(f"Targeting ~{target_m_params}M params. Calculated config: "
          f"d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, GQA_groups={n_kv_heads}")

    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=d_model,
        intermediate_size=intermediate_size,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        max_position_embeddings=8192,
        rms_norm_eps=1e-5,
        use_cache=True, # Ensure use_cache is enabled in the config
    )
    return config

def plot_results(results, args):
    """Generates a plot from the benchmark results."""
    ticks = [r[0] for r in results]
    latencies = [r[1] for r in results]
    
    plt.figure(figsize=(12, 7))
    plt.plot(ticks, latencies, marker='.', linestyle='-')
    
    y_limit = max(latencies) * 1.2 + 10
    plt.ylim(0, y_limit)
    
    real_time_budget_ms = 1000 / args.tickrate
    plt.axhline(y=real_time_budget_ms, color='r', linestyle='--', 
                label=f'{real_time_budget_ms:.2f}ms Real-time Budget ({args.tickrate}Hz)')
    
    if args.attn == 'flash':
        attn_title = "FlashAttention-2"
    elif args.attn == 'sdpa':
        attn_title = "PyTorch SDPA"
    elif args.attn == 'eager':
        attn_title = "Standard Eager Attention"
    elif args.attn == 'tensorrt':
        attn_title = "TensorRT Engine"
        
    plt.title(f'Recurrent Transformer Latency ({args.size}M params, {attn_title})', fontsize=16)
    plt.xlabel('Total Ticks Processed (Cache Size)', fontsize=12)
    plt.ylabel('Average Latency per Update (ms)', fontsize=12)
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()


# --- New TensorRT-specific Functions ---

def build_tensorrt_engine(model, config, onnx_path, engine_path):
    """Builds and saves a TensorRT engine from a PyTorch model."""
    print("Building TensorRT engine. This may take a few minutes...")
    
    # 1. Export to ONNX
    print(f"Exporting model to ONNX at {onnx_path}...")
    batch_size = 1
    sequence_length = 1 # For single-token decoding
    past_sequence_length = 1 # Initial past length

    # Dummy inputs for ONNX export
    input_ids = torch.ones((batch_size, sequence_length), dtype=torch.int64).cuda()
    
    # Create dummy past_key_values matching the model's structure
    past_key_values = []
    for _ in range(config.num_hidden_layers):
        key = torch.zeros((batch_size, config.num_key_value_heads, past_sequence_length, config.hidden_size // config.num_attention_heads), dtype=torch.float16).cuda()
        value = torch.zeros((batch_size, config.num_key_value_heads, past_sequence_length, config.hidden_size // config.num_attention_heads), dtype=torch.float16).cuda()
        past_key_values.append((key, value))
    
    # Define input and output names for clarity
    input_names = ["input_ids"] + [f"past_key_values.{i}" for i in range(config.num_hidden_layers * 2)]
    output_names = ["logits"] + [f"present_key_values.{i}" for i in range(config.num_hidden_layers * 2)]
    
    # Flatten the past_key_values for ONNX export
    flat_past_key_values = [item for sublist in past_key_values for item in sublist]
    
    # Define dynamic axes for variable sequence lengths (CRITICAL for KV Cache)
    dynamic_axes = {'input_ids': {1: 'sequence_length'}}
    for i in range(config.num_hidden_layers * 2):
        # The third dimension (index 2) is the sequence length of the cache
        dynamic_axes[f"past_key_values.{i}"] = {2: 'past_sequence_length'}
        dynamic_axes[f"present_key_values.{i}"] = {2: 'total_sequence_length'}

    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_ids, tuple(past_key_values)),
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            export_params=True
        )
    print("ONNX export complete.")

    # 2. Build Engine with TensorRT
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"Parsing ONNX model from {onnx_path}...")
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print("ONNX parsing complete.")

    # 3. Configure the builder and create optimization profile
    builder_config = builder.create_builder_config()
    builder_config.set_flag(trt.BuilderFlag.FP16) # Enable FP16 precision
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 * 1024 * 1024) # 4GB workspace

    profile = builder.create_optimization_profile()
    
    # Define min, opt, and max shapes for dynamic inputs
    # This tells TensorRT how to optimize for different KV cache sizes
    max_context = config.max_position_embeddings
    
    profile.set_shape("input_ids", (1, 1), (1, 1), (1, 1)) # We always feed one new token
    
    # Shape of one K or V tensor in the cache
    kv_shape = (batch_size, config.num_key_value_heads, 1, config.hidden_size // config.num_attention_heads)

    for i in range(config.num_hidden_layers * 2):
        profile.set_shape(
            f"past_key_values.{i}",
            min=(kv_shape[0], kv_shape[1], 1, kv_shape[3]),             # Min: cache has 1 token
            opt=(kv_shape[0], kv_shape[1], 512, kv_shape[3]),           # Opt: cache has 512 tokens
            max=(kv_shape[0], kv_shape[1], max_context - 1, kv_shape[3])# Max: cache is full
        )
    builder_config.add_optimization_profile(profile)

    print("Building serialized TensorRT engine...")
    serialized_engine = builder.build_serialized_network(network, builder_config)
    
    if serialized_engine is None:
        print("ERROR: Failed to build the TensorRT engine.")
        return None
        
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
        
    print(f"TensorRT engine saved to {engine_path}")
    return engine_path

def benchmark_tensorrt(args, config, model):
    """Runs the benchmark using a compiled TensorRT engine."""
    engine_path = f"model_{args.size}M.engine"
    onnx_path = f"model_{args.size}M.onnx"

    if not os.path.exists(engine_path):
        build_tensorrt_engine(model, config, onnx_path, engine_path)

    print("\n--- Loading and Benchmarking TensorRT Engine ---")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Allocate GPU buffers for inputs and outputs
    bindings = [None] * engine.num_bindings
    device = torch.device("cuda")

    # We need to allocate buffers large enough for the MAXIMUM possible size
    max_context = config.max_position_embeddings
    batch_size = 1
    
    # Prepare past_key_values buffers (these will be updated in the loop)
    past_key_values_trt = []
    for i in range(config.num_hidden_layers):
        # Shape: [batch, num_kv_heads, max_seq_len, head_dim]
        key_shape = (batch_size, config.num_key_value_heads, max_context, config.hidden_size // config.num_attention_heads)
        value_shape = key_shape
        past_key_values_trt.append(torch.zeros(key_shape, dtype=torch.float16, device=device))
        past_key_values_trt.append(torch.zeros(value_shape, dtype=torch.float16, device=device))

    # Output logits buffer
    logits_buffer = torch.empty((batch_size, 1, config.vocab_size), dtype=torch.float16, device=device)

    # Set up bindings
    # Order must match the ONNX export order: input_ids, past_kvs..., logits, present_kvs...
    binding_idx = 0
    # Input bindings
    context.set_binding_shape(binding_idx, (batch_size, 1)) # input_ids
    bindings[binding_idx] = 0 # Placeholder, will be set in loop
    binding_idx += 1
    for i in range(len(past_key_values_trt)):
        context.set_binding_shape(binding_idx, past_key_values_trt[i].shape) # Initially shape is max
        bindings[binding_idx] = past_key_values_trt[i].data_ptr()
        binding_idx += 1
    # Output bindings
    bindings[binding_idx] = logits_buffer.data_ptr()
    binding_idx += 1
    # For outputs, the 'present' KV cache tensors will write back into our 'past' buffers
    for i in range(len(past_key_values_trt)):
        bindings[binding_idx] = past_key_values_trt[i].data_ptr()
        binding_idx += 1
    
    # --- Benchmark Loop ---
    all_results = []
    total_ticks = 0
    current_cache_len = 0
    
    print(f"Simulating game for {args.duration} seconds at {args.tickrate}Hz...")
    print(f"Adding {args.tokens_per_tick} token(s) per tick.")
    print("\nFormat: Second | Avg Latency (ms) | Cache Length")

    for second in range(1, args.duration + 1):
        latencies_this_second = []
        for tick in range(args.tickrate):
            total_ticks += 1
            input_ids = torch.randint(0, config.vocab_size, (1, args.tokens_per_tick), device=device, dtype=torch.int64)

            # Update context shapes for the current iteration
            context.set_binding_shape(0, input_ids.shape) # input_ids
            bindings[0] = input_ids.data_ptr()
            
            # For the first token, past cache is size 0, but TRT needs a min size of 1.
            # We handle this by setting the past shape to 1 and growing from there.
            past_len = max(1, current_cache_len)

            for i in range(len(past_key_values_trt)):
                # The shape of the PAST cache for this iteration
                shape = list(past_key_values_trt[i].shape)
                shape[2] = past_len
                context.set_binding_shape(1 + i, tuple(shape))

            # Precise timing
            torch.cuda.synchronize()
            start_time = time.time()
            
            context.execute_v2(bindings=bindings) # Run inference
            
            torch.cuda.synchronize()
            end_time = time.time()
            latencies_this_second.append((end_time - start_time) * 1000)
            
            current_cache_len += args.tokens_per_tick
            
        avg_latency = sum(latencies_this_second) / len(latencies_this_second)
        all_results.append((total_ticks, avg_latency))
        
        if second % 10 == 0 or second == 1:
            print(f"{second:6d} | {avg_latency:16.2f} | {current_cache_len:12d}")
            
    print("\n--- Benchmark Complete ---")
    return all_results, args

def benchmark_pytorch(args, config, model):
    """Main function to run the benchmark for PyTorch backends."""
    model.eval()
    device = model.device

    print("\n--- Starting PyTorch Benchmark ---")
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

# --- Main Execution Logic ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Recurrent Transformer Architectures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--size', type=int, default=300, 
                        help='Target model size in millions of parameters.')
    parser.add_argument('--duration', type=int, default=180, 
                        help='Duration of the simulation in seconds.')
    parser.add_argument('--tokens_per_tick', type=int, default=1, 
                        help='Number of new tokens to add at each tick.')
    parser.add_argument('--tickrate', type=int, default=30, 
                        help='Number of ticks (updates) per second.')
    parser.add_argument(
        '--attn', 
        type=str, 
        default='sdpa',
        choices=['flash', 'sdpa', 'eager', 'tensorrt'],
        help="The attention implementation to use: 'flash', 'sdpa', 'eager', or 'tensorrt'."
    )
    
    cli_args = parser.parse_args()

    # --- Common setup for all modes ---
    config = configure_llama_model(cli_args.size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        exit()
    print(f"Using device: {torch.cuda.get_device_name(0)}")

    # --- Create PyTorch model (needed for all backends, even TRT for the initial export) ---
    attn_impl_map = {'flash': 'flash_attention_2', 'sdpa': 'sdpa', 'eager': 'eager', 'tensorrt': 'sdpa'}
    attn_impl = attn_impl_map[cli_args.attn] # Use SDPA as base for TRT export
    print(f"INFO: Using '{attn_impl}' as the base PyTorch model implementation.")

    try:
        model = AutoModelForCausalLM.from_config(
            config,
            attn_implementation=attn_impl,
            torch_dtype=torch.float16,
        ).to(device)
        print(f"Model successfully created with '{attn_impl}' backend.")
    except Exception as e:
        print(f"\n!!! FAILED TO CREATE MODEL with '{attn_impl}' backend: {e} !!!")
        print("Ensure your environment supports the selected attention implementation.")
        exit()

    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameter count: {num_params / 1_000_000:.2f}M")

    # --- Route to the correct benchmark function ---
    results = None
    if cli_args.attn == 'tensorrt':
        if trt is None:
            print("\nERROR: TensorRT library not found. Cannot run --attn tensorrt.")
            print("Please install it from NVIDIA's website or PyPI.")
            exit()
        results, args = benchmark_tensorrt(cli_args, config, model)
    else:
        results, args = benchmark_pytorch(cli_args, config, model)

    # --- Plot results if successful ---
    if results:
        try:
            import matplotlib.pyplot as plt
            plot_results(results, args)
        except ImportError:
            print("\nMatplotlib not found, skipping plot. To install: pip install matplotlib")
        except Exception as e:
            print(f"\nAn error occurred during plotting: {e}")