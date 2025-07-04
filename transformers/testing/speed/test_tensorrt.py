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

# --- Helper Functions ---

def configure_llama_model(target_m_params):
    """Approximates a LlamaConfig for a given target parameter size."""
    if target_m_params <= 150:
        n_layers, n_heads, n_kv_heads = 12, 12, 4
    elif target_m_params <= 400:
        n_layers, n_heads, n_kv_heads = 24, 16, 4
    elif target_m_params <= 800:
        n_layers, n_heads, n_kv_heads = 32, 32, 8
    else:
        n_layers, n_heads, n_kv_heads = 40, 40, 8
    d_model_approx = int((target_m_params * 1_000_000 / (12 * n_layers))**0.5)
    d_model = round(d_model_approx / 64) * 64
    intermediate_size = int(d_model * 2.6)
    print(f"Targeting ~{target_m_params}M params. Calculated config: "
          f"d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, GQA_groups={n_kv_heads}")
    return LlamaConfig(
        vocab_size=32000,
        hidden_size=d_model,
        intermediate_size=intermediate_size,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        max_position_embeddings=60000,
        rms_norm_eps=1e-5,
        use_cache=True
    )

class LlamaOnnxWrapper(torch.nn.Module):
    """Wraps Llama model for ONNX export, handling DynamicCache."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, *past_key_values):
        pkv_tuples = tuple(
            past_key_values[i:i+2] for i in range(0, len(past_key_values), 2)
        )
        cache_obj = DynamicCache.from_legacy_cache(past_key_values=pkv_tuples)
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=cache_obj,
            use_cache=True
        )
        logits = outputs.logits
        present_cache = outputs.past_key_values
        present_kv = tuple(
            item for sub in present_cache.to_legacy_cache() for item in sub
        )
        return (logits,) + present_kv

# --- TensorRT Engine Build ---

def build_tensorrt_engine(model, llama_config, onnx_path, engine_path, tokens_per_tick):
    try:
        os.remove(engine_path)
        print(f"Removed existing engine file: {engine_path}")
    except FileNotFoundError:
        pass
    print(f"Building TensorRT engine for fixed tokens_per_tick={tokens_per_tick}...")
    onnx_model = LlamaOnnxWrapper(model)
    print(f"Exporting model to ONNX at {onnx_path}...")
    batch_size, past_seq = 1, 1
    input_ids = torch.ones((batch_size, tokens_per_tick), dtype=torch.int64).cuda()
    dummy_cache = []
    for _ in range(llama_config.num_hidden_layers):
        shape = (
            batch_size,
            llama_config.num_key_value_heads,
            past_seq,
            llama_config.hidden_size // llama_config.num_attention_heads
        )
        dummy_cache.extend([
            torch.zeros(shape, dtype=torch.float16).cuda(),
            torch.zeros(shape, dtype=torch.float16).cuda()
        ])
    with torch.no_grad():
        torch.onnx.export(
            onnx_model,
            (input_ids, *dummy_cache),
            onnx_path,
            input_names=["input_ids"] + [f"past_key_values.{i}" for i in range(len(dummy_cache))],
            output_names=["logits"] + [f"present_key_values.{i}" for i in range(len(dummy_cache))],
            opset_version=17,
            dynamic_axes={
                "input_ids": {0: "batch_size"},
                "logits": {0: "batch_size"},
                **{f"past_key_values.{i}": {0: "batch_size", 2: "past_seq_len"} for i in range(len(dummy_cache))},
                **{f"present_key_values.{i}": {0: "batch_size", 2: "total_seq_len"} for i in range(len(dummy_cache))}
            }
        )
    print("ONNX export complete.")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config_builder = builder.create_builder_config()
    # Increase workspace size if needed
    config_builder.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, 4 * 1024 * 1024 * 1024
    )
    config_builder.set_flag(trt.BuilderFlag.FP16)

    print(f"Parsing ONNX model from {onnx_path}...")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse ONNX model.")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None
    print("ONNX parsing complete.")

    profile = builder.create_optimization_profile()
    max_pos = llama_config.max_position_embeddings
    profile.set_shape("input_ids", (1, tokens_per_tick), (1, tokens_per_tick), (1, tokens_per_tick))
    kv_shape = (
        1,
        llama_config.num_key_value_heads,
        1,
        llama_config.hidden_size // llama_config.num_attention_heads
    )
    for i in range(llama_config.num_hidden_layers * 2):
        profile.set_shape(
            f"past_key_values.{i}",
            kv_shape,
            (1, kv_shape[1], 512, kv_shape[3]),
            (1, kv_shape[1], max_pos-1, kv_shape[3])
        )
    config_builder.add_optimization_profile(profile)

    print("Building serialized TensorRT engine...")
    engine_bytes = builder.build_serialized_network(network, config_builder)
    if engine_bytes is None:
        print("ERROR: Failed to build engine.")
        return None
    with open(engine_path, 'wb') as f:
        f.write(engine_bytes)
    print(f"Engine saved to {engine_path}")
    return engine_path

# --- Benchmark Loop ---

def benchmark_tensorrt(args, config, model):
    engine_path = f"model_{args.size}M_t{args.tokens_per_tick}.engine"
    onnx_path  = f"model_{args.size}M_t{args.tokens_per_tick}.onnx"
    build_tensorrt_engine(model, config, onnx_path, engine_path, args.tokens_per_tick)

    print(f"\n--- Benchmarking TensorRT Engine (t={args.tokens_per_tick}) ---")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    rt = trt.Runtime(TRT_LOGGER)
    with open(engine_path, 'rb') as f:
        engine = rt.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    device = torch.device("cuda")

    # Pre-allocate separate past/present KV caches
    past_buffers, present_buffers = [], []
    for _ in range(config.num_hidden_layers * 2):
        buf = torch.zeros(
            (1, config.num_key_value_heads, config.max_position_embeddings, config.hidden_size // config.num_attention_heads),
            dtype=torch.float16, device=device
        )
        past_buffers.append(buf.clone())
        present_buffers.append(buf.clone())

    logits_buf = torch.empty(
        (1, args.tokens_per_tick, config.vocab_size), dtype=torch.float16, device=device
    )

    # Bind addresses once
    for i in range(len(past_buffers)):
        context.set_tensor_address(f"past_key_values.{i}",    past_buffers[i].data_ptr())
        context.set_tensor_address(f"present_key_values.{i}", present_buffers[i].data_ptr())
    context.set_tensor_address("logits", logits_buf.data_ptr())

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)

    all_results, total_ticks, cache_len = [], 0, 0
    print(f"Simulating {args.duration}s @ {args.tickrate}Hz:")
    for second in range(1, args.duration+1):
        latencies = []
        for _ in range(args.tickrate):
            total_ticks += 1
            input_ids = torch.randint(
                0, config.vocab_size, (1, args.tokens_per_tick),
                dtype=torch.int64, device=device
            )
            past_len = max(1, cache_len)

                        # Set dynamic shapes for each input
            context.set_input_shape("input_ids", input_ids.shape)
            for i in range(len(past_buffers)):
                context.set_input_shape(
                    f"past_key_values.{i}",
                    (1, config.num_key_value_heads, past_len, config.hidden_size // config.num_attention_heads)
                )

            context.set_tensor_address("input_ids", input_ids.data_ptr())

            start_ev.record()
            context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
            end_ev.record()
            torch.cuda.synchronize()

            latencies.append(start_ev.elapsed_time(end_ev))
            cache_len += args.tokens_per_tick

            # swap past/present for next tick
            past_buffers, present_buffers = present_buffers, past_buffers
            for i in range(len(past_buffers)):
                context.set_tensor_address(f"past_key_values.{i}", past_buffers[i].data_ptr())
                context.set_tensor_address(f"present_key_values.{i}", present_buffers[i].data_ptr())

        avg = sum(latencies)/len(latencies)
        all_results.append((total_ticks, avg))
        if second == 1 or second % 10 == 0:
            print(f"{second:4d}s | avg {avg:.2f}ms | cache {cache_len}")

    print("\n--- Benchmark Complete ---")
    return all_results, args

# --- PyTorch Benchmark (unchanged) ---

def benchmark_pytorch(args, config, model):
    model.eval()
    device = model.device
    past = None
    print("\n--- Starting PyTorch Benchmark ---")
    with torch.no_grad():
        for _ in range(50):
            inp = torch.randint(0, config.vocab_size, (1,1), device=device)
            out = model(inp, past_key_values=past, use_cache=True)
            past = out.past_key_values
    torch.cuda.synchronize()
    print("Warmup done.")
    all_res, ticks = [], 0
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)
    print(f"Simulating {args.duration}s @ {args.tickrate}Hz:")
    for sec in range(1, args.duration+1):
        lats = []
        for _ in range(args.tickrate):
            ticks += 1
            inp = torch.randint(0, config.vocab_size, (1,args.tokens_per_tick), device=device)
            start_ev.record()
            with torch.no_grad():
                out = model(inp, past_key_values=past, use_cache=True)
                past = out.past_key_values
            end_ev.record()
            torch.cuda.synchronize()
            lats.append(start_ev.elapsed_time(end_ev))
        avg = sum(lats)/len(lats)
        all_res.append((ticks, avg))
        if sec==1 or sec%10==0:
            print(f"{sec:4d}s | avg {avg:.2f}ms | seq {past.get_seq_length()}")
    print("\n--- Benchmark Complete ---")
    return all_res, args

# --- Main ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=300)
    parser.add_argument('--duration', type=int, default=60)
    parser.add_argument('--tokens_per_tick', type=int, default=1)
    parser.add_argument('--tickrate', type=int, default=30)
    parser.add_argument('--attn', type=str, choices=['flash','sdpa','eager','tensorrt'], default='sdpa')
    args = parser.parse_args()

    if args.attn=='tensorrt' and (trt is None or DynamicCache is None):
        print("ERROR: TensorRT or transformers cache_utils missing.")
        exit(1)

    cfg = configure_llama_model(args.size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type=='cpu': print("ERROR: CUDA unavailable."); exit(1)

    print(f"Using device: {torch.cuda.get_device_name(0)}")
    impl = {'flash':'flash_attention_2','sdpa':'sdpa','eager':'eager','tensorrt':'sdpa'}[args.attn]
    print(f"INFO: Using '{impl}' as PyTorch backend.")

    model = AutoModelForCausalLM.from_config(cfg, attn_implementation=impl, torch_dtype=torch.float16).to(device)
    model.eval()
    print(f"Model size: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    if args.attn=='tensorrt':
        results, cfg_args = benchmark_tensorrt(args, cfg, model)
    else:
        results, cfg_args = benchmark_pytorch(args, cfg, model)

    if results:
        try:
            import matplotlib.pyplot as plt
            ticks, lats = zip(*results)
            plt.plot(ticks, lats, marker='.')
            plt.show()
        except ImportError:
            print("Matplotlib not found; skipping plot.")
