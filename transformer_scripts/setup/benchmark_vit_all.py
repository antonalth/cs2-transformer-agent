import torch
from transformers import ViTForImageClassification
import time
import os
import argparse
import tensorrt as trt
from itertools import product

# --- Model Maps ---
# Using a nested dictionary for easier management
MODEL_ZOO = {
    16: { # Patch Size 16
        224: { # Resolution 224
            "base": "google/vit-base-patch16-224",
            "large": "google/vit-large-patch16-224",
            "huge": "google/vit-huge-patch14-224-in21k"
        },
        384: { # Resolution 384
            "base": "google/vit-base-patch16-384",
            "large": "google/vit-large-patch16-384",
            "huge": "google/vit-huge-patch14-224-in21k" # Fallback
        }
    },
    32: { # Patch Size 32
        224: {
            "base": "google/vit-base-patch32-224",
            "large": "google/vit-large-patch32-224",
            "huge": None # No official huge patch32 model
        },
        384: {
            "base": "google/vit-base-patch32-384",
            "large": "google/vit-large-patch32-384",
            "huge": None # No official huge patch32 model
        }
    }
}


def benchmark_pytorch(model, dummy_input, precision):
    print(f"--- Benchmarking PyTorch ({precision}) ---")
    NUM_WARMUP, NUM_TESTS = 20, 100
    with torch.no_grad():
        for _ in range(NUM_WARMUP): _ = model(dummy_input)
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(NUM_TESTS): _ = model(dummy_input)
    torch.cuda.synchronize()
    return NUM_TESTS / (time.perf_counter() - start_time)

def benchmark_tensorrt(engine_path, dummy_input, num_labels):
    print("--- Benchmarking TensorRT ---")
    NUM_WARMUP, NUM_TESTS = 20, 100
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    with engine.create_execution_context() as context:
        context.set_input_shape("input", dummy_input.shape)
        output_shape = (dummy_input.shape[0], num_labels)
        output = torch.empty(output_shape, dtype=dummy_input.dtype, device=dummy_input.device)
        context.set_tensor_address("input", dummy_input.data_ptr())
        context.set_tensor_address("output", output.data_ptr())
        stream = torch.cuda.current_stream().cuda_stream
        for _ in range(NUM_WARMUP): context.execute_async_v3(stream_handle=stream)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(NUM_TESTS): context.execute_async_v3(stream_handle=stream)
        torch.cuda.synchronize()
    return NUM_TESTS / (time.perf_counter() - start_time)

def build_tensorrt_engine(onnx_path, engine_path, use_fp16, batch_size):
    print("--- Building TensorRT Engine ---")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30) # 2 GB
    if use_fp16: config.set_flag(trt.BuilderFlag.FP16)
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors): print(parser.get_error(error))
            raise ValueError("Failed to parse ONNX file.")
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    min_shape = (1, 3, input_tensor.shape[2], input_tensor.shape[3])
    opt_shape = (batch_size, 3, input_tensor.shape[2], input_tensor.shape[3])
    max_shape = (batch_size * 2, 3, input_tensor.shape[2], input_tensor.shape[3])
    profile.set_shape(input_tensor.name, min=min_shape, opt=opt_shape, max=max_shape)
    config.add_optimization_profile(profile)
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None: raise RuntimeError("Failed to build TensorRT engine.")
    with open(engine_path, "wb") as f: f.write(serialized_engine)
    print(f"Engine saved to: {engine_path}")

def run_single_benchmark(config):
    """Runs a single benchmark configuration and returns the FPS results."""
    model_size = config['model_size']
    resolution = config['resolution']
    patch_size = config['patch_size']
    use_tensorrt = config['tensorrt']
    precision = 'fp16' # Hardcoding fp16 as it's the most relevant
    use_fp16 = True
    batch_size = 1
    
    # Check if model exists in our zoo
    model_name = MODEL_ZOO.get(patch_size, {}).get(resolution, {}).get(model_size)
    if not model_name:
        print(f"Skipping: No model found for {model_size}/p{patch_size}@{resolution}px")
        return None, None
        
    config_str = f"Model: {model_size}/p{patch_size} @ {resolution}px, Mode: {'TensorRT' if use_tensorrt else 'PyTorch'}"
    print(f"\n\n{'='*80}\nRUNNING: {config_str}\n{'='*80}")
    
    try:
        device = torch.device("cuda")
        print(f"Loading model '{model_name}'...")
        model = ViTForImageClassification.from_pretrained(model_name).to(device).eval()
        input_size = model.config.image_size
        num_labels = model.config.num_labels
        dummy_input = torch.randn(batch_size, 3, input_size, input_size, device=device)
        if use_fp16:
            model.half()
            dummy_input = dummy_input.half()

        pytorch_fps = benchmark_pytorch(model, dummy_input, precision)
        tensorrt_fps = None

        if use_tensorrt:
            filename_prefix = f"vit-{model_size}-p{patch_size}-{resolution}-{precision}"
            onnx_filename = f"{filename_prefix}.onnx"
            engine_filename = f"{filename_prefix}.engine"
            
            if not os.path.exists(engine_filename):
                if not os.path.exists(onnx_filename):
                    print(f"Exporting to ONNX: {onnx_filename}")
                    torch.onnx.export(model, dummy_input, onnx_filename, input_names=['input'], output_names=['output'], opset_version=17, dynamic_axes={'input': {0: 'batch_size'}})
                build_tensorrt_engine(onnx_filename, engine_filename, use_fp16, batch_size)
            
            tensorrt_fps = benchmark_tensorrt(engine_filename, dummy_input, num_labels)
        
        return pytorch_fps, tensorrt_fps

    except Exception as e:
        print(f"\n[ERROR] Failed to run benchmark for config: {config_str}")
        print(f"Error details: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Vision Transformer Benchmark Suite")
    parser.add_argument('--all', action='store_true', help="Run all combinations of models and settings.")
    # Individual run flags (ignored if --all is used)
    parser.add_argument('--res384', action='store_true', help="Use 384x384 resolution.")
    parser.add_argument('--patch32', action='store_true', help="Use 32x32 patch size.")
    model_group = parser.add_mutually_exclusive_group(required=False)
    model_group.add_argument('--base', action='store_true', help="Use ViT-Base model (default).")
    model_group.add_argument('--large', action='store_true', help="Use ViT-Large model.")
    model_group.add_argument('--huge', action='store_true', help="Use ViT-Huge model.")
    parser.add_argument('--tensorrt', action='store_true', help="Enable TensorRT benchmark.")
    args = parser.parse_args()

    results = []

    if args.all:
        # Define the parameter space for the --all run
        # Note: We exclude 'huge' from patch32 as it doesn't exist.
        resolutions = [224, 384]
        patch_sizes = [16, 32]
        model_sizes = ["base", "large"] # Sticking to base/large for full coverage
        
        for res, patch, size in product(resolutions, patch_sizes, model_sizes):
            # PyTorch Run
            config = {'resolution': res, 'patch_size': patch, 'model_size': size, 'tensorrt': False}
            pytorch_fps, _ = run_single_benchmark(config)
            
            # TensorRT Run
            config['tensorrt'] = True
            _, tensorrt_fps = run_single_benchmark(config)

            results.append({
                'model': f"{size}/p{patch}@{res}",
                'pytorch_fps': pytorch_fps,
                'tensorrt_fps': tensorrt_fps
            })
    else:
        # Run a single configuration based on flags
        config = {
            'resolution': 384 if args.res384 else 224,
            'patch_size': 32 if args.patch32 else 16,
            'model_size': "large" if args.large else "huge" if args.huge else "base",
            'tensorrt': args.tensorrt
        }
        # For a single run, we just print the final summary like before
        pytorch_fps, tensorrt_fps = run_single_benchmark(config)
        
        if pytorch_fps is not None:
             print("\n\n--- BENCHMARK SUMMARY ---")
             model_name = MODEL_ZOO.get(config['patch_size'], {}).get(config['resolution'], {}).get(config['model_size'])
             print(f"Model: {model_name} @ {config['resolution']}x{config['resolution']}")
             print(f"PyTorch Native FPS: {pytorch_fps:.2f}")
             if tensorrt_fps:
                 print(f"TensorRT Engine FPS: {tensorrt_fps:.2f}")
                 print(f"\nTensorRT Speedup: {(tensorrt_fps / pytorch_fps):.2f}x")
             print("---------------------------\n")
        return # Exit after single run

    # Print the final results table for the --all run
    print("\n\n" + "="*80)
    print(" " * 25 + "COMPREHENSIVE BENCHMARK RESULTS")
    print("="*80)
    print(f"{'Model Configuration':<25} | {'PyTorch FPS':<15} | {'TensorRT FPS':<15} | {'Speedup':<10}")
    print("-"*80)
    
    for res in sorted(results, key=lambda x: (x['model'].split('@')[1], x['model'].split('/')[1])):
        model = res['model']
        p_fps = f"{res['pytorch_fps']:.2f}" if res['pytorch_fps'] is not None else "N/A"
        t_fps = f"{res['tensorrt_fps']:.2f}" if res['tensorrt_fps'] is not None else "N/A"
        
        speedup = "N/A"
        if res['pytorch_fps'] and res['tensorrt_fps']:
            speedup_val = res['tensorrt_fps'] / res['pytorch_fps']
            speedup = f"{speedup_val:.2f}x"
            
        print(f"{model:<25} | {p_fps:<15} | {t_fps:<15} | {speedup:<10}")
        
    print("="*80)


if __name__ == "__main__":
    main()