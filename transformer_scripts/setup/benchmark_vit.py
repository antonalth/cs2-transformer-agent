import torch
from transformers import ViTForImageClassification
import time
import os
import argparse
import tensorrt as trt

# --- Configuration & Model Mapping ---
# Maps simple names to Hugging Face model identifiers
MODEL_MAP = {
    "base": "google/vit-base-patch16-224",
    "large": "google/vit-large-patch16-224",
    "huge": "google/vit-huge-patch14-224-in21k"
}

# --- Benchmark Functions ---

def benchmark_pytorch(model, dummy_input, precision):
    """Benchmarks the native PyTorch model."""
    print(f"\n--- Benchmarking PyTorch ({precision}) ---")
    NUM_WARMUP = 20
    NUM_TESTS = 100

    print(f"Running {NUM_WARMUP} warm-up iterations...")
    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            _ = model(dummy_input)
    torch.cuda.synchronize()

    print(f"Running {NUM_TESTS} benchmark iterations...")
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(NUM_TESTS):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_latency_ms = (total_time / NUM_TESTS) * 1000
    fps = NUM_TESTS / total_time
    
    print(f"PyTorch Average Latency: {avg_latency_ms:.3f} ms")
    print(f"PyTorch Throughput: {fps:.2f} FPS")
    return fps

def benchmark_tensorrt(engine_path, dummy_input, precision):
    """Benchmarks the optimized TensorRT engine."""
    print(f"\n--- Benchmarking TensorRT ({precision}) ---")
    NUM_WARMUP = 20
    NUM_TESTS = 100
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    with engine.create_execution_context() as context:
        # Allocate GPU memory for input and output using PyTorch for convenience
        output = torch.empty_like(dummy_input) # We don't care about the output values
        
        # Get memory pointers
        input_ptr = dummy_input.data_ptr()
        output_ptr = output.data_ptr()
        
        # Set up bindings
        bindings = [input_ptr, output_ptr]
        stream = torch.cuda.current_stream().cuda_stream

        print(f"Running {NUM_WARMUP} warm-up iterations...")
        for _ in range(NUM_WARMUP):
            context.execute_async_v2(bindings=bindings, stream_handle=stream)
        torch.cuda.synchronize()

        print(f"Running {NUM_TESTS} benchmark iterations...")
        start_time = time.perf_counter()
        for _ in range(NUM_TESTS):
            context.execute_async_v2(bindings=bindings, stream_handle=stream)
        torch.cuda.synchronize()
        end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_latency_ms = (total_time / NUM_TESTS) * 1000
    fps = NUM_TESTS / total_time
    
    print(f"TensorRT Average Latency: {avg_latency_ms:.3f} ms")
    print(f"TensorRT Throughput: {fps:.2f} FPS")
    return fps

def build_tensorrt_engine(onnx_path, engine_path, use_fp16):
    """Builds a TensorRT engine from an ONNX file."""
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()
    
    # Set max workspace size (adjust as needed for your GPU)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB

    if use_fp16:
        print("Building engine with FP16 precision...")
        config.set_flag(trt.BuilderFlag.FP16)

    print(f"Loading ONNX file from: {onnx_path}")
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError("Failed to parse the ONNX file.")
    
    print(f"Building TensorRT engine. This may take a few minutes...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build the TensorRT engine.")

    print(f"Saving TensorRT engine to: {engine_path}")
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description="Advanced Vision Transformer Benchmark Tool")
    model_group = parser.add_mutually_exclusive_group(required=False)
    model_group.add_argument('--base', action='store_true', help="Use ViT-Base model (default)")
    model_group.add_argument('--large', action='store_true', help="Use ViT-Large model")
    model_group.add_argument('--huge', action='store_true', help="Use ViT-Huge model")
    
    parser.add_argument('--tensorrt', action='store_true', help="Enable TensorRT benchmark comparison")
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp16', 'fp32'], help="Set model precision")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for inference")
    
    args = parser.parse_args()

    # Determine model size
    model_size = "base"
    if args.large: model_size = "large"
    if args.huge: model_size = "huge"

    model_name = MODEL_MAP[model_size]
    precision = args.precision
    use_fp16 = (precision == 'fp16')
    
    print("--- Benchmark Configuration ---")
    print(f"Model: {model_name}")
    print(f"Precision: {precision}")
    print(f"Batch Size: {args.batch_size}")
    print(f"TensorRT Enabled: {args.tensorrt}")
    print("-------------------------------")

    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        raise SystemError("This script requires a CUDA-enabled GPU.")
    
    print(f"Loading model '{model_name}' to {device}...")
    model = ViTForImageClassification.from_pretrained(model_name).to(device).eval()

    # Determine input size from model config
    input_size = model.config.image_size
    patch_size = model.config.patch_size
    if model_size == 'huge': # Special case for ViT-Huge
        patch_size = 14
        
    # Create dummy input
    dummy_input = torch.randn(args.batch_size, 3, input_size, input_size, device=device)
    
    if use_fp16:
        model.half()
        dummy_input = dummy_input.half()

    # --- Run Benchmarks ---
    pytorch_fps = benchmark_pytorch(model, dummy_input, precision)
    tensorrt_fps = None

    if args.tensorrt:
        # Define file paths for ONNX and TRT engines
        onnx_filename = f"vit-{model_size}-{precision}.onnx"
        engine_filename = f"vit-{model_size}-{precision}.engine"
        
        # 1. Export to ONNX
        print(f"\n--- Exporting to ONNX: {onnx_filename} ---")
        torch.onnx.export(model,
                          dummy_input,
                          onnx_filename,
                          input_names=['input'],
                          output_names=['output'],
                          opset_version=17, # A commonly supported version
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        
        # 2. Build or Load TensorRT Engine
        if not os.path.exists(engine_filename):
            print(f"Engine file not found. Building a new one...")
            build_tensorrt_engine(onnx_filename, engine_filename, use_fp16)
        else:
            print(f"Found existing engine file: {engine_filename}. Loading it.")
        
        # 3. Benchmark TensorRT
        tensorrt_fps = benchmark_tensorrt(engine_filename, dummy_input, precision)

    # --- Final Summary ---
    print("\n\n--- BENCHMARK SUMMARY ---")
    print(f"Model: {model_name} @ {precision}")
    print(f"PyTorch Native FPS: {pytorch_fps:.2f}")
    if tensorrt_fps:
        print(f"TensorRT Engine FPS: {tensorrt_fps:.2f}")
        speedup = tensorrt_fps / pytorch_fps
        print(f"\nTensorRT Speedup: {speedup:.2f}x")
    print("---------------------------\n")


if __name__ == "__main__":
    main()