import torch
import timm
import time
import os
import argparse
import tensorrt as trt
from timm.data import resolve_model_data_config

# This is the specific model used in model.py
MODEL_NAME = "vit_base_patch14_dinov2.lvd142m"

def benchmark_pytorch(model, dummy_input, precision, use_compile):
    """Benchmarks PyTorch performance, with an option for torch.compile()."""
    mode = "Compiled" if use_compile else "Eager"
    print(f"\n--- Benchmarking PyTorch ({mode}, {precision}) ---")
    
    if use_compile:
        print("Compiling the model with torch.compile()...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile() failed: {e}")
            return 0.0

    NUM_WARMUP, NUM_TESTS = 20, 100
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
    fps = NUM_TESTS / total_time
    print(f"PyTorch ({mode}) Average Latency: {(total_time / NUM_TESTS) * 1000:.3f} ms")
    print(f"PyTorch ({mode}) Throughput: {fps:.2f} FPS")
    return fps

def benchmark_tensorrt(engine_path, dummy_input, output_shape):
    """Benchmarks the optimized TensorRT engine."""
    print(f"\n--- Benchmarking TensorRT ---")
    NUM_WARMUP, NUM_TESTS = 20, 100
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    try:
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
    except Exception as e:
        print(f"Error loading TensorRT engine: {e}")
        return 0.0

    with engine.create_execution_context() as context:
        # Set input shape for this execution context
        context.set_input_shape("input", dummy_input.shape)

        # Allocate memory for output
        output = torch.empty(output_shape, dtype=dummy_input.dtype, device=dummy_input.device)

        # Set bindings using the modern API
        context.set_tensor_address("input", dummy_input.data_ptr())
        context.set_tensor_address("output", output.data_ptr())

        stream = torch.cuda.current_stream().cuda_stream

        print(f"Running {NUM_WARMUP} warm-up iterations...")
        for _ in range(NUM_WARMUP):
            context.execute_async_v3(stream_handle=stream)
        torch.cuda.synchronize()

        print(f"Running {NUM_TESTS} benchmark iterations...")
        start_time = time.perf_counter()
        for _ in range(NUM_TESTS):
            context.execute_async_v3(stream_handle=stream)
        torch.cuda.synchronize()
        end_time = time.perf_counter()

    total_time = end_time - start_time
    fps = NUM_TESTS / total_time
    print(f"TensorRT Average Latency: {(total_time / NUM_TESTS) * 1000:.3f} ms")
    print(f"TensorRT Throughput: {fps:.2f} FPS")
    return fps

def build_tensorrt_engine(onnx_path, engine_path, use_fp16, batch_size, input_shape_tuple):
    """Builds and saves a TensorRT engine from an ONNX file."""
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    # Explicit batch is required for ONNX files with dynamic batch sizes
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Allocate 1GB of workspace memory
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if use_fp16:
        print("Building engine with FP16 precision...")
        config.set_flag(trt.BuilderFlag.FP16)

    print(f"\nLoading ONNX file from: {onnx_path}")
    with open(onnx_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            print("--- ONNX PARSING FAILED ---")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError("Failed to parse the ONNX file.")
    print("ONNX file parsed successfully.")

    print("Creating optimization profile for dynamic shapes...")
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    
    # Define the range of supported batch sizes
    min_shape = (1, *input_shape_tuple[1:])
    opt_shape = (batch_size, *input_shape_tuple[1:])
    max_shape = (batch_size * 2, *input_shape_tuple[1:])
    
    print(f"Defining profile for input '{input_name}': Min={min_shape}, Opt={opt_shape}, Max={max_shape}")
    profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)
    config.add_optimization_profile(profile)

    print(f"\nBuilding TensorRT engine. This may take a few minutes...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build the TensorRT engine.")

    print(f"Saving TensorRT engine to: {engine_path}")
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print("Engine saved successfully.")


def main():
    parser = argparse.ArgumentParser(description=f"Benchmark script for {MODEL_NAME} as used in model.py")
    parser.add_argument('--tensorrt', action='store_true', help="Enable TensorRT benchmark comparison.")
    parser.add_argument('--compile', action='store_true', help="Enable torch.compile() for the PyTorch benchmark.")
    parser.add_argument('--precision', type=str, default='bf16', choices=['fp16', 'bf16', 'fp32'], help="Set model precision.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for inference.")
    args = parser.parse_args()

    # --- Setup Device and DType ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        raise SystemError("This script requires a CUDA-enabled GPU.")

    use_fp16 = args.precision == 'fp16'
    use_bf16 = args.precision == 'bf16'
    
    if use_bf16 and not torch.cuda.is_bf16_supported():
        print("[WARNING] BF16 is not supported on this GPU. Falling back to FP32.")
        args.precision = 'fp32'
        use_bf16 = False
    
    dtype_map = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}
    precision_dtype = dtype_map[args.precision]

    print("--- Benchmark Configuration ---")
    print(f"Model: {MODEL_NAME}")
    print(f"Precision: {args.precision.upper()}")
    print(f"Batch Size: {args.batch_size}")
    print(f"PyTorch Compile: {args.compile}")
    print(f"TensorRT Enabled: {args.tensorrt}")
    print("-------------------------------")

    print(f"Loading model '{MODEL_NAME}' to {device}...")
    # This matches the usage in model.py: load with timm, no classification head
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0).to(device).eval()

    # Determine input size from the model's config, just like in model.py
    data_config = resolve_model_data_config(model)
    input_size = data_config['input_size'] # e.g., (3, 224, 224)
    
    # Create dummy input tensor
    dummy_input = torch.randn(args.batch_size, *input_size, device=device)
    
    if use_fp16 or use_bf16:
        model = model.to(precision_dtype)
        dummy_input = dummy_input.to(precision_dtype)

    # --- Run Benchmarks ---
    pytorch_fps = benchmark_pytorch(model, dummy_input, args.precision, use_compile=args.compile)
    tensorrt_fps = None
    
    if args.tensorrt:
        model_id = MODEL_NAME.split('.')[0] # e.g., 'vit_base_patch14_dinov2'
        onnx_filename = f"{model_id}-{args.precision}.onnx"
        engine_filename = f"{model_id}-{args.precision}.engine"
        
        # Determine the model's output feature size
        output_features = model.num_features
        output_shape = (args.batch_size, output_features)
        
        # Export to ONNX if the file doesn't exist
        if not os.path.exists(onnx_filename):
            print(f"\n--- Exporting to ONNX: {onnx_filename} ---")
            # Note: opset_version=17 is recommended for modern features
            torch.onnx.export(
                model, 
                dummy_input, 
                onnx_filename, 
                input_names=['input'], 
                output_names=['output'], 
                opset_version=17,
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            print("ONNX export complete.")
        else:
            print(f"\nFound existing ONNX file: {onnx_filename}. Skipping export.")

        # Build TensorRT engine if it doesn't exist
        if not os.path.exists(engine_filename):
            print(f"Engine file '{engine_filename}' not found. Building a new one...")
            build_tensorrt_engine(onnx_filename, engine_filename, use_fp16, args.batch_size, dummy_input.shape)
        else:
            print(f"Found existing engine file: {engine_filename}. Loading it.")
        
        tensorrt_fps = benchmark_tensorrt(engine_filename, dummy_input, output_shape)

    # --- Final Summary ---
    print("\n\n--- BENCHMARK SUMMARY ---")
    print(f"Model: {MODEL_NAME} @ {args.precision.upper()}, Batch Size: {args.batch_size}")
    if pytorch_fps > 0:
        mode = "Compiled" if args.compile else "Eager"
        print(f"PyTorch ({mode}) FPS: {pytorch_fps:.2f}")
    if tensorrt_fps:
        print(f"TensorRT Engine FPS: {tensorrt_fps:.2f}")
        if pytorch_fps > 0:
            speedup = tensorrt_fps / pytorch_fps
            print(f"\nTensorRT Speedup vs PyTorch ({mode}): {speedup:.2f}x")
    print("---------------------------\n")

if __name__ == "__main__":
    main()