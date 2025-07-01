import torch
import torch.nn as nn
import time
import math
import os
import numpy as np

# --- Check for TensorRT and offer guidance if not found ---
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    
# --- Model Definition (No Changes Needed) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

def setup_pytorch_model(sequence_length, batch_size=1):
    print("--- 1. Setting up PyTorch Model ---")
    if not torch.cuda.is_available(): raise EnvironmentError("CUDA not available.")
    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(0)}")

    d_model, nhead, d_hid, nlayers = 1024, 16, 4096, 12
    encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_hid, batch_first=True, activation=nn.GELU())
    pos_encoder = PositionalEncoding(d_model, max_len=sequence_length)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)
    
    model = nn.Sequential(nn.Embedding(30000, d_model), pos_encoder, transformer_encoder).to(device)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model configured with ~{num_params / 1_000_000:.2f}M parameters.")

    dummy_input_ids = torch.randint(0, 30000, (batch_size, sequence_length), device=device)
    print(f"Created dummy input tensor with shape: {dummy_input_ids.shape}")
    return model, dummy_input_ids

def run_pytorch_benchmark(model, inputs, warmup_runs, benchmark_runs):
    print("\n--- 2. Benchmarking PyTorch + FlashAttention ---")
    print("Verifying available backends...")
    has_flash = torch.backends.cuda.flash_sdp_enabled()
    print(f"  FlashAttention backend enabled: {has_flash}")
    if not has_flash: print("  Warning: No optimized attention backend detected.")

    with torch.no_grad():
        print(f"Performing {warmup_runs} warm-up runs...")
        for _ in range(warmup_runs): _ = model(inputs)
        torch.cuda.synchronize()

        print(f"Running {benchmark_runs} benchmark runs...")
        start_time = time.perf_counter()
        for _ in range(benchmark_runs): _ = model(inputs)
        torch.cuda.synchronize()
        end_time = time.perf_counter()

    total_time = end_time - start_time
    fps = benchmark_runs / total_time
    print("\n--- PyTorch Benchmark Results ---")
    print(f"  Average latency: {total_time * 1000 / benchmark_runs:.2f} ms")
    print(f"  Inference FPS: {fps:.2f}")
    return fps

def export_to_onnx(model, dummy_input, onnx_file_path):
    print(f"\n--- 3. Exporting model to ONNX: {onnx_file_path} ---")
    model.to("cpu")
    dummy_input = dummy_input.to("cpu")
    try:
        torch.onnx.export(model, dummy_input, onnx_file_path, export_params=True,
                          opset_version=17, do_constant_folding=True,
                          input_names=['input_ids'], output_names=['output'])
        print("ONNX export successful.")
        return True
    except Exception as e:
        print(f"An error occurred during ONNX export: {e}")
        return False

def build_tensorrt_engine(onnx_file_path, engine_file_path, use_fp16=True):
    print(f"\n--- 4. Building TensorRT Engine (FP16: {use_fp16}) ---")
    print("This may take several minutes...")
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    if use_fp16 and builder.platform_has_fast_fp16:
        print("FP16 mode enabled.")
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        print("FP16 not supported or not enabled. Using FP32.")

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("Failed to build the engine.")
        return False

    with open(engine_file_path, 'wb') as f:
        f.write(serialized_engine)
    print(f"TensorRT engine built and saved to: {engine_file_path}")
    return True

def run_tensorrt_benchmark(engine_file_path, inputs, warmup_runs, benchmark_runs):
    print("\n--- 5. Benchmarking TensorRT Engine ---")
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    with engine.create_execution_context() as context:
        # Allocate input and output host/device buffers
        inputs_trt, outputs_trt, bindings, stream = [], [], [], torch.cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            device_mem = torch.cuda.empty(size=(size,), dtype=torch.from_numpy(np.dtype(dtype)).to(torch.device('cuda')))
            bindings.append(device_mem.data_ptr())
            if engine.binding_is_input(binding):
                inputs_trt.append({'name': binding, 'host': None, 'device': device_mem})
            else:
                outputs_trt.append({'name': binding, 'host': torch.empty(size=(size,), dtype=torch.from_numpy(np.dtype(dtype)).to(torch.device('cpu'))), 'device': device_mem})

        # Prepare input data
        input_data_cpu = inputs.cpu().numpy().ravel()
        inputs_trt[0]['host'] = input_data_cpu
        
        # Warm-up
        print(f"Performing {warmup_runs} warm-up runs...")
        for _ in range(warmup_runs):
            torch.cuda.memcpy_async(inputs_trt[0]['device'], torch.from_numpy(inputs_trt[0]['host']).to('cuda'), stream=stream.cuda_stream)
            context.execute_async_v2(bindings=bindings, stream_handle=stream.cuda_stream)
        stream.synchronize()

        # Measurement using CUDA events for precision
        print(f"Running {benchmark_runs} benchmark runs...")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(benchmark_runs):
            context.execute_async_v2(bindings=bindings, stream_handle=stream.cuda_stream)
        end_event.record()
        stream.synchronize()
        
        total_time_ms = start_event.elapsed_time(end_event)
        fps = benchmark_runs / (total_time_ms / 1000)

    print("\n--- TensorRT Benchmark Results ---")
    print(f"  Average latency: {total_time_ms / benchmark_runs:.2f} ms")
    print(f"  Inference FPS: {fps:.2f}")
    return fps

if __name__ == "__main__":
    # --- Configuration ---
    SEQUENCE_LENGTH = 4096
    BATCH_SIZE = 1
    WARMUP_RUNS = 20
    BENCHMARK_RUNS = 100
    ONNX_FILE = "transformer.onnx"
    TENSORRT_ENGINE_FILE = "transformer.engine"

    # --- Run Full Benchmark ---
    pytorch_fps, tensorrt_fps = 0, 0
    try:
        # 1. PyTorch Benchmark
        model, dummy_inputs = setup_pytorch_model(SEQUENCE_LENGTH, BATCH_SIZE)
        pytorch_fps = run_pytorch_benchmark(model, dummy_inputs, WARMUP_RUNS, BENCHMARK_RUNS)

        # Check for TensorRT before proceeding
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT Python package not found. Skipping TensorRT benchmark.")

        # 2. ONNX Export
        if not export_to_onnx(model, dummy_inputs, ONNX_FILE):
             raise RuntimeError("ONNX export failed. Aborting.")
        
        # 3. Build TensorRT Engine
        if not build_tensorrt_engine(ONNX_FILE, TENSORRT_ENGINE_FILE, use_fp16=True):
            raise RuntimeError("TensorRT engine build failed. Aborting.")
        
        # 4. TensorRT Benchmark
        tensorrt_fps = run_tensorrt_benchmark(TENSORRT_ENGINE_FILE, dummy_inputs, WARMUP_RUNS, BENCHMARK_RUNS)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # --- Final Comparison ---
        print("\n" + "="*40)
        print("          FINAL PERFORMANCE SUMMARY          ")
        print("="*40)
        print(f"PyTorch + FlashAttention FPS : {pytorch_fps:.2f}")
        print(f"TensorRT (FP16) FPS        : {tensorrt_fps:.2f}")
        if pytorch_fps > 0 and tensorrt_fps > 0:
            speedup = tensorrt_fps / pytorch_fps
            print(f"\nTensorRT Speedup: {speedup:.2f}x")
        print("="*40)
        
        # Clean up generated files
        if os.path.exists(ONNX_FILE): os.remove(ONNX_FILE)
        if os.path.exists(TENSORRT_ENGINE_FILE): os.remove(TENSORRT_ENGINE_FILE)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print("\nBenchmark complete. Cleaned up temporary files.")