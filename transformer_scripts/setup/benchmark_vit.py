import torch
from transformers import ViTImageProcessor, ViTForImageClassification
import time

# --- Configuration ---
MODEL_NAME = "google/vit-base-patch16-224-in21k"
BATCH_SIZE = 1  # Simulate one frame at a time
NUM_WARMUP = 100 # Number of initial runs to discard
NUM_TESTS = 5000 # Number of runs to average for the benchmark

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    print("Warning: No GPU found. Benchmarking on CPU will be very slow.")

# Load a pre-trained ViT model and its processor
# Note: We use ViTForImageClassification, but we only care about the forward pass speed
model = ViTForImageClassification.from_pretrained(MODEL_NAME).to(device)
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

# Use float16 for a significant speedup on modern GPUs
model.half() 

# Create dummy input data that matches the model's expectations
# (batch_size, num_channels, height, width)
dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224, device=device, dtype=torch.float16)

print(f"--- Starting Benchmark ---")
print(f"Model: {MODEL_NAME}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Precision: float16")
print(f"Batch Size: {BATCH_SIZE}")

# --- Warm-up Phase ---
# The first few inferences are always slower due to CUDA kernel loading.
# We run them once to get them out of the way.
print(f"\nRunning {NUM_WARMUP} warm-up iterations...")
with torch.no_grad():
    for _ in range(NUM_WARMUP):
        _ = model(dummy_input)
torch.cuda.synchronize() # Wait for all GPU operations to complete

# --- Benchmark Phase ---
print(f"Running {NUM_TESTS} benchmark iterations...")
start_time = time.perf_counter()
with torch.no_grad():
    for _ in range(NUM_TESTS):
        _ = model(dummy_input)
torch.cuda.synchronize() # Wait for the last run to finish
end_time = time.perf_counter()

# --- Results ---
total_time = end_time - start_time
total_frames = NUM_TESTS * BATCH_SIZE
avg_latency_ms = (total_time / NUM_TESTS) * 1000
fps = total_frames / total_time

print("\n--- Benchmark Results ---")
print(f"Average Latency: {avg_latency_ms:.3f} ms per batch")
print(f"Total Throughput (FPS): {fps:.2f}")