# benchmark.py
# A script to benchmark the performance of individual model components.

import torch
import torch.nn as nn
import argparse
import time
import math
from transformers import ViTModel, ViTConfig

def get_model_params(model: nn.Module) -> float:
    """Calculates the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def benchmark_vit(vit_version: str, batch_size: int, device: torch.device, reps: int):
    """
    Benchmarks a specified Vision Transformer model.

    Args:
        vit_version (str): The version of ViT to test ('224' or '384').
        batch_size (int): The number of images to process in a batch.
        device (torch.device): The device (CPU or CUDA) to run on.
        reps (int): The number of repetitions for the benchmark.
    """
    if vit_version == '224':
        model_name = "google/vit-large-patch16-224-in21k"
        image_size = 224
    elif vit_version == '384':
        model_name = "google/vit-large-patch16-384"
        image_size = 384
    else:
        raise ValueError("Invalid ViT version specified. Use '224' or '384'.")

    print(f"\n--- Benchmarking ViT Model: {model_name} ---")
    print(f"Batch Size: {batch_size}, Image Size: {image_size}x{image_size}, Device: {device}, Reps: {reps}")

    # Load model and set to evaluation mode
    model = ViTModel.from_pretrained(model_name, add_pooling_layer=False).to(device)
    model.eval()

    # Create dummy input data
    dummy_data = torch.randn(batch_size, 3, image_size, image_size, device=device)
    
    # Get model size
    params_m = get_model_params(model)
    print(f"Model Parameters: {params_m:.2f}M")

    with torch.no_grad():
        # Warmup for accurate GPU timing
        if device.type == 'cuda':
            print("Warming up GPU...")
            for _ in range(5):
                _ = model(dummy_data)
            torch.cuda.synchronize()

        # Timed benchmark loop
        print("Running benchmark...")
        start_time = time.perf_counter()
        for _ in range(reps):
            _ = model(dummy_data)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.perf_counter()

    # Calculate and display results
    total_time = end_time - start_time
    avg_latency = (total_time / reps) * 1000  # in ms
    throughput = (reps * batch_size) / total_time  # in images/sec

    print("--- Results ---")
    print(f"Average Latency: {avg_latency:.3f} ms")
    print(f"Throughput: {throughput:.2f} images/sec")
    print("---------------")


def benchmark_transformer(param_choice: str, seq_len: int, batch_size: int, device: torch.device, reps: int):
    """
    Benchmarks a standard nn.TransformerEncoder at a given parameter scale.

    Args:
        param_choice (str): The desired parameter scale ('50M', '150M', '500M', '1B').
        seq_len (int): The length of the input sequence.
        batch_size (int): The number of sequences to process in a batch.
        device (torch.device): The device (CPU or CUDA) to run on.
        reps (int): The number of repetitions for the benchmark.
    """
    # Define model configurations for different parameter counts
    TRANSFORMER_CONFIGS = {
        '50M':   {'layers': 6,  'dim': 1024, 'heads': 16},
        '150M':  {'layers': 12, 'dim': 1280, 'heads': 16},
        '500M':  {'layers': 16, 'dim': 2048, 'heads': 32},
        '1B':    {'layers': 24, 'dim': 2560, 'heads': 32}
    }

    if param_choice not in TRANSFORMER_CONFIGS:
        raise ValueError(f"Invalid param choice. Use one of {list(TRANSFORMER_CONFIGS.keys())}")

    config = TRANSFORMER_CONFIGS[param_choice]
    d_model = config['dim']
    n_head = config['heads']
    n_layers = config['layers']
    dim_feedforward = d_model * 4

    print(f"\n--- Benchmarking Transformer Model: ~{param_choice} ---")
    print(f"Config: {n_layers} layers, {d_model} dim, {n_head} heads")
    print(f"Batch Size: {batch_size}, Sequence Length: {seq_len}, Device: {device}, Reps: {reps}")

    # Create the model
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_head,
        dim_feedforward=dim_feedforward,
        dropout=0.1,
        batch_first=False # Use [S, B, E] format
    )
    model = nn.TransformerEncoder(encoder_layer, num_layers=n_layers).to(device)
    model.eval()

    # Create dummy input data
    # Shape: [Sequence Length, Batch Size, Embedding Dim]
    dummy_data = torch.randn(seq_len, batch_size, d_model, device=device)

    # Get model size
    params_m = get_model_params(model)
    print(f"Actual Model Parameters: {params_m:.2f}M")
    
    with torch.no_grad():
        # Warmup
        if device.type == 'cuda':
            print("Warming up GPU...")
            for _ in range(5):
                _ = model(dummy_data)
            torch.cuda.synchronize()

        # Timed benchmark loop
        print("Running benchmark...")
        start_time = time.perf_counter()
        for _ in range(reps):
            _ = model(dummy_data)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.perf_counter()

    # Calculate and display results
    total_time = end_time - start_time
    avg_latency = (total_time / reps) * 1000  # in ms
    throughput_seq = (reps * batch_size) / total_time  # in sequences/sec
    throughput_tok = (reps * batch_size * seq_len) / total_time # in tokens/sec

    print("--- Results ---")
    print(f"Average Latency: {avg_latency:.3f} ms")
    print(f"Throughput (Sequences/sec): {throughput_seq:.2f}")
    print(f"Throughput (Tokens/sec): {throughput_tok:,.0f}")
    print("---------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark script for core model components.")
    
    # General arguments
    parser.add_argument('--model', type=str, required=True, choices=['vit', 'transformer'], help="The component to benchmark.")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help="The device to run the benchmark on.")
    parser.add_argument('--batch-size', type=int, default=8, help="Batch size for the benchmark.")
    parser.add_argument('--reps', type=int, default=20, help="Number of repetitions for timing.")

    # ViT-specific arguments
    parser.add_argument('--vit-version', type=str, default='384', choices=['224', '384'], help="Version of the ViT model to test.")

    # Transformer-specific arguments
    parser.add_argument('--seq-len', type=int, default=1024, help="Sequence length for the transformer benchmark.")
    parser.add_argument('--params', type=str, default='150M', choices=['50M', '150M', '500M', '1B'], help="Target parameter count for the transformer.")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available on this system. Falling back to CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    # Run the selected benchmark
    if args.model == 'vit':
        benchmark_vit(
            vit_version=args.vit_version,
            batch_size=args.batch_size,
            device=device,
            reps=args.reps
        )
    elif args.model == 'transformer':
        benchmark_transformer(
            param_choice=args.params,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            device=device,
            reps=args.reps
        )