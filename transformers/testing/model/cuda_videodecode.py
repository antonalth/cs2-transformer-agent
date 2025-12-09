#!/usr/bin/env python3
"""
test_cuda_decoding.py

Usage:
    python test_cuda_decoding.py /path/to/video.mp4
"""

import argparse
import torch
from torchcodec.decoders import VideoDecoder

def main():
    parser = argparse.ArgumentParser(description="Test torchcodec CUDA video decoding.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    args = parser.parse_args()

    # 1. Check PyTorch CUDA availability
    if not torch.cuda.is_available():
        print("[-] PyTorch cannot see a CUDA device. Aborting.")
        return

    print(f"[+] PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"[+] Attempting to decode: {args.video_path} on 'cuda'")

    try:
        # 2. Initialize Decoder on GPU
        decoder = VideoDecoder(args.video_path, device="cuda")
        
        # 3. Decode a small range (Frames 0 to 4)
        # We use get_frames_in_range as it is confirmed to exist in your version
        frames = decoder.get_frames_in_range(0, 4).data
        
        print("\n[SUCCESS] Video decoded on GPU!")
        print(f"   Tensor Shape: {frames.shape} (N, C, H, W)")
        print(f"   Tensor Device: {frames.device}")
        print(f"   Memory Location: {'VRAM' if frames.is_cuda else 'RAM'}")

    except RuntimeError as e:
        print("\n[FAILURE] RuntimeError caught:")
        print(f"   {e}")
        print("-" * 60)
        if "Unsupported device: cuda" in str(e):
            print(">> DIAGNOSIS: Your torchcodec/ffmpeg build does NOT support NVDEC.")
            print(">> FIX: You must use device='cpu' or recompile torchcodec with CUDA flags.")
            print("        (See: https://github.com/pytorch/torchcodec#gpu-support)")
    except Exception as e:
        print(f"\n[FAILURE] Unexpected error: {e}")

if __name__ == "__main__":
    main()