#!/usr/bin/env python3
import argparse
import json
import time

import numpy as np
import cv2
import torch
from doctr.models import ocr_predictor, recognition_predictor
from torch.profiler import profile, record_function, ProfilerActivity

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run OCR on specified bounding boxes using docTR, with optional scaling, CPU/GPU selection, detection skip, and debug logging."
    )
    parser.add_argument("screenshot", help="Path to input image (PNG/JPG/etc.)")
    parser.add_argument(
        "json_rois",
        help='Path to JSON file of [{"x","y","w","h"}, ...]'
    )
    parser.add_argument(
        "--benchmark",
        type=int,
        metavar="N",
        help="Repeat OCR N times and report avg FPS"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run PyTorch profiler on one full pass and print summary"
    )
    parser.add_argument(
        "--det_arch",
        default="fast_tiny",
        help="Detection backbone (default: fast_tiny). Options include fast_tiny, fast_small, fast_base, db_mobilenet_v3_large, db_resnet50, linknet_resnet34, linknet_resnet50"
    )
    parser.add_argument(
        "--reco_arch",
        default="crnn_mobilenet_v3_small",
        help="Recognition backbone (default: crnn_mobilenet_v3_small). Options include crnn_mobilenet_v3_small, crnn_mobilenet_v3_large, sar_resnet31, vitstr_small"
    )
    parser.add_argument(
        "--skip-detect",
        action="store_true",
        help="Skip text detection stage and apply recognition directly on each ROI (only if ROI guaranteed)."
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scaling factor for each ROI (e.g., 0.2 for 20%)"
    )
    parser.add_argument(
        "--min-size",
        type=str,
        default="32x32",
        help="Minimum width x height in pixels: if scaled ROI is smaller in any dimension, skip scaling"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Force use of GPU. If unavailable or CUDA not enabled, the script will error."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information about crop and resize steps"
    )
    return parser.parse_args()


def parse_min_size(min_size_str):
    try:
        w_str, h_str = min_size_str.lower().split('x')
        return int(w_str), int(h_str)
    except Exception:
        raise ValueError(f"Invalid --min-size format: '{min_size_str}'. Use WIDTHxHEIGHT, e.g. 32x32")


def load_image(path, debug=False):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if debug:
        print(f"Loaded image {path} with shape {img.shape[1]}x{img.shape[0]}")
    return img


def load_rois(path, debug=False):
    with open(path, 'r') as f:
        rois = json.load(f)
    if debug:
        print(f"Loaded {len(rois)} ROIs from {path}")
    return rois


def preprocess_crop(crop, scale, min_w, min_h, debug=False, interp=cv2.INTER_LINEAR):
    """
    Downscale `crop` by `scale` (keeping aspect ratio), but if resulting size is below (min_w, min_h), skip scaling.
    """
    h, w = crop.shape[:2]
    if debug:
        print(f"  Original crop size: {w}x{h}")
    if scale != 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        if new_w >= min_w and new_h >= min_h:
            crop = cv2.resize(crop, (new_w, new_h), interpolation=interp)
            if debug:
                print(f"  Scaled crop to: {new_w}x{new_h} (scale={scale})")
        else:
            if debug:
                print(f"  Skipped scaling since {new_w}x{new_h} < min_size {min_w}x{min_h}")
    return crop


def main():
    args = parse_args()
    min_w, min_h = parse_min_size(args.min_size)
    debug = args.debug

    # Determine device
    if args.gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available; cannot use --gpu.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if debug:
        print(f"Using device: {device}")

    # Initialize predictor
    if args.skip_detect:
        predictor = recognition_predictor(
            reco_arch=args.reco_arch,
            pretrained=True
        ).to(device)
        if debug:
            print("Using recognition-only model for speed.")
    else:
        predictor = ocr_predictor(
            det_arch=args.det_arch,
            reco_arch=args.reco_arch,
            pretrained=True
        ).to(device)
        if debug:
            print(f"Using OCR model: det={args.det_arch}, reco={args.reco_arch}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Load inputs
    img = load_image(args.screenshot, debug)
    rois = load_rois(args.json_rois, debug)

    # Prepare crops
    crops = []
    for idx, roi in enumerate(rois):
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        if debug:
            print(f"ROI {idx}: position=({x},{y}), size={w}x{h}")
        crop = img[y:y+h, x:x+w]
        crop = preprocess_crop(crop, args.scale, min_w, min_h, debug)
        crops.append(crop)

    # Profile mode
    if args.profile:
        if debug:
            print("Profiling a single pass with torch.profiler...")
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True
        ) as prof:
            with record_function("full_ocr_pass"):
                _ = predictor(crops)
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=10
        ))
        return

    # Benchmark mode
    if args.benchmark:
        n = args.benchmark
        total_infer = total_post = 0.0
        start_all = time.time()
        for _ in range(n):
            t_inf = time.time()
            doc = predictor(crops)
            if device.type == "cuda": torch.cuda.synchronize()
            total_infer += time.time() - t_inf

            t_pp = time.time()
            if args.skip_detect:
                _ = list(doc)
            else:
                for page in doc.pages:
                    _ = page.export()
            total_post += time.time() - t_pp

        total_time = time.time() - start_all
        fps = n / total_time
        print(f"Average FPS over {n} runs: {fps:.2f}")
        print(f"  Inference total:   {total_infer:.3f}s")
        print(f"  Post-process total:{total_post:.3f}s")
        return

    # Default mode
    doc = predictor(crops)
    if args.skip_detect:
        results = list(doc)
    else:
        results = [page.export() for page in doc.pages]
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
