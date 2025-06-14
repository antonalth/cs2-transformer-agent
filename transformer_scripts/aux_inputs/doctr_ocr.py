#!/usr/bin/env python3
import argparse
import json
import time

import numpy as np
import cv2
import torch
from doctr.models import ocr_predictor
from torch.profiler import profile, record_function, ProfilerActivity

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run OCR on specified bounding boxes using docTR (with GPU accel), with optional scaling."
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
        default="db_resnet50",
        help="Detection backbone (default: db_resnet50)"
    )
    parser.add_argument(
        "--reco_arch",
        default="crnn_vgg16_bn",
        help="Recognition backbone (default: crnn_vgg16_bn)"
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
    return parser.parse_args()


def parse_min_size(min_size_str):
    try:
        w_str, h_str = min_size_str.lower().split('x')
        return int(w_str), int(h_str)
    except Exception:
        raise ValueError(f"Invalid --min-size format: '{min_size_str}'. Use WIDTHxHEIGHT, e.g. 32x32")


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_rois(path):
    with open(path, 'r') as f:
        return json.load(f)


def preprocess_crop(crop, scale, min_w, min_h, interp=cv2.INTER_LINEAR):
    """
    Downscale `crop` by `scale` (keeping aspect ratio), but if the resulting width or height
    would be below (min_w, min_h), skip scaling and return the original crop.
    """
    if scale != 1.0:
        h, w = crop.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        if new_w >= min_w and new_h >= min_h:
            crop = cv2.resize(crop, (new_w, new_h), interpolation=interp)
    return crop


def ocr_on_crops(crops, predictor):
    results = []
    for crop in crops:
        out = predictor([crop])
        results.append(out.export())
    return results


def main():
    args = parse_args()
    min_w, min_h = parse_min_size(args.min_size)

    # Initialize OCR predictor with GPU accel if available
    predictor = ocr_predictor(
        det_arch=args.det_arch,
        reco_arch=args.reco_arch,
        pretrained=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = predictor.to(device)
    torch.backends.cudnn.benchmark = True

    # Load inputs
    img = load_image(args.screenshot)
    rois = load_rois(args.json_rois)

    # Preprocess all crops (crop + optional scale)
    crops = []
    for roi in rois:
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        crop = img[y:y+h, x:x+w]
        crop = preprocess_crop(crop, args.scale, min_w, min_h)
        crops.append(crop)

    # Profiling mode
    if args.profile:
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

    # Benchmark mode (include crop + resize in timing)
    if args.benchmark:
        n = args.benchmark
        total_crop = total_to_gpu = total_infer = total_post = 0.0
        start_all = time.time()
        for _ in range(n):
            for roi in rois:
                x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
                # Crop timing
                t0 = time.time()
                crop = img[y:y+h, x:x+w]
                total_crop += time.time() - t0
                # Scale timing
                t1 = time.time()
                crop = preprocess_crop(crop, args.scale, min_w, min_h)
                total_crop += time.time() - t1
                # Transfer timing
                t2 = time.time()
                batch = [torch.from_numpy(crop).to(device, non_blocking=True)]
                if device.type == "cuda":
                    torch.cuda.synchronize()
                total_to_gpu += time.time() - t2
                # Inference timing
                t3 = time.time()
                out = predictor(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                total_infer += time.time() - t3
                # Post-processing timing
                t4 = time.time()
                _ = out.export()
                total_post += time.time() - t4

        total_time = time.time() - start_all
        fps = n / total_time
        print(f"Average FPS over {n} runs: {fps:.2f}")
        print(f"  Crop+resize total: {total_crop:.3f}s")
        print(f"  Transfer total:    {total_to_gpu:.3f}s")
        print(f"  Inference total:   {total_infer:.3f}s")
        print(f"  Post-process total:{total_post:.3f}s")
        return

    # Default: single pass OCR
    results = ocr_on_crops(crops, predictor)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
