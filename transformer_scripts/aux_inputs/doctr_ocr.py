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
        description="Run OCR on specified bounding boxes using docTR (with GPU accel)."
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
    return parser.parse_args()

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_rois(path):
    with open(path, 'r') as f:
        return json.load(f)

def ocr_on_rois(img, rois, predictor):
    results = []
    for roi in rois:
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        crop = img[y:y+h, x:x+w]
        out = predictor([crop])
        results.append(out.export())
    return results

def main():
    args = parse_args()

    # Initialize OCR predictor
    predictor = ocr_predictor(
        det_arch=args.det_arch,
        reco_arch=args.reco_arch,
        pretrained=True
    )
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = predictor.to(device)

    # Load data
    img = load_image(args.screenshot)
    rois = load_rois(args.json_rois)
    crops = [
        img[roi["y"]:roi["y"]+roi["h"], roi["x"]:roi["x"]+roi["w"]]
        for roi in rois
    ]

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

    # Benchmark mode
    if args.benchmark:
        n = args.benchmark
        total_crop = total_to_gpu = total_infer = total_post = 0.0
        start_all = time.time()
        for _ in range(n):
            for roi in rois:
                # Crop timing
                t0 = time.time()
                crop = img[roi["y"]:roi["y"]+roi["h"], roi["x"]:roi["x"]+roi["w"]]
                total_crop += time.time() - t0

                # Transfer timing
                t1 = time.time()
                batch = [torch.from_numpy(crop).to(device, non_blocking=True)]
                if device.type == "cuda":
                    torch.cuda.synchronize()
                total_to_gpu += time.time() - t1

                # Inference timing
                t2 = time.time()
                out = predictor(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                total_infer += time.time() - t2

                # Post-processing timing
                t3 = time.time()
                _ = out.export()
                total_post += time.time() - t3

        total_time = time.time() - start_all
        fps = n / total_time
        print(f"Average FPS over {n} runs: {fps:.2f}")
        print(f"  Crop total:         {total_crop:.3f}s")
        print(f"  Transfer total:     {total_to_gpu:.3f}s")
        print(f"  Inference total:    {total_infer:.3f}s")
        print(f"  Post-processing:    {total_post:.3f}s")
        return

    # Normal OCR pass
    results = ocr_on_rois(img, rois, predictor)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
