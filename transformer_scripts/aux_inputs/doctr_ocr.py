#!/usr/bin/env python3
import argparse
import json
import time

import numpy as np
import cv2
import torch
from doctr.models import ocr_predictor

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run OCR on specified bounding boxes using docTR (with GPU accel)."
    )
    parser.add_argument("screenshot", help="Path to input image (PNG/JPG/etc.)")
    parser.add_argument("json_rois", help='Path to JSON file of [{"x","y","w","h"}, ...]')
    parser.add_argument(
        "--benchmark", type=int, metavar="N",
        help="Repeat OCR N times and report avg FPS"
    )
    parser.add_argument(
        "--det_arch", default="db_resnet50",
        help="Detection backbone (default: db_resnet50)"
    )
    parser.add_argument(
        "--reco_arch", default="crnn_vgg16_bn",
        help="Recognition backbone (default: crnn_vgg16_bn)"
    )
    return parser.parse_args()

def load_image(path):
    # Read with OpenCV (BGR), convert to RGB
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
        # docTR expects a list of pages (ndarrays) even for one image
        out = predictor([crop])
        # export() gives a serializable dict of detections & recognized text
        results.append(out.export())
    return results

def main():
    args = parse_args()

    # Load model and move to GPU
    predictor = ocr_predictor(
        det_arch=args.det_arch,
        reco_arch=args.reco_arch,
        pretrained=True
    ).cuda()  # GPU accel :contentReference[oaicite:1]{index=1}

    img = load_image(args.screenshot)
    rois = load_rois(args.json_rois)

    if args.benchmark:
        n = args.benchmark
        start = time.time()
        for _ in range(n):
            _ = ocr_on_rois(img, rois, predictor)
        total = time.time() - start
        fps = n / total
        print(f"Average FPS over {n} runs: {fps:.2f}")
    else:
        results = ocr_on_rois(img, rois, predictor)
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
