#!/usr/bin/env python3
import argparse
import json
import time

import numpy as np
import cv2
from paddleocr import PaddleOCR
import torch
from torch.profiler import profile, record_function, ProfilerActivity

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run OCR on specified bounding boxes using PaddleOCR v3 mobile models, with optional scaling, CPU/GPU, and debug logging."
    )
    parser.add_argument("screenshot", help="Path to input image (PNG/JPG/etc.)")
    parser.add_argument("json_rois", help='Path to JSON file of [{"x","y","w","h"}, ...]')
    parser.add_argument("--benchmark", type=int, metavar="N", help="Repeat OCR N times and report avg FPS")
    parser.add_argument("--profile", action="store_true", help="Run PyTorch profiler on one full pass and print summary")
    parser.add_argument("--skip-detect", action="store_true", help="Skip detection; run recognition-only on each ROI.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scaling factor for each ROI (e.g., 0.2 for 20%)")
    parser.add_argument("--min-size", type=str, default="32x32", help="Minimum width x height in pixels: if scaled ROI is smaller, skip scaling")
    parser.add_argument("--gpu", action="store_true", help="Force use of GPU. If unavailable, error.")
    parser.add_argument("--debug", action="store_true", help="Print debug info on crop and resize steps.")
    return parser.parse_args()


def parse_min_size(ms):
    w,h = ms.lower().split('x')
    return int(w), int(h)


def load_image(path, debug=False):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if debug:
        h,w = img.shape[:2]
        print(f"Loaded image {path}: {w}x{h}")
    return img


def load_rois(path, debug=False):
    with open(path) as f:
        rois = json.load(f)
    if debug:
        print(f"Loaded {len(rois)} ROIs from {path}")
    return rois


def preprocess_crop(crop, scale, min_w, min_h, debug=False):
    h,w = crop.shape[:2]
    if debug:
        print(f"  Original crop: {w}x{h}")
    if scale!=1.0:
        nw,nh = int(w*scale), int(h*scale)
        if nw>=min_w and nh>=min_h:
            crop = cv2.resize(crop,(nw,nh),interpolation=cv2.INTER_LINEAR)
            if debug:
                print(f"  Scaled to: {nw}x{nh}")
        else:
            if debug:
                print(f"  Skipped scale: {nw}x{nh} < min {min_w}x{min_h}")
    return crop


def initialize_ocr(skip_detect, gpu, debug):
    # use mobile v3 models
    params = {
        'ocr_version':'PP-OCRv3',
        'text_detection_model_name':'PP-OCRv3_mobile_det',
        'text_recognition_model_name':'PP-OCRv3_mobile_rec',
        'use_angle_cls':False,
        'lang':'en'
    }
    if skip_detect:
        params['text_detection']=False
    if debug:
        print("Initializing PaddleOCR with params:",params)
    # PaddleOCR auto-detects GPU if paddlepaddle-gpu installed
    ocr = PaddleOCR(**params)
    return ocr


def run_ocr(ocr, crops, debug):
    results = []
    for idx,crop in enumerate(crops):
        if debug:
            print(f"OCR on crop {idx}, size={crop.shape[1]}x{crop.shape[0]}")
        res = ocr.ocr(crop, cls=False)
        # each res entry: [[bbox], (text,conf)]
        entries = []
        for line in res:
            bbox, (txt,conf) = line
            entries.append({'text':txt,'confidence':float(conf),'bbox':bbox})
        results.append(entries)
    return results


def main():
    args = parse_args()
    min_w,min_h = parse_min_size(args.min_size)
    debug=args.debug

    img = load_image(args.screenshot,debug)
    rois = load_rois(args.json_rois,debug)

    # prepare crops
    crops=[]
    for i,roi in enumerate(rois):
        x,y,w,h = roi['x'],roi['y'],roi['w'],roi['h']
        if debug:
            print(f"ROI {i}: pos=({x},{y}), size={w}x{h}")
        crop = img[y:y+h,x:x+w]
        crop = preprocess_crop(crop,args.scale,min_w,min_h,debug)
        crops.append(crop)

    # initialize OCR engine
    ocr = initialize_ocr(args.skip_detect,args.gpu,debug)
    # warm-up
    dummy = crops[0] if crops else None
    if dummy is not None:
        _ = ocr.ocr(dummy,cls=False)

    # profile
    if args.profile:
        if debug: print("Profiling OCR...")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],record_shapes=True) as prof:
            with record_function("paddle_ocr_pass"):
                _ = run_ocr(ocr,crops,debug)
        print(prof.key_averages().table(sort_by="self_cuda_time_total",row_limit=10))
        return

    # benchmark
    if args.benchmark:
        n=args.benchmark
        start=time.time()
        for _ in range(n):
            _ = run_ocr(ocr,crops,debug)
        total=time.time()-start
        fps=n/total
        print(f"Average FPS over {n} runs: {fps:.2f}")
        return

    # default
    res=run_ocr(ocr,crops,debug)
    print(json.dumps(res,indent=2))

if __name__=='__main__':
    main()
