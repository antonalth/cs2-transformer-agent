import argparse
import json
import time
from pathlib import Path

from PIL import Image
import numpy as np
from paddleocr import PaddleOCR  # Legacy 2.x API

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark PaddleOCR 2.x on pre-cropped regions with batching"
    )
    parser.add_argument(
        "-i", "--image", type=Path, required=True,
        help="Path to input JPEG"
    )
    parser.add_argument(
        "-b", "--boxes_json", type=Path, required=True,
        help="Path to JSON list of {x,y,w,h}"
    )
    parser.add_argument(
        "-n", "--benchmark", type=int, default=1,
        help="Iterations to run over all regions"
    )
    parser.add_argument(
        "--batch_size", type=int, default=5,
        help="Number of regions to process in one batch"
    )
    return parser.parse_args()

def load_boxes(json_path: Path):
    """Load list of dicts and convert to [x1, y1, x2, y2]."""
    with open(json_path, 'r', encoding='utf-8') as f:
        rects = json.load(f)
    return [
        [r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"]]
        for r in rects
    ]

def crop_to_numpy(img: Image.Image, box):
    """Crop a region and convert to NumPy array."""
    return np.array(img.crop(tuple(box)))

def chunk_list(lst, size):
    """Yield successive chunks of given size."""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def benchmark_batches(ocr, np_regions, batch_size, iterations):
    """Benchmark OCR on batches of regions; return avg total time per iteration."""
    batches = list(chunk_list(np_regions, batch_size))
    # Warm-up: recognition only on first batch
    _ = ocr.ocr(batches[0], det=False, rec=True, cls=False)  # :contentReference[oaicite:3]{index=3}

    total_time = 0.0
    for _ in range(iterations):
        for batch in batches:
            t0 = time.time()
            # Batch recognition; detection disabled
            _ = ocr.ocr(batch, det=False, rec=True, cls=False)  # :contentReference[oaicite:4]{index=4}
            total_time += (time.time() - t0)

    # Average total time for all regions per iteration
    return total_time / iterations

def main():
    args = parse_args()
    image = Image.open(args.image).convert("RGB")
    boxes = load_boxes(args.boxes_json)

    # Initialize PaddleOCR 2.x with high-performance inference flags
    ocr = PaddleOCR(
        use_gpu=True,                   # GPU inference :contentReference[oaicite:5]{index=5}
        ir_optim=True,                  # IR graph optimizations :contentReference[oaicite:6]{index=6}
        use_tensorrt=True,              # TensorRT subgraph engine :contentReference[oaicite:7]{index=7}
        precision='fp16',               # FP16 precision for TensorRT :contentReference[oaicite:8]{index=8}
        enable_mkldnn=False,            # Disable MKL-DNN when using TRT :contentReference[oaicite:9]{index=9}
        cpu_threads=4,                  # CPU threads if fallback :contentReference[oaicite:10]{index=10}
        rec_batch_num=args.batch_size,  # Recognition batch size :contentReference[oaicite:11]{index=11}
        rec_image_shape='3,48,160',     # Crop size for recognition :contentReference[oaicite:12]{index=12}
        det_db_score_mode='fast',       # Fast detection post-processing :contentReference[oaicite:13]{index=13}
        det_db_thresh=0.3,              # Detection pixel threshold :contentReference[oaicite:14]{index=14}
        det_db_box_thresh=0.5,          # Detection box score threshold :contentReference[oaicite:15]{index=15}
        det_db_unclip_ratio=1.1,        # DB unclip expansion :contentReference[oaicite:16]{index=16}
        max_batch_size=args.batch_size, # Detection batch size (unused here) :contentReference[oaicite:17]{index=17}
    )

    # Pre-crop all ROIs
    np_regions = [crop_to_numpy(image, box) for box in boxes]

    # Run batched benchmark
    avg_total_s = benchmark_batches(
        ocr,
        np_regions,
        batch_size=args.batch_size,
        iterations=args.benchmark
    )

    print(
        f"Avg total latency for {len(boxes)} regions: "
        f"{avg_total_s * 1000:.2f} ms over {args.benchmark} iterations"
    )

if __name__ == "__main__":
    main()
