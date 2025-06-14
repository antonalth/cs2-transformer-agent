import argparse
import json
import time
from pathlib import Path

from PIL import Image
import numpy as np
from paddleocr import PaddleOCR  # Pipeline API


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark PaddleOCR on image regions with batching"
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
    region = img.crop(tuple(box))
    return np.array(region)


def chunk_list(lst, size):
    """Yield successive chunks of given size."""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def benchmark_batches(ocr, np_regions, batch_size, iterations):
    """Benchmark OCR on batches of regions and return avg sum-time per iteration."""
    batches = list(chunk_list(np_regions, batch_size))
    # Warm-up on first batch
    _ = ocr.predict(batches[0])

    total_time = 0.0
    for _ in range(iterations):
        for batch in batches:
            t0 = time.time()
            _ = ocr.predict(batch)
            total_time += (time.time() - t0)

    # average total time per iteration (summing all regions)
    avg_batch_time = total_time / iterations
    return avg_batch_time


def main():
    args = parse_args()
    image = Image.open(args.image).convert("RGB")
    boxes = load_boxes(args.boxes_json)

    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        device="gpu:0",
        precision="fp16",
        enable_mkldnn=False,
                # ─── Optional Extra Speed Tweaks ───────────────────────────
        # (keep these tuned as before)
        text_det_limit_side_len=128,
        text_det_limit_type="max",
        text_det_thresh=0.3,
        text_det_box_thresh=0.5,
        text_det_unclip_ratio=1.1,
        text_recognition_batch_size=8,
        text_rec_input_shape=(3, 48, 160),
        text_rec_score_thresh=0.0,

        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    # Pre-crop all ROIs once
    np_regions = [crop_to_numpy(image, box) for box in boxes]

    # Benchmark with batching
    avg_total_s = benchmark_batches(
        ocr,
        np_regions,
        batch_size=args.batch_size,
        iterations=args.benchmark
    )

    avg_total_ms = avg_total_s * 1000
    print(
        f"Avg total latency for {len(boxes)} regions: "
        f"{avg_total_ms:.2f} ms over {args.benchmark} iterations"
    )


if __name__ == "__main__":
    main()
