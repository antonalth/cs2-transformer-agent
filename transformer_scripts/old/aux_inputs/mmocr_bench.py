import argparse
import json
import time
import os
import cv2
import torch
import numpy as np

# Suppress a specific warning from MMOCR's dependency
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='mmengine.utils.manager')

try:
    # UPDATED IMPORT
    from mmocr.apis import MMOCRInferencer
except ImportError:
    print("Error: MMOCR (or MMOCRInferencer) is not installed or not found.")
    print("Please ensure MMOCR is installed in your active Python environment.")
    print("Installation instructions (example):")
    print("1. conda activate your_env_name  OR  python -m venv myenv; source myenv/bin/activate")
    print("2. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 (adjust for your CUDA/CPU)")
    print("3. pip install openmim")
    print("4. mim install mmcv mmengine")
    print("5. pip install \"mmocr>=1.0.0\"")
    exit(1)


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run MMOCR (using MMOCRInferencer) on specified regions of interest (ROIs) in an image.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'image_path',
        type=str,
        help="Path to the input image file."
    )
    parser.add_argument(
        'roi_path',
        type=str,
        help="Path to the JSON file defining the ROIs."
    )
    parser.add_argument(
        '--benchmark',
        type=int,
        metavar='N',
        help="Run a benchmark for N iterations and report the average FPS."
    )
    parser.add_argument(
        '--output',
        type=str,
        metavar='FILE_PATH',
        help="Optional. Path to save the output image with ROIs and results visualized."
    )
    return parser.parse_args()

def process_rois(ocr_inferencer, image, rois):
    """
    Crops the image based on ROIs and runs OCR on each crop using MMOCRInferencer.

    Args:
        ocr_inferencer (MMOCRInferencer): The initialized MMOCRInferencer model.
        image (np.ndarray): The full input image in OpenCV format (BGR).
        rois (list): A list of ROI dictionaries, each with 'x', 'y', 'w', 'h'.

    Returns:
        list: A list of dictionaries, where each contains the original ROI,
              the recognized texts, and the raw MMOCRInferencer prediction result for that crop.
    """
    results = []
    img_h, img_w = image.shape[:2]

    for roi in rois:
        x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']

        # Clamp ROI coordinates to be within image boundaries
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        if x2 <= x1 or y2 <= y1:
            results.append({'roi': roi, 'texts': [], 'raw_result': {}})
            continue

        img_crop = image[y1:y2, x1:x2]

        if img_crop.size == 0: # Handle cases where crop is empty
            results.append({'roi': roi, 'texts': [], 'raw_result': {}})
            continue

        # Run inference on the cropped image using MMOCRInferencer
        # The inferencer call itself handles batching if a list of images is passed,
        # but here we pass one crop at a time.
        # `show=False` and `print_result=False` are often part of a visualize or postprocess step,
        # or parameters to __call__. For MMOCRInferencer, we pass the image directly.
        # Default behavior is usually not to show or print unless specified.
        # The result is a dictionary, typically with 'predictions' and 'visualization' keys.
        try:
            # For MMOCRInferencer, the `out_rec_fields` and `out_det_fields` can be used during
            # initialization to control what's in the prediction output if needed, but defaults are usually good.
            # We are interested in the 'predictions' part.
            # The __call__ method of MMOCRInferencer takes inputs and returns a dict.
            # `save_vis=False`, `save_pred=False` are default in postprocess call
            # Let's ensure no pop-up windows during benchmark/processing.
            # MMOCRInferencer's __call__ has `show` which defaults to False.
            # It also has `print_result` which defaults to False.
            inferencer_output = ocr_inferencer(img_crop, show=False, print_result=False, save_vis=False, save_pred=False)

            # The output structure for a single image input:
            # inferencer_output = {'predictions': [pred_dict_for_img_crop], 'visualization': [vis_array_or_None]}
            # pred_dict_for_img_crop = {'rec_texts': [...], 'rec_scores': [...], 'det_polygons': [...], ...}
            
            if inferencer_output and 'predictions' in inferencer_output and \
               len(inferencer_output['predictions']) > 0:
                prediction_for_crop = inferencer_output['predictions'][0]
                recognized_texts = prediction_for_crop.get('rec_texts', [])
                raw_pred_data = prediction_for_crop
            else:
                recognized_texts = []
                raw_pred_data = {}
                print(f"Warning: MMOCRInferencer returned unexpected output for ROI {roi}: {inferencer_output}")

        except Exception as e:
            print(f"Error during MMOCRInferencer processing for ROI {roi}: {e}")
            recognized_texts = []
            raw_pred_data = {}

        results.append({
            'roi': roi,
            'texts': recognized_texts,
            'raw_result': raw_pred_data # Store the relevant part of the prediction
        })

    return results

def visualize_results(image, results, output_path):
    """
    Draws ROIs and recognized text onto the image and saves it.
    (This function remains largely the same as it operates on our processed results list)
    """
    vis_image = image.copy()

    for res in results:
        roi = res['roi']
        texts = res['texts']

        x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        full_text = " ".join(texts) if texts else "" # Ensure full_text is a string

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6 # Adjusted for potentially smaller ROIs / fitting text
        font_thickness = 1
        text_color = (255, 255, 255)
        bg_color = (0, 128, 0)

        (text_width, text_height), baseline = cv2.getTextSize(full_text, font, font_scale, font_thickness)
        
        # Position text above ROI, or below if no space above
        text_x_pos = x
        text_y_pos = y - 10
        if text_y_pos - text_height < 0 : # If text goes off top of image
             text_y_pos = y + h + text_height + 5

        # Background rectangle for text
        cv2.rectangle(vis_image,
                      (text_x_pos, text_y_pos - text_height - baseline//2),
                      (text_x_pos + text_width, text_y_pos + baseline//2),
                      bg_color, -1)
        cv2.putText(vis_image, full_text, (text_x_pos, text_y_pos), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    cv2.imwrite(output_path, vis_image)
    print(f"\nVisualized output saved to: {output_path}")


def main():
    args = parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image path not found at '{args.image_path}'")
        return
    if not os.path.exists(args.roi_path):
        print(f"Error: ROI JSON path not found at '{args.roi_path}'")
        return

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Initializing MMOCRInferencer... (This may take a moment)")
    try:
        # UPDATED MODEL INITIALIZATION
        # Specify device during initialization
        ocr = MMOCRInferencer(det='DBNet', rec='SAR', device=device)
    except Exception as e:
        print(f"Error initializing MMOCRInferencer: {e}")
        print("Please ensure your MMOCR installation and dependencies (PyTorch, MMCV, MMEngine) are correct for the selected device.")
        return
    print("MMOCRInferencer loaded successfully.")

    image_orig = cv2.imread(args.image_path)
    if image_orig is None:
        print(f"Error: Could not read image from '{args.image_path}'")
        return

    with open(args.roi_path, 'r') as f:
        rois = json.load(f)

    if args.benchmark:
        num_runs = args.benchmark
        print(f"\n--- Starting Benchmark: {num_runs} iterations ---")

        print("Performing one warm-up run...")
        _ = process_rois(ocr, image_orig, rois) # Pass the inferencer instance
        print("Warm-up complete.")

        start_time = time.perf_counter()
        for i in range(num_runs):
            process_rois(ocr, image_orig, rois) # Pass the inferencer instance
            print(f"\rIteration {i + 1}/{num_runs}", end="")
        end_time = time.perf_counter()
        print("\nBenchmark finished.")

        total_time = end_time - start_time
        avg_time_per_run = total_time / num_runs if num_runs > 0 else 0
        avg_fps = 1.0 / avg_time_per_run if avg_time_per_run > 0 else float('inf')

        print("\n--- Benchmark Results ---")
        print(f"Total iterations: {num_runs}")
        print(f"Total time:       {total_time:.4f} seconds")
        print(f"Average time/run: {avg_time_per_run:.4f} seconds")
        print(f"Average FPS:      {avg_fps:.2f}")

    else:
        print("\n--- Running OCR on ROIs using MMOCRInferencer ---")
        # Ensure image_orig is used for processing and visualization
        results = process_rois(ocr, image_orig, rois) # Pass the inferencer instance

        print("\n--- OCR Results ---")
        for i, res in enumerate(results):
            roi_str = f"({res['roi']['x']}, {res['roi']['y']}, {res['roi']['w']}, {res['roi']['h']})"
            text_str = " ".join(res['texts']) if res['texts'] else "[NO TEXT DETECTED]"
            print(f"ROI #{i+1} {roi_str}: '{text_str}'")

        if args.output:
            visualize_results(image_orig, results, args.output)

if __name__ == "__main__":
    main()