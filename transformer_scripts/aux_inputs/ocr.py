import cv2
from paddleocr import PaddleOCR
import time
import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="OCR on selected ROIs or entire image with timing using PP-OCRv2")
    parser.add_argument('image_path', type=str, help="Path to the input image (e.g., screenshot.jpeg)")
    parser.add_argument('--full', '-f', action='store_true',
                        help="Perform OCR on the entire image instead of selecting ROIs")
    args = parser.parse_args()

    # Initialize the PP-OCRv2 reader (recognition only)
    reader = PaddleOCR(use_angle_cls=False, lang='en', det=False, rec=True, use_gpu=False)

    # Load the test image
    img = cv2.imread(args.image_path)
    if img is None:
        raise RuntimeError(f"Failed to load image '{args.image_path}'")

    # Determine crops: entire image or user-selected ROIs
    if args.full:
        crops = [img]
        bboxes = [(0, 0, img.shape[1], img.shape[0])]
        print("Performing OCR on the entire image.")
    else:
        bboxes = cv2.selectROIs("Select regions for OCR", img, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select regions for OCR")
        crops = [img[y:y+h, x:x+w] for (x, y, w, h) in bboxes]
        print(f"Number of ROIs selected: {len(crops)}")

    # Warm-up run
    for crop in crops:
        _ = reader.ocr(crop, cls=False, det=False)

    # Time 100 iterations over all crops
    iterations = 100
    start_time = time.time()
    for _ in range(iterations):
        for crop in crops:
            _ = reader.ocr(crop, cls=False, det=False)
    total_time = time.time() - start_time

    avg_time = total_time / iterations
    fps = 1.0 / avg_time

    print(f"\nTiming over {iterations} iterations:")
    print(f"Total time: {total_time:.3f} s")
    print(f"Average time per iteration: {avg_time:.4f} s")
    print(f"Equivalent FPS: {fps:.1f} FPS\n")

    # Final detailed OCR pass and display results
    results = []
    for box, crop in zip(bboxes, crops):
        ocr_res = reader.ocr(crop, cls=False, det=False)
        # ocr_res format: [[(x1,y1), (x2,y2), ...], (text, confidence)]
        text = " ".join([line[1][0] for line in ocr_res])
        conf = sum([line[1][1] for line in ocr_res]) / max(len(ocr_res), 1)
        results.append({'box': box, 'text': text, 'conf': conf})

    for r in results:
        x, y, w, h = r['box']
        print(f"ROI {(x, y, w, h)} → \"{r['text']}\"  (avg conf {r['conf']:.2f})")
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, r['text'], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("OCR Results", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
