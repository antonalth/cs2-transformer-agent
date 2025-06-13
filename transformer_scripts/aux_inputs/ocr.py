import cv2
import easyocr
import sys
import time

# --- Handle CLI argument ---
if len(sys.argv) < 2:
    print("Usage: python ocr_benchmark.py <image.jpg>")
    sys.exit(1)

image_path = sys.argv[1]
image = cv2.imread(image_path)

# --- Region selection ---
print("Select one or more regions. Press ENTER or SPACE after each. ESC when done.")
regions = []
while True:
    roi = cv2.selectROI("Select Region (ESC to finish)", image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Region (ESC to finish)")
    if sum(roi) == 0:
        break
    regions.append(roi)

if not regions:
    print("No regions selected.")
    sys.exit(1)

# --- Setup OCR ---
reader = easyocr.Reader(['en'], gpu=True)

# --- Run Inference Loop ---
print(f"Running OCR 100 times on {len(regions)} regions...")
start = time.time()

for _ in range(100):
    for (x, y, w, h) in regions:
        roi = image[y:y+h, x:x+w]
        _ = reader.readtext(roi, detail=0, paragraph=False)

end = time.time()

# --- Results ---
total_regions = 100 * len(regions)
fps = total_regions / (end - start)

print(f"OCR Inference on {total_regions} regions took {end - start:.2f} seconds")
print(f"Approx. FPS: {fps:.2f}")
