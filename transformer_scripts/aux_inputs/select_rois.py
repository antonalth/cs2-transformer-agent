import cv2
import json
import sys
import os

# --- Check for input image path ---
if len(sys.argv) < 2:
    print("Usage: python select_rois_to_json.py <image.jpg>")
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.isfile(image_path):
    print(f"File not found: {image_path}")
    sys.exit(1)

image = cv2.imread(image_path)
if image is None:
    print("Could not load image.")
    sys.exit(1)

# --- Select ROIs interactively ---
print("Select multiple ROIs. Press ENTER/SPACE to confirm each. ESC when done.")

rois = []
while True:
    roi = cv2.selectROI("Select ROI - ESC to finish", image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI - ESC to finish")
    if sum(roi) == 0:
        break
    x, y, w, h = roi
    rois.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

if not rois:
    print("No regions selected.")
    sys.exit(0)

# --- Save to JSON ---
json_filename = os.path.splitext(os.path.basename(image_path))[0] + "_rois.json"
with open(json_filename, "w") as f:
    json.dump(rois, f, indent=2)

print(f"Saved {len(rois)} ROIs to {json_filename}")
