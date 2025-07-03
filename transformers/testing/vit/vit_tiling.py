import cv2
import numpy as np
import argparse

# --- Image Processing Functions ---

def letterbox_resize(frame: np.ndarray, target_size: int) -> np.ndarray:
    """Resizes an image to a square, preserving aspect ratio by padding."""
    original_h, original_w = frame.shape[:2]
    scale = target_size / max(original_h, original_w)
    new_w, new_h = int(original_w * scale), int(original_h * scale)
    resized_img = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_size, target_size, 3), 0, dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_img
    return canvas

def create_labeled_tile(tile: np.ndarray, label: str) -> np.ndarray:
    """Creates a combined image with a tile and a text label on the side."""
    h, w = tile.shape[:2]
    label_canvas = np.full((h, 150, 3), 255, dtype=np.uint8)
    # Adjust font size based on tile height for better scaling
    font_scale = h / 224 * 0.6
    cv2.putText(label_canvas, label, (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
    return cv2.hconcat([tile, label_canvas])

# --- Main Visualization Logic ---

def main():
    parser = argparse.ArgumentParser(description="Visualize different ViT input strategies.")
    parser.add_argument('--image', type=str, default="cs_screenshot.png", help="Path to the input image.")
    # --- NEW ARGUMENT FOR RESOLUTION ---
    parser.add_argument('--res384', action='store_true', help="Use 384x384 tile size instead of 224x224.")
    args = parser.parse_args()

    # --- DYNAMIC TILE SIZE ---
    TILE_SIZE = 384 if args.res384 else 224
    print(f"--- Visualizing with {TILE_SIZE}x{TILE_SIZE} tiles ---")

    try:
        source_img = cv2.imread(args.image)
        if source_img is None: raise FileNotFoundError
    except (FileNotFoundError, cv2.error):
        print(f"Error: Could not load image at '{args.image}'. Creating a dummy image.")
        source_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.putText(source_img, "Dummy Image - Place your screenshot here", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
    H, W = source_img.shape[:2]
    
    # Check if tile size is larger than image dimensions
    if TILE_SIZE > H or TILE_SIZE > W:
        print(f"\nWarning: Tile size ({TILE_SIZE}px) is larger than the source image ({W}x{H}px).")
        print("Bounding boxes may extend beyond the image borders.")
        
    # --- 1. Foveated Vision (Center + Letterbox) Visualization ---
    vis_foveated = source_img.copy()
    center_x, center_y = W // 2, H // 2
    x1_c, y1_c = center_x - TILE_SIZE // 2, center_y - TILE_SIZE // 2
    x2_c, y2_c = center_x + TILE_SIZE // 2, center_y + TILE_SIZE // 2
    
    context_tile = letterbox_resize(source_img, TILE_SIZE)
    focus_tile = source_img[max(0, y1_c):y2_c, max(0, x1_c):x2_c]
    # Handle cases where the crop is smaller than the tile size
    if focus_tile.shape[0] != TILE_SIZE or focus_tile.shape[1] != TILE_SIZE:
        focus_tile = letterbox_resize(focus_tile, TILE_SIZE)

    cv2.rectangle(vis_foveated, (0, 0), (W-1, H-1), (0, 255, 255), 2)
    cv2.rectangle(vis_foveated, (x1_c, y1_c), (x2_c, y2_c), (0, 255, 0), 2)
    
    labeled_context = create_labeled_tile(context_tile, "Global Context")
    labeled_focus = create_labeled_tile(focus_tile, f"Center {TILE_SIZE}x{TILE_SIZE}")
    foveated_tiles_display = cv2.vconcat([labeled_context, labeled_focus])
    h_tiles = foveated_tiles_display.shape[0]
    vis_foveated_resized = cv2.resize(vis_foveated, (int(W * h_tiles / H), h_tiles))
    final_foveated_display = cv2.hconcat([vis_foveated_resized, foveated_tiles_display])
    
    # --- 2. Foveated Tiling (Overlapping + Global) Visualization ---
    vis_tiling = source_img.copy()
    regions = {
        "Global Context": ((0, 0, W, H), (0, 255, 255)),
        f"Center {TILE_SIZE}x{TILE_SIZE}": ((x1_c, y1_c, TILE_SIZE, TILE_SIZE), (0, 255, 0)),
        "Top-Left": ((0, 0, TILE_SIZE, TILE_SIZE), (255, 0, 0)),
        "Top-Right": ((W - TILE_SIZE, 0, TILE_SIZE, TILE_SIZE), (255, 0, 0)),
        "Bottom-Left": ((0, H - TILE_SIZE, TILE_SIZE, TILE_SIZE), (255, 0, 0)),
        "Bottom-Right": ((W - TILE_SIZE, H - TILE_SIZE, TILE_SIZE, TILE_SIZE), (255, 0, 0))
    }
    
    tiling_tiles = []
    for label, (rect, color) in regions.items():
        x, y, w, h = rect
        cv2.rectangle(vis_tiling, (x, y), (x + w -1, y + h -1), color, 2)
        if label == "Global Context":
            tile = letterbox_resize(source_img, TILE_SIZE)
        else:
            tile = source_img[max(0, y):y+h, max(0, x):x+w]
            if tile.shape[0] != TILE_SIZE or tile.shape[1] != TILE_SIZE:
                tile = letterbox_resize(tile, TILE_SIZE)
        tiling_tiles.append(create_labeled_tile(tile, label))

    col1 = cv2.vconcat(tiling_tiles[0:3])
    col2 = cv2.vconcat(tiling_tiles[3:6])
    tiling_tiles_display = cv2.hconcat([col1, col2])
    h_tiles_2 = tiling_tiles_display.shape[0]
    vis_tiling_resized = cv2.resize(vis_tiling, (int(W * h_tiles_2 / H), h_tiles_2))
    final_tiling_display = cv2.hconcat([vis_tiling_resized, tiling_tiles_display])
    
    # --- Display Results ---
    cv2.imshow(f"1. Foveated Vision ({TILE_SIZE}px)", final_foveated_display)
    cv2.imshow(f"2. Foveated Tiling ({TILE_SIZE}px)", final_tiling_display)
    
    print("\nVisualizations displayed. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()