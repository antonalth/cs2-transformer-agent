"""
window_capture.py

Capture only the up‐to‐date client area (no title bar or borders) of a window by its exact title,
and save it as a JPEG with specified quality.

Dependencies:
    pip install pywin32 Pillow

Usage:
    from window_capture import capture

    # Captures only the inside of the "Calculator" window, excluding title bar and borders
    capture("Calculator", "calc_client.jpg", 85)
"""

import os
import ctypes
import time
import win32con
import win32gui
import win32ui
from PIL import Image

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

# Flag for PrintWindow: request full-content rendering (Windows 8+)
PW_RENDERFULLCONTENT = 0x00000002


def _get_window_image(hwnd):
    """
    Internal helper that:
      - Calls PrintWindow to render the entire window (chrome + client) into a bitmap
      - Converts that bitmap into a PIL Image
      - Returns (full_img_PIL, win_left, win_top)
    """
    # Get full window rectangle
    win_left, win_top, win_right, win_bottom = win32gui.GetWindowRect(hwnd)
    full_w = win_right - win_left
    full_h = win_bottom - win_top

    # Create DCs and a full-window bitmap
    hdc_window = user32.GetWindowDC(hwnd)
    dc_full = gdi32.CreateCompatibleDC(hdc_window)
    bmp_full = gdi32.CreateCompatibleBitmap(hdc_window, full_w, full_h)
    obj_full = gdi32.SelectObject(dc_full, bmp_full)

    # Force the window to render into our full-window DC
    user32.PrintWindow(hwnd, dc_full, PW_RENDERFULLCONTENT)

    # Convert bmp_full to a PIL Image
    # Prepare a buffer for raw bits (full_w × full_h × 4 bytes)
    bytes_per_pixel = 4
    raw_size = full_w * full_h * bytes_per_pixel
    buffer = (ctypes.c_char * raw_size)()

    # Prepare BITMAPINFOHEADER for GetDIBits
    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", ctypes.c_uint32),
            ("biWidth", ctypes.c_int32),
            ("biHeight", ctypes.c_int32),
            ("biPlanes", ctypes.c_uint16),
            ("biBitCount", ctypes.c_uint16),
            ("biCompression", ctypes.c_uint32),
            ("biSizeImage", ctypes.c_uint32),
            ("biXPelsPerMeter", ctypes.c_int32),
            ("biYPelsPerMeter", ctypes.c_int32),
            ("biClrUsed", ctypes.c_uint32),
            ("biClrImportant", ctypes.c_uint32),
        ]

    bmi = BITMAPINFOHEADER()
    ctypes.memset(ctypes.byref(bmi), 0, ctypes.sizeof(bmi))
    bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.biWidth = full_w
    bmi.biHeight = -full_h  # negative so origin is top-left
    bmi.biPlanes = 1
    bmi.biBitCount = 32
    bmi.biCompression = win32con.BI_RGB
    bmi.biSizeImage = full_w * full_h * bytes_per_pixel

    # Retrieve raw bits from bmp_full
    gdi32.GetDIBits(
        dc_full,
        bmp_full,
        0,
        full_h,
        buffer,
        ctypes.byref(bmi),
        win32con.DIB_RGB_COLORS
    )

    # Build a PIL Image from the buffer (BGRX → RGB)
    full_img = Image.frombuffer(
        "RGB",
        (full_w, full_h),
        buffer,
        "raw",
        "BGRX",
        0,
        1
    )

    # Cleanup the DC and full-window bitmap (we'll delete bmp_full later)
    gdi32.SelectObject(dc_full, obj_full)
    gdi32.DeleteDC(dc_full)
    user32.ReleaseDC(hwnd, hdc_window)

    return full_img, win_left, win_top, bmp_full


def capture(window_name: str, output_path: str, quality: int):
    """
    Capture the client area of the window with the given exact title (excluding title bar/borders),
    forcing a fresh render, and save it as a JPEG.

    Args:
        window_name (str): Exact title of the window to capture.
        output_path (str): Path (including filename) where the JPEG will be saved.
                           Must end in .jpg or .jpeg.
        quality (int): JPEG quality (1–100).

    Raises:
        ValueError: If quality is out of range or output_path has invalid extension.
        RuntimeError: If no window with the given title is found or window is minimized.
    """
    # Validate quality
    try:
        quality = int(quality)
    except (TypeError, ValueError):
        raise ValueError("Quality must be an integer between 1 and 100.")
    if not (1 <= quality <= 100):
        raise ValueError("Quality must be between 1 and 100.")

    # Validate output extension
    ext = os.path.splitext(output_path)[1].lower()
    if ext not in {".jpg", ".jpeg"}:
        raise ValueError("Output path must end with .jpg or .jpeg")

    # Find the window handle by exact title
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd == 0:
        raise RuntimeError(f"No window found with title: '{window_name}'")

    # If minimized, cannot capture properly
    if user32.IsIconic(hwnd):
        raise RuntimeError("Window is minimized; restore it before capturing.")

    # Force a redraw to ensure up-to-date content
    win32gui.RedrawWindow(
        hwnd,
        None,
        None,
        win32con.RDW_INVALIDATE | win32con.RDW_UPDATENOW | win32con.RDW_ERASE
    )
    time.sleep(0.03)

    # Get the freshly rendered full-window image + its top-left coordinates
    full_img, win_left, win_top, bmp_full = _get_window_image(hwnd)

    # Get client-area rectangle (in client-relative coords)
    cli_left, cli_top, cli_right, cli_bottom = win32gui.GetClientRect(hwnd)
    client_w = cli_right - cli_left
    client_h = cli_bottom - cli_top

    # Compute client-area top-left in screen coordinates
    client_x_screen, client_y_screen = win32gui.ClientToScreen(hwnd, (0, 0))

    # Crop out title bar and borders by using the offset relative to the full window
    offset_x = client_x_screen - win_left
    offset_y = client_y_screen - win_top
    crop_box = (offset_x, offset_y, offset_x + client_w, offset_y + client_h)
    client_img = full_img.crop(crop_box)

    # Save the cropped client area as JPEG
    client_img.save(output_path, format="JPEG", quality=quality)

    # Cleanup the full-window bitmap
    gdi32.DeleteObject(bmp_full)

    #print(f"Saved client area of '{window_name}' → {output_path} (quality={quality})")

#import timeit
#start = timeit.default_timer()
#capture("Counter-Strike 2", "test3.jpg",100)
#stop = timeit.default_timer()
#print('Time: ', stop - start)  