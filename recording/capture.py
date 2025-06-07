#!/usr/bin/env python3
"""
capture.py: Capture a specific window's monitor area using GStreamer on Windows
with hardware acceleration, robust linking and element checks.
"""

import argparse
import sys
import ctypes
from ctypes import wintypes
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst

# --- Win32 API setup via ctypes ---
user32 = ctypes.WinDLL('user32', use_last_error=True)
WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
user32.EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.MonitorFromWindow.argtypes = [wintypes.HWND, ctypes.c_uint]
MONITOR_DEFAULTTONEAREST = 2


def find_window_by_name(name):
    target = name.lower()
    found = {'hwnd': None}

    @WNDENUMPROC
    def enum_proc(hwnd, lParam):
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                if target in buf.value.lower():
                    found['hwnd'] = hwnd
                    return False
        return True

    user32.EnumWindows(enum_proc, 0)
    if not found['hwnd']:
        raise RuntimeError(f"No window found matching: '{name}'")
    return found['hwnd']


def choose_src_element():
    for elem in ('d3d11screencapturesrc', 'dxgiscreencapsrc'):
        if Gst.ElementFactory.find(elem):
            return elem
    raise RuntimeError("No supported source found. Install appropriate GStreamer plugins.")


def make_element(factory_name, name=None):
    elem = Gst.ElementFactory.make(factory_name, name or factory_name)
    if not elem:
        raise RuntimeError(f"Element '{factory_name}' not found. Ensure its plugin is installed.")
    return elem


def build_pipeline(hwnd, fps, output):
    Gst.init(None)
    pipeline = Gst.Pipeline.new('capture')
    if not pipeline:
        raise RuntimeError("Failed creating pipeline")

    # Source
    src_name = choose_src_element()
    src = make_element(src_name, 'src')
    props = {p.name for p in src.list_properties()}

    # Set handle property
    if 'window-handle' in props:
        src.set_property('window-handle', ctypes.c_uint64(hwnd).value)
    else:
        if 'monitor-handle' in props:
            mon = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
            src.set_property('monitor-handle', ctypes.c_uint64(mon).value)
        elif 'monitor-index' in props:
            src.set_property('monitor-index', 0)
        else:
            raise RuntimeError(f"No window/monitor handle props on {src_name}: {props}")
    if src_name == 'd3d11screencapturesrc':
        for prop, val in (('capture-api', 0), ('window-capture-mode', 1)):
            if prop in props:
                src.set_property(prop, val)

    # Build other elements
    queue1 = make_element('queue', 'q1')
    videorate = make_element('videorate', 'rate')
    capsfilter = make_element('capsfilter', 'caps')
    caps = Gst.Caps.from_string(f"video/x-raw,format=BGRA,framerate={fps}/1")
    capsfilter.set_property('caps', caps)
    videoconvert = make_element('videoconvert', 'conv')
    queue2 = make_element('queue', 'q2')
    encoder = make_element('nvh264enc', 'enc')
    queue3 = make_element('queue', 'q3')  # buffer encoder->parser
    parser = make_element('h264parse', 'parse')
    queue4 = make_element('queue', 'q4')  # buffer parser->mux
    mux = make_element('mp4mux', 'mux')
    sink = make_element('filesink', 'sink')
    sink.set_property('location', output)

    # Add elements
    for elem in (src, queue1, videorate, capsfilter, videoconvert, queue2,
                 encoder, queue3, parser, queue4, mux, sink):
        pipeline.add(elem)

    # Link sequence with explicit error points
    if not Gst.Element.link(src, queue1):
        raise RuntimeError('Failed to link src->queue1')
    if not Gst.Element.link(queue1, videorate):
        raise RuntimeError('Failed to link queue1->videorate')
    if not Gst.Element.link(videorate, capsfilter):
        raise RuntimeError('Failed to link videorate->capsfilter')
    if not Gst.Element.link(capsfilter, videoconvert):
        raise RuntimeError('Failed to link capsfilter->videoconvert')
    if not Gst.Element.link(videoconvert, queue2):
        raise RuntimeError('Failed to link videoconvert->queue2')
    if not Gst.Element.link(queue2, encoder):
        raise RuntimeError('Failed to link queue2->encoder')
    if not Gst.Element.link(encoder, queue3):
        raise RuntimeError('Failed to link encoder->queue3')
    if not Gst.Element.link(queue3, parser):
        raise RuntimeError('Failed to link queue3->parser')
    if not Gst.Element.link(parser, queue4):
        raise RuntimeError('Failed to link parser->queue4')
    if not Gst.Element.link(queue4, mux):
        raise RuntimeError('Failed to link queue4->mux')
    if not Gst.Element.link(mux, sink):
        raise RuntimeError('Failed to link mux->sink')

    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Capture a window or its monitor via GStreamer")
    parser.add_argument('-w', '--window', required=True, help='Window title substring')
    parser.add_argument('-f', '--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('-o', '--output', required=True, help='Output file path')
    args = parser.parse_args()

    try:
        hwnd = find_window_by_name(args.window)
        print(f"Found HWND: {hwnd}", file=sys.stderr)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    try:
        pipeline = build_pipeline(hwnd, args.fps, args.output)
    except Exception as e:
        print(f"Pipeline build error: {e}", file=sys.stderr)
        sys.exit(1)

    pipeline.set_state(Gst.State.PLAYING)
    print("Capture started", file=sys.stderr)

    bus = pipeline.get_bus()
    while True:
        msg = bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS)
        if msg:
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                print(f"Error: {err} {debug}", file=sys.stderr)
            else:
                print("Capture finished", file=sys.stderr)
            break

    pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    main()
