#!/usr/bin/env python3
import argparse
import json
import queue
import select
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from evdev import ecodes  # noqa: E402
from snegg import ei  # noqa: E402


KEY_NAME_TO_CODE = {
    "space": ecodes.KEY_SPACE,
    "enter": ecodes.KEY_ENTER,
    "return": ecodes.KEY_ENTER,
    "tab": ecodes.KEY_TAB,
    "esc": ecodes.KEY_ESC,
    "escape": ecodes.KEY_ESC,
    "backspace": ecodes.KEY_BACKSPACE,
    "left": ecodes.KEY_LEFT,
    "right": ecodes.KEY_RIGHT,
    "up": ecodes.KEY_UP,
    "down": ecodes.KEY_DOWN,
    "lshift": ecodes.KEY_LEFTSHIFT,
    "rshift": ecodes.KEY_RIGHTSHIFT,
    "lctrl": ecodes.KEY_LEFTCTRL,
    "rctrl": ecodes.KEY_RIGHTCTRL,
    "lalt": ecodes.KEY_LEFTALT,
    "ralt": ecodes.KEY_RIGHTALT,
    "minus": ecodes.KEY_MINUS,
    "equals": ecodes.KEY_EQUAL,
    "comma": ecodes.KEY_COMMA,
    "period": ecodes.KEY_DOT,
    "slash": ecodes.KEY_SLASH,
    "semicolon": ecodes.KEY_SEMICOLON,
    "apostrophe": ecodes.KEY_APOSTROPHE,
    "leftbracket": ecodes.KEY_LEFTBRACE,
    "rightbracket": ecodes.KEY_RIGHTBRACE,
    "backslash": ecodes.KEY_BACKSLASH,
    "grave": ecodes.KEY_GRAVE,
    "capslock": ecodes.KEY_CAPSLOCK,
    "f1": ecodes.KEY_F1,
    "f2": ecodes.KEY_F2,
    "f3": ecodes.KEY_F3,
    "f4": ecodes.KEY_F4,
    "f5": ecodes.KEY_F5,
    "f6": ecodes.KEY_F6,
    "f7": ecodes.KEY_F7,
    "f8": ecodes.KEY_F8,
    "f9": ecodes.KEY_F9,
    "f10": ecodes.KEY_F10,
    "f11": ecodes.KEY_F11,
    "f12": ecodes.KEY_F12,
}

for c in "abcdefghijklmnopqrstuvwxyz":
    KEY_NAME_TO_CODE[c] = getattr(ecodes, f"KEY_{c.upper()}")

for d in "0123456789":
    KEY_NAME_TO_CODE[d] = getattr(ecodes, f"KEY_{d}")

MOUSE_BUTTONS = {
    1: ecodes.BTN_LEFT,
    2: ecodes.BTN_MIDDLE,
    3: ecodes.BTN_RIGHT,
    4: ecodes.BTN_SIDE,
    5: ecodes.BTN_EXTRA,
}


class LibeiInjector:
    def __init__(self, eis_socket: Optional[str], width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.ctx = ei.Sender.create_for_socket(
            Path(eis_socket) if eis_socket else None,
            name="gamescope-remote-input",
        )
        self.pointer_abs = None
        self.pointer_rel = None
        self.keyboard = None
        self.last_abs = (0.5, 0.5)
        self._bootstrap()

    def _drain(self, timeout: float = 0.1) -> None:
        rlist, _, _ = select.select([self.ctx.fd], [], [], timeout)
        if not rlist:
            return
        self.ctx.dispatch()
        events = list(self.ctx.events)
        for ev in events:
            et = ev.event_type
            if et == ei.EventType.SEAT_ADDED:
                ev.seat.bind((
                    ei.DeviceCapability.POINTER,
                    ei.DeviceCapability.POINTER_ABSOLUTE,
                    ei.DeviceCapability.BUTTON,
                    ei.DeviceCapability.KEYBOARD,
                ))
            elif et == ei.EventType.DEVICE_ADDED:
                dev = ev.device
                caps = set(dev.capabilities)
                if ei.DeviceCapability.KEYBOARD in caps and self.keyboard is None:
                    self.keyboard = dev
                    self.keyboard.start_emulating()
                if ei.DeviceCapability.POINTER_ABSOLUTE in caps and self.pointer_abs is None:
                    self.pointer_abs = dev
                    self.pointer_abs.start_emulating()
                elif ei.DeviceCapability.POINTER in caps and self.pointer_rel is None:
                    self.pointer_rel = dev
                    self.pointer_rel.start_emulating()
            elif et == ei.EventType.DEVICE_RESUMED:
                if ev.device is not None:
                    ev.device.start_emulating()

    def _bootstrap(self) -> None:
        deadline = time.time() + 5.0
        while time.time() < deadline:
            self._drain(0.2)
            if self.keyboard and (self.pointer_abs or self.pointer_rel):
                return
        raise RuntimeError("Timed out waiting for libei keyboard/pointer devices")

    def send_key(self, key_name: str, is_down: bool) -> None:
        code = KEY_NAME_TO_CODE.get(key_name.lower())
        if code is None or self.keyboard is None:
            return
        self._drain(0.0)
        self.keyboard.keyboard_key(code, is_down).frame()

    def send_button(self, button: int, is_down: bool) -> None:
        code = MOUSE_BUTTONS.get(button)
        pointer = self.pointer_abs or self.pointer_rel
        if code is None or pointer is None:
            return
        self._drain(0.0)
        pointer.button_button(code, is_down).frame()
    def send_mouse_abs(self, nx: float, ny: float) -> None:
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        oldx, oldy = self.last_abs
        self.last_abs = (nx, ny)
        self._drain(0.0)

        if self.pointer_abs is not None:
            x = nx * self.width
            y = ny * self.height
            self.pointer_abs.pointer_motion_absolute(x, y).frame()
            return

        if self.pointer_rel is not None:
            dx = (nx - oldx) * self.width
            dy = (ny - oldy) * self.height
            self.pointer_rel.pointer_motion(dx, dy).frame()

    def send_mouse_rel(self, dx: float, dy: float) -> None:
        if self.pointer_rel is None:
            return
        self._drain(0.0)
        self.pointer_rel.pointer_motion(dx, dy).frame()


class FrameStreamer:
    def __init__(self, node_id: int, width: int, height: int, fps: int, jpeg_quality: int) -> None:
        self.node_id = node_id
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.latest_jpeg: Optional[bytes] = None
        self.cursor_norm = (0.5, 0.5)
        self.cursor_visible = True
        self.lock = threading.Lock()

        Gst.init(None)
        pipeline_desc = f"""
            pipewiresrc path={node_id} do-timestamp=true !
            video/x-raw,format=BGRx,width={width},height={height} !
            videorate !
            video/x-raw,framerate={fps}/1 !
            videoconvert !
            video/x-raw,format=BGR !
            appsink name=appsink emit-signals=true sync=false max-buffers=1 drop=true
        """
        self.pipeline = Gst.parse_launch(pipeline_desc)
        self.appsink = self.pipeline.get_by_name("appsink")
        self.appsink.connect("new-sample", self._on_sample)

    def set_cursor(self, nx: float, ny: float, visible: bool) -> None:
        with self.lock:
            self.cursor_norm = (
                max(0.0, min(1.0, nx)),
                max(0.0, min(1.0, ny)),
            )
            self.cursor_visible = visible

    def _draw_cursor(self, frame: np.ndarray) -> np.ndarray:
        with self.lock:
            nx, ny = self.cursor_norm
            visible = self.cursor_visible
        if not visible:
            return frame

        x = int(nx * (frame.shape[1] - 1))
        y = int(ny * (frame.shape[0] - 1))
        out = frame.copy()
        cv2.circle(out, (x, y), 8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(out, (x - 12, y), (x + 12, y), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(out, (x, y - 12), (x, y + 12), (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def _on_sample(self, sink):
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps()
        s = caps.get_structure(0)
        width = s.get_value("width")
        height = s.get_value("height")

        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK

        try:
            arr = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((height, width, 3))
            arr = self._draw_cursor(arr)
            ok2, enc = cv2.imencode(
                ".jpg",
                arr,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if ok2:
                with self.lock:
                    self.latest_jpeg = enc.tobytes()
        finally:
            buf.unmap(mapinfo)

        return Gst.FlowReturn.OK

    def start(self) -> None:
        self.pipeline.set_state(Gst.State.PLAYING)

    def stop(self) -> None:
        self.pipeline.set_state(Gst.State.NULL)

    def get_latest(self) -> Optional[bytes]:
        with self.lock:
            return self.latest_jpeg


def video_server(host: str, port: int, streamer: FrameStreamer) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)

    while True:
        conn, addr = srv.accept()
        print(f"[video] client connected from {addr}")
        try:
            while True:
                frame = streamer.get_latest()
                if frame is None:
                    time.sleep(0.005)
                    continue
                header = struct.pack("!I", len(frame))
                conn.sendall(header + frame)
                time.sleep(0.001)
        except (BrokenPipeError, ConnectionResetError):
            print("[video] client disconnected")
        finally:
            conn.close()


def input_server(host: str, port: int, injector: LibeiInjector, streamer: FrameStreamer) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)

    while True:
        conn, addr = srv.accept()
        print(f"[input] client connected from {addr}")
        file = conn.makefile("r", encoding="utf-8")
        try:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                evt = json.loads(line)
                t = evt.get("t")

                if t == "cursor":
                    streamer.set_cursor(
                        float(evt["x"]),
                        float(evt["y"]),
                        bool(evt.get("visible", True)),
                    )
                elif t == "mouse_abs":
                    x = float(evt["x"])
                    y = float(evt["y"])
                    visible = bool(evt.get("visible", True))
                    streamer.set_cursor(x, y, visible)
                    injector.send_mouse_abs(x, y)
                elif t == "mouse_rel":
                    injector.send_mouse_rel(float(evt["dx"]), float(evt["dy"]))
                elif t == "mouse_btn":
                    injector.send_button(int(evt["button"]), bool(evt["down"]))
                elif t == "key":
                    injector.send_key(str(evt["key"]), bool(evt["down"]))
        except (BrokenPipeError, ConnectionResetError, json.JSONDecodeError):
            print("[input] client disconnected")
        finally:
            conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen-host", default="0.0.0.0")
    ap.add_argument("--video-port", type=int, default=5500)
    ap.add_argument("--input-port", type=int, default=5501)
    ap.add_argument("--node-id", type=int, required=True)
    ap.add_argument("--eis-socket", default=None, help="Path to the libei/EIS Unix socket for this gamescope instance")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--jpeg-quality", type=int, default=80)
    args = ap.parse_args()

    streamer = FrameStreamer(args.node_id, args.width, args.height, args.fps, args.jpeg_quality)
    streamer.start()

    injector = LibeiInjector(args.eis_socket, args.width, args.height)

    t1 = threading.Thread(target=video_server, args=(args.listen_host, args.video_port, streamer), daemon=True)
    t2 = threading.Thread(target=input_server, args=(args.listen_host, args.input_port, injector, streamer), daemon=True)
    t1.start()
    t2.start()

    print(f"Streaming node {args.node_id} on video port {args.video_port}, input port {args.input_port}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        streamer.stop()


if __name__ == "__main__":
    main()
