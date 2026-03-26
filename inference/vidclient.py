#!/usr/bin/env python3
import argparse
import json
import socket
import struct
import threading
import time
from typing import Optional

import cv2
import numpy as np
import pygame


def recv_exact(sock: socket.socket, n: int) -> bytes:
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Socket closed")
        data.extend(chunk)
    return bytes(data)


class FrameReceiver:
    def __init__(self, host: str, port: int) -> None:
        self.sock = socket.create_connection((host, port))
        self.lock = threading.Lock()
        self.latest: Optional[np.ndarray] = None
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while self.running:
            hdr = recv_exact(self.sock, 4)
            (size,) = struct.unpack("!I", hdr)
            payload = recv_exact(self.sock, size)
            arr = np.frombuffer(payload, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self.lock:
                self.latest = frame

    def get_latest(self) -> Optional[np.ndarray]:
        with self.lock:
            return None if self.latest is None else self.latest.copy()

    def close(self) -> None:
        self.running = False
        try:
            self.sock.close()
        except OSError:
            pass


class InputSender:
    def __init__(self, host: str, port: int) -> None:
        self.sock = socket.create_connection((host, port))
        self.lock = threading.Lock()

    def send(self, obj: dict) -> None:
        line = (json.dumps(obj) + "\n").encode("utf-8")
        with self.lock:
            self.sock.sendall(line)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


PYGAME_KEYMAP = {
    pygame.K_SPACE: "space",
    pygame.K_RETURN: "enter",
    pygame.K_TAB: "tab",
    pygame.K_ESCAPE: "esc",
    pygame.K_BACKSPACE: "backspace",
    pygame.K_LEFT: "left",
    pygame.K_RIGHT: "right",
    pygame.K_UP: "up",
    pygame.K_DOWN: "down",
    pygame.K_LSHIFT: "lshift",
    pygame.K_RSHIFT: "rshift",
    pygame.K_LCTRL: "lctrl",
    pygame.K_RCTRL: "rctrl",
    pygame.K_LALT: "lalt",
    pygame.K_RALT: "ralt",
    pygame.K_MINUS: "minus",
    pygame.K_EQUALS: "equals",
    pygame.K_COMMA: "comma",
    pygame.K_PERIOD: "period",
    pygame.K_SLASH: "slash",
    pygame.K_SEMICOLON: "semicolon",
    pygame.K_QUOTE: "apostrophe",
    pygame.K_LEFTBRACKET: "leftbracket",
    pygame.K_RIGHTBRACKET: "rightbracket",
    pygame.K_BACKSLASH: "backslash",
    pygame.K_BACKQUOTE: "grave",
    pygame.K_CAPSLOCK: "capslock",
    pygame.K_F1: "f1",
    pygame.K_F2: "f2",
    pygame.K_F3: "f3",
    pygame.K_F4: "f4",
    pygame.K_F5: "f5",
    pygame.K_F6: "f6",
    pygame.K_F7: "f7",
    pygame.K_F8: "f8",
    pygame.K_F9: "f9",
    pygame.K_F10: "f10",
    pygame.K_F11: "f11",
    pygame.K_F12: "f12",
}

for c in "abcdefghijklmnopqrstuvwxyz":
    PYGAME_KEYMAP[getattr(pygame, f"K_{c}")] = c
for d in "0123456789":
    PYGAME_KEYMAP[getattr(pygame, f"K_{d}")] = d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("host")
    ap.add_argument("--video-port", type=int, default=5500)
    ap.add_argument("--input-port", type=int, default=5501)
    ap.add_argument("--fullscreen", action="store_true", default=True)
    args = ap.parse_args()

    pygame.init()
    flags = pygame.FULLSCREEN if args.fullscreen else 0
    screen = pygame.display.set_mode((0, 0), flags)
    pygame.display.set_caption("sim-client")
    pygame.mouse.set_visible(False)

    receiver = FrameReceiver(args.host, args.video_port)
    sender = InputSender(args.host, args.input_port)

    fullscreen_active = args.fullscreen
    clock = pygame.time.Clock()
    last_cursor_send = 0.0

    try:
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return

                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_F11:
                    fullscreen_active = not fullscreen_active
                    if fullscreen_active:
                        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                        pygame.mouse.set_visible(False)
                    else:
                        screen = pygame.display.set_mode((1280, 720))
                        pygame.mouse.set_visible(True)
                    continue

                if not fullscreen_active:
                    continue

                if ev.type in (pygame.KEYDOWN, pygame.KEYUP):
                    key_name = PYGAME_KEYMAP.get(ev.key)
                    if key_name is not None:
                        sender.send({
                            "t": "key",
                            "key": key_name,
                            "down": ev.type == pygame.KEYDOWN,
                        })

                elif ev.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
                    sender.send({
                        "t": "mouse_btn",
                        "button": ev.button,
                        "down": ev.type == pygame.MOUSEBUTTONDOWN,
                    })

                elif ev.type == pygame.MOUSEMOTION:
                    now = time.time()
                    if now - last_cursor_send >= 1.0 / 60.0:
                        x, y = ev.pos
                        w, h = screen.get_size()
                        nx = x / max(1, w - 1)
                        ny = y / max(1, h - 1)
                        sender.send({
                            "t": "mouse_abs",
                            "x": nx,
                            "y": ny,
                            "visible": True,
                        })
                        last_cursor_send = now

            frame = receiver.get_latest()
            if frame is not None:
                fh, fw = frame.shape[:2]
                surf = pygame.image.frombuffer(frame.tobytes(), (fw, fh), "RGB")
                scaled = pygame.transform.smoothscale(surf, screen.get_size())
                screen.blit(scaled, (0, 0))

                if fullscreen_active:
                    mx, my = pygame.mouse.get_pos()
                    pygame.draw.circle(screen, (255, 255, 255), (mx, my), 8, 1)
                    pygame.draw.line(screen, (255, 255, 255), (mx - 12, my), (mx + 12, my), 1)
                    pygame.draw.line(screen, (255, 255, 255), (mx, my - 12), (mx, my + 12), 1)

                pygame.display.flip()

            clock.tick(120)

    finally:
        receiver.close()
        sender.close()
        pygame.quit()


if __name__ == "__main__":
    main()