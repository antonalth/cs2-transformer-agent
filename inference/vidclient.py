#!/usr/bin/env python3
import argparse
import json
import socket
import struct
import threading
import time
from typing import Optional
import io
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
        self.latest = None
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while self.running:
            hdr = recv_exact(self.sock, 4)
            (size,) = struct.unpack("!I", hdr)
            payload = recv_exact(self.sock, size)

            try:
                surf = pygame.image.load(io.BytesIO(payload), "frame.jpg").convert()
            except pygame.error:
                continue

            with self.lock:
                self.latest = surf

    def get_latest(self):
        with self.lock:
            return self.latest.copy() if self.latest is not None else None

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


def compute_frame_rect(screen_size: tuple[int, int], frame_size: tuple[int, int]) -> pygame.Rect:
    screen_w, screen_h = screen_size
    frame_w, frame_h = frame_size

    if screen_w <= 0 or screen_h <= 0 or frame_w <= 0 or frame_h <= 0:
        return pygame.Rect(0, 0, max(1, screen_w), max(1, screen_h))

    scale = min(screen_w / frame_w, screen_h / frame_h)
    scaled_w = max(1, int(round(frame_w * scale)))
    scaled_h = max(1, int(round(frame_h * scale)))
    x = (screen_w - scaled_w) // 2
    y = (screen_h - scaled_h) // 2
    return pygame.Rect(x, y, scaled_w, scaled_h)


def clamp_norm_from_rect(pos: tuple[int, int], rect: pygame.Rect) -> tuple[float, float]:
    if rect.width <= 1 or rect.height <= 1:
        return 0.5, 0.5

    rel_x = min(max(pos[0], rect.left), rect.right - 1) - rect.left
    rel_y = min(max(pos[1], rect.top), rect.bottom - 1) - rect.top
    nx = rel_x / max(1, rect.width - 1)
    ny = rel_y / max(1, rect.height - 1)
    return nx, ny


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("host")
    ap.add_argument("--video-port", type=int, default=5500)
    ap.add_argument("--input-port", type=int, default=5501)
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--window-width", type=int, default=1280)
    ap.add_argument("--window-height", type=int, default=720)
    args = ap.parse_args()

    pygame.init()
    flags = pygame.FULLSCREEN if args.fullscreen else 0
    initial_size = (0, 0) if args.fullscreen else (args.window_width, args.window_height)
    screen = pygame.display.set_mode(initial_size, flags)
    pygame.display.set_caption("sim-client")
    pygame.mouse.set_visible(not args.fullscreen)

    receiver = FrameReceiver(args.host, args.video_port)
    sender = InputSender(args.host, args.input_port)

    fullscreen_active = args.fullscreen
    clock = pygame.time.Clock()
    last_cursor_send = 0.0
    frame_rect = pygame.Rect(0, 0, *screen.get_size())
    latest_frame = None

    try:
        while True:
            frame = receiver.get_latest()
            if frame is not None:
                latest_frame = frame

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return

                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_F11:
                    fullscreen_active = not fullscreen_active
                    if fullscreen_active:
                        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                        pygame.mouse.set_visible(False)
                    else:
                        screen = pygame.display.set_mode((args.window_width, args.window_height))
                        pygame.mouse.set_visible(True)
                    frame_rect = pygame.Rect(0, 0, *screen.get_size())
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
                        nx, ny = clamp_norm_from_rect(ev.pos, frame_rect)
                        sender.send({
                            "t": "mouse_abs",
                            "x": nx,
                            "y": ny,
                            "visible": True,
                        })
                        last_cursor_send = now

            screen.fill((0, 0, 0))
            if latest_frame is not None:
                frame_rect = compute_frame_rect(screen.get_size(), latest_frame.get_size())
                scaled = pygame.transform.smoothscale(latest_frame, frame_rect.size)
                screen.blit(scaled, frame_rect.topleft)

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
