from __future__ import annotations

import json
import socket
import struct

from .config import SlotConfig


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("socket closed")
        data.extend(chunk)
    return bytes(data)


def _request_json(port: int, payload: dict, timeout_s: float) -> dict:
    with socket.create_connection(("127.0.0.1", port), timeout=timeout_s) as sock:
        sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        file = sock.makefile("r", encoding="utf-8")
        line = file.readline()
        if not line:
            raise ConnectionError("empty response")
        return json.loads(line)


def request_status(slot: SlotConfig, timeout_s: float = 1.0) -> dict:
    return _request_json(slot.input_port, {"op": "status"}, timeout_s)


def request_stop(slot: SlotConfig, timeout_s: float = 1.0) -> dict:
    return _request_json(slot.input_port, {"op": "stop"}, timeout_s)


def send_input(slot: SlotConfig, payload: dict, timeout_s: float = 1.0) -> dict:
    return _request_json(slot.input_port, {"op": "input", "event": payload}, timeout_s)


def request_frame(slot: SlotConfig, timeout_s: float = 1.0) -> bytes | None:
    with socket.create_connection(("127.0.0.1", slot.video_port), timeout=timeout_s) as sock:
        header = _recv_exact(sock, 4)
        (size,) = struct.unpack("!I", header)
        if size == 0:
            return None
        return _recv_exact(sock, size)
