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
    item = request_frame_packet(slot, timeout_s)
    if item is None:
        return None
    return item[2]


def request_frame_packet(slot: SlotConfig, timeout_s: float = 1.0) -> tuple[int, int, bytes] | None:
    with socket.create_connection(("127.0.0.1", slot.video_port), timeout=timeout_s) as sock:
        header = _recv_exact(sock, 20)
        seq, ts_ns, size = struct.unpack("!QQI", header)
        if size == 0:
            return None
        return seq, ts_ns, _recv_exact(sock, size)


def request_audio(slot: SlotConfig, timeout_s: float = 1.0) -> tuple[int, int, bytes] | None:
    with socket.create_connection(("127.0.0.1", slot.audio_port), timeout=timeout_s) as sock:
        header = _recv_exact(sock, 20)
        seq, ts_ns, size = struct.unpack("!QQI", header)
        if size == 0:
            return None
        return seq, ts_ns, _recv_exact(sock, size)
