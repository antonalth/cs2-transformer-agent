"""
ported_console.py  (with robust reconnect)

Single-file Python 3 client for Valve’s VConsole2 (Counter-Strike 2).

Changes:
 * VConsoleConnection._open_socket() now retries on ConnectionRefusedError
   up to self._max_retries times, sleeping 0.2s between attempts.
 * send() catches BrokenPipeError/OSError, attempts to reconnect (also with retries).
"""

from __future__ import annotations

import queue
import socket
import struct
import threading
import time
from typing import Deque, Dict, Optional, Tuple

__all__ = ["connect", "VConsoleConnection"]

# ─────────────────────────────────────────────────────────────────────────────
_PROTOCOL = 0x00D4
_HEADER_SIZE = 12
_DEFAULT_ADDR: Tuple[str, int] = ("127.0.0.1", 29000)

_pack_u16 = lambda v: struct.pack("!H", v)
_pack_u32 = lambda v: struct.pack("!I", v)
_unpack_u16 = lambda b, off=0: struct.unpack_from("!H", b, off)[0]
_unpack_u32 = lambda b, off=0: struct.unpack_from("!I", b, off)[0]


class _ByteBuffer(bytearray):
    """A bytearray with convenience methods for incremental parsing."""

    def take(self, n: int) -> bytes:
        if len(self) < n:
            raise ValueError("buffer underflow")
        chunk = bytes(self[:n])
        del self[:n]
        return chunk


class VConsoleConnection:
    """
    Represents a TCP connection to VConsole2 (CS2). Auto-reconnects on send.

    Parameters
    ----------
    host : str
    port : int
    max_retries : int
        How many times to retry connecting if the socket is closed or refused.
    timeout : float
        Timeout (sec) for socket.create_connection().
    """

    def send(self, text: str) -> None:
        """Send a console command (automatically NUL-terminated). Retries on failure."""
        payload = text.encode() + b"\x00"
        pkt_len = len(payload) + _HEADER_SIZE
        header = (
            b"CMND"
            + _pack_u16(_PROTOCOL)
            + b"\x00\x00"
            + _pack_u16(pkt_len)
            + b"\x00\x00"
        )
        for attempt in range(self._max_retries):
            if not self._alive:
                # force a reconnect attempt
                try:
                    self._open_socket()
                except Exception:
                    # if this was the last attempt, rethrow
                    if attempt + 1 >= self._max_retries:
                        raise
                    time.sleep(0.2)
                    continue
            try:
                self._sock.sendall(header + payload)
                return
            except (BrokenPipeError, OSError):
                self._alive = False
                if attempt + 1 >= self._max_retries:
                    raise
                time.sleep(0.2)
                try:
                    self._open_socket()
                except Exception:
                    if attempt + 1 >= self._max_retries:
                        raise
                    time.sleep(0.2)
                    continue

    def read(self, *, block: bool = True, timeout: Optional[float] = None) -> Optional[Tuple[str, str]]:
        """
        Return the next (channel, line) tuple.
        If no data and block=False, return None.
        """
        try:
            return self._queue.get(block, timeout)
        except queue.Empty:
            return None

    def close(self):
        """Close the TCP socket and stop background thread."""
        self._alive = False
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __init__(self, host: str, port: int, max_retries: int = 3, timeout: float = 5.0):
        self._host = host
        self._port = port
        self._max_retries = max_retries
        self._timeout = timeout

        self._queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self._channels: Dict[int, str] = {}
        self._buf: _ByteBuffer = _ByteBuffer()
        self._alive = False

        # Attempt initial connection
        self._open_socket()

    def _open_socket(self):
        """
        (Re)open the TCP socket and spawn the RX thread.
        Retries on ConnectionRefusedError / OSError up to self._max_retries.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                # If there's an existing socket, close it first
                try:
                    self._sock.shutdown(socket.SHUT_RDWR)  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    self._sock.close()  # type: ignore[attr-defined]
                except Exception:
                    pass

                sock = socket.create_connection((self._host, self._port), self._timeout)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._sock = sock
                self._alive = True
                threading.Thread(target=self._rx_loop, daemon=True).start()
                return
            except (ConnectionRefusedError, OSError) as e:
                last_exc = e
                time.sleep(0.2)
                continue
        # If we exit loop without connecting, rethrow the last exception
        raise last_exc

    def _rx_loop(self):
        recv = self._sock.recv
        while self._alive:
            try:
                chunk = recv(4096)
                if not chunk:
                    break  # peer closed
                self._buf.extend(chunk)
                self._parse_buffer()
            except OSError:
                break
        self._alive = False

    def _parse_buffer(self):
        buf = self._buf
        while len(buf) >= _HEADER_SIZE:
            msg_type = buf[:4].decode("ascii", "replace")
            version = _unpack_u16(buf, 4)
            pkt_len = _unpack_u16(buf, 8)
            if version != _PROTOCOL or pkt_len < _HEADER_SIZE:
                buf.take(1)
                continue
            if len(buf) < pkt_len:
                return
            header = buf.take(_HEADER_SIZE)
            body_len = pkt_len - _HEADER_SIZE
            body = buf.take(body_len) if body_len else b""
            handle = _unpack_u16(header, 10)
            self._dispatch(msg_type, body, handle)

    def _dispatch(self, msg_type: str, body: bytes, handle: int):
        if msg_type == "PRNT":
            self._parse_prnt(body)
        elif msg_type == "CHAN":
            self._parse_chan(body)

    def _parse_chan(self, body: bytes):
        buf = memoryview(body)
        count = _unpack_u16(buf, 0)
        offset = 2
        for _ in range(count):
            if len(buf) < offset + 58:
                break
            chan_id = _unpack_u32(buf, offset)
            offset += 4 + 16
            offset += 4
            raw_name = bytes(buf[offset : offset + 34])
            offset += 34
            name = raw_name.split(b"\x00", 1)[0].decode("utf-8", "replace")
            self._channels[chan_id] = name or f"chan_{chan_id}"

    def _parse_prnt(self, body: bytes):
        if len(body) < 28:
            return
        chan_id = _unpack_u32(body, 0)
        msg = body[28:].split(b"\x00", 1)[0].decode("utf-8", "replace")
        channel_name = self._channels.get(chan_id, str(chan_id))
        self._queue.put((channel_name, msg))


def connect(
    host: str = _DEFAULT_ADDR[0],
    port: int = _DEFAULT_ADDR[1],
    *,
    timeout: float = 5.0,
    max_retries: int = 10,
) -> VConsoleConnection:
    """
    Open a TCP connection to the CS 2 network console and return a VConsoleConnection.
    Retries up to max_retries if the initial connect fails.
    """
    return VConsoleConnection(host, port, max_retries=max_retries, timeout=timeout)
