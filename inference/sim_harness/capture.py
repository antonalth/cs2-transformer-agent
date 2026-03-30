from __future__ import annotations

from collections import deque
import subprocess
import threading
import time

from .config import SlotConfig
from .models import DiscoveredEndpoint, InputEvent


class AudioCaptureBridge:
    def __init__(self, slot: SlotConfig) -> None:
        self.slot = slot
        self._proc: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest_pcm: bytes | None = None
        self._latest_seq = 0
        self._latest_time_ns = 0
        self._chunk_bytes = self.slot.audio_frame_samples * self.slot.audio_channels * 2
        self._chunk_duration_ns = int(1_000_000_000 * self.slot.audio_frame_samples / self.slot.audio_sample_rate)

    def start(self) -> None:
        command = [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "pulse",
            "-i",
            self.slot.audio_monitor_name,
            "-ac",
            str(self.slot.audio_channels),
            "-ar",
            str(self.slot.audio_sample_rate),
            "-f",
            "s16le",
            "pipe:1",
        ]
        self._proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self) -> None:
        if self._proc is None or self._proc.stdout is None:
            return
        stdout = self._proc.stdout
        while not self._stop.is_set():
            chunk = stdout.read(self._chunk_bytes)
            if not chunk or len(chunk) < self._chunk_bytes:
                break
            # Timestamp the chunk by its midpoint rather than read completion time.
            chunk_time_ns = max(0, time.time_ns() - (self._chunk_duration_ns // 2))
            with self._lock:
                self._latest_pcm = chunk
                self._latest_seq += 1
                self._latest_time_ns = chunk_time_ns

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=2.0)
            self._proc = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def latest_pcm(self) -> tuple[int, int, bytes] | None:
        with self._lock:
            if self._latest_pcm is None:
                return None
            seq = self._latest_seq
            ts_ns = self._latest_time_ns
            pcm = self._latest_pcm
        return seq, ts_ns, pcm


class SlotCaptureBridge:
    def __init__(self, slot: SlotConfig, endpoint: DiscoveredEndpoint) -> None:
        self.slot = slot
        self.endpoint = endpoint
        self._streamer = None
        self._injector = None
        self._audio: AudioCaptureBridge | None = None
        self._events: deque[InputEvent] = deque(maxlen=64)

    def start(self) -> None:
        # Lazy import keeps config/API modules importable even before the venv is active.
        from vidserver import FrameStreamer, LibeiInjector

        self._streamer = FrameStreamer(
            node_id=self.endpoint.pipewire_node_id,
            width=self.slot.width,
            height=self.slot.height,
            fps=self.slot.capture_fps,
            jpeg_quality=self.slot.jpeg_quality,
        )
        self._streamer.start()
        self._injector = LibeiInjector(
            self.endpoint.eis_socket,
            self.slot.width,
            self.slot.height,
        )
        try:
            self._audio = AudioCaptureBridge(self.slot)
            self._audio.start()
        except Exception:
            self._audio = None

    def stop(self) -> None:
        if self._streamer is not None:
            self._streamer.stop()
            self._streamer = None
        if self._injector is not None:
            self._injector.close()
            self._injector = None
        if self._audio is not None:
            self._audio.stop()
            self._audio = None

    def latest_jpeg(self) -> bytes | None:
        if self._streamer is None:
            return None
        return self._streamer.get_latest()

    def latest_frame_packet(self) -> tuple[int, int, bytes] | None:
        if self._streamer is None:
            return None
        seq, ts_ns, payload = self._streamer.get_latest_packet()
        if payload is None:
            return None
        return seq, ts_ns, payload

    def latest_audio_pcm(self) -> tuple[int, int, bytes] | None:
        if self._audio is None:
            return None
        return self._audio.latest_pcm()

    def recent_events(self) -> list[InputEvent]:
        return list(self._events)

    def apply_input_event(self, payload: dict) -> None:
        if self._injector is None or self._streamer is None:
            raise RuntimeError("capture bridge is not started")

        self._events.append(InputEvent(payload=payload))
        event_type = payload.get("t")
        if event_type == "cursor":
            self._streamer.set_cursor(
                float(payload["x"]),
                float(payload["y"]),
                bool(payload.get("visible", True)),
            )
            return
        if event_type == "mouse_abs":
            x = float(payload["x"])
            y = float(payload["y"])
            visible = bool(payload.get("visible", True))
            self._streamer.set_cursor(x, y, visible)
            self._injector.send_mouse_abs(x, y)
            return
        if event_type == "mouse_rel":
            self._injector.send_mouse_rel(float(payload["dx"]), float(payload["dy"]))
            return
        if event_type == "mouse_btn":
            self._injector.send_button(int(payload["button"]), bool(payload["down"]))
            return
        if event_type == "key":
            self._injector.send_key(str(payload["key"]), bool(payload["down"]))
            return
        raise ValueError(f"unsupported input event: {event_type}")
