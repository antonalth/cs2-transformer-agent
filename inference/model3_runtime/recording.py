from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess
import threading
from typing import Any

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:
    cv2 = None
    np = None


@dataclass(slots=True)
class RecordingStatus:
    active: bool = False
    path: str = ""
    fps: float = 0.0


@dataclass(slots=True)
class CaptureVideoStatus:
    active: bool = False
    output_dir: str = ""
    fps: float = 0.0
    slot_paths: dict[str, str] | None = None


@dataclass(slots=True)
class _CaptureSlotWriter:
    path: str
    process: subprocess.Popen | None = None
    stdin: Any = None
    start_time_ns: int | None = None
    frames_written: int = 0
    last_frame: Any = None


class CompositeRecorder:
    def __init__(self, output_dir: str, *, fps: float = 30.0, crf: int = 28, preset: str = "veryfast") -> None:
        self.output_dir = Path(output_dir)
        self.fps = float(fps)
        self.crf = int(crf)
        self.preset = str(preset)
        self._process = None
        self._stdin = None
        self._path = ""
        self._start_time_ns: int | None = None
        self._last_timestamp_ns: int | None = None
        self._frames_written = 0
        self._last_frame = None
        self._lock = threading.Lock()

    def start(self, path: str | None = None) -> str:
        _require_cv()
        with self._lock:
            if self._path != "":
                return self._path
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if path is None:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = str((self.output_dir / f"model3_runtime_{stamp}.mp4").resolve())
            self._path = path
            self._start_time_ns = None
            self._last_timestamp_ns = None
            self._frames_written = 0
            self._last_frame = None
            return self._path

    def write_jpeg(self, payload: bytes | None, *, timestamp_ns: int | None = None) -> None:
        _require_cv()
        if not payload:
            return
        arr = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        with self._lock:
            if self._path == "":
                return
            if timestamp_ns is not None:
                ts = int(timestamp_ns)
                if self._last_timestamp_ns is not None and ts <= self._last_timestamp_ns:
                    return
                if self._start_time_ns is None:
                    self._start_time_ns = ts
                    target_index = 0
                else:
                    target_index = max(
                        self._frames_written,
                        int(round((ts - self._start_time_ns) * self.fps / 1_000_000_000.0)),
                    )
                self._last_timestamp_ns = ts
            else:
                target_index = self._frames_written

            if self._process is None or self._stdin is None:
                height, width = frame.shape[:2]
                command = [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "bgr24",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    f"{self.fps:.6f}",
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    self.preset,
                    "-crf",
                    str(self.crf),
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    self._path,
                ]
                self._process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._stdin = self._process.stdin

            if self._last_frame is not None:
                while self._frames_written < target_index:
                    self._write_raw_frame_locked(self._last_frame)
                    self._frames_written += 1

            self._write_raw_frame_locked(frame)
            self._frames_written += 1
            self._last_frame = frame.copy()

    def stop(self) -> str:
        with self._lock:
            path = self._path
            if self._stdin is not None:
                self._stdin.close()
                self._stdin = None
            if self._process is not None:
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5)
                finally:
                    self._process = None
            self._path = ""
            self._start_time_ns = None
            self._last_timestamp_ns = None
            self._frames_written = 0
            self._last_frame = None
            return path

    def status(self) -> RecordingStatus:
        with self._lock:
            return RecordingStatus(active=self._path != "", path=self._path, fps=self.fps)

    def _write_raw_frame_locked(self, frame) -> None:
        if self._stdin is None:
            raise RuntimeError("composite recorder stdin is unavailable")
        try:
            self._stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError("ffmpeg composite recorder terminated unexpectedly") from exc


class CaptureVideoRecorder:
    def __init__(self, output_dir: str, *, fps: float = 30.0, crf: int = 28, preset: str = "veryfast") -> None:
        self.output_dir = Path(output_dir)
        self.fps = float(fps)
        self.crf = int(crf)
        self.preset = str(preset)
        self._lock = threading.Lock()
        self._active_dir = Path()
        self._slot_writers: dict[str, _CaptureSlotWriter] = {}

    def start(self, slot_names: list[str] | tuple[str, ...]) -> CaptureVideoStatus:
        _require_cv()
        with self._lock:
            self._stop_locked()
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._active_dir = (self.output_dir / f"capture_{stamp}").resolve()
            self._active_dir.mkdir(parents=True, exist_ok=True)
            self._slot_writers = {
                slot_name: _CaptureSlotWriter(path=str((self._active_dir / f"{slot_name}.mp4").resolve()))
                for slot_name in slot_names
            }
            return self._status_locked()

    def write_observation(self, observation) -> None:
        _require_cv()
        with self._lock:
            if not self._slot_writers:
                return
            timestamp_ns = int(getattr(observation, "server_time_ns", 0))
            for slot in observation.slots:
                writer = self._slot_writers.get(str(slot.name))
                if writer is None or not slot.frame_jpeg:
                    continue
                arr = np.frombuffer(slot.frame_jpeg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                self._write_frame_locked(writer, frame, timestamp_ns=timestamp_ns)

    def frame_jpeg(self, slot_name: str, frame_index: int) -> bytes:
        _require_cv()
        with self._lock:
            writer = self._slot_writers.get(slot_name)
            path = writer.path if writer is not None else ""
        if not path:
            raise RuntimeError("capture video for the requested slot is unavailable")
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise RuntimeError("failed to open capture video")
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None:
            raise RuntimeError("failed to decode the requested capture frame")
        ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError("failed to encode the requested capture frame")
        return enc.tobytes()

    def frame_index_for_time_ns(self, timestamp_ns: int) -> int:
        with self._lock:
            if not self._slot_writers:
                return 0
            starts = [writer.start_time_ns for writer in self._slot_writers.values() if writer.start_time_ns is not None]
            if not starts:
                return 0
            start_time_ns = min(int(value) for value in starts)
        return max(0, int(round((int(timestamp_ns) - start_time_ns) * self.fps / 1_000_000_000.0)))

    def stop(self) -> CaptureVideoStatus:
        with self._lock:
            self._stop_locked()
            return self._status_locked()

    def clear(self) -> CaptureVideoStatus:
        with self._lock:
            self._stop_locked()
            self._active_dir = Path()
            self._slot_writers = {}
            return self._status_locked()

    def status(self) -> CaptureVideoStatus:
        with self._lock:
            return self._status_locked()

    def _status_locked(self) -> CaptureVideoStatus:
        active = bool(self._slot_writers)
        slot_paths = {slot_name: writer.path for slot_name, writer in self._slot_writers.items()}
        return CaptureVideoStatus(
            active=active,
            output_dir=str(self._active_dir) if active else "",
            fps=self.fps,
            slot_paths=slot_paths,
        )

    def _write_frame_locked(self, writer: _CaptureSlotWriter, frame, *, timestamp_ns: int) -> None:
        if writer.start_time_ns is None:
            writer.start_time_ns = int(timestamp_ns)
            target_index = 0
        else:
            target_index = max(
                writer.frames_written,
                int(round((int(timestamp_ns) - writer.start_time_ns) * self.fps / 1_000_000_000.0)),
            )
        if writer.process is None or writer.stdin is None:
            height, width = frame.shape[:2]
            command = [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{self.fps:.6f}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                self.preset,
                "-crf",
                str(self.crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                writer.path,
            ]
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            writer.process = process
            writer.stdin = process.stdin
        if writer.last_frame is not None:
            while writer.frames_written < target_index:
                self._write_raw_frame(writer, writer.last_frame)
                writer.frames_written += 1
        self._write_raw_frame(writer, frame)
        writer.frames_written += 1
        writer.last_frame = frame.copy()

    def _write_raw_frame(self, writer: _CaptureSlotWriter, frame) -> None:
        if writer.stdin is None:
            raise RuntimeError("capture writer stdin is unavailable")
        try:
            writer.stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError("ffmpeg capture writer terminated unexpectedly") from exc

    def _stop_locked(self) -> None:
        for writer in self._slot_writers.values():
            if writer.stdin is not None:
                writer.stdin.close()
                writer.stdin = None
            if writer.process is not None:
                try:
                    writer.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    writer.process.kill()
                    writer.process.wait(timeout=5)
                finally:
                    writer.process = None
                    writer.last_frame = None


def _require_cv():
    if cv2 is None or np is None:
        raise RuntimeError("opencv-python and numpy are required for recording")
