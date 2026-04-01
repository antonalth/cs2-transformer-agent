from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import threading

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


class CompositeRecorder:
    def __init__(self, output_dir: str, *, fps: float = 10.0) -> None:
        self.output_dir = Path(output_dir)
        self.fps = float(fps)
        self._writer = None
        self._path = ""
        self._lock = threading.Lock()

    def start(self, path: str | None = None) -> str:
        _require_cv()
        with self._lock:
            if self._writer is not None:
                return self._path
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if path is None:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = str((self.output_dir / f"model3_runtime_{stamp}.mp4").resolve())
            self._path = path
            return self._path

    def write_jpeg(self, payload: bytes | None) -> None:
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
            if self._writer is None:
                height, width = frame.shape[:2]
                self._writer = cv2.VideoWriter(
                    self._path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.fps,
                    (width, height),
                )
            self._writer.write(frame)

    def stop(self) -> str:
        with self._lock:
            path = self._path
            if self._writer is not None:
                self._writer.release()
                self._writer = None
            self._path = ""
            return path

    def status(self) -> RecordingStatus:
        with self._lock:
            return RecordingStatus(active=self._path != "", path=self._path, fps=self.fps)


def _require_cv():
    if cv2 is None or np is None:
        raise RuntimeError("opencv-python and numpy are required for recording")
