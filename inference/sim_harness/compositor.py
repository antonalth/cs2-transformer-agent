from __future__ import annotations

from dataclasses import asdict

import cv2
import numpy as np

from .models import SlotSnapshot


class CompositeRenderer:
    def __init__(self, tile_width: int = 640, tile_height: int = 360) -> None:
        self.tile_width = tile_width
        self.tile_height = tile_height

    def _decode_jpeg(self, payload: bytes | None) -> np.ndarray:
        if not payload:
            return np.zeros((self.tile_height, self.tile_width, 3), dtype=np.uint8)
        arr = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return np.zeros((self.tile_height, self.tile_width, 3), dtype=np.uint8)
        return cv2.resize(frame, (self.tile_width, self.tile_height))

    def _annotate(self, frame: np.ndarray, snapshot: SlotSnapshot) -> np.ndarray:
        text_lines = [
            snapshot.name,
            f"status={snapshot.status}",
            f"user={snapshot.user}",
        ]
        if snapshot.pipewire_node_id is not None:
            text_lines.append(f"node={snapshot.pipewire_node_id}")
        if snapshot.error:
            text_lines.append(f"error={snapshot.error}")
        y = 24
        for line in text_lines:
            cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y += 24
        return frame

    def render_jpeg(self, frames: list[bytes | None], snapshots: list[SlotSnapshot], quality: int = 80) -> bytes:
        tiles: list[np.ndarray] = []
        for payload, snapshot in zip(frames, snapshots):
            tile = self._decode_jpeg(payload)
            tiles.append(self._annotate(tile, snapshot))

        while len(tiles) < 6:
            tiles.append(np.zeros((self.tile_height, self.tile_width, 3), dtype=np.uint8))

        row_a = np.hstack(tiles[:3])
        row_b = np.hstack(tiles[3:6])
        canvas = np.vstack([row_a, row_b])
        ok, enc = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("failed to encode composite jpeg")
        return enc.tobytes()
