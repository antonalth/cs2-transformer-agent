from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:
    cv2 = None
    np = None


@dataclass(slots=True)
class RuntimeOverlayRenderer:
    tile_width: int = 640
    tile_height: int = 360

    def render_jpeg(
        self,
        *,
        observation,
        slot_names: list[str] | tuple[str, ...],
        action_summaries: dict[str, Any],
        cache: dict[str, Any],
        metrics: dict[str, Any],
        status: dict[str, Any],
        checkpoint_path: str,
        quality: int = 80,
    ) -> bytes:
        _require_cv()
        slot_map = {slot.name: slot for slot in observation.slots}
        tiles: list[np.ndarray] = []

        for slot_name in slot_names:
            slot = slot_map.get(slot_name)
            frame = self._decode_frame(slot.frame_jpeg if slot is not None else None)
            summary = action_summaries.get(slot_name) or {}
            self._annotate_slot(frame, slot_name, slot, summary)
            tiles.append(frame)

        tiles.append(self._render_global_panel(cache, metrics, status, checkpoint_path))
        while len(tiles) < 6:
            tiles.append(np.zeros((self.tile_height, self.tile_width, 3), dtype=np.uint8))

        row_a = np.hstack(tiles[:3])
        row_b = np.hstack(tiles[3:6])
        canvas = np.vstack([row_a, row_b])
        ok, enc = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("failed to encode model3 runtime composite")
        return enc.tobytes()

    def _decode_frame(self, payload: bytes | None):
        _require_cv()
        if not payload:
            return np.zeros((self.tile_height, self.tile_width, 3), dtype=np.uint8)
        arr = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return np.zeros((self.tile_height, self.tile_width, 3), dtype=np.uint8)
        return cv2.resize(frame, (self.tile_width, self.tile_height))

    def _draw_lines(self, frame, lines: list[str], *, x: int = 12, y: int = 24, color=(255, 255, 255), scale: float = 0.5):
        line_height = max(20, int(24 * scale))
        current_y = y
        for line in lines:
            cv2.putText(frame, line, (x, current_y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (x, current_y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            current_y += line_height

    def _annotate_slot(self, frame, slot_name: str, slot, summary: dict[str, Any]) -> None:
        status = "missing" if slot is None else str(slot.status)
        lines = [
            f"{slot_name} [{status}]",
            f"frame_seq={0 if slot is None else int(slot.frame_seq)}",
            f"audio_seq={0 if slot is None else int(slot.audio_seq)}",
        ]
        active_actions = summary.get("active_actions") or []
        held_keys = summary.get("held_keys") or []
        held_buttons = summary.get("held_buttons") or []
        mouse_dx = float(summary.get("mouse_dx") or 0.0)
        mouse_dy = float(summary.get("mouse_dy") or 0.0)
        emitted_events = summary.get("emitted_events") or []
        if active_actions:
            lines.append("act: " + " ".join(active_actions[:6]))
        if held_keys:
            lines.append("keys: " + " ".join(held_keys[:8]))
        if held_buttons:
            lines.append("btns: " + " ".join(str(v) for v in held_buttons))
        lines.append(f"mouse=({mouse_dx:.1f}, {mouse_dy:.1f})")
        lines.append(f"events={len(emitted_events)}")
        self._draw_lines(frame, lines, color=(235, 242, 245))

    def _render_global_panel(
        self,
        cache: dict[str, Any],
        metrics: dict[str, Any],
        status: dict[str, Any],
        checkpoint_path: str,
    ):
        panel = np.zeros((self.tile_height, self.tile_width, 3), dtype=np.uint8)
        ckpt_name = checkpoint_path.rsplit("/", 1)[-1]
        last_timings = metrics.get("last_timings_ms", {}) or {}
        avg_timings = metrics.get("avg_timings_ms", {}) or {}
        lines = [
            "MODEL3 RUNTIME",
            ckpt_name,
            f"connected={status.get('connected', False)} running={status.get('running', False)}",
            f"recording={status.get('recording', False)}",
            f"cache_frames={cache.get('cached_frames', 0)}/{cache.get('max_cache_frames', 'no-drop')}",
            f"cache_tokens={cache.get('cached_tokens', 0)} total_frames={cache.get('total_frames_processed', 0)}",
            f"model_ms={float(metrics.get('last_inference_ms', 0.0)):.1f} avg={float(metrics.get('avg_inference_ms', 0.0)):.1f}",
            f"step_ms={float(metrics.get('last_step_ms', 0.0)):.1f} avg={float(metrics.get('avg_step_ms', 0.0)):.1f}",
            (
                "decode/prepare/model "
                f"{float(last_timings.get('decode_ms', 0.0)):.1f}/"
                f"{float(last_timings.get('prepare_ms', 0.0)):.1f}/"
                f"{float(last_timings.get('model_ms', 0.0)):.1f}"
            ),
            (
                "xfer/act/step "
                f"{float(last_timings.get('transfer_ms', 0.0)):.1f}/"
                f"{float(last_timings.get('action_decode_ms', 0.0)):.1f}/"
                f"{float(last_timings.get('end_to_end_ms', 0.0)):.1f}"
            ),
            (
                "avg decode/model/step "
                f"{float(avg_timings.get('decode_ms', 0.0)):.1f}/"
                f"{float(avg_timings.get('model_ms', 0.0)):.1f}/"
                f"{float(avg_timings.get('end_to_end_ms', 0.0)):.1f}"
            ),
            f"obs={metrics.get('observations_received', 0)} sent_events={metrics.get('actions_sent', 0)}",
        ]
        last_error = str(metrics.get("last_error", "") or "")
        if last_error:
            lines.append("error:")
            lines.append(last_error[:80])
        self._draw_lines(panel, lines, color=(110, 220, 240), scale=0.6)
        return panel


def _require_cv():
    if cv2 is None or np is None:
        raise RuntimeError("opencv-python and numpy are required for runtime overlays")
