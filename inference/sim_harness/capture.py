from __future__ import annotations

from collections import deque

from .config import SlotConfig
from .models import DiscoveredEndpoint, InputEvent


class SlotCaptureBridge:
    def __init__(self, slot: SlotConfig, endpoint: DiscoveredEndpoint) -> None:
        self.slot = slot
        self.endpoint = endpoint
        self._streamer = None
        self._injector = None
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

    def stop(self) -> None:
        if self._streamer is not None:
            self._streamer.stop()
            self._streamer = None
        if self._injector is not None:
            self._injector.close()
            self._injector = None

    def latest_jpeg(self) -> bytes | None:
        if self._streamer is None:
            return None
        return self._streamer.get_latest()

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
