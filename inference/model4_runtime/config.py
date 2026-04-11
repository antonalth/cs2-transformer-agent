from __future__ import annotations

from dataclasses import dataclass, field

from .actions import ActionDecodeConfig


DEFAULT_SLOT_NAMES = ("steam1",)


@dataclass(slots=True)
class Model4RuntimeConfig:
    checkpoint_path: str
    harness_url: str
    device: str = "cuda"
    data_root: str = "./dataset0"
    verify_ssl: bool = True
    transport: str = "http"
    poll_interval_s: float = 1.0 / 32.0
    slot_names: tuple[str, ...] = DEFAULT_SLOT_NAMES
    cache_window_frames: int | None = None
    videos_dir: str = "./videos"
    browser_fps: float = 2.0
    recording_fps: float = 30.0
    capture_video_fps: float = 30.0
    capture_video_crf: int = 28
    settings_path: str = "./videos/runtime_settings.json"
    enable_fast_vision_preprocess: bool = True
    action_decode: ActionDecodeConfig = field(default_factory=ActionDecodeConfig)
