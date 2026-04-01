from __future__ import annotations

from dataclasses import dataclass, field

from .actions import ActionDecodeConfig


DEFAULT_SLOT_NAMES = ("steam1", "steam2", "steam3", "steam4", "steam5")


@dataclass(slots=True)
class Model3RuntimeConfig:
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
    recording_fps: float = 10.0
    enable_fast_vision_preprocess: bool = True
    action_decode: ActionDecodeConfig = field(default_factory=ActionDecodeConfig)
