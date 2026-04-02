from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib


@dataclass(slots=True)
class WebConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    browser_fps: int = 1
    auto_start_slots: bool = False
    ssl_certfile: str = ""
    ssl_keyfile: str = ""


@dataclass(slots=True)
class RecordingConfig:
    output_dir: str = "./recordings"
    composite_fps: int = 10
    codec: str = "h264_nvenc"
    crf: int = 23


@dataclass(slots=True)
class RuntimeConfig:
    runas_path: str = "/usr/local/bin/runas"
    discovery_timeout_s: float = 20.0
    discovery_poll_interval_s: float = 0.25
    restart_backoff_s: float = 2.0


@dataclass(slots=True)
class ServerConfig:
    enabled: bool = True
    user: str = "server"
    tmux_session: str = "cs2-ds"
    start_command: list[str] = field(default_factory=lambda: ["./inference/server/start_server.sh"])
    scenario_config_path: str = (
        "/home/server/steam-lib/steamapps/common/Counter-Strike Global Offensive/"
        "game/csgo/addons/counterstrikesharp/plugins/Cs2SimHarness/cs2-sim-harness.json"
    )
    plugin_state_path: str = (
        "/home/server/steam-lib/steamapps/common/Counter-Strike Global Offensive/"
        "game/csgo/addons/counterstrikesharp/plugins/Cs2SimHarness/cs2-sim-state.json"
    )
    connect_address: str = "127.0.0.1:27015"
    console_toggle_key: str = "grave"
    console_toggle_with_shift: bool = True
    connect_command_template: str = "connect {address}"
    connect_submit_key: str = "enter"
    console_close_key: str = "esc"
    pause_command: str = "css_sim_freeze"
    resume_command: str = "css_sim_unfreeze"
    connect_open_delay_s: float = 0.2
    connect_keystroke_delay_s: float = 0.012
    connect_post_enter_delay_s: float = 0.75
    plugin_state_refresh_delay_s: float = 0.15
    log_lines: int = 200


@dataclass(slots=True)
class SlotConfig:
    name: str
    user: str
    tmux_session: str
    launch_command: list[str]
    gamescope_args: list[str] = field(default_factory=lambda: ["--backend", "headless"])
    width: int = 1280
    height: int = 720
    capture_fps: int = 30
    jpeg_quality: int = 80
    video_port: int = 5500
    input_port: int = 5501
    audio_port: int = 5502
    audio_sample_rate: int = 24000
    audio_channels: int = 2
    audio_frame_hz: int = 32
    audio_monitor_name: str = "auto_null.monitor"
    inspect_fps: int = 1
    startup_grace_s: float = 2.0

    @property
    def audio_frame_samples(self) -> int:
        return max(1, self.audio_sample_rate // self.audio_frame_hz)

    def effective_gamescope_args(self) -> list[str]:
        args = list(self.gamescope_args)
        flags = set(args)
        if "-W" not in flags and "--output-width" not in flags:
            args.extend(["-W", str(self.width)])
        if "-H" not in flags and "--output-height" not in flags:
            args.extend(["-H", str(self.height)])
        if "-w" not in flags and "--nested-width" not in flags:
            args.extend(["-w", str(self.width)])
        if "-h" not in flags and "--nested-height" not in flags:
            args.extend(["-h", str(self.height)])
        return args


@dataclass(slots=True)
class HarnessConfig:
    source_path: str = ""
    web: WebConfig = field(default_factory=WebConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    slots: list[SlotConfig] = field(default_factory=list)

    def slot_map(self) -> dict[str, SlotConfig]:
        return {slot.name: slot for slot in self.slots}


def _load_web_config(data: dict) -> WebConfig:
    return WebConfig(**data) if data else WebConfig()


def _load_recording_config(data: dict) -> RecordingConfig:
    return RecordingConfig(**data) if data else RecordingConfig()


def _load_runtime_config(data: dict) -> RuntimeConfig:
    return RuntimeConfig(**data) if data else RuntimeConfig()


def _load_server_config(data: dict) -> ServerConfig:
    return ServerConfig(**data) if data else ServerConfig()


def _load_slots(data: list[dict]) -> list[SlotConfig]:
    return [SlotConfig(**slot) for slot in data]


def load_config(path: str | Path) -> HarnessConfig:
    cfg_path = Path(path)
    raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    return HarnessConfig(
        source_path=str(cfg_path.resolve()),
        web=_load_web_config(raw.get("web", {})),
        recording=_load_recording_config(raw.get("recording", {})),
        runtime=_load_runtime_config(raw.get("runtime", {})),
        server=_load_server_config(raw.get("server", {})),
        slots=_load_slots(raw.get("slots", [])),
    )
