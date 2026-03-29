from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from time import time


class SlotStatus(StrEnum):
    STOPPED = "stopped"
    LAUNCHING = "launching"
    DISCOVERING = "discovering"
    READY = "ready"
    ERROR = "error"
    STOPPING = "stopping"


@dataclass(slots=True)
class DiscoveredEndpoint:
    uid: int
    pipewire_node_id: int
    pipewire_client_id: int
    process_id: int
    gamescope_socket: str
    eis_socket: str


@dataclass(slots=True)
class InputEvent:
    payload: dict
    timestamp: float = field(default_factory=time)


@dataclass(slots=True)
class SlotSnapshot:
    name: str
    status: str
    user: str
    tmux_session: str
    process_id: int | None = None
    pipewire_node_id: int | None = None
    pipewire_client_id: int | None = None
    gamescope_socket: str | None = None
    eis_socket: str | None = None
    error: str | None = None
    updated_at: float = field(default_factory=time)
