from __future__ import annotations

from dataclasses import dataclass
import json
import struct
import time
from typing import Any


MAGIC = b"CS2M"
VERSION = 1


@dataclass(slots=True)
class SlotObservation:
    name: str
    status: str
    error: str | None
    frame_seq: int
    frame_time_ns: int
    frame_jpeg: bytes
    audio_seq: int
    audio_time_ns: int
    audio_pcm_s16le: bytes
    audio_sample_rate: int
    audio_channels: int


@dataclass(slots=True)
class HarnessObservation:
    server_time_ns: int
    slots: list[SlotObservation]


@dataclass(slots=True)
class StreamMessage:
    kind: str
    raw_size: int
    observation: HarnessObservation | None = None
    payload: dict[str, Any] | None = None


def encode_observation(metadata: dict, blobs: list[bytes]) -> bytes:
    meta_bytes = json.dumps(metadata, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    header = MAGIC + struct.pack("!HI", VERSION, len(meta_bytes))
    return header + meta_bytes + b"".join(blobs)


def decode_observation(payload: bytes) -> HarnessObservation:
    if len(payload) < 10:
        raise ValueError("observation payload too short")
    if payload[:4] != MAGIC:
        raise ValueError("invalid observation magic")
    version, meta_len = struct.unpack("!HI", payload[4:10])
    if version != VERSION:
        raise ValueError(f"unsupported observation version: {version}")
    meta_start = 10
    meta_end = meta_start + meta_len
    metadata = json.loads(payload[meta_start:meta_end].decode("utf-8"))
    cursor = meta_end
    slots: list[SlotObservation] = []
    for item in metadata["slots"]:
        frame_size = int(item["frame_size"])
        frame = payload[cursor : cursor + frame_size]
        cursor += frame_size
        audio_size = int(item["audio_size"])
        audio = payload[cursor : cursor + audio_size]
        cursor += audio_size
        slots.append(
            SlotObservation(
                name=item["name"],
                status=item["status"],
                error=item.get("error"),
                frame_seq=int(item["frame_seq"]),
                frame_time_ns=int(item["frame_time_ns"]),
                frame_jpeg=frame,
                audio_seq=int(item["audio_seq"]),
                audio_time_ns=int(item["audio_time_ns"]),
                audio_pcm_s16le=audio,
                audio_sample_rate=int(item["audio_sample_rate"]),
                audio_channels=int(item["audio_channels"]),
            )
        )
    return HarnessObservation(server_time_ns=int(metadata["server_time_ns"]), slots=slots)


def observation_metadata_item(
    *,
    name: str,
    status: str,
    error: str | None,
    frame_seq: int,
    frame_time_ns: int,
    frame_size: int,
    audio_seq: int,
    audio_time_ns: int,
    audio_size: int,
    audio_sample_rate: int,
    audio_channels: int,
) -> dict:
    return {
        "name": name,
        "status": status,
        "error": error,
        "frame_seq": frame_seq,
        "frame_time_ns": frame_time_ns,
        "frame_size": frame_size,
        "audio_seq": audio_seq,
        "audio_time_ns": audio_time_ns,
        "audio_size": audio_size,
        "audio_sample_rate": audio_sample_rate,
        "audio_channels": audio_channels,
    }


def new_metadata() -> dict:
    return {
        "server_time_ns": time.time_ns(),
        "slots": [],
    }


def encode_stream_actions(actions: dict[str, list[dict] | dict], client_send_ns: int | None = None) -> str:
    return json.dumps(
        {
            "op": "actions",
            "client_send_ns": client_send_ns if client_send_ns is not None else time.time_ns(),
            "actions": actions,
        },
        separators=(",", ":"),
        ensure_ascii=True,
    )


def encode_stream_ack(
    *,
    client_send_ns: int | None,
    server_recv_ns: int,
    server_send_ns: int,
    results: dict[str, Any],
) -> str:
    return json.dumps(
        {
            "op": "actions_ack",
            "client_send_ns": client_send_ns,
            "server_recv_ns": server_recv_ns,
            "server_send_ns": server_send_ns,
            "results": results,
        },
        separators=(",", ":"),
        ensure_ascii=True,
    )


def encode_stream_error(*, code: str, detail: str) -> str:
    return json.dumps(
        {
            "op": "error",
            "code": code,
            "detail": detail,
            "server_time_ns": time.time_ns(),
        },
        separators=(",", ":"),
        ensure_ascii=True,
    )


def encode_stream_pong(client_send_ns: int | None = None) -> str:
    return json.dumps(
        {
            "op": "pong",
            "client_send_ns": client_send_ns,
            "server_time_ns": time.time_ns(),
        },
        separators=(",", ":"),
        ensure_ascii=True,
    )


def decode_stream_text(payload: str | bytes) -> dict[str, Any]:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    item = json.loads(payload)
    if not isinstance(item, dict):
        raise ValueError("stream text payload must be a JSON object")
    return item
