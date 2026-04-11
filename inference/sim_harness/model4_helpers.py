from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

from .remote_protocol import HarnessObservation


@dataclass(slots=True)
class Model4Batch:
    images: torch.Tensor
    audio: torch.Tensor
    slot_name: str
    frame_seq: int
    frame_time_ns: int
    audio_seq: int
    audio_time_ns: int

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        image_dtype: torch.dtype | None = None,
        audio_dtype: torch.dtype | None = None,
        non_blocking: bool = True,
    ) -> Model4Batch:
        _require_torch()
        images = self.images
        audio = self.audio
        if device is not None:
            images = images.to(device=device, non_blocking=non_blocking)
            audio = audio.to(device=device, non_blocking=non_blocking)
        if image_dtype is not None:
            images = images.to(dtype=image_dtype)
        if audio_dtype is not None:
            audio = audio.to(dtype=audio_dtype)
        return Model4Batch(
            images=images,
            audio=audio,
            slot_name=self.slot_name,
            frame_seq=int(self.frame_seq),
            frame_time_ns=int(self.frame_time_ns),
            audio_seq=int(self.audio_seq),
            audio_time_ns=int(self.audio_time_ns),
        )


def observation_to_model4_batch(
    observation: HarnessObservation,
    *,
    slot_name: str,
    audio_sample_rate: int = 24000,
    frame_rate_hz: int = 32,
) -> Model4Batch:
    _require_torch()
    audio_samples_per_frame = max(1, int(audio_sample_rate) // int(frame_rate_hz))
    slot_map = {slot.name: slot for slot in observation.slots}
    slot = slot_map.get(slot_name)
    if slot is None:
        raise RuntimeError(f"slot {slot_name} missing from observation")
    if not slot.frame_jpeg:
        raise RuntimeError(f"slot {slot_name} has no frame in observation")

    image = _decode_frame_jpeg(slot.frame_jpeg).unsqueeze(0).unsqueeze(0).contiguous()
    audio = _decode_audio_pcm(
        slot.audio_pcm_s16le,
        channels=slot.audio_channels,
        expected_samples=audio_samples_per_frame,
    ).unsqueeze(0).contiguous()
    return Model4Batch(
        images=image,
        audio=audio,
        slot_name=slot_name,
        frame_seq=int(slot.frame_seq),
        frame_time_ns=int(slot.frame_time_ns),
        audio_seq=int(slot.audio_seq),
        audio_time_ns=int(slot.audio_time_ns),
    )


def _decode_frame_jpeg(frame_jpeg: bytes) -> torch.Tensor:
    t = _require_torch()
    if not frame_jpeg:
        raise ValueError("empty frame payload")
    arr = np.frombuffer(frame_jpeg, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("failed to decode JPEG frame")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1))
    return t.from_numpy(np.ascontiguousarray(chw))


def _decode_audio_pcm(
    pcm_s16le: bytes,
    *,
    channels: int = 2,
    expected_samples: int | None = None,
) -> torch.Tensor:
    t = _require_torch()
    if not pcm_s16le:
        samples = expected_samples or 0
        return t.zeros((channels, samples), dtype=t.float32)
    pcm = np.frombuffer(pcm_s16le, dtype=np.int16)
    if pcm.size % channels != 0:
        raise ValueError(f"PCM payload length {pcm.size} is not divisible by channels={channels}")
    interleaved = pcm.reshape(-1, channels)
    planar = interleaved.transpose(1, 0).astype(np.float32) / 32768.0
    tensor = t.from_numpy(np.ascontiguousarray(planar))
    if expected_samples is None:
        return tensor
    if tensor.shape[1] < expected_samples:
        pad = t.zeros((channels, expected_samples - tensor.shape[1]), dtype=t.float32)
        return t.cat([tensor, pad], dim=1)
    return tensor[:, :expected_samples]


def _require_torch():
    if torch is None:
        raise RuntimeError("torch is required for model4 harness helpers")
    return torch
