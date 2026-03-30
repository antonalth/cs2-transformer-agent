from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

from .remote_protocol import HarnessObservation


DEFAULT_SLOT_NAMES = ("steam1", "steam2", "steam3", "steam4", "steam5")


@dataclass(slots=True)
class Model3Batch:
    images: torch.Tensor
    audio: torch.Tensor
    slot_names: list[str]
    frame_seq: torch.Tensor
    frame_time_ns: torch.Tensor
    audio_seq: torch.Tensor
    audio_time_ns: torch.Tensor

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        image_dtype: torch.dtype | None = None,
        audio_dtype: torch.dtype | None = None,
        non_blocking: bool = True,
    ) -> Model3Batch:
        _require_torch()
        images = self.images
        audio = self.audio
        frame_seq = self.frame_seq
        frame_time_ns = self.frame_time_ns
        audio_seq = self.audio_seq
        audio_time_ns = self.audio_time_ns
        if device is not None:
            images = images.to(device=device, non_blocking=non_blocking)
            audio = audio.to(device=device, non_blocking=non_blocking)
            frame_seq = frame_seq.to(device=device, non_blocking=non_blocking)
            frame_time_ns = frame_time_ns.to(device=device, non_blocking=non_blocking)
            audio_seq = audio_seq.to(device=device, non_blocking=non_blocking)
            audio_time_ns = audio_time_ns.to(device=device, non_blocking=non_blocking)
        if image_dtype is not None:
            images = images.to(dtype=image_dtype)
        if audio_dtype is not None:
            audio = audio.to(dtype=audio_dtype)
        return Model3Batch(
            images=images,
            audio=audio,
            slot_names=list(self.slot_names),
            frame_seq=frame_seq,
            frame_time_ns=frame_time_ns,
            audio_seq=audio_seq,
            audio_time_ns=audio_time_ns,
        )


def observation_to_model3_batch(
    observation: HarnessObservation,
    *,
    slot_names: Sequence[str] = DEFAULT_SLOT_NAMES,
    audio_sample_rate: int = 24000,
    frame_rate_hz: int = 32,
) -> Model3Batch:
    t = _require_torch()
    audio_samples_per_frame = max(1, int(audio_sample_rate) // int(frame_rate_hz))
    slot_map = {slot.name: slot for slot in observation.slots}

    frames = []
    audio_chunks = []
    frame_seq = []
    frame_time_ns = []
    audio_seq = []
    audio_time_ns = []

    for slot_name in slot_names:
        slot = slot_map.get(slot_name)
        if slot is None:
            raise RuntimeError(f"slot {slot_name} missing from observation")
        if not slot.frame_jpeg:
            raise RuntimeError(f"slot {slot_name} has no frame in observation")

        frames.append(decode_model3_frame_jpeg(slot.frame_jpeg))
        audio_chunks.append(
            decode_model3_audio_pcm(
                slot.audio_pcm_s16le,
                channels=slot.audio_channels,
                expected_samples=audio_samples_per_frame,
            )
        )
        frame_seq.append(int(slot.frame_seq))
        frame_time_ns.append(int(slot.frame_time_ns))
        audio_seq.append(int(slot.audio_seq))
        audio_time_ns.append(int(slot.audio_time_ns))

    images = t.stack(frames, dim=0).unsqueeze(0).unsqueeze(0).contiguous()
    audio = t.stack(audio_chunks, dim=0).unsqueeze(0).contiguous()
    frame_seq_t = t.tensor(frame_seq, dtype=t.long).view(1, 1, len(slot_names))
    frame_time_ns_t = t.tensor(frame_time_ns, dtype=t.long).view(1, 1, len(slot_names))
    audio_seq_t = t.tensor(audio_seq, dtype=t.long).view(1, len(slot_names))
    audio_time_ns_t = t.tensor(audio_time_ns, dtype=t.long).view(1, len(slot_names))

    return Model3Batch(
        images=images,
        audio=audio,
        slot_names=list(slot_names),
        frame_seq=frame_seq_t,
        frame_time_ns=frame_time_ns_t,
        audio_seq=audio_seq_t,
        audio_time_ns=audio_time_ns_t,
    )


def decode_model3_frame_jpeg(frame_jpeg: bytes) -> torch.Tensor:
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


def decode_model3_audio_pcm(
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


@dataclass(slots=True)
class _SlotHistory:
    frames: deque[torch.Tensor] = field(default_factory=deque)
    audio_chunks: deque[torch.Tensor] = field(default_factory=deque)
    frame_seq: deque[int] = field(default_factory=deque)
    frame_time_ns: deque[int] = field(default_factory=deque)
    audio_seq: deque[int] = field(default_factory=deque)
    audio_time_ns: deque[int] = field(default_factory=deque)
    last_frame: torch.Tensor | None = None


class Model3WindowBuilder:
    """
    Builds model3-compatible rolling windows from websocket observations.

    Output layout matches the model training path:
    - images: [B=1, T, P=5, C, H, W]
    - audio:  [B=1, P=5, C=2, S=T*750] for 24kHz audio at 32 fps
    """

    def __init__(
        self,
        *,
        history_frames: int,
        slot_names: Sequence[str] = DEFAULT_SLOT_NAMES,
        audio_sample_rate: int = 24000,
        frame_rate_hz: int = 32,
    ) -> None:
        _require_torch()
        if history_frames <= 0:
            raise ValueError("history_frames must be positive")
        self.history_frames = int(history_frames)
        self.slot_names = list(slot_names)
        self.audio_sample_rate = int(audio_sample_rate)
        self.frame_rate_hz = int(frame_rate_hz)
        self.audio_samples_per_frame = max(1, self.audio_sample_rate // self.frame_rate_hz)
        self._history = {name: _SlotHistory() for name in self.slot_names}
        self._frame_shape_chw: tuple[int, int, int] | None = None

    @property
    def ready(self) -> bool:
        return all(len(self._history[name].frames) >= self.history_frames for name in self.slot_names)

    def push(self, observation: HarnessObservation) -> None:
        slot_map = {slot.name: slot for slot in observation.slots}
        for slot_name in self.slot_names:
            hist = self._history[slot_name]
            slot = slot_map.get(slot_name)

            frame = self._decode_or_fill_frame(slot, hist)
            audio = self._decode_or_fill_audio(slot)

            hist.frames.append(frame)
            hist.audio_chunks.append(audio)
            hist.frame_seq.append(int(slot.frame_seq) if slot is not None else 0)
            hist.frame_time_ns.append(int(slot.frame_time_ns) if slot is not None else 0)
            hist.audio_seq.append(int(slot.audio_seq) if slot is not None else 0)
            hist.audio_time_ns.append(int(slot.audio_time_ns) if slot is not None else 0)
            hist.last_frame = frame

            while len(hist.frames) > self.history_frames:
                hist.frames.popleft()
                hist.audio_chunks.popleft()
                hist.frame_seq.popleft()
                hist.frame_time_ns.popleft()
                hist.audio_seq.popleft()
                hist.audio_time_ns.popleft()

    def build_batch(self) -> Model3Batch:
        t = _require_torch()
        if not self.ready:
            raise RuntimeError(
                f"model3 window is not ready yet: need {self.history_frames} frames for {self.slot_names}"
            )

        frame_stacks = [t.stack(list(self._history[name].frames), dim=0) for name in self.slot_names]
        images = t.stack(frame_stacks, dim=1).unsqueeze(0).contiguous()

        audio_stacks = [t.cat(list(self._history[name].audio_chunks), dim=1) for name in self.slot_names]
        audio = t.stack(audio_stacks, dim=0).unsqueeze(0).contiguous()

        frame_seq = t.stack(
            [t.tensor(list(self._history[name].frame_seq), dtype=t.long) for name in self.slot_names],
            dim=1,
        ).unsqueeze(0)
        frame_time_ns = t.stack(
            [t.tensor(list(self._history[name].frame_time_ns), dtype=t.long) for name in self.slot_names],
            dim=1,
        ).unsqueeze(0)
        audio_seq = t.stack(
            [t.tensor(list(self._history[name].audio_seq), dtype=t.long) for name in self.slot_names],
            dim=1,
        ).unsqueeze(0)
        audio_time_ns = t.stack(
            [t.tensor(list(self._history[name].audio_time_ns), dtype=t.long) for name in self.slot_names],
            dim=1,
        ).unsqueeze(0)

        return Model3Batch(
            images=images,
            audio=audio,
            slot_names=list(self.slot_names),
            frame_seq=frame_seq,
            frame_time_ns=frame_time_ns,
            audio_seq=audio_seq,
            audio_time_ns=audio_time_ns,
        )

    def _decode_or_fill_frame(self, slot, hist: _SlotHistory) -> torch.Tensor:
        t = _require_torch()
        if slot is not None and slot.frame_jpeg:
            frame = decode_model3_frame_jpeg(slot.frame_jpeg)
            if self._frame_shape_chw is None:
                self._frame_shape_chw = tuple(frame.shape)
            return frame

        if hist.last_frame is not None:
            return hist.last_frame.clone()
        if self._frame_shape_chw is not None:
            return t.zeros(self._frame_shape_chw, dtype=t.uint8)
        raise RuntimeError("cannot build model3 frames before at least one real frame has been received")

    def _decode_or_fill_audio(self, slot) -> torch.Tensor:
        t = _require_torch()
        if slot is not None and slot.audio_pcm_s16le:
            return decode_model3_audio_pcm(
                slot.audio_pcm_s16le,
                channels=slot.audio_channels,
                expected_samples=self.audio_samples_per_frame,
            )
        return t.zeros((2, self.audio_samples_per_frame), dtype=t.float32)


def _require_torch():
    if torch is None:
        raise RuntimeError(
            "torch is required for model3 stream helpers. Install/use the model environment before importing "
            "or calling sim_harness.model3_helpers."
        )
    return torch
