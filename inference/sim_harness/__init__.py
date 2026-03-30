"""Host-side CS2 simulation harness."""

from .config import HarnessConfig, load_config
from .remote_client import RemoteHarnessClient, RemoteHarnessStreamClient
from .remote_protocol import HarnessObservation, SlotObservation, StreamMessage
from .supervisor import HarnessSupervisor

try:
    from .model3_helpers import (
        Model3Batch,
        Model3WindowBuilder,
        decode_model3_audio_pcm,
        decode_model3_frame_jpeg,
        observation_to_model3_batch,
    )
except Exception:
    Model3Batch = None
    Model3WindowBuilder = None
    decode_model3_audio_pcm = None
    decode_model3_frame_jpeg = None
    observation_to_model3_batch = None

__all__ = [
    "HarnessConfig",
    "HarnessObservation",
    "HarnessSupervisor",
    "Model3Batch",
    "Model3WindowBuilder",
    "RemoteHarnessClient",
    "RemoteHarnessStreamClient",
    "SlotObservation",
    "StreamMessage",
    "decode_model3_audio_pcm",
    "decode_model3_frame_jpeg",
    "observation_to_model3_batch",
    "load_config",
]
