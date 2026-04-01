from __future__ import annotations

from dataclasses import asdict, dataclass, field
import asyncio
import contextlib
import time
from typing import Any

import torch

from ..sim_harness.model3_helpers import observation_to_model3_batch
from ..sim_harness.remote_client import RemoteHarnessClient, RemoteHarnessStreamClient
from ..sim_harness.remote_protocol import HarnessObservation
from .actions import ModelActionDecoder
from .checkpoint import LoadedModel3, load_model3_checkpoint
from .config import Model3RuntimeConfig
from .overlay import RuntimeOverlayRenderer
from .recording import CompositeRecorder


@dataclass(slots=True)
class RuntimeMetrics:
    observations_received: int = 0
    inference_steps: int = 0
    actions_sent: int = 0
    last_observation_server_time_ns: int = 0
    last_inference_ms: float = 0.0
    avg_inference_ms: float = 0.0
    last_step_ms: float = 0.0
    avg_step_ms: float = 0.0
    last_timings_ms: dict[str, float] = field(default_factory=dict)
    avg_timings_ms: dict[str, float] = field(default_factory=dict)
    last_client_send_ns: int = 0
    last_error: str = ""


@dataclass(slots=True)
class RuntimeStateView:
    connected: bool
    running: bool
    harness_url: str
    checkpoint_path: str
    runtime_options: dict[str, Any]
    cache_window_frames: int | None
    cache: dict[str, Any]
    metrics: dict[str, Any]
    recording: dict[str, Any]
    last_actions: dict[str, list[dict[str, Any]]]
    last_action_summaries: dict[str, dict[str, Any]]


class Model3InferenceRuntime:
    def __init__(
        self,
        cfg: Model3RuntimeConfig,
        *,
        loaded_model: LoadedModel3 | None = None,
    ) -> None:
        self.cfg = cfg
        self.loaded_model = loaded_model or load_model3_checkpoint(
            cfg.checkpoint_path,
            device=cfg.device,
            data_root=cfg.data_root,
            enable_fast_vision_preprocess=cfg.enable_fast_vision_preprocess,
        )
        self.decoder = ModelActionDecoder(cfg=cfg.action_decode, slot_names=cfg.slot_names)
        self.renderer = RuntimeOverlayRenderer()
        self.recorder = CompositeRecorder(cfg.videos_dir, fps=cfg.recording_fps)

        self._client: RemoteHarnessStreamClient | None = None
        self._http_client: RemoteHarnessClient | None = None
        self._loop_task: asyncio.Task | None = None
        self._closing = False
        self._connected = False
        self._running = False
        self._lock = asyncio.Lock()

        self._latest_observation: HarnessObservation | None = None
        self._latest_prediction: dict[str, torch.Tensor] | None = None
        self._latest_actions: dict[str, list[dict[str, Any]]] = {}
        self._latest_action_summaries: dict[str, Any] = {}
        self._latest_composite_jpeg: bytes | None = None
        self._metrics = RuntimeMetrics()
        self._ar_state = self.loaded_model.backbone.init_autoregressive_state(
            max_cache_frames=cfg.cache_window_frames
        )

    async def connect(self) -> None:
        async with self._lock:
            if self._connected:
                return
            if self.cfg.transport == "stream":
                self._client = RemoteHarnessStreamClient(
                    base_url=self.cfg.harness_url,
                    verify_ssl=self.cfg.verify_ssl,
                )
            elif self.cfg.transport == "http":
                self._http_client = RemoteHarnessClient(
                    base_url=self.cfg.harness_url,
                    verify_ssl=self.cfg.verify_ssl,
                )
            else:
                raise ValueError(f"unsupported transport: {self.cfg.transport}")
        if self._client is not None:
            try:
                await self._client.connect()
            except Exception:
                async with self._lock:
                    self._client = None
                raise

        async with self._lock:
            self._connected = True
            self._closing = False
            self._loop_task = asyncio.create_task(self._run_loop(), name="model3-runtime-loop")

    async def disconnect(self) -> None:
        await self.pause()
        async with self._lock:
            self._closing = True
            loop_task = self._loop_task
            client = self._client
            http_client = self._http_client
            self._loop_task = None
            self._client = None
            self._http_client = None
            self._connected = False
            self._latest_composite_jpeg = None
        if client is not None:
            await client.close()
        if http_client is not None:
            await asyncio.to_thread(http_client.close)
        if loop_task is not None:
            loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await loop_task

    async def start(self) -> None:
        async with self._lock:
            if not self._connected:
                raise RuntimeError("runtime is not connected to a harness")
            self._running = True

    async def pause(self) -> None:
        async with self._lock:
            self._running = False
            client = self._client
            http_client = self._http_client
        if client is not None:
            release_actions = self.decoder.release_all()
            if release_actions:
                await client.send_actions(release_actions)
        elif http_client is not None:
            release_actions = self.decoder.release_all()
            if release_actions:
                await asyncio.to_thread(http_client.send_actions, release_actions)

    async def reset(self) -> None:
        await self.pause()
        async with self._lock:
            self._ar_state = self.loaded_model.backbone.reset_autoregressive_state(
                max_cache_frames=self.cfg.cache_window_frames
            )
            self._latest_prediction = None
            self._latest_actions = {}
            self._latest_action_summaries = {}
            self._refresh_composite_locked()

    async def set_cache_window_frames(self, cache_window_frames: int | None) -> None:
        async with self._lock:
            self.cfg.cache_window_frames = cache_window_frames
            self._ar_state.max_cache_frames = cache_window_frames
            self.loaded_model.backbone.crop_autoregressive_state(
                self._ar_state,
                max_cache_frames=cache_window_frames,
            )
            self._refresh_composite_locked()

    async def set_harness_url(self, harness_url: str) -> None:
        async with self._lock:
            was_connected = self._connected
            self.cfg.harness_url = harness_url
        if was_connected:
            await self.disconnect()

    async def start_recording(self, path: str | None = None) -> str:
        async with self._lock:
            recording_path = self.recorder.start(path)
            if self._latest_composite_jpeg is not None:
                self.recorder.write_jpeg(self._latest_composite_jpeg)
            self._refresh_composite_locked()
            return recording_path

    async def stop_recording(self) -> str:
        async with self._lock:
            path = self.recorder.stop()
            self._refresh_composite_locked()
            return path

    async def latest_composite_jpeg(self) -> bytes | None:
        async with self._lock:
            return self._latest_composite_jpeg

    async def snapshot(self) -> RuntimeStateView:
        async with self._lock:
            cache = {
                "tokens_per_frame": int(self._ar_state.tokens_per_frame),
                "cached_tokens": int(self._ar_state.cached_tokens),
                "cached_frames": int(self._ar_state.cached_frames),
                "total_frames_processed": int(self._ar_state.total_frames_processed),
                "max_cache_frames": self._ar_state.max_cache_frames,
            }
            return RuntimeStateView(
                connected=self._connected,
                running=self._running,
                harness_url=self.cfg.harness_url,
                checkpoint_path=self.loaded_model.checkpoint_path,
                runtime_options={
                    "fast_vision_preprocess": bool(self.loaded_model.backbone.video.fast_preprocess_enabled),
                },
                cache_window_frames=self.cfg.cache_window_frames,
                cache=cache,
                metrics=asdict(self._metrics),
                recording=asdict(self.recorder.status()),
                last_actions=dict(self._latest_actions),
                last_action_summaries={
                    slot_name: summary.to_dict()
                    for slot_name, summary in self._latest_action_summaries.items()
                },
            )

    async def latest_logits(self, player_index: int) -> dict[str, Any]:
        async with self._lock:
            if self._latest_prediction is None:
                return {}
            if not 0 <= player_index < len(self.cfg.slot_names):
                raise IndexError(f"player_index must be in [0, {len(self.cfg.slot_names) - 1}]")

            logits: dict[str, Any] = {}
            for head_name, tensor in self._latest_prediction.items():
                if tensor.ndim < 4:
                    continue
                if tensor.shape[2] == len(self.cfg.slot_names):
                    logits[head_name] = tensor[0, 0, player_index].tolist()
                elif tensor.shape[2] == 1:
                    logits[head_name] = tensor[0, 0, 0].tolist()
            return logits

    async def _run_loop(self) -> None:
        if self._http_client is not None:
            await self._run_http_loop()
            return
        try:
            while True:
                client = self._client
                if client is None:
                    return
                observation = await client.recv_observation()
                async with self._lock:
                    self._latest_observation = observation
                    self._metrics.observations_received += 1
                    self._metrics.last_observation_server_time_ns = int(observation.server_time_ns)
                    should_run = self._running
                    self._refresh_composite_locked()
                if not should_run:
                    continue

                prediction, actions, summaries, timings_ms = await asyncio.to_thread(
                    self._infer_observation,
                    observation,
                )
                async with self._lock:
                    still_running = self._running and self._client is client
                if not still_running:
                    continue

                send_ns = 0
                if actions:
                    send_ns = await client.send_actions(actions)

                async with self._lock:
                    self._apply_inference_result_locked(
                        prediction,
                        actions,
                        summaries,
                        timings_ms,
                        send_ns=send_ns,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            async with self._lock:
                self._metrics.last_error = str(exc)
                self._running = False
                self._connected = False

    async def _run_http_loop(self) -> None:
        try:
            while True:
                http_client = self._http_client
                if http_client is None:
                    return
                observation = await asyncio.to_thread(http_client.get_observation)
                async with self._lock:
                    self._latest_observation = observation
                    self._metrics.observations_received += 1
                    self._metrics.last_observation_server_time_ns = int(observation.server_time_ns)
                    should_run = self._running
                    self._refresh_composite_locked()
                if should_run:
                    prediction, actions, summaries, timings_ms = await asyncio.to_thread(
                        self._infer_observation,
                        observation,
                    )
                    send_ns = 0
                    if actions:
                        send_ns = time.time_ns()
                        await asyncio.to_thread(http_client.send_actions, actions)
                    async with self._lock:
                        still_running = self._running and self._http_client is http_client
                        if not still_running:
                            continue
                        self._apply_inference_result_locked(
                            prediction,
                            actions,
                            summaries,
                            timings_ms,
                            send_ns=send_ns,
                        )
                await asyncio.sleep(max(0.0, float(self.cfg.poll_interval_s)))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            async with self._lock:
                self._metrics.last_error = str(exc)
                self._running = False
                self._connected = False
                self._refresh_composite_locked()
                self._refresh_composite_locked()

    def _infer_observation(
        self,
        observation: HarnessObservation,
    ) -> tuple[dict[str, torch.Tensor], dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, float]]:
        total_start = time.perf_counter()

        decode_start = total_start
        batch = observation_to_model3_batch(
            observation,
            slot_names=self.cfg.slot_names,
            audio_sample_rate=int(self.loaded_model.global_cfg.dataset.audio_sample_rate),
            frame_rate_hz=32,
        )
        decode_ms = (time.perf_counter() - decode_start) * 1000.0

        prepare_start = time.perf_counter()
        images, audio = self.loaded_model.prepare_batch_tensors(batch.images, batch.audio)
        prepare_ms = (time.perf_counter() - prepare_start) * 1000.0

        self._synchronize_model_device()
        model_start = time.perf_counter()
        with torch.inference_mode():
            prediction, self._ar_state = self.loaded_model.backbone.forward_step(
                images,
                audio,
                self._ar_state,
                max_cache_frames=self.cfg.cache_window_frames,
            )
        self._synchronize_model_device()
        model_ms = (time.perf_counter() - model_start) * 1000.0

        transfer_start = time.perf_counter()
        prediction_cpu = {
            head_name: tensor.detach().float().cpu()
            for head_name, tensor in prediction.items()
        }
        transfer_ms = (time.perf_counter() - transfer_start) * 1000.0

        action_decode_start = time.perf_counter()
        actions, summaries = self.decoder.decode(
            prediction_cpu,
            model_cfg=self.loaded_model.global_cfg.model,
        )
        action_decode_ms = (time.perf_counter() - action_decode_start) * 1000.0
        end_to_end_ms = (time.perf_counter() - total_start) * 1000.0

        timings_ms = {
            "decode_ms": decode_ms,
            "prepare_ms": prepare_ms,
            "model_ms": model_ms,
            "transfer_ms": transfer_ms,
            "action_decode_ms": action_decode_ms,
            "end_to_end_ms": end_to_end_ms,
        }
        return prediction_cpu, actions, summaries, timings_ms

    def _synchronize_model_device(self) -> None:
        if self.loaded_model.device.type == "cuda":
            torch.cuda.synchronize(device=self.loaded_model.device)

    def _apply_inference_result_locked(
        self,
        prediction: dict[str, torch.Tensor],
        actions: dict[str, list[dict[str, Any]]],
        summaries: dict[str, Any],
        timings_ms: dict[str, float],
        *,
        send_ns: int,
    ) -> None:
        self._latest_prediction = prediction
        self._latest_actions = actions
        self._latest_action_summaries = summaries
        self._metrics.inference_steps += 1
        count = self._metrics.inference_steps

        normalized_timings = {key: float(value) for key, value in timings_ms.items()}
        self._metrics.last_timings_ms = normalized_timings
        if count == 1 or not self._metrics.avg_timings_ms:
            self._metrics.avg_timings_ms = dict(normalized_timings)
        else:
            for key, value in normalized_timings.items():
                prev = float(self._metrics.avg_timings_ms.get(key, value))
                self._metrics.avg_timings_ms[key] = prev + (value - prev) / count

        model_ms = float(normalized_timings.get("model_ms", 0.0))
        step_ms = float(normalized_timings.get("end_to_end_ms", 0.0))
        self._metrics.last_inference_ms = model_ms
        self._metrics.last_step_ms = step_ms
        if count == 1:
            self._metrics.avg_inference_ms = model_ms
            self._metrics.avg_step_ms = step_ms
        else:
            prev_model = self._metrics.avg_inference_ms
            prev_step = self._metrics.avg_step_ms
            self._metrics.avg_inference_ms = prev_model + (model_ms - prev_model) / count
            self._metrics.avg_step_ms = prev_step + (step_ms - prev_step) / count

        self._metrics.actions_sent += sum(len(events) for events in actions.values())
        self._metrics.last_client_send_ns = int(send_ns)
        self._refresh_composite_locked()

    def _refresh_composite_locked(self) -> None:
        if self._latest_observation is None:
            return
        try:
            state = {
                "connected": self._connected,
                "running": self._running,
                "recording": self.recorder.status().active,
            }
            composite = self.renderer.render_jpeg(
                observation=self._latest_observation,
                slot_names=self.cfg.slot_names,
                action_summaries={
                    slot_name: summary.to_dict()
                    for slot_name, summary in self._latest_action_summaries.items()
                },
                cache={
                    "tokens_per_frame": int(self._ar_state.tokens_per_frame),
                    "cached_tokens": int(self._ar_state.cached_tokens),
                    "cached_frames": int(self._ar_state.cached_frames),
                    "total_frames_processed": int(self._ar_state.total_frames_processed),
                    "max_cache_frames": self._ar_state.max_cache_frames,
                },
                metrics=asdict(self._metrics),
                status=state,
                checkpoint_path=self.loaded_model.checkpoint_path,
            )
            self._latest_composite_jpeg = composite
            if self.recorder.status().active:
                self.recorder.write_jpeg(composite)
        except Exception as exc:
            self._metrics.last_error = str(exc)
