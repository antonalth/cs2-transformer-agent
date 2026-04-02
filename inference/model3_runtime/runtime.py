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
from .actions import KEYBOARD_ONLY_ACTIONS, ModelActionDecoder, keyboard_action_labels, mu_law_decode
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


ROUND_STATE_LABELS = ("freeze", "live", "plant", "t_win", "ct_win")
POSITION_MIN = -4096.0
POSITION_MAX = 4096.0


def _format_axis_value(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _linear_axis_labels(min_value: float, max_value: float, count: int) -> list[str]:
    if count <= 1:
        return [_format_axis_value(min_value)]
    step = (max_value - min_value) / float(count - 1)
    return [_format_axis_value(min_value + (step * idx)) for idx in range(count)]


def _index_axis_labels(count: int, *, prefix: str = "") -> list[str]:
    return [f"{prefix}{idx}" for idx in range(count)]


def _mouse_axis_labels(model_cfg: object) -> list[str]:
    bins = int(model_cfg.mouse_bins_count)
    indices = torch.arange(bins)
    decoded = mu_law_decode(
        indices,
        mu=float(model_cfg.mouse_mu),
        max_val=float(model_cfg.mouse_max),
        bins=bins,
    )
    return [_format_axis_value(float(value)) for value in decoded.tolist()]


def _head_axis_labels(head_name: str, count: int, model_cfg: object) -> list[str]:
    if head_name == "keyboard_logits":
        return keyboard_action_labels(count)
    if head_name in {"mouse_x", "mouse_y"}:
        return _mouse_axis_labels(model_cfg)
    if head_name in {"health_logits", "armor_logits"}:
        return _linear_axis_labels(0.0, 100.0, count)
    if head_name == "money_logits":
        return _linear_axis_labels(0.0, 16000.0, count)
    if head_name in {"player_pos_x", "player_pos_y", "player_pos_z", "enemy_pos_x", "enemy_pos_y", "enemy_pos_z"}:
        return _linear_axis_labels(POSITION_MIN, POSITION_MAX, count)
    if head_name == "round_state_logits":
        return list(ROUND_STATE_LABELS[:count])
    if head_name == "round_num_logits":
        return _index_axis_labels(count)
    if head_name in {"team_alive_logits", "enemy_alive_logits"}:
        return _index_axis_labels(count)
    if head_name == "eco_purchase_logits":
        return ["no", "yes"]
    if head_name == "active_weapon_logits":
        return _index_axis_labels(count, prefix="weapon ")
    if head_name == "eco_buy_logits":
        return _index_axis_labels(count, prefix="item ")
    return _index_axis_labels(count)


def _top_categories(labels: list[str], probabilities: list[float], limit: int = 5) -> list[dict[str, Any]]:
    ranked = sorted(
        (
            {"label": str(label), "probability": float(probability), "index": int(idx)}
            for idx, (label, probability) in enumerate(zip(labels, probabilities))
        ),
        key=lambda item: item["probability"],
        reverse=True,
    )
    return ranked[:limit]


def _binary_panel(head_name: str, logit_value: float, model_cfg: object) -> dict[str, Any]:
    yes_probability = float(torch.sigmoid(torch.tensor(logit_value)).item())
    probabilities = [1.0 - yes_probability, yes_probability]
    labels = _head_axis_labels(head_name, 2, model_cfg)
    predicted_index = int(yes_probability >= 0.5)
    full_categories = [
        {
            "label": str(label),
            "probability": float(probability),
            "index": int(idx),
        }
        for idx, (label, probability) in enumerate(zip(labels, probabilities))
    ]
    return {
        "head_name": head_name,
        "kind": "binary",
        "representation": "sigmoid",
        "x_labels": labels,
        "probabilities": probabilities,
        "raw_logits": [0.0, float(logit_value)],
        "predicted_index": predicted_index,
        "predicted_label": labels[predicted_index],
        "predicted_probability": float(probabilities[predicted_index]),
        "top_categories": _top_categories(labels, probabilities, limit=2),
        "full_categories": full_categories,
    }


def _categorical_panel(head_name: str, values: torch.Tensor, model_cfg: object) -> dict[str, Any]:
    probabilities = torch.softmax(values, dim=0)
    labels = _head_axis_labels(head_name, int(values.numel()), model_cfg)
    predicted_index = int(torch.argmax(probabilities).item())
    probs_list = [float(value) for value in probabilities.tolist()]
    full_categories = [
        {
            "label": str(label),
            "probability": float(probability),
            "index": int(idx),
        }
        for idx, (label, probability) in enumerate(zip(labels, probs_list))
    ]
    return {
        "head_name": head_name,
        "kind": "categorical",
        "representation": "softmax",
        "x_labels": labels,
        "probabilities": probs_list,
        "raw_logits": [float(value) for value in values.tolist()],
        "predicted_index": predicted_index,
        "predicted_label": labels[predicted_index],
        "predicted_probability": float(probs_list[predicted_index]),
        "top_categories": _top_categories(labels, probs_list),
        "full_categories": full_categories,
    }


def _keyboard_panel(values: torch.Tensor, keyboard_threshold: float) -> dict[str, Any]:
    probabilities = torch.sigmoid(values)
    probs_list = [float(value) for value in probabilities.tolist()]
    labels = keyboard_action_labels(len(probs_list))
    active_actions = [
        label
        for label, probability in zip(labels, probs_list)
        if probability >= keyboard_threshold
    ]
    full_categories = [
        {
            "label": str(label),
            "probability": float(probability),
            "index": int(idx),
        }
        for idx, (label, probability) in enumerate(zip(labels, probs_list))
    ]
    return {
        "head_name": "keyboard_logits",
        "kind": "bernoulli",
        "representation": "sigmoid",
        "x_labels": labels,
        "probabilities": probs_list,
        "raw_logits": [float(value) for value in values.tolist()],
        "active_actions": active_actions,
        "threshold": float(keyboard_threshold),
        "top_categories": _top_categories(labels, probs_list),
        "full_categories": full_categories,
    }


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

    async def set_action_decode_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            cfg = self.cfg.action_decode
            if "keyboard_threshold" in payload:
                cfg.keyboard_threshold = float(payload["keyboard_threshold"])
            if "mouse_deadzone" in payload:
                cfg.mouse_deadzone = float(payload["mouse_deadzone"])
            if "mouse_scale" in payload:
                cfg.mouse_scale = float(payload["mouse_scale"])
            if "max_mouse_delta" in payload:
                cfg.max_mouse_delta = float(payload["max_mouse_delta"])
            if "mouse_zero_bias" in payload:
                cfg.mouse_zero_bias = float(payload["mouse_zero_bias"])
            if "mouse_temperature" in payload:
                cfg.mouse_temperature = max(1e-4, float(payload["mouse_temperature"]))
            if "mouse_top_k" in payload:
                cfg.mouse_top_k = max(1, int(payload["mouse_top_k"]))
            return asdict(cfg)

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
                    "action_decode": asdict(self.cfg.action_decode),
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

            panels: list[dict[str, Any]] = []
            model_cfg = self.loaded_model.global_cfg.model
            player_name = self.cfg.slot_names[player_index]
            for head_name, tensor in self._latest_prediction.items():
                if tensor.ndim < 4:
                    continue
                if head_name in {"enemy_pos_x", "enemy_pos_y", "enemy_pos_z"} and tensor.shape[2] == 5:
                    for enemy_index in range(5):
                        values = tensor[0, 0, enemy_index].detach().float().cpu().view(-1)
                        panel = _categorical_panel(f"{head_name}[enemy {enemy_index}]", values, model_cfg)
                        panels.append(panel)
                    continue
                selected: torch.Tensor | None = None
                if tensor.shape[2] == len(self.cfg.slot_names):
                    selected = tensor[0, 0, player_index]
                elif tensor.shape[2] == 1:
                    selected = tensor[0, 0, 0]
                if selected is None:
                    continue

                values = selected.detach().float().cpu().view(-1)
                if head_name == "keyboard_logits":
                    panels.append(_keyboard_panel(values, float(self.cfg.action_decode.keyboard_threshold)))
                elif values.numel() == 1:
                    panels.append(_binary_panel(head_name, float(values[0].item()), model_cfg))
                else:
                    panels.append(_categorical_panel(head_name, values, model_cfg))
            return {
                "player_index": int(player_index),
                "player_name": player_name,
                "action_decode": asdict(self.cfg.action_decode),
                "action_summary": self._latest_action_summaries[player_name].to_dict()
                if player_name in self._latest_action_summaries
                else {},
                "panels": panels,
            }

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
