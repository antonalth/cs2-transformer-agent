from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import asyncio
import contextlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from ..sim_harness.model3_helpers import observation_to_model3_batch
from ..sim_harness.remote_client import RemoteHarnessClient, RemoteHarnessStreamClient
from ..sim_harness.remote_protocol import HarnessObservation
from .actions import KEYBOARD_ONLY_ACTIONS, ModelActionDecoder, keyboard_action_labels, mu_law_decode
from .checkpoint import LoadedModel3, ensure_model3_import_path, load_model3_checkpoint
from .config import Model3RuntimeConfig
from .overlay import RuntimeOverlayRenderer
from .recording import CaptureVideoRecorder, CompositeRecorder


@dataclass(slots=True)
class RuntimeMetrics:
    observations_received: int = 0
    inference_steps: int = 0
    actions_sent: int = 0
    skipped_observations: int = 0
    skipped_frames: int = 0
    last_processed_frame_seq: int = 0
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
    capture: dict[str, Any]
    decoder_calibration: dict[str, Any]
    last_actions: dict[str, list[dict[str, Any]]]
    last_action_summaries: dict[str, dict[str, Any]]
    calibration: dict[str, Any]
    ui_settings: dict[str, Any]


@dataclass(slots=True)
class MouseCalibrationView:
    running: bool = False
    updated_at: float = 0.0
    last_error: str = ""
    last_result: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DecoderCalibrationView:
    running: bool = False
    updated_at: float = 0.0
    started_at: float = 0.0
    current_phase: str = ""
    progress_current: int = 0
    progress_total: int = 0
    eta_seconds: float | None = None
    last_error: str = ""
    last_result: dict[str, Any] = field(default_factory=dict)


DEFAULT_DECODER_CALIBRATION_KEYBOARD_FN_COST_OVERRIDES = {
    "IN_ATTACK": 3.0,
    "IN_ATTACK2": 2.0,
    "IN_ATTACK3": 1.5,
    "IN_JUMP": 1.4,
    "IN_DUCK": 1.6,
    "IN_FORWARD": 2.6,
    "IN_BACK": 1.8,
    "IN_USE": 1.4,
    "IN_MOVELEFT": 2.4,
    "IN_MOVERIGHT": 2.4,
    "IN_RELOAD": 1.6,
    "IN_WALK": 1.3,
    "SWITCH_1": 2.0,
    "SWITCH_2": 2.0,
    "SWITCH_3": 2.0,
    "SWITCH_4": 1.8,
    "SWITCH_5": 1.8,
}
DEFAULT_DECODER_CALIBRATION_KEYBOARD_FP_COST_OVERRIDES = {
    "IN_SCORE": 3.0,
    "IN_INSPECT": 2.0,
    "IN_TURNLEFT": 2.5,
    "IN_TURNRIGHT": 2.5,
    "IN_BULLRUSH": 2.0,
}
DECODER_KEYBOARD_METRICS = ("cost_weighted_accuracy", "balanced_accuracy", "f1")
DECODER_MOUSE_METRICS = ("balanced_accuracy", "f1")


@dataclass(slots=True)
class DecoderCalibrationSettings:
    max_games: int = 4
    max_samples: int = 8
    samples_per_game: int = 2
    frames_per_sample: int = 128
    keyboard_metric: str = "cost_weighted_accuracy"
    keyboard_false_positive_cost: float = 1.0
    keyboard_false_negative_cost: float = 1.0
    keyboard_false_positive_cost_overrides: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_DECODER_CALIBRATION_KEYBOARD_FP_COST_OVERRIDES)
    )
    keyboard_false_negative_cost_overrides: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_DECODER_CALIBRATION_KEYBOARD_FN_COST_OVERRIDES)
    )
    mouse_metric: str = "balanced_accuracy"
    mouse_move_weight: float = 0.7
    mouse_stochastic_sampling: bool = False


@dataclass(slots=True)
class RuntimeUiSettings:
    browser_fps: float = 2.0
    player_index: int = 0
    capture_player_index: int = 0
    decoder_calibration: DecoderCalibrationSettings = field(default_factory=DecoderCalibrationSettings)


def _capture_slot_summary(slot: Any) -> dict[str, Any]:
    return {
        "name": str(slot.name),
        "status": str(slot.status),
        "error": slot.error,
        "frame_seq": int(slot.frame_seq),
        "frame_time_ns": int(slot.frame_time_ns),
        "audio_seq": int(slot.audio_seq),
        "audio_time_ns": int(slot.audio_time_ns),
    }

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


def _unwrap_delta_degrees(delta: float) -> float:
    value = float(delta)
    while value <= -180.0:
        value += 360.0
    while value > 180.0:
        value -= 360.0
    return value


def _fit_linear_axis(samples: list[tuple[float, float]]) -> dict[str, float]:
    if len(samples) < 2:
        raise RuntimeError("need at least two calibration samples")
    xs = [float(x) for x, _ in samples]
    ys = [float(y) for _, y in samples]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if abs(denom) < 1e-9:
        raise RuntimeError("calibration samples are degenerate")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    intercept = mean_y - (slope * mean_x)
    residuals = [y - ((slope * x) + intercept) for x, y in samples]
    mean_abs_error = sum(abs(value) for value in residuals) / len(residuals)
    max_abs_error = max(abs(value) for value in residuals)
    return {
        "slope": slope,
        "intercept": intercept,
        "mean_abs_error": mean_abs_error,
        "max_abs_error": max_abs_error,
    }


def _predict_linear_axis(fit: dict[str, float], x_value: float) -> float:
    return float(fit["slope"]) * float(x_value) + float(fit["intercept"])


def _observation_max_frame_seq(observation: HarnessObservation) -> int:
    if not observation.slots:
        return 0
    return max(int(slot.frame_seq) for slot in observation.slots)


def _scaled_mouse_input(
    raw_delta: float,
    *,
    global_scale: float,
    axis_scale: float,
) -> tuple[float, bool]:
    scaled = float(raw_delta) * float(global_scale) * float(axis_scale)
    return scaled, False


def _summarize_angle_validation(results: list[dict[str, Any]]) -> dict[str, float | int]:
    if not results:
        return {
            "count": 0,
            "mean_abs_error": 0.0,
            "max_abs_error": 0.0,
            "max_abs_measured_angle": 0.0,
            "clamped_count": 0,
        }
    abs_errors = [abs(float(item.get("angle_error", 0.0))) for item in results]
    abs_measured = [abs(float(item.get("measured_angle", 0.0))) for item in results]
    clamped_count = sum(1 for item in results if bool(item.get("clamped", False)))
    return {
        "count": len(results),
        "mean_abs_error": sum(abs_errors) / len(abs_errors),
        "max_abs_error": max(abs_errors),
        "max_abs_measured_angle": max(abs_measured),
        "clamped_count": clamped_count,
    }


def _binary_confusion_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float | int]:
    pred_bool = pred.to(dtype=torch.bool)
    gt_bool = gt.to(dtype=torch.bool)
    tp = int((pred_bool & gt_bool).sum().item())
    fp = int((pred_bool & (~gt_bool)).sum().item())
    tn = int(((~pred_bool) & (~gt_bool)).sum().item())
    fn = int(((~pred_bool) & gt_bool).sum().item())
    total = max(1, tp + fp + tn + fn)
    positives = max(1, tp + fn)
    negatives = max(1, tn + fp)
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / positives)
    tpr = recall
    tnr = float(tn / negatives)
    fpr = float(fp / negatives)
    fnr = float(fn / positives)
    if precision + recall > 0.0:
        f1 = float((2.0 * precision * recall) / (precision + recall))
    else:
        f1 = 0.0
    balanced_accuracy = 0.5 * (tpr + tnr)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": float((tp + tn) / total),
        "precision": precision,
        "recall": recall,
        "tpr": tpr,
        "tnr": tnr,
        "fpr": fpr,
        "fnr": fnr,
        "f1": f1,
        "balanced_accuracy": float(balanced_accuracy),
        "avg_class_accuracy": float(balanced_accuracy),
        "pred_rate": float(pred_bool.float().mean().item()),
        "gt_rate": float(gt_bool.float().mean().item()),
    }


def _decoder_metric_score(
    metric_name: str,
    metrics: dict[str, Any],
    *,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 1.0,
) -> float:
    if metric_name == "f1":
        return float(metrics["f1"])
    if metric_name == "cost_weighted_accuracy":
        fp_cost = max(0.0, float(false_positive_cost))
        fn_cost = max(0.0, float(false_negative_cost))
        denom = max(1e-6, fp_cost + fn_cost)
        weighted_error = ((fn_cost * float(metrics["fnr"])) + (fp_cost * float(metrics["fpr"]))) / denom
        return max(0.0, 1.0 - weighted_error)
    return float(metrics["balanced_accuracy"])


def _soft_confusion_metrics(pred_positive_prob: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    gt_bool = gt.to(dtype=torch.bool)
    pred_positive_prob = pred_positive_prob.float().clamp(0.0, 1.0)
    pred_negative_prob = 1.0 - pred_positive_prob
    tp = float(pred_positive_prob[gt_bool].sum().item())
    fn = float(pred_negative_prob[gt_bool].sum().item())
    fp = float(pred_positive_prob[~gt_bool].sum().item())
    tn = float(pred_negative_prob[~gt_bool].sum().item())
    positives = max(1e-6, tp + fn)
    negatives = max(1e-6, tn + fp)
    total = max(1e-6, tp + fp + tn + fn)
    precision = float(tp / max(1e-6, tp + fp))
    recall = float(tp / positives)
    tpr = recall
    tnr = float(tn / negatives)
    fpr = float(fp / negatives)
    fnr = float(fn / positives)
    if precision + recall > 0.0:
        f1 = float((2.0 * precision * recall) / (precision + recall))
    else:
        f1 = 0.0
    balanced_accuracy = 0.5 * (tpr + tnr)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": float((tp + tn) / total),
        "precision": precision,
        "recall": recall,
        "tpr": tpr,
        "tnr": tnr,
        "fpr": fpr,
        "fnr": fnr,
        "f1": f1,
        "balanced_accuracy": float(balanced_accuracy),
        "avg_class_accuracy": float(balanced_accuracy),
        "pred_rate": float(pred_positive_prob.mean().item()),
        "gt_rate": float(gt_bool.float().mean().item()),
    }


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


def _keyboard_panel(
    values: torch.Tensor,
    keyboard_threshold: float,
    keyboard_thresholds: dict[str, float],
) -> dict[str, Any]:
    probabilities = torch.sigmoid(values)
    probs_list = [float(value) for value in probabilities.tolist()]
    labels = keyboard_action_labels(len(probs_list))
    resolved_thresholds = [
        float(keyboard_thresholds.get(label, keyboard_threshold))
        for label in labels
    ]
    active_actions = [
        label
        for label, probability, threshold in zip(labels, probs_list, resolved_thresholds)
        if probability >= threshold
    ]
    full_categories = [
        {
            "label": str(label),
            "probability": float(probability),
            "index": int(idx),
            "threshold": float(threshold),
        }
        for idx, (label, probability, threshold) in enumerate(zip(labels, probs_list, resolved_thresholds))
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
        "thresholds": {label: float(threshold) for label, threshold in zip(labels, resolved_thresholds)},
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
        self.capture_video = CaptureVideoRecorder(
            cfg.videos_dir,
            fps=cfg.capture_video_fps,
            crf=cfg.capture_video_crf,
        )

        self._client: RemoteHarnessStreamClient | None = None
        self._http_client: RemoteHarnessClient | None = None
        self._loop_task: asyncio.Task | None = None
        self._decoder_calibration_task: asyncio.Task | None = None
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
        self._mouse_calibration = MouseCalibrationView(updated_at=time.time())
        self._decoder_calibration = DecoderCalibrationView(updated_at=time.time())
        self._ui_settings = RuntimeUiSettings(browser_fps=float(cfg.browser_fps))
        self._capture_active = False
        self._capture_started_at = 0.0
        self._capture_stopped_at = 0.0
        self._capture_frames: list[dict[str, Any]] = []
        self._last_processed_frame_seq = -1
        self._ar_state = self.loaded_model.backbone.init_autoregressive_state(
            max_cache_frames=cfg.cache_window_frames
        )
        self._load_persisted_settings()

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
            self._last_processed_frame_seq = -1
            self._metrics.last_processed_frame_seq = 0
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
            self._last_processed_frame_seq = -1
            self._metrics.last_processed_frame_seq = 0
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
            self._last_processed_frame_seq = -1
            self._metrics.last_processed_frame_seq = 0
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
            self._persist_settings_locked()

    async def set_harness_url(self, harness_url: str) -> None:
        async with self._lock:
            was_connected = self._connected
            self.cfg.harness_url = harness_url
            self._persist_settings_locked()
        if was_connected:
            await self.disconnect()

    def _apply_action_decode_payload_locked(self, payload: dict[str, Any]) -> dict[str, Any]:
        cfg = self.cfg.action_decode
        if "keyboard_threshold" in payload:
            cfg.keyboard_threshold = float(payload["keyboard_threshold"])
        if "keyboard_thresholds" in payload:
            thresholds_payload = payload["keyboard_thresholds"]
            if not isinstance(thresholds_payload, dict):
                raise TypeError("keyboard_thresholds must be an object mapping action names to thresholds")
            normalized_thresholds: dict[str, float] = {}
            valid_labels = set(KEYBOARD_ONLY_ACTIONS)
            for key, value in thresholds_payload.items():
                action = str(key)
                if action not in valid_labels:
                    continue
                normalized_thresholds[action] = float(value)
            cfg.keyboard_thresholds = normalized_thresholds
        if "mouse_deadzone" in payload:
            cfg.mouse_deadzone = float(payload["mouse_deadzone"])
        if "mouse_scale" in payload:
            cfg.mouse_scale = float(payload["mouse_scale"])
        if "mouse_scale_x" in payload:
            cfg.mouse_scale_x = float(payload["mouse_scale_x"])
        if "mouse_scale_y" in payload:
            cfg.mouse_scale_y = float(payload["mouse_scale_y"])
        if "mouse_zero_bias" in payload:
            cfg.mouse_zero_bias = float(payload["mouse_zero_bias"])
        if "mouse_temperature" in payload:
            cfg.mouse_temperature = max(1e-4, float(payload["mouse_temperature"]))
        if "mouse_top_k" in payload:
            cfg.mouse_top_k = max(1, int(payload["mouse_top_k"]))
        return asdict(cfg)

    @staticmethod
    def _normalize_decoder_calibration_ui_settings(payload: dict[str, Any]) -> DecoderCalibrationSettings:
        settings = DecoderCalibrationSettings()
        if "max_games" in payload:
            settings.max_games = max(1, int(payload["max_games"]))
        if "max_samples" in payload:
            settings.max_samples = max(1, int(payload["max_samples"]))
        if "samples_per_game" in payload:
            settings.samples_per_game = max(1, int(payload["samples_per_game"]))
        if "frames_per_sample" in payload:
            settings.frames_per_sample = max(8, int(payload["frames_per_sample"]))
        if "keyboard_metric" in payload:
            metric = str(payload["keyboard_metric"]).strip()
            if metric not in DECODER_KEYBOARD_METRICS:
                raise ValueError(f"keyboard_metric must be one of {DECODER_KEYBOARD_METRICS}")
            settings.keyboard_metric = metric
        if "keyboard_false_positive_cost" in payload:
            settings.keyboard_false_positive_cost = max(0.0, float(payload["keyboard_false_positive_cost"]))
        if "keyboard_false_negative_cost" in payload:
            settings.keyboard_false_negative_cost = max(0.0, float(payload["keyboard_false_negative_cost"]))
        if "keyboard_false_positive_cost_overrides" in payload:
            settings.keyboard_false_positive_cost_overrides = Model3InferenceRuntime._normalize_keyboard_cost_overrides(
                payload,
                field_name="keyboard_false_positive_cost_overrides",
            )
        else:
            settings.keyboard_false_positive_cost_overrides = dict(DEFAULT_DECODER_CALIBRATION_KEYBOARD_FP_COST_OVERRIDES)
        if "keyboard_false_negative_cost_overrides" in payload:
            settings.keyboard_false_negative_cost_overrides = Model3InferenceRuntime._normalize_keyboard_cost_overrides(
                payload,
                field_name="keyboard_false_negative_cost_overrides",
            )
        else:
            settings.keyboard_false_negative_cost_overrides = dict(DEFAULT_DECODER_CALIBRATION_KEYBOARD_FN_COST_OVERRIDES)
        if "mouse_metric" in payload:
            metric = str(payload["mouse_metric"]).strip()
            if metric not in DECODER_MOUSE_METRICS:
                raise ValueError(f"mouse_metric must be one of {DECODER_MOUSE_METRICS}")
            settings.mouse_metric = metric
        if "mouse_move_weight" in payload:
            settings.mouse_move_weight = min(0.99, max(0.01, float(payload["mouse_move_weight"])))
        if "mouse_stochastic_sampling" in payload:
            settings.mouse_stochastic_sampling = bool(payload["mouse_stochastic_sampling"])
        return settings

    def _export_settings_locked(self) -> dict[str, Any]:
        return {
            "version": 1,
            "harness_url": self.cfg.harness_url,
            "cache_window_frames": self.cfg.cache_window_frames,
            "action_decode": asdict(self.cfg.action_decode),
            "ui_settings": asdict(self._ui_settings),
        }

    def _persist_settings_locked(self) -> None:
        settings_path = Path(self.cfg.settings_path)
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(self._export_settings_locked(), indent=2, sort_keys=True), encoding="utf-8")

    def _apply_settings_locked(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "harness_url" in payload:
            self.cfg.harness_url = str(payload["harness_url"]).strip()
        if "cache_window_frames" in payload:
            cache_window_frames = payload["cache_window_frames"]
            if cache_window_frames is not None:
                cache_window_frames = int(cache_window_frames)
            self.cfg.cache_window_frames = cache_window_frames
            self._ar_state.max_cache_frames = cache_window_frames
            self.loaded_model.backbone.crop_autoregressive_state(
                self._ar_state,
                max_cache_frames=cache_window_frames,
            )
        if "action_decode" in payload:
            self._apply_action_decode_payload_locked(dict(payload["action_decode"]))
        if "ui_settings" in payload:
            ui_payload = payload["ui_settings"]
            if not isinstance(ui_payload, dict):
                raise TypeError("ui_settings must be an object")
            if "browser_fps" in ui_payload:
                self._ui_settings.browser_fps = max(0.1, float(ui_payload["browser_fps"]))
            if "player_index" in ui_payload:
                self._ui_settings.player_index = max(0, int(ui_payload["player_index"]))
            if "capture_player_index" in ui_payload:
                self._ui_settings.capture_player_index = max(0, int(ui_payload["capture_player_index"]))
            if "decoder_calibration" in ui_payload:
                decoder_payload = ui_payload["decoder_calibration"]
                if not isinstance(decoder_payload, dict):
                    raise TypeError("ui_settings.decoder_calibration must be an object")
                self._ui_settings.decoder_calibration = self._normalize_decoder_calibration_ui_settings(decoder_payload)
        self._refresh_composite_locked()
        self._persist_settings_locked()
        return self._export_settings_locked()

    def _load_persisted_settings(self) -> None:
        settings_path = Path(self.cfg.settings_path)
        if not settings_path.is_file():
            return
        try:
            payload = json.loads(settings_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self._apply_settings_locked(payload)
        except Exception as exc:
            self._metrics.last_error = f"failed to load settings: {exc}"

    async def set_action_decode_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            updated = self._apply_action_decode_payload_locked(payload)
            self._persist_settings_locked()
            return updated

    async def set_ui_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            self._apply_settings_locked({"ui_settings": payload})
            return asdict(self._ui_settings)

    async def export_settings(self) -> dict[str, Any]:
        async with self._lock:
            return self._export_settings_locked()

    async def import_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            return self._apply_settings_locked(payload)

    async def start_recording(self, path: str | None = None) -> str:
        async with self._lock:
            recording_path = self.recorder.start(path)
            if self._latest_composite_jpeg is not None and self._latest_observation is not None:
                self.recorder.write_jpeg(
                    self._latest_composite_jpeg,
                    timestamp_ns=int(self._latest_observation.server_time_ns),
                )
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
                capture=self._capture_summary_locked(),
                decoder_calibration=asdict(self._decoder_calibration),
                last_actions=dict(self._latest_actions),
                last_action_summaries={
                    slot_name: summary.to_dict()
                    for slot_name, summary in self._latest_action_summaries.items()
                },
                calibration=asdict(self._mouse_calibration),
                ui_settings=asdict(self._ui_settings),
            )

    def _capture_summary_locked(self) -> dict[str, Any]:
        frames = []
        for idx, frame in enumerate(self._capture_frames):
            frames.append(
                {
                    "index": int(idx),
                    "server_time_ns": int(frame["server_time_ns"]),
                    "slots": [dict(item) for item in frame["slot_summaries"]],
                }
            )
        return {
            "active": bool(self._capture_active),
            "started_at": float(self._capture_started_at),
            "stopped_at": float(self._capture_stopped_at),
            "frame_count": len(self._capture_frames),
            "video": asdict(self.capture_video.status()),
            "frames": frames,
        }

    def _cache_snapshot_locked(self) -> dict[str, Any]:
        return {
            "tokens_per_frame": int(self._ar_state.tokens_per_frame),
            "cached_tokens": int(self._ar_state.cached_tokens),
            "cached_frames": int(self._ar_state.cached_frames),
            "total_frames_processed": int(self._ar_state.total_frames_processed),
            "max_cache_frames": self._ar_state.max_cache_frames,
        }

    async def latest_logits(self, player_index: int) -> dict[str, Any]:
        async with self._lock:
            if self._latest_prediction is None:
                return {}
            if not 0 <= player_index < len(self.cfg.slot_names):
                raise IndexError(f"player_index must be in [0, {len(self.cfg.slot_names) - 1}]")

            player_name = self.cfg.slot_names[player_index]
            return {
                "player_index": int(player_index),
                "player_name": player_name,
                "action_decode": asdict(self.cfg.action_decode),
                "action_summary": self._latest_action_summaries[player_name].to_dict()
                if player_name in self._latest_action_summaries
                else {},
                "panels": self._panels_for_prediction_locked(self._latest_prediction, player_index),
            }

    def _panels_for_prediction_locked(self, prediction: dict[str, torch.Tensor], player_index: int) -> list[dict[str, Any]]:
        panels: list[dict[str, Any]] = []
        model_cfg = self.loaded_model.global_cfg.model
        for head_name, tensor in prediction.items():
            if tensor.ndim < 4:
                continue
            if head_name in {"enemy_pos_x", "enemy_pos_y", "enemy_pos_z"} and tensor.shape[2] == 5:
                for enemy_index in range(5):
                    values = tensor[0, 0, enemy_index].detach().float().cpu().view(-1)
                    panels.append(_categorical_panel(f"{head_name}[enemy {enemy_index}]", values, model_cfg))
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
                panels.append(
                    _keyboard_panel(
                        values,
                        float(self.cfg.action_decode.keyboard_threshold),
                        dict(self.cfg.action_decode.keyboard_thresholds),
                    )
                )
            elif values.numel() == 1:
                panels.append(_binary_panel(head_name, float(values[0].item()), model_cfg))
            else:
                panels.append(_categorical_panel(head_name, values, model_cfg))
        return panels

    async def start_capture(self) -> dict[str, Any]:
        async with self._lock:
            self._capture_frames = []
            self._capture_active = True
            self._capture_started_at = time.time()
            self._capture_stopped_at = 0.0
            self.capture_video.start(self.cfg.slot_names)
            return self._capture_summary_locked()

    async def stop_capture(self) -> dict[str, Any]:
        async with self._lock:
            self._capture_active = False
            self._capture_stopped_at = time.time()
            self.capture_video.stop()
            return self._capture_summary_locked()

    async def clear_capture(self) -> dict[str, Any]:
        async with self._lock:
            self._capture_frames = []
            self._capture_active = False
            self._capture_started_at = 0.0
            self._capture_stopped_at = 0.0
            self.capture_video.clear()
            return self._capture_summary_locked()

    async def capture_snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return self._capture_summary_locked()

    async def capture_frame(self, frame_index: int, player_index: int) -> dict[str, Any]:
        async with self._lock:
            if not 0 <= player_index < len(self.cfg.slot_names):
                raise IndexError(f"player_index must be in [0, {len(self.cfg.slot_names) - 1}]")
            if not 0 <= frame_index < len(self._capture_frames):
                raise IndexError(f"frame_index must be in [0, {len(self._capture_frames) - 1}]")
            frame = self._capture_frames[frame_index]
            player_name = self.cfg.slot_names[player_index]
            slot_summaries = [dict(item) for item in frame["slot_summaries"]]
            slot_for_player = next((item for item in slot_summaries if item["name"] == player_name), None)
            return {
                "frame_index": int(frame_index),
                "player_index": int(player_index),
                "player_name": player_name,
                "server_time_ns": int(frame["server_time_ns"]),
                "video_frame_index": int(frame["video_frame_index"]),
                "timings_ms": dict(frame["timings_ms"]),
                "cache": dict(frame["cache"]),
                "slot": dict(slot_for_player) if slot_for_player is not None else None,
                "slots": slot_summaries,
                "action_summary": dict(frame["action_summaries"].get(player_name, {})),
                "actions": {slot: list(events) for slot, events in frame["actions"].items()},
                "panels": self._panels_for_prediction_locked(frame["prediction"], player_index),
            }

    async def capture_frame_jpeg(self, frame_index: int, player_index: int) -> bytes:
        async with self._lock:
            if not 0 <= player_index < len(self.cfg.slot_names):
                raise IndexError(f"player_index must be in [0, {len(self.cfg.slot_names) - 1}]")
            if not 0 <= frame_index < len(self._capture_frames):
                raise IndexError(f"frame_index must be in [0, {len(self._capture_frames) - 1}]")
            player_name = self.cfg.slot_names[player_index]
            frame = self._capture_frames[frame_index]
            return self.capture_video.frame_jpeg(player_name, int(frame["video_frame_index"]))

    async def start_decoder_calibration(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            if self._decoder_calibration.running:
                raise RuntimeError("decoder calibration is already running")
        task = asyncio.create_task(self._calibrate_decoder_task(payload), name="decoder-calibration")
        async with self._lock:
            self._decoder_calibration_task = task
        await asyncio.sleep(0)
        return {"ok": True}

    async def cancel_decoder_calibration(self) -> dict[str, Any]:
        async with self._lock:
            task = self._decoder_calibration_task
            if task is None or task.done():
                return {"ok": True, "cancelled": False}
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return {"ok": True, "cancelled": True}

    async def _calibrate_decoder_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            was_running = self._running
            self._decoder_calibration.running = True
            self._decoder_calibration.updated_at = time.time()
            self._decoder_calibration.started_at = time.time()
            self._decoder_calibration.current_phase = "initializing"
            self._decoder_calibration.progress_current = 0
            self._decoder_calibration.progress_total = 1
            self._decoder_calibration.eta_seconds = None
            self._decoder_calibration.last_error = ""
        await self.pause()
        try:
            result = await self._run_decoder_calibration(payload)
            async with self._lock:
                cfg = self.cfg.action_decode
                cfg.keyboard_thresholds = dict(result["recommended_keyboard_thresholds"])
                cfg.mouse_zero_bias = float(result["recommended_mouse_zero_bias"])
                cfg.mouse_temperature = float(result["recommended_mouse_temperature"])
                cfg.mouse_top_k = int(result["recommended_mouse_top_k"])
                cfg.mouse_deadzone = float(result["recommended_mouse_deadzone"])
                self._ui_settings.decoder_calibration = self._normalize_decoder_calibration_ui_settings(result)
                result["applied_action_decode"] = asdict(cfg)
                self._decoder_calibration.last_result = result
                self._decoder_calibration.last_error = ""
                self._decoder_calibration.updated_at = time.time()
                self._persist_settings_locked()
            if was_running:
                await self.start()
            return result
        except asyncio.CancelledError:
            async with self._lock:
                self._decoder_calibration.last_error = "cancelled by user"
                self._decoder_calibration.updated_at = time.time()
            if was_running:
                await self.start()
            raise
        except Exception as exc:
            async with self._lock:
                self._decoder_calibration.last_error = str(exc)
                self._decoder_calibration.updated_at = time.time()
            return {"ok": False, "error": str(exc)}
        finally:
            async with self._lock:
                self._decoder_calibration_task = None
                self._decoder_calibration.running = False
                self._decoder_calibration.current_phase = ""
                self._decoder_calibration.eta_seconds = None
                self._decoder_calibration.updated_at = time.time()

    async def _update_decoder_calibration_progress(
        self,
        *,
        phase: str,
        current: int,
        total: int,
    ) -> None:
        async with self._lock:
            view = self._decoder_calibration
            view.current_phase = phase
            view.progress_current = int(current)
            view.progress_total = max(1, int(total))
            elapsed = max(0.0, time.time() - float(view.started_at or time.time()))
            if current > 0:
                per_step = elapsed / float(current)
                view.eta_seconds = max(0.0, per_step * max(0, total - current))
            else:
                view.eta_seconds = None
            view.updated_at = time.time()

    def _current_keyboard_thresholds(self) -> dict[str, float]:
        return {
            action: float(self.cfg.action_decode.keyboard_thresholds.get(action, self.cfg.action_decode.keyboard_threshold))
            for action in KEYBOARD_ONLY_ACTIONS
        }

    @staticmethod
    def _normalize_keyboard_cost_overrides(payload: dict[str, Any], *, field_name: str) -> dict[str, float]:
        raw_overrides = payload.get(field_name, {})
        if raw_overrides is None:
            return {}
        if not isinstance(raw_overrides, dict):
            raise TypeError(f"{field_name} must be an object mapping action names to non-negative costs")
        normalized: dict[str, float] = {}
        valid_labels = set(KEYBOARD_ONLY_ACTIONS)
        for key, value in raw_overrides.items():
            action = str(key)
            if action not in valid_labels:
                continue
            normalized[action] = max(0.0, float(value))
        return normalized

    async def _run_decoder_calibration(self, payload: dict[str, Any]) -> dict[str, Any]:
        defaults = self._ui_settings.decoder_calibration
        max_games = max(1, int(payload.get("max_games", defaults.max_games)))
        max_samples = max(1, int(payload.get("max_samples", defaults.max_samples)))
        samples_per_game = max(1, int(payload.get("samples_per_game", defaults.samples_per_game)))
        frames_per_sample = max(8, int(payload.get("frames_per_sample", defaults.frames_per_sample)))
        keyboard_metric = str(payload.get("keyboard_metric", defaults.keyboard_metric)).strip()
        if keyboard_metric not in DECODER_KEYBOARD_METRICS:
            raise ValueError(f"keyboard_metric must be one of {DECODER_KEYBOARD_METRICS}")
        keyboard_false_positive_cost = max(
            0.0,
            float(payload.get("keyboard_false_positive_cost", defaults.keyboard_false_positive_cost)),
        )
        keyboard_false_negative_cost = max(
            0.0,
            float(payload.get("keyboard_false_negative_cost", defaults.keyboard_false_negative_cost)),
        )
        mouse_metric = str(payload.get("mouse_metric", defaults.mouse_metric)).strip()
        if mouse_metric not in DECODER_MOUSE_METRICS:
            raise ValueError(f"mouse_metric must be one of {DECODER_MOUSE_METRICS}")
        mouse_move_weight = min(
            0.99,
            max(0.01, float(payload.get("mouse_move_weight", defaults.mouse_move_weight))),
        )
        mouse_stochastic_sampling = bool(payload.get("mouse_stochastic_sampling", defaults.mouse_stochastic_sampling))
        if "keyboard_false_positive_cost_overrides" in payload:
            keyboard_false_positive_cost_overrides = self._normalize_keyboard_cost_overrides(
                payload,
                field_name="keyboard_false_positive_cost_overrides",
            )
        else:
            keyboard_false_positive_cost_overrides = dict(defaults.keyboard_false_positive_cost_overrides)
        if "keyboard_false_negative_cost_overrides" in payload:
            keyboard_false_negative_cost_overrides = self._normalize_keyboard_cost_overrides(
                payload,
                field_name="keyboard_false_negative_cost_overrides",
            )
        else:
            keyboard_false_negative_cost_overrides = dict(defaults.keyboard_false_negative_cost_overrides)

        val_ds, selected_indices, sample_labels = await asyncio.to_thread(
            self._build_decoder_calibration_dataset,
            max_games,
            max_samples,
            samples_per_game,
            frames_per_sample,
        )
        mouse_candidates = self._decoder_mouse_candidate_grid()
        total_steps = len(selected_indices) + len(KEYBOARD_ONLY_ACTIONS) + len(mouse_candidates)
        await self._update_decoder_calibration_progress(phase="collecting validation samples", current=0, total=total_steps)

        keyboard_prob_chunks: list[torch.Tensor] = []
        keyboard_gt_chunks: list[torch.Tensor] = []
        mouse_x_logit_chunks: list[torch.Tensor] = []
        mouse_y_logit_chunks: list[torch.Tensor] = []
        mouse_x_gt_chunks: list[torch.Tensor] = []
        mouse_y_gt_chunks: list[torch.Tensor] = []

        completed = 0
        for index in selected_indices:
            sample = await asyncio.to_thread(val_ds.__getitem__, index)
            batch = await asyncio.to_thread(self._collate_calibration_sample, sample)
            prediction_cpu = await asyncio.to_thread(self._forward_decoder_calibration_batch, batch.images, batch.audio)
            keyboard_prob_chunks.append(torch.sigmoid(prediction_cpu["keyboard_logits"]).reshape(-1, prediction_cpu["keyboard_logits"].shape[-1]))
            keyboard_gt_chunks.append(batch.truth.keyboard_mask.reshape(-1))
            mouse_x_logit_chunks.append(prediction_cpu["mouse_x"].reshape(-1, prediction_cpu["mouse_x"].shape[-1]))
            mouse_y_logit_chunks.append(prediction_cpu["mouse_y"].reshape(-1, prediction_cpu["mouse_y"].shape[-1]))
            mouse_x_gt_chunks.append(batch.truth.mouse_delta[..., 0].reshape(-1).float())
            mouse_y_gt_chunks.append(batch.truth.mouse_delta[..., 1].reshape(-1).float())
            completed += 1
            await self._update_decoder_calibration_progress(
                phase="collecting validation samples",
                current=completed,
                total=total_steps,
            )

        keyboard_probs = torch.cat(keyboard_prob_chunks, dim=0)
        keyboard_gt_masks = torch.cat(keyboard_gt_chunks, dim=0).to(dtype=torch.int64)
        mouse_x_logits = torch.cat(mouse_x_logit_chunks, dim=0)
        mouse_y_logits = torch.cat(mouse_y_logit_chunks, dim=0)
        mouse_x_gt = torch.cat(mouse_x_gt_chunks, dim=0)
        mouse_y_gt = torch.cat(mouse_y_gt_chunks, dim=0)

        keyboard_thresholds, keyboard_metrics = await self._search_keyboard_thresholds(
            keyboard_probs=keyboard_probs,
            keyboard_gt_masks=keyboard_gt_masks,
            metric_name=keyboard_metric,
            false_positive_cost=keyboard_false_positive_cost,
            false_negative_cost=keyboard_false_negative_cost,
            false_positive_cost_overrides=keyboard_false_positive_cost_overrides,
            false_negative_cost_overrides=keyboard_false_negative_cost_overrides,
            completed_steps=completed,
            total_steps=total_steps,
        )
        completed += len(KEYBOARD_ONLY_ACTIONS)

        mouse_result = await self._search_mouse_decoder_settings(
            mouse_x_logits=mouse_x_logits,
            mouse_y_logits=mouse_y_logits,
            mouse_x_gt=mouse_x_gt,
            mouse_y_gt=mouse_y_gt,
            metric_name=mouse_metric,
            move_weight=mouse_move_weight,
            stochastic_sampling=mouse_stochastic_sampling,
            completed_steps=completed,
            total_steps=total_steps,
        )

        return {
            "selected_samples": len(selected_indices),
            "selected_games": sorted({label.split(":")[0] for label in sample_labels}),
            "sample_labels": sample_labels,
            "frames_per_sample": frames_per_sample,
            "keyboard_metric": keyboard_metric,
            "keyboard_false_positive_cost": keyboard_false_positive_cost,
            "keyboard_false_negative_cost": keyboard_false_negative_cost,
            "keyboard_false_positive_cost_overrides": keyboard_false_positive_cost_overrides,
            "keyboard_false_negative_cost_overrides": keyboard_false_negative_cost_overrides,
            "keyboard_metric_choices": list(DECODER_KEYBOARD_METRICS),
            "mouse_metric": mouse_metric,
            "mouse_move_weight": mouse_move_weight,
            "mouse_stochastic_sampling": mouse_stochastic_sampling,
            "mouse_metric_choices": list(DECODER_MOUSE_METRICS),
            "recommended_keyboard_thresholds": keyboard_thresholds,
            "keyboard_metrics": keyboard_metrics,
            "recommended_mouse_zero_bias": float(mouse_result["best_candidate"]["mouse_zero_bias"]),
            "recommended_mouse_temperature": float(mouse_result["best_candidate"]["mouse_temperature"]),
            "recommended_mouse_top_k": int(mouse_result["best_candidate"]["mouse_top_k"]),
            "recommended_mouse_deadzone": float(mouse_result["best_candidate"]["mouse_deadzone"]),
            "mouse_metrics": mouse_result,
        }

    def _build_decoder_calibration_dataset(
        self,
        max_games: int,
        max_samples: int,
        samples_per_game: int,
        frames_per_sample: int,
    ) -> tuple[Any, list[int], list[str]]:
        ensure_model3_import_path()
        from dataset import DatasetRoot

        dataset_cfg = replace(
            self.loaded_model.global_cfg.dataset,
            run_dir="./runs/decoder_calibration",
            sample_stride=max(frames_per_sample, int(self.loaded_model.global_cfg.dataset.sample_stride)),
            epoch_round_sample_length=frames_per_sample,
            epoch_video_decoding_device="cpu",
        )
        ds_root = DatasetRoot(dataset_cfg)
        val_ds = ds_root.build_dataset("val")

        selected_indices: list[int] = []
        sample_labels: list[str] = []
        per_game_counts: dict[str, int] = {}
        for idx, sample in enumerate(val_ds.samples):
            game_name = str(sample.round.game.demo_name)
            if game_name not in per_game_counts and len(per_game_counts) >= max_games:
                continue
            if per_game_counts.get(game_name, 0) >= samples_per_game:
                continue
            per_game_counts[game_name] = per_game_counts.get(game_name, 0) + 1
            selected_indices.append(idx)
            sample_labels.append(f"{game_name}:round{sample.round.round_num}:frame{sample.start_frame}")
            if len(selected_indices) >= max_samples:
                break
        if not selected_indices:
            raise RuntimeError("decoder calibration could not find validation samples")
        return val_ds, selected_indices, sample_labels

    @staticmethod
    def _collate_calibration_sample(sample: Any) -> Any:
        ensure_model3_import_path()
        from dataset import cs2_collate_fn

        return cs2_collate_fn([sample])

    def _forward_decoder_calibration_batch(self, images: torch.Tensor, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        images, audio = self.loaded_model.prepare_batch_tensors(images, audio)
        self._synchronize_model_device()
        with torch.inference_mode():
            prediction = self.loaded_model.backbone(images, audio)
        self._synchronize_model_device()
        return {
            head_name: tensor.detach().float().cpu()
            for head_name, tensor in prediction.items()
        }

    async def _search_keyboard_thresholds(
        self,
        *,
        keyboard_probs: torch.Tensor,
        keyboard_gt_masks: torch.Tensor,
        metric_name: str,
        false_positive_cost: float,
        false_negative_cost: float,
        false_positive_cost_overrides: dict[str, float],
        false_negative_cost_overrides: dict[str, float],
        completed_steps: int,
        total_steps: int,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        threshold_candidates = [round(value, 3) for value in torch.linspace(0.02, 0.60, steps=30).tolist()]
        current_thresholds = self._current_keyboard_thresholds()
        recommended: dict[str, float] = {}
        per_key_metrics: dict[str, Any] = {}

        for action_offset, action in enumerate(KEYBOARD_ONLY_ACTIONS):
            gt = ((keyboard_gt_masks >> int(action_offset)) & 1).to(dtype=torch.bool)
            probs = keyboard_probs[:, action_offset]
            false_positive_cost_for_key = float(false_positive_cost_overrides.get(action, false_positive_cost))
            false_negative_cost_for_key = float(false_negative_cost_overrides.get(action, false_negative_cost))
            positives = int(gt.sum().item())
            negatives = int((~gt).sum().item())
            if positives == 0 or negatives == 0:
                threshold = float(current_thresholds[action])
                recommended[action] = threshold
                per_key_metrics[action] = {
                    "false_positive_cost_used": false_positive_cost_for_key,
                    "false_negative_cost_used": false_negative_cost_for_key,
                    "current_threshold": threshold,
                    "recommended_threshold": threshold,
                    "positive_frames": positives,
                    "negative_frames": negatives,
                    "score_before": None,
                    "score_after": None,
                    "skipped": True,
                }
            else:
                best_score = -1.0
                best_metrics: dict[str, Any] | None = None
                current_metrics = self._score_keyboard_threshold(
                    probs=probs,
                    gt=gt,
                    threshold=float(current_thresholds[action]),
                    metric_name=metric_name,
                    false_positive_cost=false_positive_cost_for_key,
                    false_negative_cost=false_negative_cost_for_key,
                )
                for threshold in threshold_candidates:
                    candidate_metrics = self._score_keyboard_threshold(
                        probs=probs,
                        gt=gt,
                        threshold=float(threshold),
                        metric_name=metric_name,
                        false_positive_cost=false_positive_cost_for_key,
                        false_negative_cost=false_negative_cost_for_key,
                    )
                    score = float(candidate_metrics["score"])
                    if score > best_score + 1e-12 or (
                        abs(score - best_score) <= 1e-12
                        and best_metrics is not None
                        and abs(float(threshold) - float(current_thresholds[action]))
                        < abs(float(best_metrics["threshold"]) - float(current_thresholds[action]))
                    ):
                        best_score = score
                        best_metrics = candidate_metrics
                assert best_metrics is not None
                recommended[action] = float(best_metrics["threshold"])
                per_key_metrics[action] = {
                    "false_positive_cost_used": false_positive_cost_for_key,
                    "false_negative_cost_used": false_negative_cost_for_key,
                    "current_threshold": float(current_thresholds[action]),
                    "recommended_threshold": float(best_metrics["threshold"]),
                    "metric_name": metric_name,
                    "positive_frames": positives,
                    "negative_frames": negatives,
                    "score_before": float(current_metrics["score"]),
                    "score_after": float(best_metrics["score"]),
                    "tpr_before": float(current_metrics["tpr"]),
                    "tpr_after": float(best_metrics["tpr"]),
                    "tnr_before": float(current_metrics["tnr"]),
                    "tnr_after": float(best_metrics["tnr"]),
                    "fpr_before": float(current_metrics["fpr"]),
                    "fpr_after": float(best_metrics["fpr"]),
                    "fnr_before": float(current_metrics["fnr"]),
                    "fnr_after": float(best_metrics["fnr"]),
                    "precision_before": float(current_metrics["precision"]),
                    "precision_after": float(best_metrics["precision"]),
                    "f1_before": float(current_metrics["f1"]),
                    "f1_after": float(best_metrics["f1"]),
                    "balanced_accuracy_before": float(current_metrics["balanced_accuracy"]),
                    "balanced_accuracy_after": float(best_metrics["balanced_accuracy"]),
                    "tp_before": int(current_metrics["tp"]),
                    "tp_after": int(best_metrics["tp"]),
                    "fp_before": int(current_metrics["fp"]),
                    "fp_after": int(best_metrics["fp"]),
                    "tn_before": int(current_metrics["tn"]),
                    "tn_after": int(best_metrics["tn"]),
                    "fn_before": int(current_metrics["fn"]),
                    "fn_after": int(best_metrics["fn"]),
                    "pred_rate_before": float(current_metrics["pred_rate"]),
                    "pred_rate_after": float(best_metrics["pred_rate"]),
                    "gt_rate": float(current_metrics["gt_rate"]),
                    "skipped": False,
                }
            await self._update_decoder_calibration_progress(
                phase=f"calibrating keyboard thresholds ({action})",
                current=completed_steps + action_offset + 1,
                total=total_steps,
            )

        valid_scores_before = [item["score_before"] for item in per_key_metrics.values() if item.get("score_before") is not None]
        valid_scores_after = [item["score_after"] for item in per_key_metrics.values() if item.get("score_after") is not None]
        return recommended, {
            "metric_name": metric_name,
            "global_false_positive_cost": float(false_positive_cost),
            "global_false_negative_cost": float(false_negative_cost),
            "per_key": per_key_metrics,
            "mean_score_before": float(sum(valid_scores_before) / len(valid_scores_before)) if valid_scores_before else None,
            "mean_score_after": float(sum(valid_scores_after) / len(valid_scores_after)) if valid_scores_after else None,
        }

    @staticmethod
    def _score_keyboard_threshold(
        *,
        probs: torch.Tensor,
        gt: torch.Tensor,
        threshold: float,
        metric_name: str,
        false_positive_cost: float,
        false_negative_cost: float,
    ) -> dict[str, float]:
        pred = probs >= float(threshold)
        metrics = _binary_confusion_metrics(pred, gt)
        metrics["threshold"] = float(threshold)
        metrics["metric_name"] = metric_name
        metrics["false_positive_cost"] = float(false_positive_cost)
        metrics["false_negative_cost"] = float(false_negative_cost)
        metrics["score"] = float(
            _decoder_metric_score(
                metric_name,
                metrics,
                false_positive_cost=false_positive_cost,
                false_negative_cost=false_negative_cost,
            )
        )
        return metrics

    def _decoder_mouse_candidate_grid(self) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for zero_bias in (0.0, 0.25, 0.5, 0.75, 1.0):
            for temperature in (0.75, 1.0, 1.25, 1.5):
                for top_k in (1, 2, 3, 5):
                    for deadzone in (0.10, 0.20, 0.35, 0.50, 0.75):
                        candidates.append(
                            {
                                "mouse_zero_bias": float(zero_bias),
                                "mouse_temperature": float(temperature),
                                "mouse_top_k": int(top_k),
                                "mouse_deadzone": float(deadzone),
                            }
                        )
        return candidates

    async def _search_mouse_decoder_settings(
        self,
        *,
        mouse_x_logits: torch.Tensor,
        mouse_y_logits: torch.Tensor,
        mouse_x_gt: torch.Tensor,
        mouse_y_gt: torch.Tensor,
        metric_name: str,
        move_weight: float,
        stochastic_sampling: bool,
        completed_steps: int,
        total_steps: int,
    ) -> dict[str, Any]:
        current_candidate = {
            "mouse_zero_bias": float(self.cfg.action_decode.mouse_zero_bias),
            "mouse_temperature": float(self.cfg.action_decode.mouse_temperature),
            "mouse_top_k": int(self.cfg.action_decode.mouse_top_k),
            "mouse_deadzone": float(self.cfg.action_decode.mouse_deadzone),
        }
        candidates = self._decoder_mouse_candidate_grid()
        current_metrics = self._score_mouse_candidate(
            candidate=current_candidate,
            mouse_x_logits=mouse_x_logits,
            mouse_y_logits=mouse_y_logits,
            mouse_x_gt=mouse_x_gt,
            mouse_y_gt=mouse_y_gt,
            metric_name=metric_name,
            move_weight=move_weight,
            stochastic_sampling=stochastic_sampling,
        )

        best_metrics: dict[str, Any] | None = None
        progress_base = completed_steps
        for candidate_index, candidate in enumerate(candidates):
            candidate_metrics = self._score_mouse_candidate(
                candidate=candidate,
                mouse_x_logits=mouse_x_logits,
                mouse_y_logits=mouse_y_logits,
                mouse_x_gt=mouse_x_gt,
                mouse_y_gt=mouse_y_gt,
                metric_name=metric_name,
                move_weight=move_weight,
                stochastic_sampling=stochastic_sampling,
            )
            if best_metrics is None or float(candidate_metrics["score"]) > float(best_metrics["score"]):
                best_metrics = candidate_metrics
            await self._update_decoder_calibration_progress(
                phase=(
                    "calibrating mouse decoder "
                    f"(bias={candidate['mouse_zero_bias']:.2f}, temp={candidate['mouse_temperature']:.2f}, "
                    f"k={candidate['mouse_top_k']}, deadzone={candidate['mouse_deadzone']:.2f})"
                ),
                current=progress_base + candidate_index + 1,
                total=total_steps,
            )
        assert best_metrics is not None
        return {
            "metric_name": metric_name,
            "move_weight": float(move_weight),
            "stochastic_sampling": bool(stochastic_sampling),
            "current_candidate": current_candidate,
            "current_metrics": current_metrics,
            "best_candidate": best_metrics["candidate"],
            "best_metrics": best_metrics,
        }

    def _score_mouse_candidate(
        self,
        *,
        candidate: dict[str, Any],
        mouse_x_logits: torch.Tensor,
        mouse_y_logits: torch.Tensor,
        mouse_x_gt: torch.Tensor,
        mouse_y_gt: torch.Tensor,
        metric_name: str,
        move_weight: float,
        stochastic_sampling: bool,
    ) -> dict[str, Any]:
        gt_move_threshold = 0.35
        x_metrics = self._score_mouse_axis_candidate(
            logits=mouse_x_logits,
            gt=mouse_x_gt,
            axis_name="mouse_x",
            zero_bias=float(candidate["mouse_zero_bias"]),
            temperature=float(candidate["mouse_temperature"]),
            top_k=int(candidate["mouse_top_k"]),
            deadzone=float(candidate["mouse_deadzone"]),
            gt_move_threshold=gt_move_threshold,
            metric_name=metric_name,
            move_weight=move_weight,
            stochastic_sampling=stochastic_sampling,
        )
        y_metrics = self._score_mouse_axis_candidate(
            logits=mouse_y_logits,
            gt=mouse_y_gt,
            axis_name="mouse_y",
            zero_bias=float(candidate["mouse_zero_bias"]),
            temperature=float(candidate["mouse_temperature"]),
            top_k=int(candidate["mouse_top_k"]),
            deadzone=float(candidate["mouse_deadzone"]),
            gt_move_threshold=gt_move_threshold,
            metric_name=metric_name,
            move_weight=move_weight,
            stochastic_sampling=stochastic_sampling,
        )
        score = (
            (float(x_metrics["score"]) + float(y_metrics["score"])) * 0.5
        )
        return {
            "candidate": dict(candidate),
            "move_weight": float(move_weight),
            "stochastic_sampling": bool(stochastic_sampling),
            "score": float(score),
            "mouse_x": x_metrics,
            "mouse_y": y_metrics,
        }

    @staticmethod
    def _nearest_mouse_bin_indices(decoded: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        distances = (decoded.unsqueeze(0) - gt.unsqueeze(1)).abs()
        return torch.argmin(distances, dim=-1)

    def _score_mouse_axis_candidate(
        self,
        *,
        logits: torch.Tensor,
        gt: torch.Tensor,
        axis_name: str,
        zero_bias: float,
        temperature: float,
        top_k: int,
        deadzone: float,
        gt_move_threshold: float,
        metric_name: str,
        move_weight: float,
        stochastic_sampling: bool,
    ) -> dict[str, Any]:
        adjusted = logits.float().clone()
        zero_bin_index = int((adjusted.shape[-1] - 1) // 2)
        adjusted[:, zero_bin_index] -= float(zero_bias)
        probabilities = torch.softmax(adjusted / max(1e-4, float(temperature)), dim=-1)
        limited_top_k = max(1, min(int(top_k), int(probabilities.shape[-1])))
        if limited_top_k < probabilities.shape[-1]:
            top_values, top_indices = torch.topk(probabilities, k=limited_top_k, dim=-1)
            renorm = top_values / top_values.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            probs = torch.zeros_like(probabilities).scatter(-1, top_indices, renorm)
        else:
            probs = probabilities

        bin_indices = torch.arange(probabilities.shape[-1], dtype=torch.float32)
        decoded = mu_law_decode(
            bin_indices,
            mu=float(self.loaded_model.global_cfg.model.mouse_mu),
            max_val=float(self.loaded_model.global_cfg.model.mouse_max),
            bins=int(self.loaded_model.global_cfg.model.mouse_bins_count),
        )

        move_mask = decoded.abs() >= float(deadzone)
        expected_move = (probs * move_mask.to(dtype=probs.dtype).unsqueeze(0)).sum(dim=-1)
        gt_move = gt.abs() >= float(gt_move_threshold)
        gt_bins = self._nearest_mouse_bin_indices(decoded, gt.float())

        if stochastic_sampling:
            repeats = 4
            generator = torch.Generator(device=probs.device)
            generator.manual_seed(1337)
            sampled_bins = torch.multinomial(probs, num_samples=repeats, replacement=True, generator=generator)
            sampled_move = move_mask[sampled_bins]
            sampled_bin_match = sampled_bins.eq(gt_bins.unsqueeze(1))
            move_confusion = _binary_confusion_metrics(sampled_move.reshape(-1), gt_move.unsqueeze(1).expand_as(sampled_move).reshape(-1))
            bin_accuracy = float(
                sampled_bin_match[gt_move.unsqueeze(1).expand_as(sampled_bin_match)].float().mean().item()
            ) if bool(gt_move.any().item()) else 1.0
            pred_move_rate = float(sampled_move.float().mean().item())
            effective_mode = "stochastic"
        else:
            move_confusion = _soft_confusion_metrics(expected_move, gt_move)
            move_prob_gt_bin = probs.gather(1, gt_bins.unsqueeze(1)).squeeze(1)
            bin_accuracy = float(move_prob_gt_bin[gt_move].mean().item()) if bool(gt_move.any().item()) else 1.0
            pred_move_rate = float(expected_move.mean().item())
            effective_mode = "expected"

        expected_delta = (probs * decoded.unsqueeze(0)).sum(dim=-1)
        magnitude_mae = float((expected_delta - gt).abs().mean().item())
        move_score = float(_decoder_metric_score(metric_name, move_confusion))
        total_score = (float(move_weight) * move_score) + ((1.0 - float(move_weight)) * float(bin_accuracy))

        return {
            **move_confusion,
            "metric_name": metric_name,
            "move_weight": float(move_weight),
            "mode": effective_mode,
            "bin_accuracy": float(bin_accuracy),
            "magnitude_mae": float(magnitude_mae),
            "move_score": float(move_score),
            "score": float(total_score),
            "pred_move_rate": float(pred_move_rate),
            "expected_move_rate": float(expected_move.mean().item()),
            "gt_move_rate": float(gt_move.float().mean().item()),
        }

    async def calibrate_mouse(self) -> dict[str, Any]:
        async with self._lock:
            if self._mouse_calibration.running:
                raise RuntimeError("mouse calibration is already running")
            was_running = self._running
            self._mouse_calibration.running = True
            self._mouse_calibration.last_error = ""
            self._mouse_calibration.updated_at = time.time()

        await self.pause()
        client = RemoteHarnessClient(
            base_url=self.cfg.harness_url,
            verify_ssl=self.cfg.verify_ssl,
            timeout_s=10.0,
        )
        try:
            result = await self._run_mouse_calibration(client)
            async with self._lock:
                self.cfg.action_decode.mouse_scale_x = float(result["recommended_mouse_scale_x"])
                self.cfg.action_decode.mouse_scale_y = float(result["recommended_mouse_scale_y"])
                result["applied_action_decode"] = asdict(self.cfg.action_decode)
                self._mouse_calibration.last_result = result
                self._mouse_calibration.last_error = ""
                self._mouse_calibration.updated_at = time.time()
                self._persist_settings_locked()
            if was_running:
                await self.start()
            return result
        except Exception as exc:
            async with self._lock:
                self._mouse_calibration.last_error = str(exc)
                self._mouse_calibration.updated_at = time.time()
            raise
        finally:
            client.close()
            async with self._lock:
                self._mouse_calibration.running = False
                self._mouse_calibration.updated_at = time.time()

    async def _run_mouse_calibration(self, client: RemoteHarnessClient) -> dict[str, Any]:
        slot_name = self.cfg.slot_names[0]
        slots = await asyncio.to_thread(client.get_slots)
        slot_snapshot = next((slot for slot in slots if slot.get("name") == slot_name), None)
        if slot_snapshot is None:
            raise RuntimeError(f"slot {slot_name} is not present in the harness")
        if slot_snapshot.get("status") != "ready":
            raise RuntimeError(f"slot {slot_name} is not ready for calibration")

        await self._post_server_command(client, "css_sim_unfreeze", allow_error=True)
        try:
            await self._post_server_command(client, "css_sim_reset")
            await asyncio.sleep(1.5)

            state = await self._fetch_plugin_state(client, refresh=True)
            player = self._select_calibration_player(state, slot_name=slot_name)
            baseline_pitch = float(player["pitch"])
            baseline_yaw = float(player["yaw"])
            if abs(baseline_pitch) > 10.0:
                await self._post_server_command(
                    client,
                    f"css_sim_set_view {0.0:.3f} {baseline_yaw:.3f}",
                )
                await asyncio.sleep(0.15)

            await self._post_server_command(client, "css_sim_freeze bots")
            await asyncio.sleep(0.15)
            state = await self._fetch_plugin_state(client, refresh=True)
            player = self._select_calibration_player(state, slot_name=slot_name)
            baseline_pitch = float(player["pitch"])
            baseline_yaw = float(player["yaw"])

            axis_inputs = [8.0, -8.0, 16.0, -16.0, 32.0, -32.0, 48.0, -48.0, 64.0, -64.0, 96.0, -96.0]
            yaw_trials = await self._collect_mouse_axis_trials(
                client,
                slot_name=slot_name,
                axis_name="x",
                values=axis_inputs * 2,
            )
            pitch_trials = await self._collect_mouse_axis_trials(
                client,
                slot_name=slot_name,
                axis_name="y",
                values=axis_inputs * 2,
            )

            yaw_fit = _fit_linear_axis([(float(item["input"]), float(item["delta_yaw"])) for item in yaw_trials])
            pitch_fit = _fit_linear_axis([(float(item["input"]), float(item["delta_pitch"])) for item in pitch_trials])

            yaw_slope = float(yaw_fit["slope"])
            pitch_slope = float(pitch_fit["slope"])
            if abs(yaw_slope) < 1e-5:
                raise RuntimeError("yaw calibration slope was too close to zero")
            if abs(pitch_slope) < 1e-5:
                raise RuntimeError("pitch calibration slope was too close to zero")

            model_cfg = self.loaded_model.global_cfg.model
            global_scale = float(self.cfg.action_decode.mouse_scale)
            recommended_scale_x = 1.0 / (global_scale * yaw_slope)
            recommended_scale_y = 1.0 / (global_scale * pitch_slope)

            sequence_results = await self._run_mouse_sequence_validation(
                client,
                slot_name=slot_name,
                yaw_fit=yaw_fit,
                pitch_fit=pitch_fit,
            )
            single_frame_target_tests = {
                "mouse_x": await self._run_mouse_target_validation(
                    client,
                    slot_name=slot_name,
                    axis_name="x",
                    fit=yaw_fit,
                    axis_scale=recommended_scale_x,
                    baseline_pitch=baseline_pitch,
                    baseline_yaw=baseline_yaw,
                    targets_deg=[-30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
                ),
                "mouse_y": await self._run_mouse_target_validation(
                    client,
                    slot_name=slot_name,
                    axis_name="y",
                    fit=pitch_fit,
                    axis_scale=recommended_scale_y,
                    baseline_pitch=baseline_pitch,
                    baseline_yaw=baseline_yaw,
                    targets_deg=[-30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
                ),
            }
            single_frame_bin_tests = {
                "mouse_x": await self._run_mouse_bin_validation(
                    client,
                    slot_name=slot_name,
                    axis_name="x",
                    fit=yaw_fit,
                    axis_scale=recommended_scale_x,
                    baseline_pitch=baseline_pitch,
                    baseline_yaw=baseline_yaw,
                    model_cfg=model_cfg,
                ),
                "mouse_y": await self._run_mouse_bin_validation(
                    client,
                    slot_name=slot_name,
                    axis_name="y",
                    fit=pitch_fit,
                    axis_scale=recommended_scale_y,
                    baseline_pitch=baseline_pitch,
                    baseline_yaw=baseline_yaw,
                    model_cfg=model_cfg,
                ),
            }
            final_state = await self._fetch_plugin_state(client, refresh=True)
            final_player = self._select_calibration_player(final_state, slot_name=slot_name)
            return {
                "slot_name": slot_name,
                "player_name": str(final_player["name"]),
                "freeze_mode": final_state.get("freezeMode", ""),
                "recommended_mouse_scale_x": recommended_scale_x,
                "recommended_mouse_scale_y": recommended_scale_y,
                "global_mouse_scale": global_scale,
                "yaw_fit": yaw_fit,
                "pitch_fit": pitch_fit,
                "yaw_trials": yaw_trials,
                "pitch_trials": pitch_trials,
                "sequence_results": sequence_results,
                "single_frame_target_tests": single_frame_target_tests,
                "single_frame_target_summary": {
                    axis_name: _summarize_angle_validation(results)
                    for axis_name, results in single_frame_target_tests.items()
                },
                "single_frame_bin_tests": single_frame_bin_tests,
                "single_frame_bin_summary": {
                    axis_name: _summarize_angle_validation(results)
                    for axis_name, results in single_frame_bin_tests.items()
                },
                "player_state": final_player,
            }
        finally:
            await self._post_server_command(client, "css_sim_unfreeze", allow_error=True)

    async def _post_server_command(
        self,
        client: RemoteHarnessClient,
        command: str,
        *,
        allow_error: bool = False,
    ) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(client.post_json, "/api/server/command", {"command": command})
        except Exception:
            if allow_error:
                return {"ok": False, "command": command}
            raise

    async def _fetch_plugin_state(self, client: RemoteHarnessClient, *, refresh: bool) -> dict[str, Any]:
        response = await asyncio.to_thread(client.post_json, "/api/server/plugin-state", {"refresh": refresh})
        state = response.get("state")
        if not isinstance(state, dict):
            raise RuntimeError("server plugin state response was invalid")
        return state

    @staticmethod
    def _state_value(payload: dict[str, Any], *names: str, default: Any = None) -> Any:
        for name in names:
            if name in payload:
                return payload[name]
        return default

    @classmethod
    def _normalize_player_state(cls, player: dict[str, Any]) -> dict[str, Any]:
        return {
            "slot": cls._state_value(player, "slot", "Slot", default=-1),
            "name": str(cls._state_value(player, "name", "Name", default="")),
            "isBot": bool(cls._state_value(player, "isBot", "IsBot", default=False)),
            "connected": bool(cls._state_value(player, "connected", "Connected", default=False)),
            "alive": bool(cls._state_value(player, "alive", "Alive", default=False)),
            "frozen": bool(cls._state_value(player, "frozen", "Frozen", default=False)),
            "team": str(cls._state_value(player, "team", "Team", default="")),
            "pitch": float(cls._state_value(player, "pitch", "Pitch", default=0.0)),
            "yaw": float(cls._state_value(player, "yaw", "Yaw", default=0.0)),
            "roll": float(cls._state_value(player, "roll", "Roll", default=0.0)),
            "originX": float(cls._state_value(player, "originX", "OriginX", default=0.0)),
            "originY": float(cls._state_value(player, "originY", "OriginY", default=0.0)),
            "originZ": float(cls._state_value(player, "originZ", "OriginZ", default=0.0)),
        }

    @classmethod
    def _select_calibration_player(cls, state: dict[str, Any], *, slot_name: str) -> dict[str, Any]:
        players = cls._state_value(state, "players", "Players", default=[])
        if not isinstance(players, list):
            raise RuntimeError("server plugin state did not contain a player list")
        humans = [
            cls._normalize_player_state(player)
            for player in players
            if isinstance(player, dict)
            and bool(cls._state_value(player, "connected", "Connected", default=False))
            and not bool(cls._state_value(player, "isBot", "IsBot", default=False))
            and bool(cls._state_value(player, "alive", "Alive", default=False))
            and str(cls._state_value(player, "team", "Team", default="")) in {"ct", "t"}
        ]

        if not humans:
            raise RuntimeError("expected at least one alive human player for calibration, found 0")

        preferred_names = {slot_name.lower()}
        suffix = slot_name[5:] if slot_name.lower().startswith("steam") else ""
        preferred_slot: int | None = None
        if suffix.isdigit():
            preferred_names.add(f"agent{suffix}")
            preferred_slot = int(suffix) - 1

        for player in humans:
            if player["name"].lower() in preferred_names:
                return player

        if preferred_slot is not None:
            for player in humans:
                if int(player["slot"]) == preferred_slot:
                    return player

        if len(humans) == 1:
            return humans[0]

        available = ", ".join(f"{player['name']}@slot{player['slot']}" for player in humans)
        raise RuntimeError(f"unable to identify calibration player for {slot_name}; available humans: {available}")

    async def _collect_mouse_axis_trials(
        self,
        client: RemoteHarnessClient,
        *,
        slot_name: str,
        axis_name: str,
        values: list[float],
    ) -> list[dict[str, Any]]:
        trials: list[dict[str, Any]] = []
        for value in values:
            measurement = await self._measure_mouse_delta(
                client,
                slot_name=slot_name,
                dx=float(value) if axis_name == "x" else 0.0,
                dy=float(value) if axis_name == "y" else 0.0,
            )
            trials.append({"axis": axis_name, "input": float(value), **measurement})
            await asyncio.sleep(0.05)
        return trials

    async def _measure_mouse_delta(
        self,
        client: RemoteHarnessClient,
        *,
        slot_name: str,
        dx: float,
        dy: float,
        settle_s: float = 0.12,
    ) -> dict[str, Any]:
        before = self._select_calibration_player(
            await self._fetch_plugin_state(client, refresh=True),
            slot_name=slot_name,
        )
        await asyncio.to_thread(client.send_actions, {slot_name: [{"t": "mouse_rel", "dx": float(dx), "dy": float(dy)}]})
        await asyncio.sleep(settle_s)
        after = self._select_calibration_player(
            await self._fetch_plugin_state(client, refresh=True),
            slot_name=slot_name,
        )
        delta_pitch = float(after["pitch"]) - float(before["pitch"])
        delta_yaw = _unwrap_delta_degrees(float(after["yaw"]) - float(before["yaw"]))
        return {
            "before_pitch": float(before["pitch"]),
            "before_yaw": float(before["yaw"]),
            "after_pitch": float(after["pitch"]),
            "after_yaw": float(after["yaw"]),
            "delta_pitch": delta_pitch,
            "delta_yaw": delta_yaw,
        }

    async def _recenter_calibration_view(
        self,
        client: RemoteHarnessClient,
        *,
        baseline_pitch: float,
        baseline_yaw: float,
        settle_s: float = 0.08,
    ) -> None:
        await self._post_server_command(
            client,
            f"css_sim_set_view {float(baseline_pitch):.3f} {float(baseline_yaw):.3f}",
        )
        await asyncio.sleep(settle_s)

    async def _run_mouse_target_validation(
        self,
        client: RemoteHarnessClient,
        *,
        slot_name: str,
        axis_name: str,
        fit: dict[str, float],
        axis_scale: float,
        baseline_pitch: float,
        baseline_yaw: float,
        targets_deg: list[float],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        global_scale = float(self.cfg.action_decode.mouse_scale)
        slope = float(fit["slope"])
        intercept = float(fit["intercept"])
        if abs(slope) < 1e-8:
            raise RuntimeError(f"{axis_name} validation slope was too close to zero")

        for target_deg in targets_deg:
            await self._recenter_calibration_view(
                client,
                baseline_pitch=baseline_pitch,
                baseline_yaw=baseline_yaw,
            )
            required_input = (float(target_deg) - intercept) / slope
            injected_input = float(required_input)
            measurement = await self._measure_mouse_delta(
                client,
                slot_name=slot_name,
                dx=injected_input if axis_name == "x" else 0.0,
                dy=injected_input if axis_name == "y" else 0.0,
            )
            measured_angle = float(measurement["delta_yaw"] if axis_name == "x" else measurement["delta_pitch"])
            predicted_angle = _predict_linear_axis(fit, injected_input)
            raw_model_delta = injected_input / max(1e-8, global_scale * axis_scale)
            results.append(
                {
                    "axis": axis_name,
                    "target_angle": float(target_deg),
                    "required_input": float(required_input),
                    "injected_input": float(injected_input),
                    "raw_model_delta_equivalent": float(raw_model_delta),
                    "predicted_angle": float(predicted_angle),
                    "measured_angle": float(measured_angle),
                    "angle_error": float(measured_angle - predicted_angle),
                    "clamped": False,
                }
            )
            await asyncio.sleep(0.03)
        return results

    async def _run_mouse_bin_validation(
        self,
        client: RemoteHarnessClient,
        *,
        slot_name: str,
        axis_name: str,
        fit: dict[str, float],
        axis_scale: float,
        baseline_pitch: float,
        baseline_yaw: float,
        model_cfg: object,
    ) -> list[dict[str, Any]]:
        bin_indices = torch.arange(int(model_cfg.mouse_bins_count))
        raw_deltas = mu_law_decode(
            bin_indices,
            mu=float(model_cfg.mouse_mu),
            max_val=float(model_cfg.mouse_max),
            bins=int(model_cfg.mouse_bins_count),
        ).tolist()
        global_scale = float(self.cfg.action_decode.mouse_scale)
        results: list[dict[str, Any]] = []

        for bin_index, raw_delta in enumerate(raw_deltas):
            await self._recenter_calibration_view(
                client,
                baseline_pitch=baseline_pitch,
                baseline_yaw=baseline_yaw,
            )
            injected_input, clamped = _scaled_mouse_input(
                float(raw_delta),
                global_scale=global_scale,
                axis_scale=axis_scale,
            )
            measurement = await self._measure_mouse_delta(
                client,
                slot_name=slot_name,
                dx=injected_input if axis_name == "x" else 0.0,
                dy=injected_input if axis_name == "y" else 0.0,
            )
            measured_angle = float(measurement["delta_yaw"] if axis_name == "x" else measurement["delta_pitch"])
            predicted_angle = _predict_linear_axis(fit, injected_input)
            results.append(
                {
                    "axis": axis_name,
                    "bin_index": int(bin_index),
                    "raw_model_delta": float(raw_delta),
                    "injected_input": float(injected_input),
                    "predicted_angle": float(predicted_angle),
                    "measured_angle": float(measured_angle),
                    "angle_error": float(measured_angle - predicted_angle),
                    "clamped": bool(clamped),
                }
            )
            await asyncio.sleep(0.02)
        return results

    async def _run_mouse_sequence_validation(
        self,
        client: RemoteHarnessClient,
        *,
        slot_name: str,
        yaw_fit: dict[str, float],
        pitch_fit: dict[str, float],
    ) -> list[dict[str, Any]]:
        sequences = [
            {
                "label": "yaw_small_chain",
                "events": [{"dx": 12.0, "dy": 0.0}] * 8,
            },
            {
                "label": "yaw_mixed_chain",
                "events": [
                    {"dx": 24.0, "dy": 0.0},
                    {"dx": -8.0, "dy": 0.0},
                    {"dx": 16.0, "dy": 0.0},
                    {"dx": -4.0, "dy": 0.0},
                ],
            },
            {
                "label": "pitch_small_chain",
                "events": [{"dx": 0.0, "dy": 10.0}] * 6,
            },
            {
                "label": "pitch_mixed_chain",
                "events": [
                    {"dx": 0.0, "dy": 20.0},
                    {"dx": 0.0, "dy": -8.0},
                    {"dx": 0.0, "dy": 12.0},
                    {"dx": 0.0, "dy": -4.0},
                ],
            },
        ]

        results: list[dict[str, Any]] = []
        for sequence in sequences:
            before = self._select_calibration_player(
                await self._fetch_plugin_state(client, refresh=True),
                slot_name=slot_name,
            )
            for event in sequence["events"]:
                await asyncio.to_thread(client.send_actions, {slot_name: [{"t": "mouse_rel", "dx": event["dx"], "dy": event["dy"]}]})
                await asyncio.sleep(max(0.03, float(self.cfg.poll_interval_s)))
            await asyncio.sleep(0.12)
            after = self._select_calibration_player(
                await self._fetch_plugin_state(client, refresh=True),
                slot_name=slot_name,
            )

            measured_yaw = _unwrap_delta_degrees(float(after["yaw"]) - float(before["yaw"]))
            measured_pitch = float(after["pitch"]) - float(before["pitch"])
            predicted_yaw = sum((yaw_fit["slope"] * event["dx"]) + yaw_fit["intercept"] for event in sequence["events"])
            predicted_pitch = sum((pitch_fit["slope"] * event["dy"]) + pitch_fit["intercept"] for event in sequence["events"])
            results.append(
                {
                    "label": sequence["label"],
                    "events": sequence["events"],
                    "predicted_yaw": predicted_yaw,
                    "measured_yaw": measured_yaw,
                    "yaw_error": measured_yaw - predicted_yaw,
                    "predicted_pitch": predicted_pitch,
                    "measured_pitch": measured_pitch,
                    "pitch_error": measured_pitch - predicted_pitch,
                }
            )
        return results

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
                latest_observation, drained = await self._drain_stream_backlog(client, observation)
                latest_frame_seq = _observation_max_frame_seq(latest_observation)
                async with self._lock:
                    self._latest_observation = latest_observation
                    self._metrics.observations_received += 1 + drained
                    self._metrics.skipped_observations += drained
                    self._metrics.last_observation_server_time_ns = int(latest_observation.server_time_ns)
                    should_run = self._running
                    last_processed_frame_seq = self._last_processed_frame_seq
                    self._refresh_composite_locked()
                if not should_run:
                    continue
                if latest_frame_seq <= last_processed_frame_seq:
                    async with self._lock:
                        self._metrics.skipped_observations += 1
                    continue

                prediction, actions, summaries, timings_ms = await asyncio.to_thread(
                    self._infer_observation,
                    latest_observation,
                )
                async with self._lock:
                    still_running = self._running and self._client is client
                if not still_running:
                    continue

                send_ns = 0
                if actions:
                    send_ns = await client.send_actions(actions)

                async with self._lock:
                    if last_processed_frame_seq >= 0 and latest_frame_seq > last_processed_frame_seq + 1:
                        self._metrics.skipped_frames += latest_frame_seq - last_processed_frame_seq - 1
                    self._last_processed_frame_seq = latest_frame_seq
                    self._metrics.last_processed_frame_seq = latest_frame_seq
                    self._apply_inference_result_locked(
                        latest_observation,
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
                latest_frame_seq = _observation_max_frame_seq(observation)
                async with self._lock:
                    self._latest_observation = observation
                    self._metrics.observations_received += 1
                    self._metrics.last_observation_server_time_ns = int(observation.server_time_ns)
                    should_run = self._running
                    last_processed_frame_seq = self._last_processed_frame_seq
                    self._refresh_composite_locked()
                if should_run and latest_frame_seq <= last_processed_frame_seq:
                    async with self._lock:
                        self._metrics.skipped_observations += 1
                    await asyncio.sleep(max(0.0, float(self.cfg.poll_interval_s)))
                    continue
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
                        if last_processed_frame_seq >= 0 and latest_frame_seq > last_processed_frame_seq + 1:
                            self._metrics.skipped_frames += latest_frame_seq - last_processed_frame_seq - 1
                        self._last_processed_frame_seq = latest_frame_seq
                        self._metrics.last_processed_frame_seq = latest_frame_seq
                        self._apply_inference_result_locked(
                            observation,
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

    async def _drain_stream_backlog(
        self,
        client: RemoteHarnessStreamClient,
        observation: HarnessObservation,
    ) -> tuple[HarnessObservation, int]:
        latest = observation
        drained = 0
        while True:
            try:
                latest = await asyncio.wait_for(client.recv_observation(), timeout=0.001)
                drained += 1
            except asyncio.TimeoutError:
                return latest, drained

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
        observation: HarnessObservation,
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
        if self._capture_active:
            self.capture_video.write_observation(observation)
            self._capture_frames.append(
                {
                    "server_time_ns": int(observation.server_time_ns),
                    "slot_summaries": [_capture_slot_summary(slot) for slot in observation.slots],
                    "video_frame_index": int(self.capture_video.frame_index_for_time_ns(int(observation.server_time_ns))),
                    "prediction": {
                        head_name: tensor.to(dtype=torch.float16).clone()
                        for head_name, tensor in prediction.items()
                    },
                    "actions": {slot: list(events) for slot, events in actions.items()},
                    "action_summaries": {
                        slot_name: summary.to_dict()
                        for slot_name, summary in summaries.items()
                    },
                    "timings_ms": dict(normalized_timings),
                    "cache": self._cache_snapshot_locked(),
                }
            )
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
                self.recorder.write_jpeg(
                    composite,
                    timestamp_ns=int(self._latest_observation.server_time_ns),
                )
        except Exception as exc:
            self._metrics.last_error = str(exc)
