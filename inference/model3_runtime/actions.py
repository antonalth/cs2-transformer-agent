from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import torch
except ModuleNotFoundError:
    torch = None


KEYBOARD_ONLY_ACTIONS = [
    "IN_ATTACK",
    "IN_JUMP",
    "IN_DUCK",
    "IN_FORWARD",
    "IN_BACK",
    "IN_USE",
    "IN_CANCEL",
    "IN_TURNLEFT",
    "IN_TURNRIGHT",
    "IN_MOVELEFT",
    "IN_MOVERIGHT",
    "IN_ATTACK2",
    "IN_RELOAD",
    "IN_ALT1",
    "IN_ALT2",
    "IN_SPEED",
    "IN_WALK",
    "IN_ZOOM",
    "IN_WEAPON1",
    "IN_WEAPON2",
    "IN_BULLRUSH",
    "IN_GRENADE1",
    "IN_GRENADE2",
    "IN_ATTACK3",
    "IN_SCORE",
    "IN_INSPECT",
    "SWITCH_1",
    "SWITCH_2",
    "SWITCH_3",
    "SWITCH_4",
    "SWITCH_5",
]


def keyboard_action_labels(count: int) -> list[str]:
    labels = list(KEYBOARD_ONLY_ACTIONS)
    if count <= len(labels):
        return labels[:count]
    extra = [f"UNUSED_KEYBOARD_{idx}" for idx in range(len(labels), count)]
    return labels + extra


KEY_ACTION_TO_KEY = {
    "IN_JUMP": "space",
    "IN_DUCK": "lctrl",
    "IN_FORWARD": "w",
    "IN_BACK": "s",
    "IN_USE": "e",
    "IN_MOVELEFT": "a",
    "IN_MOVERIGHT": "d",
    "IN_RELOAD": "r",
    "IN_WALK": "lshift",
    "IN_SCORE": "tab",
    "IN_INSPECT": "f",
    "SWITCH_1": "1",
    "SWITCH_2": "2",
    "SWITCH_3": "3",
    "SWITCH_4": "4",
    "SWITCH_5": "5",
}

KEY_ACTION_TO_BUTTON = {
    "IN_ATTACK": 1,
    "IN_ATTACK3": 2,
    "IN_ATTACK2": 3,
}


def mu_law_decode(y: torch.Tensor, mu: float = 255.0, max_val: float = 30.0, bins: int = 256) -> torch.Tensor:
    _require_torch()
    y = y.float()
    y_norm = (y / (bins - 1)) * 2 - 1
    x_abs = (1 / mu) * (torch.pow(1 + mu, torch.abs(y_norm)) - 1)
    x_norm = torch.sign(y_norm) * x_abs
    return x_norm * max_val


@dataclass(slots=True)
class ActionDecodeConfig:
    keyboard_threshold: float = 0.12
    keyboard_thresholds: dict[str, float] = field(default_factory=dict)
    mouse_deadzone: float = 0.35
    mouse_scale: float = 1.0
    mouse_scale_x: float = 1.0
    mouse_scale_y: float = 1.0
    mouse_zero_bias: float = 0.0
    mouse_temperature: float = 1.0
    mouse_top_k: int = 1


@dataclass(slots=True)
class SlotActionSummary:
    slot_name: str
    active_actions: list[str] = field(default_factory=list)
    held_keys: list[str] = field(default_factory=list)
    held_buttons: list[int] = field(default_factory=list)
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    emitted_events: list[dict[str, Any]] = field(default_factory=list)
    decision_trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_name": self.slot_name,
            "active_actions": list(self.active_actions),
            "held_keys": list(self.held_keys),
            "held_buttons": list(self.held_buttons),
            "mouse_dx": self.mouse_dx,
            "mouse_dy": self.mouse_dy,
            "emitted_events": list(self.emitted_events),
            "decision_trace": dict(self.decision_trace),
        }


class ModelActionDecoder:
    def __init__(self, *, cfg: ActionDecodeConfig, slot_names: list[str] | tuple[str, ...]) -> None:
        self.cfg = cfg
        self.slot_names = list(slot_names)
        self._held_keys = {slot: set() for slot in self.slot_names}
        self._held_buttons = {slot: set() for slot in self.slot_names}

    def _keyboard_threshold_for(self, action: str) -> float:
        override = self.cfg.keyboard_thresholds.get(action)
        if override is None:
            return float(self.cfg.keyboard_threshold)
        return float(override)

    @staticmethod
    def _resolve_contrary_actions(
        probabilities: dict[str, float],
        active_actions: set[str],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        contrary_pairs = (
            ("IN_FORWARD", "IN_BACK"),
            ("IN_MOVELEFT", "IN_MOVERIGHT"),
            ("IN_TURNLEFT", "IN_TURNRIGHT"),
        )
        dropped: list[dict[str, Any]] = []
        resolved = set(active_actions)
        for left, right in contrary_pairs:
            if left not in resolved or right not in resolved:
                continue
            left_prob = float(probabilities.get(left, 0.0))
            right_prob = float(probabilities.get(right, 0.0))
            if left_prob >= right_prob:
                loser, winner = right, left
                loser_prob, winner_prob = right_prob, left_prob
            else:
                loser, winner = left, right
                loser_prob, winner_prob = left_prob, right_prob
            resolved.discard(loser)
            dropped.append(
                {
                    "winner": winner,
                    "winner_probability": winner_prob,
                    "dropped": loser,
                    "dropped_probability": loser_prob,
                }
            )
        return sorted(resolved), dropped

    def _decode_keyboard(self, logits: torch.Tensor) -> tuple[list[str], dict[str, Any]]:
        probabilities = torch.sigmoid(logits)
        probs_list = [float(value) for value in probabilities.tolist()]
        labels = keyboard_action_labels(len(probs_list))
        action_probs = {
            label: prob
            for label, prob in zip(labels, probs_list)
        }
        thresholds = {
            action: self._keyboard_threshold_for(action)
            for action in labels
        }
        threshold_hits = {
            action
            for action, prob in action_probs.items()
            if prob >= thresholds[action]
        }
        resolved_actions, dropped_contrary = self._resolve_contrary_actions(action_probs, threshold_hits)
        trace = {
            "threshold": float(self.cfg.keyboard_threshold),
            "thresholds": thresholds,
            "labels": labels,
            "threshold_hits": [
                {
                    "action": action,
                    "probability": float(action_probs[action]),
                    "threshold": float(thresholds[action]),
                }
                for action in sorted(threshold_hits)
            ],
            "contrary_resolutions": dropped_contrary,
            "selected_actions": list(resolved_actions),
        }
        return resolved_actions, trace

    def _decode_mouse(
        self,
        logits: torch.Tensor,
        *,
        axis_name: str,
        model_cfg: object,
    ) -> tuple[float, int, dict[str, Any]]:
        adjusted_logits = logits.detach().float().clone()
        zero_bin_index = int((adjusted_logits.numel() - 1) // 2)
        zero_bias = float(self.cfg.mouse_zero_bias)
        if adjusted_logits.numel() > 0:
            adjusted_logits[zero_bin_index] -= zero_bias

        temperature = max(1e-4, float(self.cfg.mouse_temperature))
        sampling_logits = adjusted_logits / temperature
        probabilities = torch.softmax(sampling_logits, dim=0)

        top_k = max(1, min(int(self.cfg.mouse_top_k), int(probabilities.numel())))
        top_probs, top_indices = torch.topk(probabilities, k=top_k)
        renorm_probs = top_probs / top_probs.sum().clamp(min=1e-8)

        if top_k == 1:
            selected_rank = 0
        else:
            selected_rank = int(torch.multinomial(renorm_probs, num_samples=1).item())
        selected_bin = int(top_indices[selected_rank].item())
        selected_probability = float(probabilities[selected_bin].item())
        sampled_probability = float(renorm_probs[selected_rank].item())

        raw_delta = float(
            mu_law_decode(
                torch.tensor(selected_bin),
                mu=float(model_cfg.mouse_mu),
                max_val=float(model_cfg.mouse_max),
                bins=int(model_cfg.mouse_bins_count),
            ).item()
        )
        axis_scale = float(self.cfg.mouse_scale_x if axis_name == "mouse_x" else self.cfg.mouse_scale_y)
        delta = raw_delta * float(self.cfg.mouse_scale) * axis_scale

        trace = {
            "axis": axis_name,
            "zero_bin_index": zero_bin_index,
            "zero_bin_probability": float(probabilities[zero_bin_index].item()),
            "zero_bias": zero_bias,
            "temperature": temperature,
            "top_k": top_k,
            "selected_bin": selected_bin,
            "selected_probability": selected_probability,
            "sampled_probability_within_top_k": sampled_probability,
            "selected_raw_delta": raw_delta,
            "selected_output_delta": delta,
            "selected_delta": delta,
            "top_k_candidates": [
                {
                    "bin_index": int(index.item()),
                    "probability": float(prob.item()),
                    "sampling_weight": float(weight.item()),
                }
                for index, prob, weight in zip(top_indices, top_probs, renorm_probs)
            ],
        }
        return delta, selected_bin, trace

    def decode(
        self,
        prediction: dict[str, torch.Tensor],
        *,
        model_cfg: object,
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, SlotActionSummary]]:
        _require_torch()
        keyboard_logits = prediction["keyboard_logits"][0, 0]
        mouse_x = prediction["mouse_x"][0, 0]
        mouse_y = prediction["mouse_y"][0, 0]

        actions: dict[str, list[dict[str, Any]]] = {}
        summaries: dict[str, SlotActionSummary] = {}

        for slot_idx, slot_name in enumerate(self.slot_names):
            active_actions, keyboard_trace = self._decode_keyboard(keyboard_logits[slot_idx])
            desired_keys = {KEY_ACTION_TO_KEY[action] for action in active_actions if action in KEY_ACTION_TO_KEY}
            desired_buttons = {KEY_ACTION_TO_BUTTON[action] for action in active_actions if action in KEY_ACTION_TO_BUTTON}

            slot_events: list[dict[str, Any]] = []

            held_keys = self._held_keys[slot_name]
            for key_name in sorted(held_keys - desired_keys):
                slot_events.append({"t": "key", "key": key_name, "down": False})
            for key_name in sorted(desired_keys - held_keys):
                slot_events.append({"t": "key", "key": key_name, "down": True})
            self._held_keys[slot_name] = set(desired_keys)

            held_buttons = self._held_buttons[slot_name]
            for button in sorted(held_buttons - desired_buttons):
                slot_events.append({"t": "mouse_btn", "button": button, "down": False})
            for button in sorted(desired_buttons - held_buttons):
                slot_events.append({"t": "mouse_btn", "button": button, "down": True})
            self._held_buttons[slot_name] = set(desired_buttons)

            dx, mouse_bin_x, mouse_trace_x = self._decode_mouse(
                mouse_x[slot_idx],
                axis_name="mouse_x",
                model_cfg=model_cfg,
            )
            dy, mouse_bin_y, mouse_trace_y = self._decode_mouse(
                mouse_y[slot_idx],
                axis_name="mouse_y",
                model_cfg=model_cfg,
            )

            raw_dx = float(mouse_trace_x.get("selected_raw_delta", 0.0))
            raw_dy = float(mouse_trace_y.get("selected_raw_delta", 0.0))
            mouse_event_emitted = abs(raw_dx) >= self.cfg.mouse_deadzone or abs(raw_dy) >= self.cfg.mouse_deadzone
            if mouse_event_emitted:
                slot_events.append({"t": "mouse_rel", "dx": dx, "dy": dy})

            if slot_events:
                actions[slot_name] = slot_events

            summaries[slot_name] = SlotActionSummary(
                slot_name=slot_name,
                active_actions=active_actions,
                held_keys=sorted(self._held_keys[slot_name]),
                held_buttons=sorted(self._held_buttons[slot_name]),
                mouse_dx=dx,
                mouse_dy=dy,
                emitted_events=list(slot_events),
                decision_trace={
                    "keyboard": keyboard_trace,
                    "mouse_x": mouse_trace_x,
                    "mouse_y": mouse_trace_y,
                    "mouse_deadzone": float(self.cfg.mouse_deadzone),
                    "mouse_event_emitted": bool(mouse_event_emitted),
                    "selected_mouse_bins": {
                        "mouse_x": int(mouse_bin_x),
                        "mouse_y": int(mouse_bin_y),
                    },
                },
            )

        return actions, summaries

    def release_all(self) -> dict[str, list[dict[str, Any]]]:
        actions: dict[str, list[dict[str, Any]]] = {}
        for slot_name in self.slot_names:
            slot_events: list[dict[str, Any]] = []
            for key_name in sorted(self._held_keys[slot_name]):
                slot_events.append({"t": "key", "key": key_name, "down": False})
            for button in sorted(self._held_buttons[slot_name]):
                slot_events.append({"t": "mouse_btn", "button": button, "down": False})
            self._held_keys[slot_name].clear()
            self._held_buttons[slot_name].clear()
            if slot_events:
                actions[slot_name] = slot_events
        return actions


def _require_torch():
    if torch is None:
        raise RuntimeError("torch is required for model3 action decoding")
    return torch
