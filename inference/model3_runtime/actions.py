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
    keyboard_threshold: float = 0.5
    mouse_deadzone: float = 0.35
    mouse_scale: float = 1.0
    max_mouse_delta: float = 30.0


@dataclass(slots=True)
class SlotActionSummary:
    slot_name: str
    active_actions: list[str] = field(default_factory=list)
    held_keys: list[str] = field(default_factory=list)
    held_buttons: list[int] = field(default_factory=list)
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    emitted_events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_name": self.slot_name,
            "active_actions": list(self.active_actions),
            "held_keys": list(self.held_keys),
            "held_buttons": list(self.held_buttons),
            "mouse_dx": self.mouse_dx,
            "mouse_dy": self.mouse_dy,
            "emitted_events": list(self.emitted_events),
        }


class ModelActionDecoder:
    def __init__(self, *, cfg: ActionDecodeConfig, slot_names: list[str] | tuple[str, ...]) -> None:
        self.cfg = cfg
        self.slot_names = list(slot_names)
        self._held_keys = {slot: set() for slot in self.slot_names}
        self._held_buttons = {slot: set() for slot in self.slot_names}

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
            kb_probs = torch.sigmoid(keyboard_logits[slot_idx])
            active_actions = [
                KEYBOARD_ONLY_ACTIONS[action_idx]
                for action_idx, prob in enumerate(kb_probs.tolist())
                if prob >= self.cfg.keyboard_threshold
            ]
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

            mouse_bin_x = torch.argmax(mouse_x[slot_idx]).view(())
            mouse_bin_y = torch.argmax(mouse_y[slot_idx]).view(())
            dx = float(
                mu_law_decode(
                    mouse_bin_x,
                    mu=float(model_cfg.mouse_mu),
                    max_val=float(model_cfg.mouse_max),
                    bins=int(model_cfg.mouse_bins_count),
                ).item()
            )
            dy = float(
                mu_law_decode(
                    mouse_bin_y,
                    mu=float(model_cfg.mouse_mu),
                    max_val=float(model_cfg.mouse_max),
                    bins=int(model_cfg.mouse_bins_count),
                ).item()
            )
            dx *= float(self.cfg.mouse_scale)
            dy *= float(self.cfg.mouse_scale)
            dx = max(-self.cfg.max_mouse_delta, min(self.cfg.max_mouse_delta, dx))
            dy = max(-self.cfg.max_mouse_delta, min(self.cfg.max_mouse_delta, dy))

            if abs(dx) >= self.cfg.mouse_deadzone or abs(dy) >= self.cfg.mouse_deadzone:
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
