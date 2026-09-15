"""JSON action schema for stricter browser-action validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from mm_webagent.data.action_schema import ACTION_NAMES


@dataclass(frozen=True)
class JsonAction:
    action: str
    target: dict[str, Any] = field(default_factory=dict)
    args: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    reason: str | None = None


def parse_json_action(text: str | dict[str, Any]) -> JsonAction:
    payload = json.loads(text) if isinstance(text, str) else text
    if not isinstance(payload, dict):
        raise ValueError("JSON action must be an object.")

    action = str(payload.get("action", ""))
    if action not in ACTION_NAMES:
        raise ValueError(f"Unsupported action name: {action}")

    target = payload.get("target") or {}
    args = payload.get("args") or {}
    if not isinstance(target, dict) or not isinstance(args, dict):
        raise ValueError("JSON action target and args must be objects.")

    confidence = payload.get("confidence")
    if confidence is not None:
        confidence = float(confidence)

    reason = payload.get("reason")
    return JsonAction(action=action, target=target, args=args, confidence=confidence, reason=reason)


def json_action_to_python_call(action: JsonAction) -> str:
    """Convert a validated JSON action to the existing Python-style call."""
    values: list[Any] = []
    kwargs: dict[str, Any] = {}

    bid = action.target.get("bid")
    if bid is not None:
        values.append(str(bid))

    for key, value in action.args.items():
        if key in {"text", "url", "key", "reason"} and not values:
            values.append(value)
        else:
            kwargs[key] = value

    positional = ", ".join(repr(value) for value in values)
    keyword = ", ".join(f"{key}={value!r}" for key, value in kwargs.items())
    joined = ", ".join(part for part in (positional, keyword) if part)
    return f"{action.action}({joined})"
