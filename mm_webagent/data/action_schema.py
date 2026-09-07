"""Unified browser action schema used by SFT, GRPO, and inference."""

from __future__ import annotations

import re
from dataclasses import dataclass


ACTION_NAMES = {
    "click",
    "fill",
    "select_option",
    "hover",
    "mouse_move",
    "mouse_click",
    "mouse_down",
    "mouse_up",
    "keyboard_press",
    "keyboard_type",
    "scroll",
    "goto",
    "go_back",
    "go_forward",
    "tab_new",
    "tab_close",
    "tab_focus",
    "send_msg_to_user",
    "noop",
    "infeasible",
    "report_infeasible",
}

ACTION_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)\s*$", re.DOTALL)


@dataclass(frozen=True)
class ParsedAction:
    name: str
    arguments: str
    raw: str

    @property
    def is_terminal(self) -> bool:
        return self.name in {"send_msg_to_user", "infeasible", "report_infeasible"}


def parse_action(action: str) -> ParsedAction:
    """Parse a Python-style browser action without executing it."""
    match = ACTION_RE.match(action or "")
    if not match:
        raise ValueError(f"Invalid action syntax: {action!r}")

    name, arguments = match.groups()
    if name not in ACTION_NAMES:
        raise ValueError(f"Unsupported action name: {name}")

    return ParsedAction(name=name, arguments=arguments.strip(), raw=action.strip())


def is_valid_action(action: str) -> bool:
    try:
        parse_action(action)
    except ValueError:
        return False
    return True
