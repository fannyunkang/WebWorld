"""Parse model responses into executable browser actions."""

from __future__ import annotations

import re

from mm_webagent.data.action_schema import ParsedAction, parse_action


def extract_action(response: str) -> ParsedAction:
    match = re.search(r"<action>(.*?)</action>", response or "", re.DOTALL | re.IGNORECASE)
    action_text = match.group(1).strip() if match else (response or "").strip()
    action_text = re.sub(r"^```(?:python)?", "", action_text, flags=re.MULTILINE).strip()
    action_text = re.sub(r"```$", "", action_text, flags=re.MULTILINE).strip()
    return parse_action(action_text)
