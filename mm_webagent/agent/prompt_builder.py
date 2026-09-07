"""Prompt assembly for multimodal web-operation policies."""

from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = """You are a multimodal web operation agent.
Given the user instruction, webpage screenshot, page structure, and action history,
choose exactly one valid browser action.
Return concise reasoning in <reason> tags and one action in <action> tags."""

ACTION_SPACE = """Available actions:
click(bid, button='left', modifiers=[])
fill(bid, text, press_enter=False)
select_option(bid, options)
hover(bid)
scroll(dx, dy)
goto(url)
go_back()
go_forward()
keyboard_press(key)
keyboard_type(text)
tab_new()
tab_close()
tab_focus(index)
send_msg_to_user(text)
infeasible(reason)"""


def build_policy_prompt(
    instruction: str,
    page_state: str,
    history: list[dict[str, Any]] | None = None,
    screenshot: str | None = None,
    retrieved_context: str | None = None,
) -> str:
    history = history or []
    history_lines = []
    for idx, item in enumerate(history[-10:]):
        history_lines.append(f"Step {idx}: action={item.get('action', '')}")
        if item.get("observation"):
            history_lines.append(f"Observation after step {idx}: {item['observation']}")

    screenshot_line = screenshot or "No screenshot attached; rely on page_state."
    return f"""{SYSTEM_PROMPT}

# User instruction
{instruction}

# Screenshot
{screenshot_line}

# Current page state
{page_state}

# History
{chr(10).join(history_lines) if history_lines else "None"}

# Retrieved operation experience
{retrieved_context or "None"}

{ACTION_SPACE}

# Response format
<reason>brief plan</reason>
<action>one_valid_action(...)</action>
"""
