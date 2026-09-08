"""Dataset formatting helpers for TRL-based post-training."""

from __future__ import annotations

import json
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sft_text(row: dict) -> str:
    observation = row.get("observation", {})
    history = row.get("history", [])
    return (
        "You are a web operation Agent. Select exactly one valid browser action.\n\n"
        f"Instruction:\n{row.get('instruction', '')}\n\n"
        f"Page State:\n{observation.get('page_state', '')}\n\n"
        f"Screenshot:\n{observation.get('screenshot') or 'none'}\n\n"
        f"History:\n{json.dumps(history, ensure_ascii=False)}\n\n"
        f"<action>{row.get('target_action', '')}</action>"
    )


def dpo_prompt(row: dict) -> str:
    return row.get("prompt", "")
