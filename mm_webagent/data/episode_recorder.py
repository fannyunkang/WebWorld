"""Persist multimodal web-agent episodes for replay, training, and eval."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EpisodeStep:
    step: int
    url: str
    screenshot: str | None
    page_state: str
    action: str
    result: str = "unknown"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class EpisodeRecorder:
    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.steps: list[EpisodeStep] = []

    def add_step(self, step: EpisodeStep) -> None:
        self.steps.append(step)

    def write_jsonl(self) -> None:
        with self.output_path.open("w", encoding="utf-8") as handle:
            for step in self.steps:
                handle.write(json.dumps(asdict(step), ensure_ascii=False) + "\n")

    def to_sft_rows(self, instruction: str) -> list[dict[str, Any]]:
        rows = []
        history: list[dict[str, str]] = []
        for step in self.steps:
            rows.append(
                {
                    "id": f"episode-step-{step.step}",
                    "instruction": instruction,
                    "observation": {
                        "screenshot": step.screenshot,
                        "page_state": step.page_state,
                        "url": step.url,
                    },
                    "history": history.copy(),
                    "target_action": step.action,
                    "execution_result": step.result,
                    "execution_error": step.error,
                }
            )
            history.append({"observation": step.page_state, "action": step.action})
        return rows
