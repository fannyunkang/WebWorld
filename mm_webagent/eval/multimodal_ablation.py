"""Ablation helpers for page-state-only vs screenshot-enabled episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mm_webagent.eval.evaluator import load_episodes, summarize


def compare_episode_files(page_state_only: str | Path, multimodal: str | Path) -> dict[str, Any]:
    text_report = summarize(load_episodes(page_state_only))
    multimodal_report = summarize(load_episodes(multimodal))
    return {
        "page_state_only": text_report,
        "screenshot_plus_page_state": multimodal_report,
        "delta": {
            key: multimodal_report.get(key, 0.0) - text_report.get(key, 0.0)
            for key in sorted(set(text_report) | set(multimodal_report))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare text-only and multimodal agent episodes.")
    parser.add_argument("--page-state-only", required=True)
    parser.add_argument("--multimodal", required=True)
    args = parser.parse_args()
    print(json.dumps(compare_episode_files(args.page_state_only, args.multimodal), indent=2))


if __name__ == "__main__":
    main()
