"""Evaluate saved agent episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mm_webagent.eval.metrics import Episode, average_steps, average_tokens, invalid_action_rate, success_rate


def load_episodes(path: str | Path) -> list[Episode]:
    episodes = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            episodes.append(
                Episode(
                    task_completed=bool(row.get("task_completed")),
                    actions=list(row.get("actions", [])),
                    token_count=int(row.get("token_count", 0)),
                )
            )
    return episodes


def summarize(episodes: list[Episode]) -> dict[str, float]:
    return {
        "success_rate": success_rate(episodes),
        "invalid_action_rate": invalid_action_rate(episodes),
        "average_steps": average_steps(episodes),
        "average_tokens": average_tokens(episodes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multimodal web-agent episodes.")
    parser.add_argument("--episodes", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(load_episodes(args.episodes)), indent=2))


if __name__ == "__main__":
    main()
