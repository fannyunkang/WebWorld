"""GRPO training scaffold for optimizing web-operation policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mm_webagent.training.reward import compute_reward


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight GRPO reward dry run.")
    parser.add_argument("--rollout-seed", required=True)
    parser.add_argument("--max-records", type=int, default=5)
    args = parser.parse_args()

    rewards = []
    with Path(args.rollout_seed).open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= args.max_records:
                break
            row = json.loads(line)
            action = row.get("reference_action", "")
            rewards.append(compute_reward(action, task_completed=False, step_index=idx).total)

    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"Checked {len(rewards)} rollout seeds.")
    print(f"Average shaped reward: {avg_reward:.4f}")


if __name__ == "__main__":
    main()
