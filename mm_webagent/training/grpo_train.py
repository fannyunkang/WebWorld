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
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--train", action="store_true", help="Print the GRPO launch boundary for grouped rollouts.")
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
    print(f"Group size: {args.group_size}")
    if args.train:
        print("GRPO training samples multiple actions per state, normalizes rewards within each group, and updates the policy.")


if __name__ == "__main__":
    main()
