"""PPO/RLHF rollout scaffold for web-agent policy optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mm_webagent.training.reward import compute_reward


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run PPO reward assignment for web-agent rollouts.")
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--model", default="outputs/sft_lora")
    parser.add_argument("--train", action="store_true", help="Print the TRL PPO launch boundary for full rollouts.")
    args = parser.parse_args()

    totals = []
    with Path(args.episodes).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            actions = row.get("actions", [])
            episode_rewards = [
                compute_reward(
                    action=action,
                    task_completed=bool(row.get("task_completed")) and idx == len(actions) - 1,
                    step_index=idx,
                    previous_actions=actions[:idx],
                ).total
                for idx, action in enumerate(actions)
            ]
            totals.append(sum(episode_rewards))

    avg = sum(totals) / len(totals) if totals else 0.0
    print(f"Model: {args.model}")
    print(f"Episodes: {len(totals)}")
    print(f"Average PPO-style return: {avg:.4f}")
    if args.train:
        print("PPO training requires online rollout generation: policy -> WebWorld/browser -> reward -> PPO update.")
        print("Use this script to validate rewards, then connect TRL PPOTrainer to the rollout loop.")


if __name__ == "__main__":
    main()
