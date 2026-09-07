"""Convert WebWorld trajectories into SFT and GRPO training records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_webworld_records(path: str | Path) -> list[dict[str, Any]]:
    data_path = Path(path)
    if data_path.suffix == ".jsonl":
        return list(_iter_jsonl(data_path))
    with data_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array or JSONL file.")
    return data


def to_sft_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build supervised action-prediction records from WebWorld samples."""
    converted: list[dict[str, Any]] = []
    for idx, item in enumerate(records):
        trajectory = item.get("trajectory")
        if isinstance(trajectory, list):
            history: list[dict[str, str]] = []
            for step in trajectory:
                observation = step.get("observation") or step.get("page_state")
                action = step.get("action")
                if observation and action:
                    converted.append(
                        {
                            "id": f"trajectory-{idx}-{len(history)}",
                            "instruction": item.get("instruction", "Complete the web task."),
                            "observation": {
                                "screenshot": item.get("screenshot"),
                                "page_state": observation,
                            },
                            "history": history.copy(),
                            "target_action": action,
                        }
                    )
                    history.append({"observation": observation, "action": action})
        elif item.get("observation") and item.get("action"):
            converted.append(
                {
                    "id": f"sample-{idx}",
                    "instruction": item.get("instruction", "Predict the next browser action."),
                    "observation": {
                        "screenshot": item.get("screenshot"),
                        "page_state": item["observation"],
                    },
                    "history": [],
                    "target_action": item["action"],
                    "next_observation": item.get("next_observation"),
                }
            )
    return converted


def to_grpo_rollout_seed(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build rollout seeds with reward metadata for GRPO-style optimization."""
    seeds = []
    for row in to_sft_records(records):
        seeds.append(
            {
                "id": row["id"],
                "instruction": row["instruction"],
                "observation": row["observation"],
                "history": row["history"],
                "reference_action": row["target_action"],
                "reward_hints": {
                    "prefer_valid_action": True,
                    "penalize_repetition": True,
                    "penalize_extra_steps": True,
                },
            }
        )
    return seeds


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT/GRPO data from WebWorld trajectories.")
    parser.add_argument("--input", required=True, help="WebWorld json/jsonl trajectory file.")
    parser.add_argument("--sft-output", default="data/mm_webagent/sft_train.jsonl")
    parser.add_argument("--grpo-output", default="data/mm_webagent/grpo_rollout_seed.jsonl")
    args = parser.parse_args()

    records = load_webworld_records(args.input)
    write_jsonl(to_sft_records(records), args.sft_output)
    write_jsonl(to_grpo_rollout_seed(records), args.grpo_output)
    print(f"Wrote SFT records to {args.sft_output}")
    print(f"Wrote GRPO rollout seeds to {args.grpo_output}")


if __name__ == "__main__":
    main()
