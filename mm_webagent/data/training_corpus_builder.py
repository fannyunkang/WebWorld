"""Build a larger training corpus for SFT, DPO, and GRPO dry runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mm_webagent.eval.agent_benchmark import ACTION_BY_CLASS, TASK_CLASSES, TASK_VARIANTS


DISTRACTOR_ACTIONS = {
    "invalid": "click_bad",
    "repeat": "scroll(0, 420)",
    "unsafe_goto": "goto('https://unknown.example.com')",
    "premature_done": "send_msg_to_user('Done')",
}


def build_trajectory_records(repeats_per_class: int = 10) -> list[dict]:
    records = []
    for class_idx, task_class in enumerate(TASK_CLASSES):
        for repeat in range(repeats_per_class):
            variant = TASK_VARIANTS[repeat % len(TASK_VARIANTS)]
            target_action = ACTION_BY_CLASS[task_class]
            target_bid = 10 + class_idx
            aux_bid = 30 + repeat
            instruction = f"Complete the {task_class.replace('_', ' ')} task; condition: {variant}."
            start_state = _page_state(task_class, variant, target_bid, aux_bid, stage="start")
            progress_state = _page_state(task_class, variant, target_bid, aux_bid, stage="progress")
            done_state = _page_state(task_class, variant, target_bid, aux_bid, stage="done")

            warmup_action = _warmup_action(variant)
            records.append(
                {
                    "id": f"{task_class}-{repeat + 1:02d}",
                    "task_class": task_class,
                    "variant": variant,
                    "instruction": instruction,
                    "screenshot": f"assets/synthetic/{task_class}-{repeat + 1:02d}.png",
                    "trajectory": [
                        {
                            "observation": start_state,
                            "action": warmup_action,
                            "next_observation": progress_state,
                        },
                        {
                            "observation": progress_state,
                            "action": target_action,
                            "next_observation": done_state,
                        },
                        {
                            "observation": done_state,
                            "action": "send_msg_to_user('Done')",
                            "next_observation": done_state,
                        },
                    ],
                    "negative_actions": _negative_actions(target_action, variant),
                    "reward_hints": {
                        "prefer_valid_action": True,
                        "prefer_reference_action": target_action,
                        "penalize_repetition": variant == "repeated previous action",
                        "penalize_unsafe_navigation": variant == "navigation safety boundary",
                        "penalize_extra_steps": True,
                    },
                }
            )
    return records


def to_sft_records(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        history = []
        for step_idx, step in enumerate(record["trajectory"]):
            rows.append(
                {
                    "id": f"{record['id']}-sft-{step_idx + 1}",
                    "task_class": record["task_class"],
                    "variant": record["variant"],
                    "instruction": record["instruction"],
                    "observation": {
                        "screenshot": record.get("screenshot"),
                        "page_state": step["observation"],
                    },
                    "history": history.copy(),
                    "target_action": step["action"],
                    "next_observation": step.get("next_observation"),
                    "reward_hints": record["reward_hints"],
                }
            )
            history.append({"observation": step["observation"], "action": step["action"]})
    return rows


def to_dpo_records(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        history = []
        negatives = record["negative_actions"]
        for step_idx, step in enumerate(record["trajectory"]):
            rejected = negatives[step_idx % len(negatives)]
            prompt = (
                "You are a multimodal web operation agent. Choose exactly one valid browser action.\n\n"
                f"Instruction:\n{record['instruction']}\n\n"
                f"Task class: {record['task_class']}\n"
                f"Scenario variant: {record['variant']}\n\n"
                f"Page State:\n{step['observation']}\n\n"
                f"History:\n{json.dumps(history, ensure_ascii=False)}"
            )
            rows.append(
                {
                    "id": f"{record['id']}-dpo-{step_idx + 1}",
                    "task_class": record["task_class"],
                    "variant": record["variant"],
                    "prompt": prompt,
                    "chosen": f"<action>{step['action']}</action>",
                    "rejected": f"<action>{rejected}</action>",
                    "rejection_reason": _rejection_reason(rejected, record["variant"]),
                }
            )
            history.append({"observation": step["observation"], "action": step["action"]})
    return rows


def to_grpo_records(records: list[dict]) -> list[dict]:
    rows = []
    for sft in to_sft_records(records):
        rows.append(
            {
                "id": sft["id"].replace("-sft-", "-grpo-"),
                "task_class": sft["task_class"],
                "variant": sft["variant"],
                "instruction": sft["instruction"],
                "observation": sft["observation"],
                "history": sft["history"],
                "reference_action": sft["target_action"],
                "candidate_actions": _candidate_actions(sft["target_action"], sft["variant"]),
                "reward_hints": sft["reward_hints"],
            }
        )
    return rows


def write_json(path: str | Path, rows: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_training_corpus(
    repeats_per_class: int = 10,
    trajectory_output: str | Path = "data/mm_webagent/training_trajectories.json",
    sft_output: str | Path = "data/mm_webagent/sft_train.jsonl",
    dpo_output: str | Path = "data/mm_webagent/dpo_train.jsonl",
    grpo_output: str | Path = "data/mm_webagent/grpo_rollout_seed.jsonl",
) -> dict[str, int]:
    trajectories = build_trajectory_records(repeats_per_class)
    sft = to_sft_records(trajectories)
    dpo = to_dpo_records(trajectories)
    grpo = to_grpo_records(trajectories)
    write_json(trajectory_output, trajectories)
    write_jsonl(sft_output, sft)
    write_jsonl(dpo_output, dpo)
    write_jsonl(grpo_output, grpo)
    return {
        "trajectories": len(trajectories),
        "sft_records": len(sft),
        "dpo_records": len(dpo),
        "grpo_records": len(grpo),
        "task_classes": len(TASK_CLASSES),
        "variants_per_class": repeats_per_class,
    }


def _page_state(task_class: str, variant: str, target_bid: int, aux_bid: int, stage: str) -> str:
    return (
        f"RootWebArea '{task_class}' stage='{stage}'\n"
        f"Scenario '{variant}'\n"
        f"[{target_bid}] button 'Primary {task_class.replace('_', ' ')} action'\n"
        f"[{aux_bid}] textbox 'Task input'\n"
        f"[{aux_bid + 20}] link 'Similar distractor'\n"
        f"[{aux_bid + 40}] status 'Current stage: {stage}'"
    )


def _warmup_action(variant: str) -> str:
    if variant == "target below the fold":
        return "scroll(0, 420)"
    if variant == "slow page with wait action":
        return "noop(500)"
    if variant == "form field has existing value":
        return "keyboard_press('Control+A')"
    if variant == "modal confirmation present":
        return "click('70')"
    return "noop(200)"


def _negative_actions(target_action: str, variant: str) -> list[str]:
    negatives = [DISTRACTOR_ACTIONS["invalid"], DISTRACTOR_ACTIONS["premature_done"], "scroll(0, 420)"]
    if variant == "repeated previous action":
        negatives.insert(0, target_action)
    if variant == "navigation safety boundary":
        negatives.insert(0, DISTRACTOR_ACTIONS["unsafe_goto"])
    return negatives


def _candidate_actions(reference_action: str, variant: str) -> list[str]:
    candidates = [reference_action, "scroll(0, 420)", "noop(200)", "click_bad"]
    if variant == "navigation safety boundary":
        candidates.append(DISTRACTOR_ACTIONS["unsafe_goto"])
    if variant == "requires terminal response":
        candidates.append("send_msg_to_user('Done')")
    return candidates


def _rejection_reason(action: str, variant: str) -> str:
    if action == "click_bad":
        return "invalid action syntax"
    if action.startswith("goto"):
        return "unsafe navigation without explicit user URL"
    if action.startswith("send_msg_to_user") and variant != "requires terminal response":
        return "premature terminal response before task completion"
    return "lower progress action than the chosen reference action"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT, DPO, and GRPO data for the 20-class Agent corpus.")
    parser.add_argument("--repeats-per-class", type=int, default=10)
    parser.add_argument("--trajectory-output", default="data/mm_webagent/training_trajectories.json")
    parser.add_argument("--sft-output", default="data/mm_webagent/sft_train.jsonl")
    parser.add_argument("--dpo-output", default="data/mm_webagent/dpo_train.jsonl")
    parser.add_argument("--grpo-output", default="data/mm_webagent/grpo_rollout_seed.jsonl")
    args = parser.parse_args()

    print(
        json.dumps(
            build_training_corpus(
                repeats_per_class=args.repeats_per_class,
                trajectory_output=args.trajectory_output,
                sft_output=args.sft_output,
                dpo_output=args.dpo_output,
                grpo_output=args.grpo_output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
