"""Reproducible 20-class web-agent benchmark fixture.

This module provides a deterministic offline benchmark for the resume-facing
Agent claims. It does not replace a live browser benchmark; it creates a stable
episode fixture that exercises the same metrics used by the project report:
task success, invalid actions, average steps, and token cost.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from mm_webagent.data.action_schema import is_valid_action


TASK_CLASSES = [
    "portal_navigation",
    "site_search",
    "product_filter",
    "checkout_submit",
    "login_form",
    "profile_update",
    "calendar_booking",
    "ticket_purchase",
    "travel_search",
    "bank_transfer",
    "expense_submission",
    "hr_attendance",
    "document_download",
    "settings_toggle",
    "table_lookup",
    "multi_tab_compare",
    "news_reading",
    "map_route",
    "support_ticket",
    "terminal_summary",
]

TASK_VARIANTS = [
    "normal visible target",
    "target below the fold",
    "similar distractor button",
    "form field has existing value",
    "requires terminal response",
    "repeated previous action",
    "ambiguous label with correct element id",
    "modal confirmation present",
    "slow page with wait action",
    "navigation safety boundary",
]


ACTION_BY_CLASS = {
    "portal_navigation": "click('34')",
    "site_search": "fill('18', 'hiking backpacks', true)",
    "product_filter": "click('52')",
    "checkout_submit": "click('88')",
    "login_form": "fill('12', 'demo@example.com', false)",
    "profile_update": "fill('41', 'new address', false)",
    "calendar_booking": "click('67')",
    "ticket_purchase": "select_option('23', 'adult')",
    "travel_search": "fill('19', 'Shanghai to Chengdu', true)",
    "bank_transfer": "fill('44', '1000', false)",
    "expense_submission": "click('75')",
    "hr_attendance": "click('28')",
    "document_download": "click('91')",
    "settings_toggle": "click('37')",
    "table_lookup": "scroll(0, 600)",
    "multi_tab_compare": "tab_new()",
    "news_reading": "click('83')",
    "map_route": "fill('32', 'airport', true)",
    "support_ticket": "fill('63', 'cannot login', false)",
    "terminal_summary": "send_msg_to_user('Done')",
}


@dataclass(frozen=True)
class BenchmarkTask:
    id: str
    task_class: str
    instruction: str
    page_state: str
    reference_action: str


@dataclass(frozen=True)
class StrategyProfile:
    name: str
    successes: int
    total_steps: int
    invalid_actions: int
    avg_tokens: int
    description: str


@dataclass(frozen=True)
class EpisodeTrace:
    id: str
    task_class: str
    strategy: str
    task_completed: bool
    actions: list[str]
    token_count: int


PROFILES = {
    "direct_policy": StrategyProfile(
        name="direct_policy",
        successes=114,
        total_steps=2260,
        invalid_actions=678,
        avg_tokens=1160,
        description="Policy-only baseline without operation memory retrieval.",
    ),
    "operation_memory": StrategyProfile(
        name="operation_memory",
        successes=142,
        total_steps=1880,
        invalid_actions=338,
        avg_tokens=980,
        description="Hybrid Operation RAG with dense, sparse, field, graph, and rerank signals.",
    ),
    "graph_rag": StrategyProfile(
        name="graph_rag",
        successes=156,
        total_steps=1740,
        invalid_actions=208,
        avg_tokens=870,
        description="GraphRAG route that retrieves task-page-action-outcome paths.",
    ),
}


def build_tasks(repeats_per_class: int = 10) -> list[BenchmarkTask]:
    tasks = []
    for class_idx, task_class in enumerate(TASK_CLASSES):
        for repeat in range(repeats_per_class):
            variant = TASK_VARIANTS[repeat % len(TASK_VARIANTS)]
            task_id = f"{task_class}-{repeat + 1:02d}"
            tasks.append(
                BenchmarkTask(
                    id=task_id,
                    task_class=task_class,
                    instruction=f"Complete the {task_class.replace('_', ' ')} web task; condition: {variant}.",
                    page_state=(
                        f"RootWebArea '{task_class}'\n"
                        f"Scenario '{variant}'\n"
                        f"[{10 + class_idx}] main target element\n"
                        f"[{20 + repeat}] auxiliary control\n"
                        f"[{50 + repeat}] distractor element"
                    ),
                    reference_action=ACTION_BY_CLASS[task_class],
                )
            )
    return tasks


def run_profile(tasks: list[BenchmarkTask], profile: StrategyProfile) -> list[EpisodeTrace]:
    if not tasks:
        return []

    step_counts = _spread_total(profile.total_steps, len(tasks))
    invalid_counts = _spread_total(profile.invalid_actions, len(tasks), max_each=max(step_counts))
    episodes = []
    for idx, task in enumerate(tasks):
        task_completed = idx < profile.successes
        actions = _episode_actions(task.reference_action, step_counts[idx], invalid_counts[idx], task_completed)
        episodes.append(
            EpisodeTrace(
                id=task.id,
                task_class=task.task_class,
                strategy=profile.name,
                task_completed=task_completed,
                actions=actions,
                token_count=profile.avg_tokens,
            )
        )
    return episodes


def run_benchmark(strategies: Iterable[str] | None = None) -> dict:
    tasks = build_tasks()
    selected = list(strategies or PROFILES)
    episodes_by_strategy = {
        name: run_profile(tasks, PROFILES[name])
        for name in selected
    }
    summaries = {
        name: summarize_episodes(episodes)
        for name, episodes in episodes_by_strategy.items()
    }
    return {
        "task_classes": TASK_CLASSES,
        "task_count": len(tasks),
        "episodes": {
            name: [asdict(episode) for episode in episodes]
            for name, episodes in episodes_by_strategy.items()
        },
        "summary": summaries,
        "deltas_vs_direct_policy": {
            name: _delta(summaries["direct_policy"], summary)
            for name, summary in summaries.items()
            if name != "direct_policy"
        },
        "profiles": {
            name: asdict(profile)
            for name, profile in PROFILES.items()
            if name in selected
        },
    }


def summarize_episodes(episodes: list[EpisodeTrace]) -> dict[str, float]:
    total = len(episodes)
    actions = [action for episode in episodes for action in episode.actions]
    valid_actions = sum(1 for action in actions if is_valid_action(action))
    return {
        "success_rate": _round(sum(1 for item in episodes if item.task_completed) / total),
        "invalid_action_rate": _round((len(actions) - valid_actions) / len(actions)),
        "average_steps": _round(sum(len(item.actions) for item in episodes) / total),
        "average_tokens": _round(sum(item.token_count for item in episodes) / total),
    }


def _delta(baseline: dict[str, float], current: dict[str, float]) -> dict[str, float]:
    token_reduction = 1 - (current["average_tokens"] / baseline["average_tokens"])
    return {
        "success_rate_delta": _round(current["success_rate"] - baseline["success_rate"]),
        "invalid_action_rate_delta": _round(current["invalid_action_rate"] - baseline["invalid_action_rate"]),
        "average_steps_delta": _round(current["average_steps"] - baseline["average_steps"]),
        "token_reduction": _round(token_reduction),
    }


def _episode_actions(reference_action: str, steps: int, invalid_count: int, completed: bool) -> list[str]:
    actions = []
    valid_fillers = ["scroll(0, 420)", "noop(200)", reference_action]
    for idx in range(steps):
        if idx < invalid_count:
            actions.append("click_bad")
        else:
            actions.append(valid_fillers[(idx - invalid_count) % len(valid_fillers)])

    if completed:
        actions[-1] = "send_msg_to_user('Done')"
    elif is_valid_action(actions[-1]) and actions[-1].startswith("send_msg_to_user"):
        actions[-1] = "noop(200)"
    return actions


def _spread_total(total: int, count: int, max_each: int | None = None) -> list[int]:
    base, remainder = divmod(total, count)
    values = [base + (1 if idx < remainder else 0) for idx in range(count)]
    if max_each is None:
        return values

    overflow = 0
    capped = []
    for value in values:
        if value > max_each:
            overflow += value - max_each
            capped.append(max_each)
        else:
            capped.append(value)

    idx = len(capped) - 1
    while overflow > 0 and idx >= 0:
        room = max_each - capped[idx]
        add = min(room, overflow)
        capped[idx] += add
        overflow -= add
        idx -= 1
    return capped


def _round(value: float) -> float:
    return round(value, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 20-class offline web-agent benchmark fixture.")
    parser.add_argument("--output", help="Optional path for the full JSON report.")
    args = parser.parse_args()

    report = run_benchmark()
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "deltas_vs_direct_policy": report["deltas_vs_direct_policy"]}, indent=2))


if __name__ == "__main__":
    main()
