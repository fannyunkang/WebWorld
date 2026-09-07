"""Task-level metrics matching the resume project's evaluation claims."""

from __future__ import annotations

from dataclasses import dataclass

from mm_webagent.data.action_schema import is_valid_action


@dataclass(frozen=True)
class Episode:
    task_completed: bool
    actions: list[str]
    token_count: int = 0


def success_rate(episodes: list[Episode]) -> float:
    return _ratio(sum(1 for item in episodes if item.task_completed), len(episodes))


def invalid_action_rate(episodes: list[Episode]) -> float:
    actions = [action for episode in episodes for action in episode.actions]
    return _ratio(sum(1 for action in actions if not is_valid_action(action)), len(actions))


def average_steps(episodes: list[Episode]) -> float:
    return sum(len(item.actions) for item in episodes) / len(episodes) if episodes else 0.0


def average_tokens(episodes: list[Episode]) -> float:
    return sum(item.token_count for item in episodes) / len(episodes) if episodes else 0.0


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
