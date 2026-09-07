"""Reward functions for GRPO-style web-agent optimization."""

from __future__ import annotations

from dataclasses import dataclass

from mm_webagent.data.action_schema import is_valid_action, parse_action


@dataclass(frozen=True)
class RewardBreakdown:
    task_success: float = 0.0
    action_validity: float = 0.0
    step_penalty: float = 0.0
    repeat_penalty: float = 0.0
    invalid_action_penalty: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.task_success
            + self.action_validity
            + self.step_penalty
            + self.repeat_penalty
            + self.invalid_action_penalty
        )


def compute_reward(
    action: str,
    task_completed: bool,
    step_index: int,
    previous_actions: list[str] | None = None,
) -> RewardBreakdown:
    previous_actions = previous_actions or []
    valid = is_valid_action(action)
    repeat_count = sum(1 for prev in previous_actions[-3:] if prev == action)

    terminal_bonus = 0.0
    if valid and parse_action(action).is_terminal and task_completed:
        terminal_bonus = 0.5

    return RewardBreakdown(
        task_success=1.0 + terminal_bonus if task_completed else 0.0,
        action_validity=0.2 if valid else 0.0,
        step_penalty=-0.02 * max(step_index, 0),
        repeat_penalty=-0.15 * repeat_count,
        invalid_action_penalty=0.0 if valid else -1.0,
    )
