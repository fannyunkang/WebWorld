"""Policy wrapper for Qwen-VL or OpenAI-compatible inference backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from mm_webagent.agent.action_parser import extract_action
from mm_webagent.agent.prompt_builder import build_policy_prompt
from mm_webagent.data.action_schema import ParsedAction


ModelCaller = Callable[[str], str]


@dataclass
class PolicyAgent:
    """Small adapter that turns observations into one browser action."""

    model_caller: ModelCaller
    history: list[dict[str, Any]] = field(default_factory=list)

    def decide(
        self,
        instruction: str,
        page_state: str,
        screenshot: str | None = None,
        retrieved_context: str | None = None,
    ) -> ParsedAction:
        prompt = build_policy_prompt(
            instruction=instruction,
            page_state=page_state,
            history=self.history,
            screenshot=screenshot,
            retrieved_context=retrieved_context,
        )
        response = self.model_caller(prompt)
        action = extract_action(response)
        self.history.append({"observation": page_state, "action": action.raw})
        return action
