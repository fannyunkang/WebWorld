"""Policy wrapper for Qwen-VL or OpenAI-compatible inference backends."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any, Callable

from mm_webagent.agent.action_parser import extract_action
from mm_webagent.agent.prompt_builder import build_policy_prompt
from mm_webagent.data.action_schema import ParsedAction


ModelCaller = Callable[..., str]


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
        response = self._call_model(prompt, screenshot)
        action = extract_action(response)
        self.history.append({"observation": page_state, "action": action.raw})
        return action

    def _call_model(self, prompt: str, screenshot: str | None) -> str:
        """Call text-only and multimodal model clients through one adapter."""
        signature = inspect.signature(self.model_caller)
        parameters = signature.parameters
        accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())
        if screenshot and ("screenshot" in parameters or "screenshot_path" in parameters or accepts_kwargs):
            if "screenshot_path" in parameters and "screenshot" not in parameters:
                return self.model_caller(prompt, screenshot_path=screenshot)
            return self.model_caller(prompt, screenshot=screenshot)
        return self.model_caller(prompt)
