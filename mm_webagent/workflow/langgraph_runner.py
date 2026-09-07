"""LangGraph workflow wrapper with a dependency-free fallback runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from mm_webagent.agent.policy_agent import PolicyAgent


EnvironmentStep = Callable[[str], str]


@dataclass
class AgentState:
    instruction: str
    observation: str
    screenshot: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)
    done: bool = False


RetrieveContext = Callable[[AgentState], str]


class LangGraphWebAgentRunner:
    """Observe -> decide -> execute/simulate -> feedback loop."""

    def __init__(
        self,
        agent: PolicyAgent,
        environment_step: EnvironmentStep,
        retrieve_context: RetrieveContext | None = None,
    ):
        self.agent = agent
        self.environment_step = environment_step
        self.retrieve_context = retrieve_context

    def run(self, state: AgentState, max_steps: int = 12) -> AgentState:
        for _ in range(max_steps):
            action = self.agent.decide(
                instruction=state.instruction,
                page_state=state.observation,
                screenshot=state.screenshot,
                retrieved_context=self.retrieve_context(state) if self.retrieve_context else None,
            )
            state.history.append({"observation": state.observation, "action": action.raw})
            if action.is_terminal:
                state.done = True
                break
            state.observation = self.environment_step(action.raw)
        return state
