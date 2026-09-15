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

    def build_state_graph(self):
        """Build a real LangGraph StateGraph when langgraph is installed.

        The fallback ``run`` method above keeps demos dependency-light. This
        method exposes the same observe-decide-execute loop through LangGraph
        for environments that install the full runtime stack.
        """
        try:
            from langgraph.graph import END, StateGraph
        except ImportError as exc:
            raise RuntimeError("Install langgraph to build the StateGraph workflow.") from exc

        def decide(state: AgentState) -> AgentState:
            action = self.agent.decide(
                instruction=state.instruction,
                page_state=state.observation,
                screenshot=state.screenshot,
                retrieved_context=self.retrieve_context(state) if self.retrieve_context else None,
            )
            state.history.append({"observation": state.observation, "action": action.raw})
            if action.is_terminal:
                state.done = True
            else:
                state.observation = self.environment_step(action.raw)
            return state

        def route(state: AgentState) -> str:
            return END if state.done else "decide"

        graph = StateGraph(AgentState)
        graph.add_node("decide", decide)
        graph.set_entry_point("decide")
        graph.add_conditional_edges("decide", route)
        return graph.compile()
