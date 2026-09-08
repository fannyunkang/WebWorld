"""Runtime strategies for multimodal web agents."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RunStrategy(str, Enum):
    DIRECT_POLICY = "direct_policy"
    RAG_ENHANCED = "rag_enhanced"
    GRAPH_RAG = "graph_rag"
    REFLECTION_RETRY = "reflection_retry"
    RL_ROLLOUT = "rl_rollout"


@dataclass(frozen=True)
class StrategyDecision:
    strategy: RunStrategy
    reason: str


def choose_strategy(
    instruction: str,
    page_state: str,
    last_actions: list[str] | None = None,
    use_graph_rag: bool = True,
) -> StrategyDecision:
    """Route an episode step to the cheapest strategy that has enough context."""
    last_actions = last_actions or []
    repeated = len(last_actions) >= 2 and len(set(last_actions[-2:])) == 1
    complex_page = page_state.count("\n") > 12 or any(word in instruction.lower() for word in ["checkout", "login", "form"])

    if repeated:
        return StrategyDecision(RunStrategy.REFLECTION_RETRY, "recent repeated action detected")
    if use_graph_rag and complex_page:
        return StrategyDecision(RunStrategy.GRAPH_RAG, "complex state benefits from trajectory graph retrieval")
    if use_graph_rag:
        return StrategyDecision(RunStrategy.RAG_ENHANCED, "retrieve operation memory before policy inference")
    return StrategyDecision(RunStrategy.DIRECT_POLICY, "simple task can use policy-only inference")
