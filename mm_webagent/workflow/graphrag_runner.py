"""GraphRAG-enhanced Agent runner."""

from __future__ import annotations

from mm_webagent.agent.policy_agent import PolicyAgent
from mm_webagent.rag.graphrag import GraphRAGRetriever
from mm_webagent.workflow.langgraph_runner import AgentState, EnvironmentStep, LangGraphWebAgentRunner


def build_graphrag_runner(
    agent: PolicyAgent,
    retriever: GraphRAGRetriever,
    environment_step: EnvironmentStep,
) -> LangGraphWebAgentRunner:
    def retrieve_context(state: AgentState) -> str:
        return retriever.build_context(
            instruction=state.instruction,
            page_state=state.observation,
            last_actions=[item["action"] for item in state.history],
        )

    return LangGraphWebAgentRunner(
        agent=agent,
        environment_step=environment_step,
        retrieve_context=retrieve_context,
    )
