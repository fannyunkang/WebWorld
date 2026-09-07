"""Minimal local demo for the multimodal web-agent scaffold."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mm_webagent.agent.policy_agent import PolicyAgent
from mm_webagent.workflow.langgraph_runner import AgentState, LangGraphWebAgentRunner


def mock_policy(prompt: str) -> str:
    if "RootWebArea 'Shopping'" in prompt:
        return "<reason>The shopping section is open.</reason><action>send_msg_to_user('Done')</action>"
    if "Shopping" in prompt:
        return "<reason>Open the shopping section.</reason><action>click('34')</action>"
    return "<reason>The task is complete.</reason><action>send_msg_to_user('Done')</action>"


def mock_environment(action: str) -> str:
    return "RootWebArea 'Shopping'\\n[50] heading 'Shopping results for hiking backpacks'"


def main() -> None:
    agent = PolicyAgent(model_caller=mock_policy)
    runner = LangGraphWebAgentRunner(agent=agent, environment_step=mock_environment)
    state = AgentState(
        instruction="Open the shopping section.",
        observation="RootWebArea 'Global Start'\\n[34] link 'Shopping'",
    )
    final_state = runner.run(state, max_steps=2)
    for item in final_state.history:
        print(item["action"])


if __name__ == "__main__":
    main()
