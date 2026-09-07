"""Document the post-training flow from SFT to RLHF-style optimization."""

from __future__ import annotations


POST_TRAINING_STAGES = [
    "Qwen3-0.6B LoRA SFT validates action-format learning on small trajectories.",
    "DPO validates preference alignment: chosen correct actions beat rejected invalid or repetitive actions.",
    "PPO/RLHF assigns scalar rewards to rollouts and checks policy optimization signals.",
    "GRPO removes the value-model dependency and compares grouped candidate actions with shaped rewards.",
    "vLLM serves the selected policy for LangGraph workflow evaluation.",
]


def describe_pipeline() -> str:
    return "\n".join(f"{idx + 1}. {stage}" for idx, stage in enumerate(POST_TRAINING_STAGES))
