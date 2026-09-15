# Resume Alignment Design

This repository wraps QwenLM/WebWorld as the environment layer for a multimodal
web-operation agent training project.

## Alignment With The Resume Description

| Resume point | Repository module |
| --- | --- |
| Build a multimodal web-operation Agent from Qwen3-VL | `mm_webagent/agent`, `mm_webagent/data/screenshot_encoder.py` |
| Generate click, input, and scroll actions from screenshots, user instruction, and history | `mm_webagent/agent/prompt_builder.py`, `mm_webagent/data/action_schema.py` |
| Send real screenshot payloads to Qwen-VL-compatible inference | `mm_webagent/data/screenshot_encoder.py`, `mm_webagent/serving/vlm_client.py` |
| Observe and execute actions in a browser loop | `mm_webagent/browser/observer.py`, `mm_webagent/browser/executor.py` |
| Build Qwen-VL multimodal SFT messages and grounding screenshots | `mm_webagent/training/vl_dataset_format.py`, `mm_webagent/data/screenshot_annotator.py` |
| Build web-operation trajectory data | `mm_webagent/data/trajectory_builder.py` |
| Retrieve similar trajectories, site rules, failed actions, and reward hints | `mm_webagent/rag`, `mm_webagent/configs/rag.yaml` |
| Use LoRA for supervised fine-tuning | `mm_webagent/training/sft_train.py`, `mm_webagent/training/lora_config.py` |
| Validate SFT and DPO on Qwen3-0.6B + LoRA + TRL before scaling | `mm_webagent/post_training/dpo_train.py`, `mm_webagent/configs/rlhf.yaml` |
| Run PPO/RLHF rollout reward checks | `mm_webagent/post_training/ppo_train.py` |
| Use GRPO with task completion, action validity, step count, and repetition rewards | `mm_webagent/training/grpo_train.py`, `mm_webagent/training/reward.py` |
| Deploy model inference with vLLM | `mm_webagent/serving/vllm_server.py`, `mm_webagent/serving/client.py` |
| Manage observe, decide, execute, and feedback loops with LangGraph | `mm_webagent/workflow/langgraph_runner.py` |
| Evaluate success rate, invalid action rate, average steps, and token usage | `mm_webagent/eval/metrics.py`, `mm_webagent/eval/evaluator.py` |

## Relationship To WebWorld

WebWorld remains the world-model and simulation component:

```text
agent action + current page state -> simulated next page state
```

The `mm_webagent` package adds the policy-agent layer above it:

```text
screenshot/page state + instruction + history -> browser action
```

Hybrid RAG adds an experience layer before action generation:

```text
instruction/page state/history
  -> query rewrite
  -> Qdrant + BGE-M3 dense retrieval
  -> BM25/jieba sparse retrieval
  -> BGE reranking
  -> compressed operation context
  -> policy action
```

This RAG chain retrieves trajectories and operational constraints rather than
ordinary factual passages, so it is directly tied to element selection, repeated
action avoidance, and reward shaping.

## Training And Deployment Pipeline

```text
WebWorld trajectories
  -> SFT data
  -> Qwen3-0.6B + LoRA + TRL quick validation
  -> Qwen-VL SFT
  -> DPO preference alignment
  -> PPO/RLHF rollout optimization
  -> GRPO grouped action optimization
  -> vLLM deployment
  -> LangGraph Agent benchmark
```

The Qwen3-0.6B validation stage checks whether the data and trainer setup are
effective before expensive Qwen-VL training. It measures action-format validity,
action accuracy, invalid action rate, and repetition rate on held-out records.
The benchmark stage is broader: it runs complete tasks and reports success rate,
average steps, invalid actions, and token consumption.

This separation makes the project read as an agent-training framework while
preserving the original WebWorld role as a scalable environment for generating
and evaluating trajectories.

## Current Status

The project now contains a complete first-pass training and evaluation scaffold.
Heavyweight model training is intentionally represented as configurable entry
points so the repository can be inspected, extended, and dry-run without
requiring immediate access to Qwen-VL checkpoints or GPUs.
