# Resume Claim Evidence Map

This file maps the resume description of the multimodal web-operation Agent to
concrete repository evidence.

## Agent Input And Action Generation

| Claim | Evidence |
| --- | --- |
| Use user instruction, screenshot, page state, and action history. | `mm_webagent/agent/prompt_builder.py` assembles all four inputs into the policy prompt. |
| Generate click, fill, scroll, goto, terminal, and browser actions. | `mm_webagent/data/action_schema.py` defines the unified action set and validation logic. |
| Parse model output into executable actions. | `mm_webagent/agent/action_parser.py` extracts `<action>...</action>` responses. |
| Keep per-step action history. | `mm_webagent/agent/policy_agent.py` appends observation-action pairs after every decision. |

## Trajectory Data

| Claim | Evidence |
| --- | --- |
| Convert WebWorld trajectories into supervised action records. | `mm_webagent/data/trajectory_builder.py::to_sft_records`. |
| Build GRPO rollout seeds with reward hints. | `mm_webagent/data/trajectory_builder.py::to_grpo_rollout_seed`. |
| Provide sample train data. | `data/mm_webagent/sft_train.jsonl` and `data/mm_webagent/grpo_rollout_seed.jsonl`. |

## Retrieval

| Claim | Evidence |
| --- | --- |
| Hybrid Operation RAG. | `mm_webagent/rag/operation_retriever.py`. |
| Dense and sparse retrieval. | `mm_webagent/rag/hybrid_retriever.py`, `qdrant_store.py`, and `bm25_jieba.py`. |
| Field-aware operation matching. | `mm_webagent/rag/field_bm25.py`. |
| Late interaction over labels and action strings. | `mm_webagent/rag/late_interaction.py`. |
| GraphRAG over task, page, action, rule, failure, and outcome nodes. | `mm_webagent/rag/graphrag.py`. |
| Offline retrieval benchmark. | `mm_webagent/eval/retrieval_eval.py` and `docs/retrieval_benchmark_report.md`. |

## Training And Reward

| Claim | Evidence |
| --- | --- |
| LoRA-SFT entry point. | `mm_webagent/training/sft_train.py` and `mm_webagent/training/lora_config.py`. |
| DPO preference validation and optional training. | `mm_webagent/post_training/dpo_train.py`. |
| PPO/RLHF rollout reward validation. | `mm_webagent/post_training/ppo_train.py`. |
| GRPO grouped rollout reward validation. | `mm_webagent/training/grpo_train.py`. |
| Reward terms for completion, validity, steps, repetition, and invalid actions. | `mm_webagent/training/reward.py`. |

## Runtime And Deployment

| Claim | Evidence |
| --- | --- |
| Observe-decide-execute-feedback loop. | `mm_webagent/workflow/langgraph_runner.py`. |
| GraphRAG-enhanced runner. | `mm_webagent/workflow/graphrag_runner.py`. |
| Optional real LangGraph StateGraph. | `LangGraphWebAgentRunner.build_state_graph`. |
| Runtime strategy routing. | `mm_webagent/workflow/strategy_router.py`. |
| vLLM launch command. | `mm_webagent/serving/vllm_server.py`. |
| OpenAI-compatible client. | `mm_webagent/serving/client.py`. |

## Evaluation

| Claim | Evidence |
| --- | --- |
| Success rate, invalid action rate, steps, and token metrics. | `mm_webagent/eval/metrics.py` and `mm_webagent/eval/evaluator.py`. |
| 20-class Agent benchmark fixture. | `mm_webagent/eval/agent_benchmark.py`. |
| Resume-style result reproduction. | `docs/agent_benchmark_20class_report.md` and `experiments/agent_benchmark_20class_results.json`. |

## Honest Scope Boundary

The repository now contains reproducible local evidence for the project
architecture, retrieval benchmark, metric definitions, and 20-class offline
Agent benchmark. Full external validation still requires a GPU environment,
real Qwen-VL checkpoints, vLLM service logs, and live browser/WebWorld rollout
traces.
