# Training And Deployment Evidence

This document records the local commands that validate the training and
deployment chain for the multimodal web-operation Agent scaffold.

## 1. Build WebWorld Trajectory Data

Command:

```bash
python -m mm_webagent.cli build-data \
  --input examples/sample_trajectory.json \
  --sft-output data/mm_webagent/sft_train.jsonl \
  --grpo-output data/mm_webagent/grpo_rollout_seed.jsonl
```

Verified output:

```text
Built data/mm_webagent/sft_train.jsonl and data/mm_webagent/grpo_rollout_seed.jsonl
```

Evidence:

- `data/mm_webagent/sft_train.jsonl` contains supervised action-prediction
  records.
- `data/mm_webagent/grpo_rollout_seed.jsonl` contains grouped rollout seeds and
  reward hints.

## 2. LoRA-SFT Dry Run

Command:

```bash
python -m mm_webagent.training.sft_train \
  --data data/mm_webagent/sft_train.jsonl \
  --model Qwen/Qwen3-0.6B \
  --dry-run
```

Verified output:

```text
Model: Qwen/Qwen3-0.6B
Output: outputs/sft_lora
LoRA: {'r': 16, 'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], 'lora_alpha': 32, 'lora_dropout': 0.05}
Preview records: 2
```

What this proves:

- The SFT data is readable.
- The LoRA adapter configuration is assembled.
- The same entry point can launch real TRL SFT with `--train` in a CUDA
  environment.

## 3. DPO Preference Validation

Command:

```bash
python -m mm_webagent.post_training.dpo_train \
  --data examples/sample_preferences.jsonl \
  --model Qwen/Qwen3-0.6B
```

Verified output:

```text
Model: Qwen/Qwen3-0.6B
{
  "valid": 2,
  "invalid": 0
}
```

What this proves:

- Chosen/rejected preference data follows the expected TRL DPO format.
- The DPO training entry point is wired and can be launched with `--train` when
  model dependencies and GPU resources are available.

## 4. PPO/RLHF Reward Validation

Command:

```bash
python -m mm_webagent.post_training.ppo_train \
  --episodes examples/sample_episodes.jsonl \
  --model outputs/qwen3_0_6b_sft_lora
```

Verified output:

```text
Model: outputs/qwen3_0_6b_sft_lora
Episodes: 2
Average PPO-style return: 0.2800
```

What this proves:

- Episode actions can be scored by the reward model.
- Completion, validity, invalid-action penalty, repetition penalty, and step
  penalty terms are available before connecting online PPO rollouts.

## 5. GRPO Reward Validation

Command:

```bash
python -m mm_webagent.training.grpo_train \
  --rollout-seed data/mm_webagent/grpo_rollout_seed.jsonl \
  --group-size 4
```

Verified output:

```text
Checked 2 rollout seeds.
Average shaped reward: 0.1900
Group size: 4
```

What this proves:

- GRPO rollout seeds can be read.
- The grouped rollout reward path is ready for replacement with sampled model
  candidates in a full training environment.

## 6. vLLM Deployment Command

Command:

```bash
python -m mm_webagent.serving.vllm_server \
  --model outputs/qwen3_0_6b_sft_lora \
  --served-model-name mm-webagent \
  --port 8000
```

Verified output:

```text
vllm serve outputs/qwen3_0_6b_sft_lora --served-model-name mm-webagent --port 8000 --max-model-len 8192
```

What this proves:

- The project defines a concrete vLLM launch boundary.
- `mm_webagent/serving/client.py` can call the resulting OpenAI-compatible
  endpoint after the service is started in the target environment.

## 7. Retrieval Benchmark

Command:

```bash
python -m mm_webagent.cli eval-graphrag \
  --documents examples/operation_memory_docs.json \
  --queries examples/operation_memory_eval.json \
  --top-k 3
```

Verified output summary:

| Metric | Baseline Hybrid | Operation Memory | GraphRAG |
| --- | ---: | ---: | ---: |
| hit@3 | 0.75 | 1.00 | 1.00 |
| MRR | 0.75 | 1.00 | 1.00 |
| failure case recall | 0.50 | 0.50 | 0.50 |
| action hint hit rate | 0.75 | 1.00 | 1.00 |

What this proves:

- Operation Memory and GraphRAG retrieve the expected operation memory more
  reliably than the baseline Hybrid RAG fixture.
- This retrieval benchmark is separate from the full Agent task benchmark.

## 8. 20-Class Agent Benchmark

Command:

```bash
python -m mm_webagent.cli eval-agent-benchmark \
  --output experiments/agent_benchmark_20class_results.json
```

Verified output summary:

| Metric | Direct Policy | Operation Memory | GraphRAG |
| --- | ---: | ---: | ---: |
| success rate | 0.57 | 0.71 | 0.78 |
| invalid action rate | 0.3000 | 0.1798 | 0.1195 |
| average steps | 11.3 | 9.4 | 8.7 |
| average tokens | 1160 | 980 | 870 |

GraphRAG deltas against direct policy:

- success rate: +0.21
- invalid action rate: -0.1805
- average steps: -2.6
- token reduction: 0.25

The benchmark currently contains 20 task classes and 10 deterministic scenario
variants per class, for 200 episodes per strategy.

## Scope Boundary

The commands above validate the local project scaffold and reproducible offline
fixtures. A final production result should additionally include:

- GPU training logs for Qwen-VL LoRA-SFT and post-training.
- vLLM service startup logs and health checks.
- Live browser or WebWorld rollout traces over the same 20 task classes.
- A frozen evaluation set with seed, model checkpoint, and environment version.
