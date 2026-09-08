# Run And Train Guide

## 1. Prepare Environment

Install the lightweight dependencies first:

```bash
pip install -r requirements.txt
```

For real model training, use a CUDA environment and install versions of
`torch`, `transformers`, `peft`, `trl`, and `accelerate` that match the GPU.

## 2. Build Training Data

Convert WebWorld-style trajectories into SFT records and GRPO rollout seeds:

```bash
python -m mm_webagent.cli build-data \
  --input examples/sample_trajectory.json \
  --sft-output data/mm_webagent/sft_train.jsonl \
  --grpo-output data/mm_webagent/grpo_rollout_seed.jsonl
```

## 3. Evaluate GraphRAG Before Training

```bash
python -m mm_webagent.cli eval-graphrag \
  --documents examples/operation_memory_docs.json \
  --queries examples/operation_memory_eval.json \
  --top-k 3
```

This checks whether GraphRAG retrieves the right operation path before the
policy model is trained.

## 4. Run A Local Agent Demo

```bash
python examples/run_graphrag_demo.py
python examples/run_agent_demo.py
```

`run_graphrag_demo.py` shows the retrieved path. `run_agent_demo.py` shows the
minimal action loop.

## 5. LoRA-SFT Cold Start

Dry-run validation:

```bash
python -m mm_webagent.training.sft_train \
  --data data/mm_webagent/sft_train.jsonl \
  --model Qwen/Qwen3-0.6B \
  --dry-run
```

Real TRL SFT training:

```bash
python -m mm_webagent.training.sft_train \
  --data data/mm_webagent/sft_train.jsonl \
  --model Qwen/Qwen3-0.6B \
  --output-dir outputs/qwen3_0_6b_sft_lora \
  --epochs 1 \
  --batch-size 1 \
  --train
```

After this stage, the model should learn the browser action format and basic
element selection.

## 6. DPO Preference Alignment

Dry-run validation:

```bash
python -m mm_webagent.post_training.dpo_train \
  --data examples/sample_preferences.jsonl \
  --model Qwen/Qwen3-0.6B
```

Real TRL DPO training:

```bash
python -m mm_webagent.post_training.dpo_train \
  --data examples/sample_preferences.jsonl \
  --model Qwen/Qwen3-0.6B \
  --output-dir outputs/qwen3_0_6b_dpo_lora \
  --train
```

DPO teaches the model to prefer valid, task-progressing actions over invalid,
repeated, or low-value actions.

## 7. PPO/RLHF Rollout Validation

```bash
python -m mm_webagent.post_training.ppo_train \
  --episodes examples/sample_episodes.jsonl \
  --model outputs/qwen3_0_6b_sft_lora
```

Full PPO requires an online rollout loop:

```text
policy -> GraphRAG context -> action -> WebWorld/browser feedback -> reward -> PPO update
```

## 8. GRPO Grouped Rollout Validation

```bash
python -m mm_webagent.training.grpo_train \
  --rollout-seed data/mm_webagent/grpo_rollout_seed.jsonl \
  --group-size 4
```

Full GRPO samples multiple actions for the same state, compares shaped rewards
within the group, and updates the policy toward better actions.

## 9. Deploy With vLLM

Print the launch command:

```bash
python -m mm_webagent.serving.vllm_server \
  --model outputs/qwen3_0_6b_sft_lora \
  --served-model-name mm-webagent
```

Run the printed command in the training environment:

```bash
vllm serve outputs/qwen3_0_6b_sft_lora \
  --served-model-name mm-webagent \
  --port 8000 \
  --max-model-len 8192
```

Then call the service with `mm_webagent/serving/client.py`.

## Recommended Learning Order

```text
GraphRAG retrieval validation
  -> LoRA-SFT on Qwen3-0.6B
  -> DPO chosen/rejected action preference
  -> PPO reward validation
  -> GRPO grouped action optimization
  -> vLLM deployment
  -> full Agent benchmark
```
