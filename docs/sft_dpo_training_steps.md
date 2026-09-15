# SFT And DPO Data Training Steps

## Current Data Status

The project now has a formal 20-class training corpus under
`data/mm_webagent/`.

| File | Records | Purpose |
| --- | ---: | --- |
| `training_trajectories.json` | 200 | Source multi-step operation trajectories. |
| `sft_train.jsonl` | 600 | Supervised action prediction records. |
| `dpo_train.jsonl` | 600 | Chosen/rejected action preference records. |
| `grpo_rollout_seed.jsonl` | 600 | Rollout seeds and reward hints for GRPO-style optimization. |

The corpus uses 20 task classes and 10 scenario variants per class. Each
trajectory has three decision steps:

```text
start observation -> warmup/recovery action
progress observation -> task-specific reference action
done observation -> terminal response
```

## Step 1. Build The Training Corpus

PowerShell:

```powershell
python -m mm_webagent.cli build-training-corpus `
  --repeats-per-class 10 `
  --trajectory-output data/mm_webagent/training_trajectories.json `
  --sft-output data/mm_webagent/sft_train.jsonl `
  --dpo-output data/mm_webagent/dpo_train.jsonl `
  --grpo-output data/mm_webagent/grpo_rollout_seed.jsonl
```

Bash:

```bash
python -m mm_webagent.cli build-training-corpus \
  --repeats-per-class 10 \
  --trajectory-output data/mm_webagent/training_trajectories.json \
  --sft-output data/mm_webagent/sft_train.jsonl \
  --dpo-output data/mm_webagent/dpo_train.jsonl \
  --grpo-output data/mm_webagent/grpo_rollout_seed.jsonl
```

Expected output:

```json
{
  "trajectories": 200,
  "sft_records": 600,
  "dpo_records": 600,
  "grpo_records": 600,
  "task_classes": 20,
  "variants_per_class": 10
}
```

## Step 2. Validate SFT Data

PowerShell:

```powershell
python -m mm_webagent.training.sft_train `
  --data data/mm_webagent/sft_train.jsonl `
  --model Qwen/Qwen3-0.6B `
  --dry-run
```

Bash:

```bash
python -m mm_webagent.training.sft_train \
  --data data/mm_webagent/sft_train.jsonl \
  --model Qwen/Qwen3-0.6B \
  --dry-run
```

Expected checks:

- JSONL can be loaded.
- LoRA config can be assembled.
- Records contain instruction, screenshot path, page state, history, and target
  action.

## Step 3. Run Low-Cost Text SFT

Use this first to verify action formatting and element-selection learning before
scaling to a larger VLM.

PowerShell:

```powershell
python -m mm_webagent.training.sft_train `
  --data data/mm_webagent/sft_train.jsonl `
  --model Qwen/Qwen3-0.6B `
  --output-dir outputs/qwen3_0_6b_sft_lora `
  --epochs 1 `
  --batch-size 1 `
  --train
```

Bash:

```bash
python -m mm_webagent.training.sft_train \
  --data data/mm_webagent/sft_train.jsonl \
  --model Qwen/Qwen3-0.6B \
  --output-dir outputs/qwen3_0_6b_sft_lora \
  --epochs 1 \
  --batch-size 1 \
  --train
```

Expected output artifact:

```text
outputs/qwen3_0_6b_sft_lora
```

Recommended validation after SFT:

- action format validity
- exact action match
- invalid action rate
- repeated action rate

## Step 4. Validate DPO Data

PowerShell:

```powershell
python -m mm_webagent.post_training.dpo_train `
  --data data/mm_webagent/dpo_train.jsonl `
  --model Qwen/Qwen3-0.6B
```

Bash:

```bash
python -m mm_webagent.post_training.dpo_train \
  --data data/mm_webagent/dpo_train.jsonl \
  --model Qwen/Qwen3-0.6B
```

Expected output:

```json
{
  "valid": 600,
  "invalid": 0
}
```

Each DPO row contains:

- `prompt`: instruction, task class, scenario variant, page state, and history
- `chosen`: preferred action wrapped in `<action>...</action>`
- `rejected`: invalid, unsafe, repeated, or lower-progress action
- `rejection_reason`: human-readable reason for the rejected action

## Step 5. Run DPO Preference Alignment

Run DPO after the SFT adapter exists.

PowerShell:

```powershell
python -m mm_webagent.post_training.dpo_train `
  --data data/mm_webagent/dpo_train.jsonl `
  --model Qwen/Qwen3-0.6B `
  --output-dir outputs/qwen3_0_6b_dpo_lora `
  --train
```

Bash:

```bash
python -m mm_webagent.post_training.dpo_train \
  --data data/mm_webagent/dpo_train.jsonl \
  --model Qwen/Qwen3-0.6B \
  --output-dir outputs/qwen3_0_6b_dpo_lora \
  --train
```

Expected output artifact:

```text
outputs/qwen3_0_6b_dpo_lora
```

Expected effect:

- reduce invalid syntax such as `click_bad`
- prefer task-progressing actions over premature terminal responses
- avoid unsafe `goto` when no explicit destination URL exists
- avoid repeating the same action in repeated-action variants

## Step 6. Validate GRPO Seeds

PowerShell:

```powershell
python -m mm_webagent.training.grpo_train `
  --rollout-seed data/mm_webagent/grpo_rollout_seed.jsonl `
  --group-size 4
```

Bash:

```bash
python -m mm_webagent.training.grpo_train \
  --rollout-seed data/mm_webagent/grpo_rollout_seed.jsonl \
  --group-size 4
```

This validates that rollout seeds can be consumed by the reward function before
connecting full grouped action sampling.

## Step 7. Scale To Qwen-VL

After the text-only Qwen3-0.6B run proves the data and action format, switch to
the multimodal policy model.

PowerShell:

```powershell
python -m mm_webagent.training.sft_train `
  --data data/mm_webagent/sft_train.jsonl `
  --model Qwen/Qwen2.5-VL-7B-Instruct `
  --output-dir outputs/qwen_vl_sft_lora `
  --epochs 1 `
  --batch-size 1 `
  --train
```

Bash:

```bash
python -m mm_webagent.training.sft_train \
  --data data/mm_webagent/sft_train.jsonl \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --output-dir outputs/qwen_vl_sft_lora \
  --epochs 1 \
  --batch-size 1 \
  --train
```

For a strict multimodal training run, add a VLM collator that loads screenshot
paths, applies the Qwen-VL processor, and masks labels outside the assistant
action response.

## Step 8. Deploy The SFT/DPO Adapter

After merging or loading the adapter in the model directory:

PowerShell:

```powershell
python -m mm_webagent.serving.vllm_server `
  --model outputs/qwen3_0_6b_dpo_lora `
  --served-model-name mm-webagent `
  --port 8000
```

Bash:

```bash
python -m mm_webagent.serving.vllm_server \
  --model outputs/qwen3_0_6b_dpo_lora \
  --served-model-name mm-webagent \
  --port 8000
```

Then start the printed `vllm serve ...` command in the target runtime and call
it through `mm_webagent/serving/client.py`.

## Step 9. Evaluate The Trained Policy

Run the retrieval and Agent benchmarks:

PowerShell:

```powershell
python -m mm_webagent.cli eval-graphrag `
  --documents examples/operation_memory_docs.json `
  --queries examples/operation_memory_eval.json `
  --top-k 3

python -m mm_webagent.cli eval-agent-benchmark `
  --output experiments/agent_benchmark_20class_results.json
```

Bash:

```bash
python -m mm_webagent.cli eval-graphrag \
  --documents examples/operation_memory_docs.json \
  --queries examples/operation_memory_eval.json \
  --top-k 3

python -m mm_webagent.cli eval-agent-benchmark \
  --output experiments/agent_benchmark_20class_results.json
```

For final production evidence, replace deterministic fixture traces with live
WebWorld or browser rollout logs while keeping the same metric functions.
