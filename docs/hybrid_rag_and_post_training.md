# Hybrid RAG And Post-Training Design

## Why This RAG Is Different

This project does not use RAG to answer factual questions. It retrieves
operation experience for a web Agent:

- successful trajectories for similar tasks
- failed actions and recovery strategies
- site-level rules such as login, checkout, or search behavior
- UI element aliases and page-state descriptions
- reward hints for invalid, repeated, or high-cost actions

The retrieved context is injected before action generation so the policy model
can avoid known dead ends and copy proven interaction patterns.

## Hybrid RAG Chain

```text
instruction + page_state + recent actions
  -> query rewrite
  -> dense retrieval: Qdrant + BGE-M3 embeddings
  -> sparse retrieval: BM25 + jieba tokenization
  -> fielded retrieval: OpenSearch-style BM25 over instruction/page/action/site/rule/failure
  -> late interaction: ColBERT-style token max-sim for element labels and action strings
  -> trajectory graph retrieval: page_state -> action -> next_state transition evidence
  -> score fusion
  -> BGE Reranker
  -> context compression
  -> Agent prompt
```

This chain is retained as an auxiliary recall path. The main runtime route can
be switched to GraphRAG through `mm_webagent/configs/rag.yaml`.

Dense retrieval finds semantically similar tasks. Sparse retrieval catches exact
Chinese terms, element names, button labels, and action strings. The reranker
uses the full query/document pair to pick the examples that are most useful for
the current page. Context compression keeps only the task-relevant action
pattern, failed-action warning, and reward hint.

## Upgraded Operation Memory Retrieval

The upgraded retriever is implemented in `mm_webagent/rag/operation_retriever.py`.
It is designed for web operations rather than document QA:

```text
dense semantic match
  + exact element/action match
  + field boosts for page_state/action/failure/rule
  + token-level late interaction
  + trajectory graph priors
  + rerank
```

The expected optimization effect is measured before model training by an
offline retrieval benchmark:

```bash
python -m mm_webagent.eval.retrieval_eval \
  --documents examples/operation_memory_docs.json \
  --queries examples/operation_memory_eval.json \
  --top-k 3
```

Metrics:

- `hit@k`: whether the expected operation memory is retrieved
- `mrr`: whether useful memory is ranked early
- `failure_case_recall`: whether failure cases are available to prevent loops
- `action_hint_hit_rate`: whether retrieved memory contains the expected next action

This is not the final Agent benchmark. It measures whether the retrieval layer
can provide better action hints before SFT/PPO/GRPO training.

The current offline report is saved in `docs/retrieval_benchmark_report.md`.

## Memory Layout

| Memory | Storage | Content | Usage |
| --- | --- | --- | --- |
| working | prompt history | recent observations and actions | next action decision |
| episodic | Qdrant | full task trajectories | retrieve similar successful paths |
| semantic | Qdrant | site rules, UI aliases, constraints | avoid unsafe or invalid actions |
| negative | Qdrant/BM25 | failed actions and loop patterns | reduce repeated and illegal actions |

## Post-Training Flow

```text
trajectory collection
  -> SFT action prediction
  -> DPO preference alignment
  -> PPO/RLHF rollout optimization
  -> GRPO grouped action optimization
  -> vLLM deployment
  -> LangGraph benchmark evaluation
```

SFT teaches the model the basic input-output format:

```text
observation + instruction + history -> next action
```

DPO uses chosen/rejected pairs to prefer correct actions over invalid,
repetitive, or unsafe actions. PPO/RLHF uses rollout rewards from the simulated
environment. GRPO then compares multiple candidate actions for the same state
without requiring a separate value model.

## Qwen3-0.6B + LoRA + TRL Validation

This stage is a low-cost effectiveness check, not the final benchmark. The
small Qwen3-0.6B model verifies that:

- the SFT data teaches valid action formatting
- LoRA adapters can learn element selection
- TRL DPO/PPO trainers can consume preference or rollout data
- held-out action validity and action accuracy improve before scaling to Qwen-VL

Validation compares pre-training and post-training metrics on held-out records:

```text
JSON/action validity
action accuracy
invalid action rate
repetition rate
```

Benchmark evaluation is different: it runs the full Agent loop on complete
tasks and measures success rate, average steps, invalid action rate, token
consumption, and task completion behavior.
