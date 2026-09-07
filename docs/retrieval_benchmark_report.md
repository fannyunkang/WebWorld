# Operation Memory Retrieval Benchmark Report

## Setup

Command:

```bash
python -m mm_webagent.eval.retrieval_eval \
  --documents examples/operation_memory_docs.json \
  --queries examples/operation_memory_eval.json \
  --top-k 3
```

The benchmark compares the original Hybrid RAG retriever with the upgraded
Operation Memory Retrieval chain.

## Retrieval Chains

Baseline:

```text
Qdrant/BGE-M3-style dense retrieval
  + BM25/jieba sparse retrieval
  + BGE-style reranking
```

Upgraded:

```text
Qdrant/BGE-M3-style dense retrieval
  + BM25/jieba sparse retrieval
  + OpenSearch-style fielded BM25
  + ColBERT-style late interaction
  + trajectory graph retrieval
  + BGE-style reranking
```

## Results

| Metric | Baseline | Upgraded | Delta |
| --- | ---: | ---: | ---: |
| hit@3 | 0.75 | 1.00 | +0.25 |
| MRR@3 | 0.75 | 1.00 | +0.25 |
| action hint hit rate@3 | 0.75 | 1.00 | +0.25 |
| failure case recall@3 | 0.50 | 0.50 | +0.00 |
| hit@1 | 0.75 | 1.00 | +0.25 |
| MRR@1 | 0.75 | 1.00 | +0.25 |
| action hint hit rate@1 | 0.75 | 1.00 | +0.25 |
| failure case recall@1 | 0.25 | 0.25 | +0.00 |

## Interpretation

The upgraded retriever improves exact operation-memory retrieval on this
offline benchmark because fielded BM25 and late interaction can use structured
signals that are weak or absent in plain document text:

- element IDs such as `[88]`
- action strings such as `click('88')`
- page-state fields such as `Checkout Review`
- route and safety rules such as `Place order`

Failure-case recall did not improve in this small benchmark. That is expected:
the evaluation set has only one repeated-action failure case, and both
retrievers can already retrieve it. A larger failure-memory corpus is needed to
measure improvement on invalid-action and loop avoidance.

## Relation To Agent Performance

This benchmark evaluates retrieval quality before model training. It does not
claim final task success-rate improvement. The expected downstream effect is:

```text
better retrieved trajectory/action hints
  -> fewer invalid or repeated actions
  -> lower average steps
  -> higher task success after SFT/PPO/GRPO
```

Final Agent performance still requires full episode evaluation over the
20-class web-task benchmark.
