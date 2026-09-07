"""Offline retrieval evaluation for operation-memory RAG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Protocol

from mm_webagent.rag.hybrid_retriever import HybridWebAgentRetriever
from mm_webagent.rag.operation_retriever import OperationMemoryRetriever
from mm_webagent.rag.query_rewrite import rewrite_query
from mm_webagent.rag.schema import RagDocument


class TextRetriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5, prefetch_k: int = 12):
        ...


def load_documents(path: str | Path) -> list[RagDocument]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        RagDocument(
            id=row["id"],
            text=row["text"],
            source=row.get("source", "eval"),
            doc_type=row.get("doc_type", "unknown"),
            metadata=row.get("metadata", {}),
        )
        for row in rows
    ]


def load_queries(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_base(retriever: TextRetriever, queries: list[dict], top_k: int) -> dict[str, float]:
    ranks = []
    failure_hits = 0
    action_hits = 0
    for query in queries:
        rewritten = rewrite_query(query["instruction"], query["page_state"], query.get("last_actions", []))
        hits = retriever.retrieve(rewritten, top_k=top_k, prefetch_k=max(top_k, 12))
        ids = [hit.document.id for hit in hits]
        ranks.append(_rank(ids, query["expected_doc_id"]))
        failure_hits += int(any(hit.document.doc_type == "negative_action" for hit in hits))
        action_hits += int(any(hit.document.metadata.get("action") == query.get("expected_action") for hit in hits))
    return _summary(ranks, failure_hits, action_hits, len(queries), top_k)


def evaluate_operation(retriever: OperationMemoryRetriever, queries: list[dict], top_k: int) -> dict[str, float]:
    ranks = []
    failure_hits = 0
    action_hits = 0
    for query in queries:
        hits = retriever.retrieve(
            query["instruction"],
            query["page_state"],
            query.get("last_actions", []),
            top_k=top_k,
        )
        ids = [hit.document.id for hit in hits]
        ranks.append(_rank(ids, query["expected_doc_id"]))
        failure_hits += int(any(hit.document.doc_type == "negative_action" for hit in hits))
        action_hits += int(any(hit.document.metadata.get("action") == query.get("expected_action") for hit in hits))
    return _summary(ranks, failure_hits, action_hits, len(queries), top_k)


def _rank(ids: list[str], expected: str) -> int | None:
    try:
        return ids.index(expected) + 1
    except ValueError:
        return None


def _summary(
    ranks: list[int | None],
    failure_hits: int,
    action_hits: int,
    total: int,
    top_k: int,
) -> dict[str, float]:
    hits = [rank for rank in ranks if rank is not None and rank <= top_k]
    reciprocal = [(1 / rank) if rank is not None else 0.0 for rank in ranks]
    return {
        f"hit@{top_k}": len(hits) / total if total else 0.0,
        "mrr": sum(reciprocal) / total if total else 0.0,
        "failure_case_recall": failure_hits / total if total else 0.0,
        "action_hint_hit_rate": action_hits / total if total else 0.0,
    }


def improvement(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {key: round(after.get(key, 0.0) - value, 4) for key, value in before.items()}


def compare(documents: list[RagDocument], queries: list[dict], top_k: int) -> dict[str, dict[str, float]]:
    base = HybridWebAgentRetriever(documents)
    upgraded = OperationMemoryRetriever(documents)
    baseline = evaluate_base(base, queries, top_k)
    optimized = evaluate_operation(upgraded, queries, top_k)
    return {
        "baseline_hybrid_rag": baseline,
        "upgraded_operation_memory": optimized,
        "delta": improvement(baseline, optimized),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline Hybrid RAG with upgraded Operation Memory Retrieval.")
    parser.add_argument("--documents", default="examples/operation_memory_docs.json")
    parser.add_argument("--queries", default="examples/operation_memory_eval.json")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    documents = load_documents(args.documents)
    queries = load_queries(args.queries)
    report = {
        f"top_{args.top_k}": compare(documents, queries, args.top_k),
        "top_1": compare(documents, queries, 1),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
