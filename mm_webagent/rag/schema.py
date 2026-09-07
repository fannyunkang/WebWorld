"""Data structures for web-agent Hybrid RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RagDocument:
    id: str
    text: str
    source: str
    doc_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalHit:
    document: RagDocument
    dense_score: float = 0.0
    sparse_score: float = 0.0
    field_score: float = 0.0
    late_interaction_score: float = 0.0
    graph_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
