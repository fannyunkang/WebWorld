"""Qdrant dense-vector adapter with an in-memory fallback for dry runs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from mm_webagent.rag.schema import RagDocument


def simple_embedding(text: str, dim: int = 128) -> list[float]:
    """Deterministic hashing embedding used when BGE-M3 is unavailable."""
    vec = [0.0] * dim
    for idx, char in enumerate(text):
        vec[(ord(char) + idx) % dim] += 1.0
    norm = math.sqrt(sum(item * item for item in vec)) or 1.0
    return [item / norm for item in vec]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


@dataclass
class InMemoryQdrantStore:
    documents: list[RagDocument] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)

    def upsert(self, documents: list[RagDocument]) -> None:
        for document in documents:
            self.documents.append(document)
            self.vectors.append(simple_embedding(document.text))

    def search(self, query: str, top_k: int = 8) -> list[tuple[RagDocument, float]]:
        query_vec = simple_embedding(query)
        scored = [
            (doc, cosine(query_vec, vector))
            for doc, vector in zip(self.documents, self.vectors)
        ]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
