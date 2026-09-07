"""OpenSearch-style fielded BM25 for web-agent operation records."""

from __future__ import annotations

from collections import defaultdict

from mm_webagent.rag.bm25_jieba import BM25JiebaIndex
from mm_webagent.rag.schema import RagDocument


FIELD_WEIGHTS = {
    "instruction": 2.0,
    "page_state": 2.5,
    "action": 3.0,
    "site": 1.5,
    "rule": 2.0,
    "failure": 2.5,
    "text": 1.0,
}


class FieldedBM25Index:
    """Approximate OpenSearch BM25 with per-field boosts."""

    def __init__(self, documents: list[RagDocument], field_weights: dict[str, float] | None = None):
        self.documents = documents
        self.field_weights = field_weights or FIELD_WEIGHTS
        self.indexes: dict[str, BM25JiebaIndex] = {}
        fields = set(self.field_weights)
        for field in fields:
            self.indexes[field] = BM25JiebaIndex([self._field_text(doc, field) for doc in documents])

    def score(self, query: str) -> list[float]:
        totals = defaultdict(float)
        for field, index in self.indexes.items():
            weight = self.field_weights.get(field, 1.0)
            for idx, score in enumerate(index.score(query)):
                totals[idx] += weight * score
        return [totals[idx] for idx in range(len(self.documents))]

    def _field_text(self, document: RagDocument, field: str) -> str:
        if field == "text":
            return document.text
        value = document.metadata.get(field)
        if isinstance(value, list):
            return " ".join(str(item) for item in value)
        return str(value or "")
