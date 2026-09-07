"""Trajectory graph retrieval for state-action transition experience."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from mm_webagent.rag.bm25_jieba import tokenize
from mm_webagent.rag.schema import RagDocument


@dataclass(frozen=True)
class GraphSuggestion:
    document_id: str
    action: str
    score: float


class TrajectoryGraph:
    """Builds page-state -> action evidence from successful episodes."""

    def __init__(self, documents: list[RagDocument]):
        self.documents = documents
        self.action_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.doc_actions: dict[str, str] = {}
        for doc in documents:
            action = str(doc.metadata.get("action", ""))
            page_state = str(doc.metadata.get("page_state", doc.text))
            if action:
                signature = self._signature(page_state)
                self.action_counts[signature][action] += 1
                self.doc_actions[doc.id] = action

    def score_documents(self, query: str) -> dict[str, float]:
        query_tokens = set(tokenize(query))
        scores = {}
        for doc in self.documents:
            page_tokens = set(tokenize(str(doc.metadata.get("page_state", doc.text))))
            action = self.doc_actions.get(doc.id)
            if not action:
                continue
            overlap = len(query_tokens & page_tokens) / max(len(query_tokens), 1)
            action_prior = self.action_counts[self._signature(str(doc.metadata.get("page_state", "")))][action]
            scores[doc.id] = overlap + 0.1 * action_prior
        return scores

    def suggest_actions(self, query: str, top_k: int = 3) -> list[GraphSuggestion]:
        scores = self.score_documents(query)
        suggestions = [
            GraphSuggestion(doc_id, self.doc_actions[doc_id], score)
            for doc_id, score in scores.items()
        ]
        return sorted(suggestions, key=lambda item: item.score, reverse=True)[:top_k]

    def _signature(self, page_state: str) -> str:
        tokens = tokenize(page_state)
        return " ".join(tokens[:24])
