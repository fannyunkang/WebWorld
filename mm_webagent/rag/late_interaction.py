"""ColBERT-style late interaction scoring for element/action matching."""

from __future__ import annotations

from mm_webagent.rag.bm25_jieba import tokenize


def token_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if left in right or right in left:
        return 0.6
    overlap = set(left) & set(right)
    return len(overlap) / max(len(set(left) | set(right)), 1)


class LateInteractionScorer:
    """Scores a query by max-sim over document tokens, like a tiny ColBERT proxy."""

    def score(self, query: str, document_text: str) -> float:
        query_tokens = tokenize(query)
        doc_tokens = tokenize(document_text)
        if not query_tokens or not doc_tokens:
            return 0.0
        total = 0.0
        for query_token in query_tokens:
            total += max(token_similarity(query_token, doc_token) for doc_token in doc_tokens)
        return total / len(query_tokens)
