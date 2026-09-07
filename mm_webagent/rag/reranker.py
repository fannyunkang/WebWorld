"""BGE reranker adapter with lexical fallback."""

from __future__ import annotations

from mm_webagent.rag.bm25_jieba import tokenize
from mm_webagent.rag.schema import RagDocument


class BGEReranker:
    """Cross-encoder reranker facade.

    In production, replace `score` with BAAI/bge-reranker model inference.
    The fallback keeps local validation dependency-light.
    """

    def score(self, query: str, document: RagDocument) -> float:
        query_terms = set(tokenize(query))
        doc_terms = set(tokenize(document.text))
        if not query_terms:
            return 0.0
        return len(query_terms & doc_terms) / len(query_terms)
