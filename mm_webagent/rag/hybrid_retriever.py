"""Hybrid retrieval over trajectories, page rules, and action feedback."""

from __future__ import annotations

from mm_webagent.rag.bm25_jieba import BM25JiebaIndex
from mm_webagent.rag.qdrant_store import InMemoryQdrantStore
from mm_webagent.rag.reranker import BGEReranker
from mm_webagent.rag.schema import RagDocument, RetrievalHit


class HybridWebAgentRetriever:
    """Qdrant/BGE-M3 dense + BM25/jieba sparse + BGE rerank retrieval chain."""

    def __init__(self, documents: list[RagDocument]):
        self.documents = documents
        self.dense_store = InMemoryQdrantStore()
        self.dense_store.upsert(documents)
        self.sparse_index = BM25JiebaIndex([doc.text for doc in documents])
        self.reranker = BGEReranker()

    def retrieve(self, query: str, top_k: int = 5, prefetch_k: int = 12) -> list[RetrievalHit]:
        dense_hits = self.dense_store.search(query, top_k=prefetch_k)
        sparse_scores = self.sparse_index.score(query)

        by_id: dict[str, RetrievalHit] = {}
        for doc, dense_score in dense_hits:
            by_id[doc.id] = RetrievalHit(document=doc, dense_score=dense_score)

        sparse_ranked = sorted(
            enumerate(sparse_scores),
            key=lambda item: item[1],
            reverse=True,
        )[:prefetch_k]
        for idx, sparse_score in sparse_ranked:
            doc = self.documents[idx]
            existing = by_id.get(doc.id)
            if existing:
                by_id[doc.id] = RetrievalHit(
                    document=doc,
                    dense_score=existing.dense_score,
                    sparse_score=sparse_score,
                )
            else:
                by_id[doc.id] = RetrievalHit(document=doc, sparse_score=sparse_score)

        reranked = []
        for hit in by_id.values():
            rerank_score = self.reranker.score(query, hit.document)
            final = 0.45 * hit.dense_score + 0.25 * hit.sparse_score + 0.30 * rerank_score
            reranked.append(
                RetrievalHit(
                    document=hit.document,
                    dense_score=hit.dense_score,
                    sparse_score=hit.sparse_score,
                    rerank_score=rerank_score,
                    final_score=final,
                )
            )

        return sorted(reranked, key=lambda item: item.final_score, reverse=True)[:top_k]
