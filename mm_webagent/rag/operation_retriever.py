"""Upgraded operation-memory retriever for web agents."""

from __future__ import annotations

from mm_webagent.rag.context_compressor import compress_hits
from mm_webagent.rag.field_bm25 import FieldedBM25Index
from mm_webagent.rag.hybrid_retriever import HybridWebAgentRetriever
from mm_webagent.rag.late_interaction import LateInteractionScorer
from mm_webagent.rag.query_rewrite import rewrite_query
from mm_webagent.rag.schema import RagDocument, RetrievalHit
from mm_webagent.rag.trajectory_graph import TrajectoryGraph


class OperationMemoryRetriever:
    """Dense + field BM25 + late interaction + trajectory graph + rerank."""

    def __init__(self, documents: list[RagDocument]):
        self.documents = documents
        self.base = HybridWebAgentRetriever(documents)
        self.field_index = FieldedBM25Index(documents)
        self.late = LateInteractionScorer()
        self.graph = TrajectoryGraph(documents)

    def retrieve(
        self,
        instruction: str,
        page_state: str,
        last_actions: list[str] | None = None,
        top_k: int = 5,
        prefetch_k: int = 12,
    ) -> list[RetrievalHit]:
        query = rewrite_query(instruction, page_state, last_actions)
        base_hits = self.base.retrieve(query, top_k=prefetch_k, prefetch_k=prefetch_k)
        field_scores = self.field_index.score(query)
        graph_scores = self.graph.score_documents(query)

        doc_to_base = {hit.document.id: hit for hit in base_hits}
        candidates = {hit.document.id for hit in base_hits}
        field_ranked = sorted(enumerate(field_scores), key=lambda item: item[1], reverse=True)[:prefetch_k]
        candidates.update(self.documents[idx].id for idx, _ in field_ranked)
        candidates.update(graph_scores)

        results = []
        for doc in self.documents:
            if doc.id not in candidates:
                continue
            base = doc_to_base.get(doc.id, RetrievalHit(document=doc))
            field_score = field_scores[self.documents.index(doc)]
            late_score = self.late.score(query, self._late_text(doc))
            graph_score = graph_scores.get(doc.id, 0.0)
            final = (
                0.25 * base.dense_score
                + 0.20 * base.sparse_score
                + 0.20 * field_score
                + 0.15 * late_score
                + 0.10 * graph_score
                + 0.10 * base.rerank_score
            )
            results.append(
                RetrievalHit(
                    document=doc,
                    dense_score=base.dense_score,
                    sparse_score=base.sparse_score,
                    field_score=field_score,
                    late_interaction_score=late_score,
                    graph_score=graph_score,
                    rerank_score=base.rerank_score,
                    final_score=final,
                )
            )
        return sorted(results, key=lambda item: item.final_score, reverse=True)[:top_k]

    def build_context(self, instruction: str, page_state: str, last_actions: list[str] | None = None) -> str:
        return compress_hits(self.retrieve(instruction, page_state, last_actions))

    def _late_text(self, document: RagDocument) -> str:
        metadata_text = " ".join(str(value) for value in document.metadata.values())
        return f"{document.text}\n{metadata_text}"
