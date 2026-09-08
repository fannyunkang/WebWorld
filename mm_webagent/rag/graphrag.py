"""GraphRAG retriever for web-operation agents."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from mm_webagent.rag.bm25_jieba import tokenize
from mm_webagent.rag.schema import RagDocument, RetrievalHit


@dataclass(frozen=True)
class GraphNode:
    id: str
    kind: str
    text: str


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    relation: str
    weight: float = 1.0


@dataclass
class GraphRAGResult:
    hit: RetrievalHit
    path: list[GraphNode] = field(default_factory=list)
    suggested_action: str | None = None


class WebAgentGraph:
    """Heterogeneous graph over tasks, pages, elements, actions, outcomes, and failures."""

    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []
        self.adjacent: dict[str, list[GraphEdge]] = defaultdict(list)
        self.doc_nodes: dict[str, list[str]] = defaultdict(list)

    def add_node(self, node_id: str, kind: str, text: str, document_id: str | None = None) -> None:
        self.nodes[node_id] = GraphNode(node_id, kind, text)
        if document_id:
            self.doc_nodes[document_id].append(node_id)

    def add_edge(self, source: str, target: str, relation: str, weight: float = 1.0) -> None:
        edge = GraphEdge(source, target, relation, weight)
        self.edges.append(edge)
        self.adjacent[source].append(edge)


class GraphRAGRetriever:
    """Subgraph retrieval over operation memory."""

    def __init__(self, documents: list[RagDocument]):
        self.documents = documents
        self.graph = self._build_graph(documents)

    def retrieve(
        self,
        instruction: str,
        page_state: str,
        last_actions: list[str] | None = None,
        top_k: int = 5,
    ) -> list[GraphRAGResult]:
        query = f"{instruction}\n{page_state}\n{' '.join(last_actions or [])}"
        query_tokens = set(tokenize(query))
        scored = []
        for doc in self.documents:
            node_ids = self.graph.doc_nodes.get(doc.id, [])
            node_score = sum(self._node_score(self.graph.nodes[node_id], query_tokens) for node_id in node_ids)
            edge_score = self._edge_score(node_ids)
            failure_bonus = 0.25 if doc.doc_type == "negative_action" and self._has_repetition(last_actions or []) else 0.0
            final = node_score + edge_score + failure_bonus
            path = self._best_path(node_ids)
            hit = RetrievalHit(
                document=doc,
                graph_score=edge_score,
                late_interaction_score=node_score,
                final_score=final,
            )
            scored.append(GraphRAGResult(hit=hit, path=path, suggested_action=doc.metadata.get("action")))
        return sorted(scored, key=lambda item: item.hit.final_score, reverse=True)[:top_k]

    def build_context(
        self,
        instruction: str,
        page_state: str,
        last_actions: list[str] | None = None,
        top_k: int = 3,
    ) -> str:
        results = self.retrieve(instruction, page_state, last_actions, top_k=top_k)
        blocks = []
        for result in results:
            path = " -> ".join(f"{node.kind}:{node.text}" for node in result.path[:5])
            blocks.append(
                f"[GraphRAG:{result.hit.document.id}] score={result.hit.final_score:.3f}\n"
                f"path: {path}\n"
                f"suggested_action: {result.suggested_action}\n"
                f"memory: {result.hit.document.text}"
            )
        return "\n\n".join(blocks)

    def _build_graph(self, documents: list[RagDocument]) -> WebAgentGraph:
        graph = WebAgentGraph()
        for doc in documents:
            task_id = f"{doc.id}:task"
            page_id = f"{doc.id}:page"
            action_id = f"{doc.id}:action"
            outcome_id = f"{doc.id}:outcome"
            rule_id = f"{doc.id}:rule"
            graph.add_node(task_id, "task", str(doc.metadata.get("instruction", doc.text)), doc.id)
            graph.add_node(page_id, "page", str(doc.metadata.get("page_state", "")), doc.id)
            graph.add_node(action_id, "action", str(doc.metadata.get("action", "")), doc.id)
            graph.add_node(outcome_id, doc.doc_type, doc.text, doc.id)
            graph.add_node(rule_id, "rule", str(doc.metadata.get("rule", doc.metadata.get("failure", ""))), doc.id)
            graph.add_edge(task_id, page_id, "starts_on", 1.0)
            graph.add_edge(page_id, action_id, "selects_action", 1.4)
            graph.add_edge(action_id, outcome_id, "leads_to", 1.2)
            graph.add_edge(rule_id, action_id, "constrains", 0.8)
        return graph

    def _node_score(self, node: GraphNode, query_tokens: set[str]) -> float:
        node_tokens = set(tokenize(node.text))
        if not query_tokens:
            return 0.0
        kind_weight = {
            "page": 1.5,
            "action": 1.4,
            "task": 1.2,
            "rule": 1.1,
            "negative_action": 1.3,
            "successful_trajectory": 1.0,
        }.get(node.kind, 1.0)
        return kind_weight * len(query_tokens & node_tokens) / max(len(query_tokens), 1)

    def _edge_score(self, node_ids: list[str]) -> float:
        node_set = set(node_ids)
        return sum(edge.weight for edge in self.graph.edges if edge.source in node_set and edge.target in node_set) / 10.0

    def _best_path(self, node_ids: list[str]) -> list[GraphNode]:
        if not node_ids:
            return []
        node_set = set(node_ids)
        start = node_ids[0]
        seen = {start}
        queue = deque([(start, [start])])
        best = [start]
        while queue:
            node_id, path = queue.popleft()
            if len(path) > len(best):
                best = path
            for edge in self.graph.adjacent.get(node_id, []):
                if edge.target in node_set and edge.target not in seen:
                    seen.add(edge.target)
                    queue.append((edge.target, path + [edge.target]))
        return [self.graph.nodes[node_id] for node_id in best]

    def _has_repetition(self, last_actions: list[str]) -> bool:
        return len(last_actions) >= 2 and len(set(last_actions[-2:])) == 1
