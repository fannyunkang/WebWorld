"""Demo for GraphRAG operation-memory retrieval."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mm_webagent.rag.graphrag import GraphRAGRetriever
from mm_webagent.rag.schema import RagDocument


def load_docs() -> list[RagDocument]:
    rows = json.loads((PROJECT_ROOT / "examples" / "operation_memory_docs.json").read_text(encoding="utf-8"))
    return [
        RagDocument(
            id=row["id"],
            text=row["text"],
            source=row.get("source", "demo"),
            doc_type=row.get("doc_type", "memory"),
            metadata=row.get("metadata", {}),
        )
        for row in rows
    ]


def main() -> None:
    retriever = GraphRAGRetriever(load_docs())
    context = retriever.build_context(
        instruction="Submit the order after reviewing checkout details.",
        page_state="RootWebArea 'Checkout Review'\n[88] button 'Place order'",
        last_actions=["fill('41', 'address', false)", "click('66')"],
    )
    print(context)


if __name__ == "__main__":
    main()
