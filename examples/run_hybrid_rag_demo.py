"""Demo for Hybrid RAG over web-operation experience."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mm_webagent.rag.context_compressor import compress_hits
from mm_webagent.rag.hybrid_retriever import HybridWebAgentRetriever
from mm_webagent.rag.query_rewrite import rewrite_query
from mm_webagent.rag.schema import RagDocument


def main() -> None:
    documents = [
        RagDocument(
            id="traj-shopping-001",
            doc_type="successful_trajectory",
            source="episode",
            text="When the page contains link 'Shopping', click that link before selecting product filters.",
        ),
        RagDocument(
            id="neg-repeat-001",
            doc_type="negative_action",
            source="reward_log",
            text="Do not click the same Shopping link repeatedly after the page title already changed to Shopping.",
        ),
        RagDocument(
            id="site-search-001",
            doc_type="site_rule",
            source="manual_rule",
            text="Search boxes should be filled before pressing enter; avoid goto unless the task gives an explicit URL.",
        ),
    ]
    query = rewrite_query(
        instruction="Open the shopping section.",
        page_state="RootWebArea 'Global Start'\n[34] link 'Shopping'",
        last_actions=[],
    )
    retriever = HybridWebAgentRetriever(documents)
    hits = retriever.retrieve(query, top_k=2)
    print(compress_hits(hits))


if __name__ == "__main__":
    main()
