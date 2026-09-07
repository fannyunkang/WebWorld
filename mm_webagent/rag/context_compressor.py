"""Compress retrieved experience into compact agent context."""

from __future__ import annotations

from mm_webagent.rag.schema import RetrievalHit


def compress_hits(hits: list[RetrievalHit], max_chars: int = 1600) -> str:
    blocks = []
    used = 0
    for hit in hits:
        block = (
            f"[{hit.document.doc_type}:{hit.document.id}] "
            f"score={hit.final_score:.3f}\n"
            f"{hit.document.text.strip()}"
        )
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)
