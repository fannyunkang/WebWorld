"""Query rewriting for web-agent retrieval."""

from __future__ import annotations


def rewrite_query(instruction: str, page_state: str, last_actions: list[str] | None = None) -> str:
    """Build a retrieval query focused on task intent, UI entities, and failure context."""
    last_actions = last_actions or []
    page_lines = [line.strip() for line in page_state.splitlines() if line.strip()]
    visible_elements = " ".join(page_lines[:20])
    recent_actions = " ".join(last_actions[-5:])
    return (
        f"task: {instruction}\n"
        f"visible elements: {visible_elements}\n"
        f"recent actions: {recent_actions}\n"
        "retrieve similar successful trajectories, site rules, invalid action fixes, and reward hints"
    )
