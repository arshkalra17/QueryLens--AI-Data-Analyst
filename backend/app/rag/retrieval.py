"""Retrieval entry points used by the agent.

``retrieval_context(question)`` returns a compact text block combining:
- the most relevant *table profiles* (semantic match against the vector store)
- the most relevant *past Q&A turns* (so follow-ups ground on prior answers)

This block is injected into the prompt before SQL/chart generation. Raw data
is never sent to the LLM.
"""
from __future__ import annotations

import logging

from ..config import get_settings
from . import store

logger = logging.getLogger(__name__)


def retrieval_context(question: str, session_id: str | None = None, dataset_ids: list[str] | None = None) -> str:
    """Build the grounded context string for a question.

    Retrieval is scoped to the datasets attached to the current session
    (``dataset_ids``) and the session's own past Q&A (``session_id``), so
    profiles from files uploaded in other sessions never leak into the prompt.
    """
    if not get_settings().enable_rag:
        return ""
    parts = []
    profiles = store.query_table_profiles(question, dataset_ids=dataset_ids)
    if profiles:
        lines = ["## Retrieved table context (semantic search)"]
        for p in profiles:
            snippet = _truncate(p["text"], 1200)
            lines.append(f"[dataset:{p['metadata'].get('name','')}]")
            lines.append(snippet)
        parts.append("\n".join(lines))
    history = store.query_qa_history(question, session_id=session_id)
    if history:
        lines = ["## Similar past Q&A (retrieved for grounding)"]
        for h in history:
            lines.append(_truncate(h["text"], 700))
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _truncate(text: str, limit: int) -> str:
    return text[:limit] + ("…" if len(text) > limit else "")


_rag = None


def get_rag():
    global _rag
    if _rag is None:
        from ..core.registry import get_registry

        from . import store as store_mod

        _rag = _RAG(registry=get_registry(), store=store_mod)
    return _rag


def reset_rag() -> None:
    global _rag
    _rag = None
    store.reset()
    from . import embeddings

    embeddings.reset()


class _RAG:
    """Small facade so the agent has one place to talk to the RAG layer."""

    def __init__(self, registry, store):
        self.registry = registry
        self.store = store

    def index_all(self) -> int:
        records = self.registry.list()
        valid = {r.dataset_id for r in records}
        try:
            self.store.prune_table_profiles(valid)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("RAG prune failed: %s", exc)
        n = 0
        for record in records:
            self.store.upsert_table_profile(record)
            n += 1
        return n

    def index_record(self, record) -> None:
        self.store.upsert_table_profile(record)

    def drop_record(self, dataset_id: str) -> None:
        self.store.delete_table_profile(dataset_id)

    def context(self, question: str, session_id: str | None = None, dataset_ids: list[str] | None = None) -> str:
        return retrieval_context(question, session_id, dataset_ids)

    def remember(self, session_id: str, question: str, sql: str | None, summary: str) -> None:
        self.store.store_turn(session_id, question, sql, summary)
