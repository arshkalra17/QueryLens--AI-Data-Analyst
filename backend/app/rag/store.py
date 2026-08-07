"""Persistent ChromaDB vector store.

Two collections:
- ``table_profiles`` : one entry per uploaded dataset (embedding of its profile)
- ``qa_history``     : one entry per conversation turn (question + sql + result
  summary) so follow-up questions can retrieve *similar past answers*.

Embeddings come from the local sentence-transformers model — ChromaDB is used
as the store + similarity index, not the embedding provider.
"""
from __future__ import annotations

import logging
import threading
import uuid
from typing import Any

import chromadb

from ..config import get_settings
from . import embeddings

logger = logging.getLogger(__name__)

TABLE_COLLECTION = "table_profiles"
QA_COLLECTION = "qa_history"

_lock = threading.Lock()
_client: Any = None


def _get_client():
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                settings = get_settings()
                settings.rag_store_dir.mkdir(parents=True, exist_ok=True)
                _client = chromadb.PersistentClient(path=str(settings.rag_store_dir))
    return _client


def _collection(name: str):
    client = _get_client()
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )


# ---------------------------------------------------------------- table profiles
def upsert_table_profile(record) -> None:
    """Embed + store (or replace) a dataset's profile."""
    if not get_settings().enable_rag:
        return
    from .profiling import build_table_profile

    text = build_table_profile(record)
    vector = embeddings.embed_one(text)
    coll = _collection(TABLE_COLLECTION)
    coll.upsert(
        ids=[record.dataset_id],
        embeddings=[vector],
        documents=[text],
        metadatas=[{
            "name": record.name,
            "dataset_id": record.dataset_id,
            "row_count": record.row_count,
            "col_count": record.col_count,
        }],
    )


def delete_table_profile(dataset_id: str) -> None:
    if not get_settings().enable_rag:
        return
    try:
        _collection(TABLE_COLLECTION).delete(ids=[dataset_id])
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("RAG delete skipped for %s: %s", dataset_id, exc)


def prune_table_profiles(valid_ids: set[str]) -> int:
    """Drop profiles whose dataset is no longer registered, so the store never
    accumulates entries for files the user no longer has."""
    if not get_settings().enable_rag:
        return 0
    try:
        coll = _collection(TABLE_COLLECTION)
        ids = coll.get(include=[])["ids"]
    except Exception:
        return 0
    stale = [i for i in ids if i not in valid_ids]
    for i in stale:
        try:
            coll.delete(ids=[i])
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("RAG prune skipped for %s: %s", i, exc)
    if stale:
        logger.info("Pruned %d stale table profile(s) from the vector store", len(stale))
    return len(stale)


def query_table_profiles(text: str, k: int | None = None, dataset_ids: list[str] | None = None) -> list[dict]:
    """Return the most relevant dataset profiles for a query.

    When ``dataset_ids`` is given, retrieval is scoped to those datasets only,
    so context never leaks from files uploaded outside the current session.
    """
    if not get_settings().enable_rag:
        return []
    if dataset_ids is not None and not dataset_ids:
        return []
    settings = get_settings()
    vector = embeddings.embed_one(text)
    coll = _collection(TABLE_COLLECTION)
    if coll.count() == 0:
        return []
    where = {"dataset_id": {"$in": dataset_ids}} if dataset_ids else None
    res = coll.query(query_embeddings=[vector], n_results=min(k or settings.rag_k_table, coll.count()), where=where)
    return _hits(res)


# ---------------------------------------------------------------- QA history
def store_turn(session_id: str, question: str, sql: str | None, answer_summary: str) -> None:
    """Embed + store a conversation turn for cross-turn retrieval."""
    if not get_settings().enable_rag:
        return
    text = f"Q: {question}\nSQL: {sql or 'n/a'}\nA: {answer_summary[:500]}"
    vector = embeddings.embed_one(text)
    _collection(QA_COLLECTION).add(
        ids=[f"{session_id}:{uuid.uuid4().hex[:8]}"],
        embeddings=[vector],
        documents=[text],
        metadatas=[{
            "session_id": session_id,
            "question": question[:300],
        }],
    )


def query_qa_history(text: str, k: int | None = None, session_id: str | None = None) -> list[dict]:
    """Return similar past Q&A turns for grounding follow-ups.

    When ``session_id`` is given, only that conversation's turns are searched,
    so past answers from other sessions never leak into the prompt.
    """
    if not get_settings().enable_rag:
        return []
    settings = get_settings()
    vector = embeddings.embed_one(text)
    coll = _collection(QA_COLLECTION)
    if coll.count() == 0:
        return []
    where = {"session_id": session_id} if session_id else None
    res = coll.query(query_embeddings=[vector], n_results=min(k or settings.rag_k_history, coll.count()), where=where)
    return _hits(res)


def reset() -> None:
    global _client
    _client = None


def _hits(res: dict) -> list[dict]:
    out = []
    if not res.get("ids"):
        return out
    for i, doc_id in enumerate(res["ids"][0]):
        out.append({
            "id": doc_id,
            "text": res["documents"][0][i],
            "distance": res["distances"][0][i] if res.get("distances") else None,
            "metadata": res["metadatas"][0][i] if res.get("metadatas") else {},
        })
    return out
