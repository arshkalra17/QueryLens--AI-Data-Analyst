"""Tests for the RAG layer (profiling, embedding, ChromaDB retrieval, grounding).

These opt into RAG with a temp store dir and a lightweight local embedding
model, so they run offline and deterministically.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pandas as pd
import pytest

# Must be set before importing app modules
os.environ["LLM_PROVIDER"] = "none"
os.environ["ENABLE_RAG"] = "true"
os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"
os.environ["RAG_STORE_DIR"] = "/tmp/ai_data_analyst_rag_test"

from app.config import get_settings  # noqa: E402
from app.core.registry import classify_columns  # noqa: E402
from app.rag import get_rag, reset_rag  # noqa: E402
from app.rag.profiling import build_table_profile  # noqa: E402

get_settings.cache_clear()


@pytest.fixture(scope="module", autouse=True)
def _clean_rag(tmp_path_factory):
    reset_rag()
    # Clear any collections left over from a previous run (the store is a
    # persistent Chroma client). Deleting the directory instead would race with
    # an open SQLite handle, so drop collections through the API.
    from app.rag import store as store_mod

    try:
        client = store_mod._get_client()
        for name in (store_mod.TABLE_COLLECTION, store_mod.QA_COLLECTION):
            try:
                client.delete_collection(name)
            except Exception:
                pass
    except Exception:
        pass
    yield
    reset_rag()


def _rec(df: pd.DataFrame, name: str, dataset_id: str):
    return SimpleNamespace(
        dataset_id=dataset_id,
        name=name,
        df=df,
        row_count=len(df),
        col_count=len(df.columns),
        columns=classify_columns(df),
    )


def test_build_table_profile_has_structure():
    df = pd.DataFrame(
        {
            "region": ["North", "South", "East", "West"],
            "revenue": [100.0, 200.0, 150.0, 300.0],
            "date": pd.date_range("2024-01-01", periods=4),
        }
    )
    text = build_table_profile(_rec(df, "sales.csv", "x1"))
    assert "sales.csv" in text
    assert "region" in text
    assert "revenue" in text
    assert "nulls" in text or "unique" in text


def test_index_and_retrieve_relevant_table():
    rag = get_rag()
    sales = pd.DataFrame({"region": ["North", "South", "East", "West"] * 10, "revenue": range(40)})
    cust = pd.DataFrame({"customer_id": [f"C{i}" for i in range(20)], "total": range(20)})
    rag.index_record(_rec(sales, "sales.csv", "sales1"))
    rag.index_record(_rec(cust, "customers.csv", "cust1"))
    ctx = rag.context("Which region generated the highest revenue?")
    assert "sales.csv" in ctx
    ctx2 = rag.context("who is the best customer")
    assert "customers.csv" in ctx2


def test_retrieval_scoped_to_session_datasets():
    rag = get_rag()
    sales = pd.DataFrame({"region": ["North", "South", "East", "West"] * 10, "revenue": range(40)})
    cust = pd.DataFrame({"customer_id": [f"C{i}" for i in range(20)], "total": range(20)})
    rag.index_record(_rec(sales, "sales.csv", "sales1"))
    rag.index_record(_rec(cust, "customers.csv", "cust1"))
    # A session that only uploaded the customers file must not see the sales
    # profile, and its own profile is not repeated (its schema is already in
    # the prompt).
    ctx = rag.context("Which region generated the highest revenue?", "s1", dataset_ids=["cust1"])
    assert "sales.csv" not in ctx and "customers.csv" not in ctx
    # profile retrieval itself is still scoped correctly at the store level
    from app.rag import store

    scoped = store.query_table_profiles("Which region generated the highest revenue?", dataset_ids=["cust1"])
    assert [h["metadata"]["name"] for h in scoped] == ["customers.csv"]
    # Empty scope (nothing attached) returns nothing rather than global leakage.
    assert rag.context("Which region generated the highest revenue?", "s1", dataset_ids=[]) == ""


def test_turn_memory_retrieves_past_qa():
    rag = get_rag()
    rag.remember("s1", "Which region generated the highest revenue?", "SELECT region, SUM(revenue) ...", "North")
    ctx = rag.context("top region again", "s1")
    assert "SELECT region" in ctx


def test_retrieval_disabled_returns_empty():
    from app.config import get_settings as gs

    gs.cache_clear()
    os.environ["ENABLE_RAG"] = "false"
    rag = get_rag()
    assert rag.context("anything") == ""
    os.environ["ENABLE_RAG"] = "true"
    gs.cache_clear()


def test_embedding_cache_key_is_stable_across_processes():
    """The disk cache key must not depend on Python's per-process hash seed."""
    import subprocess
    import sys

    from app.rag.embeddings import cache_key

    code = "from app.rag.embeddings import cache_key; print(cache_key('revenue by region'))"
    keys = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout.strip()
        for seed in ("1", "2")
    }
    assert keys == {cache_key("revenue by region")}
    assert cache_key("a") != cache_key("b")


def test_attached_dataset_profiles_not_repeated_in_context():
    """Attached datasets' schemas are already in the prompt; retrieval must not repeat them."""
    from app.rag.retrieval import retrieval_context

    get_rag()  # store initialised by the module fixture
    assert "Retrieved table context" not in retrieval_context("revenue by region", None, dataset_ids=["anything"])


def test_qa_history_relevance_cutoff(monkeypatch):
    from app.rag import retrieval, store

    monkeypatch.setattr(store, "query_qa_history", lambda q, session_id=None: [
        {"id": "a", "text": "Q: close", "distance": 0.2, "metadata": {}},
        {"id": "b", "text": "Q: unrelated", "distance": 0.95, "metadata": {}},
    ])
    monkeypatch.setattr(store, "query_table_profiles", lambda *a, **k: [])
    ctx = retrieval.retrieval_context("q", "s1", dataset_ids=[])
    assert "Q: close" in ctx and "Q: unrelated" not in ctx


def test_evicted_sessions_history_is_pruned(tmp_path):
    from app.core.sessions import SessionManager
    from app.rag import store

    store.store_turn("old-session", "what is revenue?", None, "42")
    assert store.query_qa_history("what is revenue?", session_id="old-session")
    mgr = SessionManager(max_sessions=1)
    mgr._sessions["old-session"] = SimpleNamespace(session_id="old-session")
    mgr.create()  # evicts old-session
    assert store.query_qa_history("what is revenue?", session_id="old-session") == []
