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
    # A session that only uploaded the customers file must not see the sales profile.
    ctx = rag.context("Which region generated the highest revenue?", "s1", dataset_ids=["cust1"])
    assert "sales.csv" not in ctx
    assert "customers.csv" in ctx
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
