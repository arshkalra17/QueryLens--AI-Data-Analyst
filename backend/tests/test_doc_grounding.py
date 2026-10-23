"""Document RAG wired into chat: routing, definition-grounded SQL, citations."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.registry import reset_registry
from app.core.sessions import reset_sessions
from app.engine.conditions import extract_where

SAMPLES = Path(__file__).resolve().parents[2] / "sample_data"


@pytest.fixture()
def client(docs_store):
    reset_registry()
    reset_sessions()
    from app.main import app

    c = TestClient(app)
    for name in ("sales_data.csv", "retail_transactions.csv", "web_traffic.csv"):
        with open(SAMPLES / name, "rb") as fh:
            c.post("/api/datasets", files={"files": (name, fh, "text/csv")})
    c.ds = {d["name"]: d["dataset_id"] for d in c.get("/api/datasets").json()["datasets"]}
    yield c
    reset_registry()


def ask(client, dataset, question):
    return client.post("/api/chat", json={"message": question, "dataset_ids": [client.ds[dataset]]}).json()


def test_defined_term_becomes_where_clause_with_citation(client):
    r = ask(client, "retail_transactions.csv", "How many high-value transactions were there in Mumbai?")
    assert r["route"] == "data"
    assert '"total" > 10000' in r["sql"] and "\"city\" = 'Mumbai'" in r["sql"]
    assert "Definition applied:** High-value transaction" in r["answer"]
    assert r["citations"][0]["label"] == "[metric_definitions.md, § High-value transaction]"


def test_documented_metric_expression_is_computed(client):
    r = ask(client, "sales_data.csv", "What is the gross margin by region?")
    assert 'SUM("revenue") - SUM("cost") AS gross_margin' in r["sql"]
    assert r["citations"] and "Gross margin" in r["citations"][0]["definition"]


def test_ranking_by_documented_metric(client):
    r = ask(client, "sales_data.csv", "Which product category has the lowest margin rate?")
    assert "AS margin_rate" in r["sql"] and r["sql"].rstrip().endswith("ASC")


def test_documentation_question_routes_to_documents(client):
    r = ask(client, "retail_transactions.csv", "What counts as a high-value transaction?")
    assert r["route"] == "documents" and r["sql"] is None
    assert "10,000 rupees" in r["answer"] and "[metric_definitions.md" in r["answer"]


def test_policy_question_cites_pdf_page(client):
    r = ask(client, "web_traffic.csv", "What is the target bounce rate?")
    assert r["route"] == "documents" and "55 percent" in r["answer"]
    assert r["citations"][0]["page"] == 2


def test_data_questions_unaffected_by_documents(client):
    r = ask(client, "sales_data.csv", "What is the total revenue by region?")
    assert r["route"] == "data" and r["citations"] == []
    assert 'SUM("revenue")' in r["sql"]


def test_documents_only_session(docs_store):
    reset_registry()
    reset_sessions()
    from app.main import app

    r = TestClient(app).post("/api/chat", json={"message": "When does the fiscal year start?"}).json()
    assert r["route"] == "documents" and "1 April" in r["answer"]


def test_definition_filters_respect_dataset(docs_store):
    from app.core.registry import DataFrameRegistry
    import pandas as pd

    retail = DataFrameRegistry().register(pd.DataFrame({"total": [1.0], "rating": [5]}), "retail_transactions.csv", "t")
    other = DataFrameRegistry().register(pd.DataFrame({"total": [1.0], "rating": [5]}), "other.csv", "t")
    assert extract_where("count low-rated transactions", retail) == '"rating" <= 2'
    assert extract_where("count low-rated transactions", other) is None


def test_documents_disabled_changes_nothing(monkeypatch, docs_store):
    from app.config import get_settings
    from app.core.registry import DataFrameRegistry
    import pandas as pd

    monkeypatch.setattr(get_settings(), "enable_docs", False)
    retail = DataFrameRegistry().register(pd.DataFrame({"total": [1.0]}), "retail_transactions.csv", "t")
    assert extract_where("how many high-value transactions", retail) is None
