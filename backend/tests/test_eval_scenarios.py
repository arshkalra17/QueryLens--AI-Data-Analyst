"""Evaluation set (pytest): deterministic scenarios run against the rule engine.

These run with ``LLM_PROVIDER=none`` (set in conftest), so they are fast,
offline, and regression-safe in CI. Each scenario asserts expected behaviour in
the answer (tokens) and, where required, that a chart was produced.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_DATA = ROOT / "sample_data"

SCENARIOS = [
    {"name": "top region by revenue", "question": "Which region generated the highest revenue?", "expected": ["north", "south", "east", "west"], "require_chart": False},
    {"name": "top 5 customers", "question": "What are the top five customers?", "expected": ["cust"], "require_chart": False},
    {"name": "underperformers", "question": "Which products are underperforming?", "expected": ["charger", "monitor", "headset"], "require_chart": False},
    {"name": "monthly trend chart", "question": "Show monthly sales trends.", "expected": ["trend"], "require_chart": True},
    {"name": "anomaly in named column", "question": "Detect anomalies in revenue.", "expected": ["anomal", "revenue"], "require_chart": False},
    {"name": "forecast", "question": "Forecast revenue for the next 6 months.", "expected": ["forecast", "month"], "require_chart": True},
    {"name": "SQL generation", "question": "Generate SQL for this analysis.", "expected": ["select"], "require_chart": False},
    {"name": "pie chart", "question": "Show a pie chart of product categories.", "expected": [], "require_chart": True},
    {"name": "data quality", "question": "Check the data quality of this dataset.", "expected": ["score", "/100"], "require_chart": False},
    {"name": "insights", "question": "Give me business insights about this data.", "expected": ["insight"], "require_chart": False},
]


def _upload(client: TestClient, name: str) -> str:
    with open(SAMPLE_DATA / name, "rb") as fh:
        resp = client.post("/api/datasets", files={"files": (name, io.BytesIO(fh.read()), "text/csv")})
    assert resp.status_code == 200, resp.text
    return resp.json()["imported"][0]["dataset_id"]


@pytest.fixture()
def ids():
    # Isolate from other test files: clear persisted uploads so this module's
    # client reloads nothing, and use a single function-scoped client so the
    # registry contains exactly the two files uploaded here (no duplicates).
    uploads = get_settings().upload_dir_resolved
    for path in uploads.glob("*.csv"):
        path.unlink()
    with TestClient(app) as client:
        dids = [_upload(client, "sales_data.csv"), _upload(client, "retail_transactions.csv")]
        yield dids, client


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["name"] for s in SCENARIOS])
def test_eval_scenario(ids, scenario):
    dids, client = ids
    resp = client.post("/api/chat", json={"message": scenario["question"], "dataset_ids": dids})
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    answer = payload.get("answer", "").lower()
    for tok in scenario["expected"]:
        assert tok in answer, f"expected '{tok}' in answer for '{scenario['question']}'"
    if scenario["require_chart"]:
        assert payload.get("chart_ids"), f"expected a chart for '{scenario['question']}'"
