"""End-to-end API tests (TestClient)."""
from __future__ import annotations

import io
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core.registry import reset_registry
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def upload_sales(client):
    reset_registry()
    rng = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=60, freq="D"),
            "region": ["North", "South", "East", "West"] * 15,
            "revenue": [100.0, 200.0, 300.0, 400.0] * 15,
            "cost": [50.0, 100.0, 150.0, 200.0] * 15,
        }
    )
    csv = rng.to_csv(index=False).encode()
    resp = client.post("/api/datasets", files={"files": ("sales.csv", io.BytesIO(csv), "text/csv")})
    assert resp.status_code == 200, resp.text
    return resp.json()["imported"][0]["dataset_id"]


def test_upload_same_filename_dedupes(client):
    reset_registry()
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    csv = df.to_csv(index=False).encode()
    r1 = client.post("/api/datasets", files={"files": ("same.csv", io.BytesIO(csv), "text/csv")})
    r2 = client.post("/api/datasets", files={"files": ("same.csv", io.BytesIO(csv), "text/csv")})
    assert r1.status_code == 200 and r2.status_code == 200
    same_name = [d for d in client.get("/api/datasets").json()["datasets"] if d["name"] == "same.csv"]
    assert len(same_name) == 1
    # re-upload replaces the old entry with a fresh id (and keeps a single copy)
    assert r1.json()["imported"][0]["dataset_id"] != r2.json()["imported"][0]["dataset_id"]
    assert same_name[0]["dataset_id"] == r2.json()["imported"][0]["dataset_id"]


def test_chat_uses_most_recently_uploaded_dataset(client):
    reset_registry()
    sales = pd.DataFrame({"city": ["Alpha", "Beta"], "revenue": [10.0, 20.0]})
    org = pd.DataFrame({"org_name": ["Acme", "Globex", "Initech"], "headcount": [5, 6, 7]})
    sales_id = client.post(
        "/api/datasets", files={"files": ("sales.csv", io.BytesIO(sales.to_csv(index=False).encode()), "text/csv")}
    ).json()["imported"][0]["dataset_id"]
    org_id = client.post(
        "/api/datasets", files={"files": ("org.csv", io.BytesIO(org.to_csv(index=False).encode()), "text/csv")}
    ).json()["imported"][0]["dataset_id"]

    r = client.post("/api/chat", json={"message": "Show the first 10 rows.", "dataset_ids": [sales_id, org_id]})
    assert r.status_code == 200
    data = r.json()
    # the most recently uploaded dataset (org.csv) must be the active one
    assert "org.csv" in data["answer"]
    assert "Acme" in data["answer"]
    assert "Alpha" not in data["answer"]


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_upload_and_list(client, upload_sales):
    r = client.get("/api/datasets")
    assert r.status_code == 200
    ids = [d["dataset_id"] for d in r.json()["datasets"]]
    assert upload_sales in ids


def test_preview_and_export(client, upload_sales):
    prev = client.get(f"/api/datasets/{upload_sales}/preview")
    assert prev.status_code == 200
    assert len(prev.json()["rows"]) > 0

    export = client.get(f"/api/datasets/{upload_sales}/export")
    assert export.status_code == 200
    assert "region" in export.text


def test_chat_blocking(client, upload_sales):
    r = client.post("/api/chat", json={"message": "Which region has the highest revenue?", "dataset_ids": [upload_sales]})
    assert r.status_code == 200
    data = r.json()
    assert data["answer"]
    assert "North" in data["answer"]
    assert data["session_id"]


def test_chat_stream(client, upload_sales):
    with client.stream(
        "POST", "/api/chat/stream", json={"message": "Show monthly revenue trends.", "dataset_ids": [upload_sales]}
    ) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    assert "event: meta" in body
    assert "event: tool_start" in body or "event: plan" in body
    assert "event: final" in body


def test_chat_without_datasets(client):
    r = client.post("/api/chat", json={"message": "hello"})
    assert r.status_code == 200
    assert r.json()["answer"]


def test_session_roundtrip(client, upload_sales):
    create = client.post("/api/chat/session", json={"dataset_ids": [upload_sales]})
    assert create.status_code == 200
    sid = create.json()["session_id"]
    hist = client.get(f"/api/chat/session/{sid}/history")
    assert hist.status_code == 200
    reset = client.post(f"/api/chat/session/{sid}/reset")
    assert reset.status_code == 200


def test_chat_context_persists(client, upload_sales):
    client.post("/api/chat", json={"message": "Which region has the highest revenue?", "dataset_ids": [upload_sales]})
    follow = client.post(
        "/api/chat", json={"message": "Now forecast its revenue for 2 periods.", "dataset_ids": [upload_sales]}
    )
    assert follow.status_code == 200
    assert follow.json()["answer"]


def test_insights_endpoint(client, upload_sales):
    r = client.post("/api/insights", json={"dataset_id": upload_sales})
    assert r.status_code == 200
    assert len(r.json()["insights"]) > 0


def test_chart_endpoints(client, upload_sales):
    r = client.post("/api/charts", json={"dataset_id": upload_sales, "kind": "bar", "x": "region", "y": "revenue"})
    assert r.status_code == 200
    chart_id = r.json()["chart_id"]
    img = client.get(f"/api/charts/{chart_id}/image")
    assert img.status_code == 200
    assert img.content.startswith(b"\x89PNG")
    html = client.get(f"/api/charts/{chart_id}/html")
    assert html.status_code == 200
    assert "<!DOCTYPE html>" in html.text


def test_dashboard_endpoint(client, upload_sales):
    r = client.post("/api/dashboards", json={"dataset_ids": [upload_sales], "title": "Sales Board"})
    assert r.status_code == 200
    dash_id = r.json()["dashboard_id"]
    html = client.get(f"/api/dashboards/{dash_id}/html")
    assert html.status_code == 200
    assert "Sales Board" in html.text


def test_report_endpoints(client, upload_sales):
    md = client.post(f"/api/reports/{upload_sales}/markdown")
    assert md.status_code == 200
    assert md.text.startswith("#")
    h = client.post(f"/api/reports/{upload_sales}/html")
    assert h.status_code == 200
    assert "<!DOCTYPE html>" in h.text


def test_metrics_and_cache(client):
    r = client.get("/api/metrics")
    assert r.status_code == 200
    assert "counters" in r.json()
    c = client.get("/api/cache/stats")
    assert c.status_code == 200


def test_auth_required_when_enabled(client, upload_sales, monkeypatch):
    import app.main as main
    from fastapi import HTTPException

    def fake_check_auth(request):
        if request.headers.get("authorization") != "Bearer test-secret":
            raise HTTPException(status_code=401, detail="Missing bearer token")

    monkeypatch.setattr(main, "is_auth_enabled", lambda: True)
    monkeypatch.setattr(main, "check_auth", fake_check_auth)

    r = client.get("/api/datasets")
    assert r.status_code == 401
    r2 = client.get("/api/datasets", headers={"Authorization": "Bearer test-secret"})
    assert r2.status_code == 200


def test_invalid_sql_returns_helpful_error(client, upload_sales):
    r = client.post(
        "/api/chat",
        json={
            "message": "Write SQL for: SELECT * FROM sales WHERE 1=1",
            "dataset_ids": [upload_sales],
            "focus": "sql",
        },
    )
    assert r.status_code == 200
    assert r.json()["answer"]


def test_web_app_assets_load():
    """index.html's stylesheet and script must resolve (they 404'd when referenced relatively)."""
    import re

    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    html = client.get("/").text
    assets = re.findall(r'(?:href|src)="(/[^"]+\.(?:css|js))"', html)
    assert sorted(assets) == ["/app/app.js", "/app/styles.css"]
    for path in assets:
        assert client.get(path).status_code == 200
