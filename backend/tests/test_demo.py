"""Public-demo safeguards: per-IP rate limit, delete protection, seeded samples."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.core import demo
from app.core.demo import RateLimiter, client_ip
from app.core.registry import get_registry
from app.main import app


def test_rate_limiter_sliding_window():
    rl = RateLimiter(limit=2, window_seconds=60)
    assert rl.allow("a", now=0)[0] and rl.allow("a", now=1)[0]
    allowed, retry = rl.allow("a", now=2)
    assert not allowed and retry == 59
    assert rl.allow("b", now=2)[0]            # limits are per client
    assert rl.allow("a", now=60.5)[0]          # oldest hit expired


def test_client_ip_prefers_first_forwarded_hop():
    assert client_ip({"x-forwarded-for": "1.2.3.4, 10.0.0.1"}, "10.0.0.9") == "1.2.3.4"
    assert client_ip({}, "10.0.0.9") == "10.0.0.9"


@pytest.fixture()
def demo_settings(monkeypatch):
    monkeypatch.setattr(get_settings(), "demo_mode", True)
    monkeypatch.setattr(get_settings(), "rate_limit_per_hour", 2)
    monkeypatch.setattr(demo, "_limiter", None)
    yield
    demo._limiter = None


def test_demo_limits_posts_and_blocks_deletes(demo_settings):
    client = TestClient(app)
    headers = {"x-forwarded-for": "203.0.113.7"}
    for _ in range(2):
        assert client.post("/api/chat/session", headers=headers).status_code != 429
    over = client.post("/api/chat/session", headers=headers)
    assert over.status_code == 429 and "Retry-After" in over.headers
    assert client.get("/api/health", headers=headers).status_code == 200   # reads are never limited
    assert client.delete("/api/datasets/abc").status_code == 403
    assert client.get("/api/health").json()["demo"] == {"enabled": True, "rate_limit_per_hour": 2}


def test_seed_samples_loads_bundled_csvs_once(demo_settings):
    demo.seed_samples()
    names = [r.name for r in get_registry().list()]
    assert "sales_data.csv" in names
    demo.seed_samples()
    assert [r.name for r in get_registry().list()] == names   # idempotent
    for r in get_registry().list():
        get_registry().delete(r.dataset_id)
