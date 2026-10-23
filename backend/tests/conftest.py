"""Shared pytest fixtures."""
from __future__ import annotations

import os

# Force the offline rule-based engine for deterministic, fast tests regardless
# of any .env LLM configuration (must happen before app config is read).
os.environ["LLM_PROVIDER"] = "none"
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("GROQ_API_KEY", None)
os.environ.pop("GEMINI_API_KEY", None)
# Disable RAG in the general suite (dedicated RAG tests opt in with tmp dirs)
os.environ["ENABLE_RAG"] = "false"
# Keep the shared session manager in memory; persistence has dedicated tests.
os.environ["SESSION_STORE"] = "memory"
# Document RAG off by default; tests/test_knowledge.py opts in with a tmp store.
os.environ["ENABLE_DOCS"] = "false"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.core.registry import DataFrameRegistry, reset_registry  # noqa: E402
from app.core.sessions import SessionManager, reset_sessions  # noqa: E402

get_settings.cache_clear()


@pytest.fixture(scope="session", autouse=True)
def _clean_upload_dir():
    """Wipe files persisted by tests so they don't pollute real usage."""
    uploads = get_settings().upload_dir_resolved
    for path in uploads.glob("*.csv"):
        path.unlink()
    yield
    for path in uploads.glob("*.csv"):
        path.unlink()


@pytest.fixture()
def sales_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2023-01-01", periods=500, freq="D")
    regions = ["North", "South", "East", "West"]
    df = pd.DataFrame(
        {
            "date": dates,
            "region": rng.choice(regions, 500),
            "product": rng.choice(["A", "B", "C"], 500),
            "revenue": rng.uniform(100, 900, 500),
            "cost": rng.uniform(50, 500, 500),
            "units": rng.integers(1, 40, 500),
        }
    )
    # inject clear anomalies in revenue
    df.loc[10, "revenue"] = 9500.0
    df.loc[300, "revenue"] = 11000.0
    df.loc[301, "cost"] = 8000.0
    return df


@pytest.fixture()
def simple_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "city": ["Mumbai", "Delhi", "Mumbai", "Pune", "Delhi", "Mumbai"],
            "category": ["A", "A", "B", "B", "C", "C"],
            "value": [100.0, 200.0, 150.0, 300.0, 250.0, 175.0],
            "name": ["acme corp", "beta ltd", "acme corp", "gamma inc", "delta llc", "acme corp"],
        }
    )


@pytest.fixture()
def registry() -> DataFrameRegistry:
    reset_registry()
    reg = DataFrameRegistry()
    return reg


@pytest.fixture()
def sessions() -> SessionManager:
    reset_sessions()
    return SessionManager()


@pytest.fixture()
def settings():
    return get_settings()
