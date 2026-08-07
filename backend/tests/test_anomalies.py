"""Tests for anomaly detection."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.engine.anomalies import detect_anomalies


def test_anomalies_found_via_zscore(sales_df):
    flags = detect_anomalies(sales_df, columns=["revenue"], method="zscore")
    assert len(flags) >= 2
    # the injected spikes are 9500/11000 vs mean ~500 -> huge z-scores
    explained = {f.row_index: f for f in flags}
    assert 10 in explained
    assert 300 in explained
    assert "standard deviation" in explained[10].explanation.lower()
    assert explained[10].severity == "critical"


def test_iqr_method(sales_df):
    flags = detect_anomalies(sales_df, columns=["revenue"], method="iqr")
    assert len(flags) >= 2
    assert all("IQR" in f.explanation for f in flags)


def test_rolling_method_on_time_series(sales_df):
    flags = detect_anomalies(sales_df, columns=["revenue"], date_column="date", method="rolling")
    assert len(flags) >= 1
    assert any(f.row_index in (10, 300) for f in flags)


def test_group_by_scope(sales_df):
    flags = detect_anomalies(sales_df, columns=["revenue"], group_by="region", method="zscore")
    assert all(f.group in ("North", "South", "East", "West") for f in flags)


def test_no_false_positives_on_clean_data():
    df = pd.DataFrame({"v": np.random.default_rng(1).normal(50, 5, 200)})
    flags = detect_anomalies(df, columns=["v"], method="zscore", z_threshold=3.5)
    assert len(flags) == 0


def test_empty_dataset():
    assert detect_anomalies(pd.DataFrame({"a": []}), columns=["a"]) == []
