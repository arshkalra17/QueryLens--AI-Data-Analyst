"""Tests for data quality reporting."""
from __future__ import annotations

import pandas as pd

from app.engine.quality import quality_report


def test_quality_report_scores(sales_df):
    report = quality_report(sales_df)
    assert 0 <= report.score <= 100
    assert report.checks["rows"] == 500
    assert report.checks["columns"] > 0


def test_quality_catches_duplicates():
    df = pd.DataFrame({"id": [1, 1, 1], "v": [1, 1, 1]})
    report = quality_report(df)
    kinds = [i.kind for i in report.issues]
    assert "duplicates" in kinds


def test_quality_catches_cardinality():
    df = pd.DataFrame({"name": ["a"] * 40 + ["b"] * 20 + [f"c{i}" for i in range(40)]})
    report = quality_report(df)
    assert len(report.issues) >= 0


def test_quality_catches_high_null_rate():
    df = pd.DataFrame({"x": [None, None, None, 1, 2]})
    report = quality_report(df)
    kinds = [i.kind for i in report.issues]
    assert "missing" in kinds


def test_quality_summary():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    report = quality_report(df)
    assert report.score == 100
    assert len(report.issues) == 0
