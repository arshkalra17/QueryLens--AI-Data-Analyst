"""Tests for insights generation."""
from __future__ import annotations

from app.engine.insights import generate_insights


def test_insights_types(sales_df):
    insights = generate_insights(sales_df)
    kinds = {i.category for i in insights}
    assert {"trend", "anomalies", "distribution"} <= kinds


def test_insights_are_explainable(sales_df):
    insights = generate_insights(sales_df)
    for ins in insights:
        assert ins.title and ins.detail
        assert ins.evidence
        assert 0 <= ins.importance <= 1


def test_insights_respect_limit(sales_df):
    insights = generate_insights(sales_df, limit=3)
    assert len(insights) <= 3


def test_insights_on_tiny_dataset(simple_df):
    insights = generate_insights(simple_df)
    assert isinstance(insights, list)
    assert all(ins.title for ins in insights)
