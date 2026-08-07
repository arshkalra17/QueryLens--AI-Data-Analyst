"""Tests for forecasting."""
from __future__ import annotations

import pandas as pd

from app.engine.forecast import forecast


def test_forecast_output_shape(sales_df):
    res = forecast(sales_df, date_column="date", value_column="revenue", periods=12)
    assert res["periods"] == 12
    fc = [r for r in res["table"] if r["type"] == "forecast"]
    assert len(fc) == 12
    assert all(r["lower"] <= r["value"] <= r["upper"] for r in fc)
    assert res["summary"]
    assert res["model"]["direction"] in ("upward", "downward")


def test_forecast_seasonal_variants(sales_df):
    res_d = forecast(sales_df, date_column="date", value_column="revenue", periods=3, seasonality="dayofweek")
    res_m = forecast(sales_df, date_column="date", value_column="revenue", periods=3, seasonality="month")
    res_n = forecast(sales_df, date_column="date", value_column="revenue", periods=3, seasonality="none")
    assert all(r["type"] == "forecast" for r in res_d["table"][-3:])
    assert len(res_m["table"]) > 3
    assert len(res_n["table"]) > 3


def test_forecast_contains_actuals(sales_df):
    res = forecast(sales_df, date_column="date", value_column="revenue", periods=2)
    actuals = [r for r in res["table"] if r["type"] == "actual"]
    assert len(actuals) >= 3


def test_forecast_raises_on_tiny_data(simple_df):
    import pytest

    with pytest.raises(ValueError):
        forecast(simple_df, date_column="city", value_column="value", periods=2)
