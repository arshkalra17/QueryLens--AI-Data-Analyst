"""Tests for chart generation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.engine.charts import CHART_KINDS, make_chart


def test_all_chart_kinds_render(sales_df):
    configs = [
        ("bar", {"x": "region", "y": "revenue", "agg": "sum"}),
        ("barh", {"x": "region", "y": "revenue", "agg": "mean"}),
        ("line", {"x": "date", "y": "revenue"}),
        ("pie", {"x": "region"}),
        ("scatter", {"x": "units", "y": "revenue"}),
        ("hist", {"y": "revenue"}),
        ("box", {"y": "revenue", "group_by": "region"}),
        ("area", {"x": "date", "y": "revenue"}),
        ("heatmap", {}),
    ]
    for kind, kwargs in configs:
        chart = make_chart(sales_df, kind=kind, **kwargs)
        assert chart.png.startswith(b"\x89PNG")
        assert chart.caption
        assert chart.html.startswith("<!DOCTYPE html>")


def test_invalid_kind_rejected(sales_df):
    with pytest.raises(ValueError):
        make_chart(sales_df, kind="radar")


def test_count_agg_ignores_missing_y_column(simple_df):
    # "which country has the most customers" charts row counts: y="row_count"
    # is not a source column, but agg="count" must count rows per x anyway.
    chart = make_chart(simple_df, kind="bar", x="city", y="row_count", agg="count")
    assert chart.png.startswith(b"\x89PNG")
    assert chart.caption



def test_unknown_columns_error(sales_df):
    with pytest.raises(Exception):
        make_chart(sales_df, kind="bar", x="nope", y="revenue")


def test_chart_store_roundtrip(sales_df):
    from app.engine.charts import ChartStore

    store = ChartStore()
    chart = make_chart(sales_df, kind="bar", x="region", y="revenue", agg="sum")
    store.put(chart)
    assert store.get(chart.chart_id) is chart
    assert store.get("missing") is None


def test_chart_kinds_set():
    assert {"bar", "line", "pie", "scatter"} <= CHART_KINDS


# --- correctness: a wrong chart must never be produced --------------------------


def test_make_chart_refuses_empty_data():
    """A chart must refuse to render when given an empty/failed query result."""
    empty = pd.DataFrame({"country": pd.Series(dtype=str), "value": pd.Series(dtype=float)})
    with pytest.raises(ValueError, match="[Nn]o data to chart"):
        make_chart(empty, kind="bar", x="country")


def test_make_chart_refuses_missing_y_column():
    """A chart must refuse when the requested y column does not exist."""
    df = pd.DataFrame({"country": ["A", "B"], "value": [1.0, 2.0]})
    with pytest.raises(ValueError, match="does not exist"):
        make_chart(df, kind="bar", x="country", y="row_count")


def test_make_chart_refuses_non_numeric_y():
    df = pd.DataFrame({"country": ["A", "B"], "name": ["x", "y"]})
    with pytest.raises(ValueError, match="not numeric"):
        make_chart(df, kind="bar", x="country", y="name")


def test_make_chart_understands_sql_count_expression():
    """'customers by country' style requests may arrive as y='count(*)' (as an
    LLM would write it); it must be treated as a count aggregation, not a crash."""
    df = pd.DataFrame(
        {"country": ["China", "China", "Liechtenstein", "Gabon", "Gabon", "Gabon"]}
    )
    chart = make_chart(df, kind="bar", x="country", y="COUNT(*)")
    spec = chart.plotly_json
    data = spec["data"][0]
    pairs = dict(zip(data.get("x"), data.get("y")))
    assert pairs == {"China": 2, "Gabon": 3, "Liechtenstein": 1}, pairs


def test_chart_values_match_real_aggregation():
    """Chart values must equal an independent pandas groupby-count on the real
    customers-1000_2.csv: max is Liechtenstein=12, not a 1000+ row index sum."""
    path = Path(__file__).resolve().parents[2] / "sample_data" / "customers-1000_2.csv"
    if not path.exists():
        pytest.skip("sample customers-1000_2.csv not present")
    df = pd.read_csv(path)
    expected = df["Country"].value_counts().sort_values(ascending=False)
    assert expected.idxmax() == "Liechtenstein" and expected.max() == 12, expected.head(3)

    # fallback-style request: bar chart of customers by country (no numeric y)
    chart = make_chart(df, kind="bar", x="Country", agg="count")
    spec = chart.plotly_json
    data = spec["data"][0]
    pairs = dict(zip(data.get("x"), data.get("y")))
    for country, count in expected.head(5).items():
        assert pairs.get(country) == count, (country, pairs.get(country), count)
    assert max(pairs.values()) == 12 and pairs["Liechtenstein"] == 12


def test_tool_generate_chart_no_chart_on_failure():
    """A failed/failed-result chart request must NOT emit a chart into the store."""
    from app.core.registry import DataFrameRegistry
    from app.engine.charts import ChartStore
    from app.engine.tools import ToolError, run_tool

    store = ChartStore()
    reg = DataFrameRegistry()
    rec = reg.register(pd.DataFrame({"country": ["A", "B"], "value": [1.0, 2.0]}), "c.csv", "test")
    ctx = {
        "registry": reg,
        "chart_store": store,
        "pinned": rec.dataset_id,
        "dataset_ids": [rec.dataset_id],
        "metrics": [],
    }
    with pytest.raises(ToolError):
        run_tool(ctx, "generate_chart", {"dataset_id": rec.dataset_id, "kind": "bar", "x": "country", "y": "row_count"})
    assert len(store._charts) == 0, "a failed chart request must not be stored"

    # an empty dataset also refuses
    rec2 = reg.register(pd.DataFrame({"country": pd.Series(dtype=str)}), "empty.csv", "test")
    with pytest.raises(ToolError):
        run_tool(ctx, "generate_chart", {"dataset_id": rec2.dataset_id, "kind": "bar", "x": "country"})
    assert len(store._charts) == 0


def test_bare_bar_chart_infers_count_dimension():
    """'bar chart' with no x/y/agg must infer a dimension and count rows per it
    (real data: Liechtenstein=12), never error out or plot a row index."""
    path = Path(__file__).resolve().parents[2] / "sample_data" / "customers-1000_2.csv"
    if not path.exists():
        pytest.skip("sample customers-1000_2.csv not present")
    df = pd.read_csv(path)
    chart = make_chart(df, kind="bar")
    spec = chart.plotly_json
    data = spec["data"][0]
    pairs = dict(zip(data.get("x"), data.get("y")))
    assert pairs.get("Liechtenstein") == 12, pairs
    assert max(pairs.values()) == 12


def test_tool_accepts_llm_aliases_and_case_insensitive_columns():
    """LLMs use 'chart_type'/'type', empty-string y, and lowercase column names;
    the tool must still produce a correct count chart."""
    from app.core.registry import DataFrameRegistry
    from app.engine.charts import ChartStore
    from app.engine.tools import run_tool

    df = pd.DataFrame(
        {"Country": ["China", "China", "Liechtenstein", "Gabon", "Gabon", "Gabon"]}
    )
    reg = DataFrameRegistry()
    rec = reg.register(df, "cc.csv", "test")
    store = ChartStore()
    ctx = {"registry": reg, "chart_store": store, "pinned": rec.dataset_id, "dataset_ids": [rec.dataset_id], "metrics": []}

    res = run_tool(
        ctx, "generate_chart",
        {"dataset_id": rec.dataset_id, "chart_type": "bar", "x": "country", "y": "", "agg": "count"},
    )
    cr = store.get(res["chart_id"])
    spec = cr.plotly_json["data"][0]
    assert dict(zip(spec.get("x"), spec.get("y"))) == {"China": 2, "Gabon": 3, "Liechtenstein": 1}
