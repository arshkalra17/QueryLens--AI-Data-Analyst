"""Rule-engine behaviour found and fixed via the SQL accuracy benchmark.

Each test pins a failure mode the benchmark exposed, checked by executing the
generated SQL rather than matching its text.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.core.registry import DataFrameRegistry
from app.core.sessions import SessionManager
from app.engine.charts import get_chart_store
from app.engine.conditions import extract_where
from app.engine.fallback import narrate, plan
from app.engine.sql_exec import run_sql
from app.engine.tools import _frames, _safe_name, run_tool


@pytest.fixture()
def sales():
    reg = DataFrameRegistry()
    df = pd.DataFrame({
        "region": ["West", "West", "East", "East", "North"],
        "sales_rep": ["Ann", "Bob", "Ann", None, "Cy"],
        "customer_id": ["C1", "C1", "C2", "C3", "C4"],
        "units_sold": [2, 3, 5, 1, 4],
        "unit_price": [10.0, 20.0, 10.0, 30.0, 5.0],
        "revenue": [20.0, 60.0, 50.0, 30.0, 20.0],
        "rating": [5, 3, 5, 1, 4],
    })
    rec = reg.register(df, "sales.csv", "t")
    return reg, rec


def _ctx(reg, rec):
    return {
        "registry": reg, "sessions": SessionManager(), "session_id": None,
        "pinned": rec.dataset_id, "dataset_ids": [rec.dataset_id],
        "settings": type("S", (), {"allow_pandas_exec": True})(),
        "chart_store": get_chart_store(), "metrics": [],
    }


def _answer(reg, rec, question):
    """(plan, rows of the plan's SQL) for ``question``."""
    p = plan(_ctx(reg, rec), question)
    sql = next(c["args"]["sql"] for c in p["tool_calls"] if c["name"] == "run_sql")
    result = run_sql(sql, _frames({"registry": reg, "sessions": None}, None))
    assert result.error is None, (sql, result.error)
    return p, [tuple(r.values()) for r in result.rows]


def test_count_of_sales_counts_rows_not_a_related_column(sales):
    # "sales" used to match "sales_rep" and count distinct reps
    _, rows = _answer(*sales, "How many sales were made in the West region?")
    assert rows == [(2,)]


def test_count_of_quantity_column_is_summed(sales):
    p, rows = _answer(*sales, "How many units were sold in total?")
    assert rows == [(15,)]
    assert p["count_measure"] == "units_sold"


@pytest.mark.parametrize("question, expected", [
    ("What is the total revenue?", [(180.0,)]),
    ("What is the average rating?", [(3.6,)]),
    ("What is the highest single revenue value?", [(60.0,)]),
    ("What is the lowest unit price?", [(5.0,)]),
    ("What is the maximum number of units sold?", [(5,)]),
    ("Total revenue in the West region", [(80.0,)]),
])
def test_scalar_aggregates(sales, question, expected):
    p, rows = _answer(*sales, question)
    assert p["narrative"] == "aggregate"
    assert rows == expected


def test_grouped_aggregate(sales):
    _, rows = _answer(*sales, "What is the total revenue by region?")
    assert sorted(rows) == [("East", 80.0), ("North", 20.0), ("West", 80.0)]


def test_grouping_without_function_defaults_by_metric_kind(sales):
    _, summed = _answer(*sales, "Revenue per region")
    assert ("West", 80.0) in summed
    _, averaged = _answer(*sales, "Unit price per region")
    assert ("West", 15.0) in averaged


def test_across_all_is_scope_not_grouping(sales):
    _, rows = _answer(*sales, "Sum of revenue across all sales")
    assert rows == [(180.0,)]


def test_ranking_excludes_null_groups(sales):
    _, rows = _answer(*sales, "Which sales rep has the lowest revenue?")
    assert rows[0][0] == "Cy" and all(r[0] is not None for r in rows)


def test_fewest_ranks_ascending(sales):
    _, rows = _answer(*sales, "Which region has the fewest sales?")
    assert rows[0][0] == "North"


@pytest.mark.parametrize("question, cond", [
    ("How many sales have a rating of 5?", '"rating" = 5'),
    ("How many sales had units sold greater than 3?", '"units_sold" > 3'),
    ("How many sales had a unit price below 15?", '"unit_price" < 15'),
    ("How many sales had units sold at least 4?", '"units_sold" >= 4'),
])
def test_numeric_comparison_conditions(sales, question, cond):
    reg, rec = sales
    assert extract_where(question, rec) == cond


def test_comparison_never_fires_on_top_n(sales):
    assert extract_where("Show the top 3 regions by revenue", sales[1]) is None


def test_attached_dataset_is_sticky_against_weak_name_match():
    reg = DataFrameRegistry()
    reg.register(pd.DataFrame({"region": ["W"], "revenue": [1.0]}), "sales_data.csv", "t")
    retail = reg.register(pd.DataFrame({"city": ["X", "Y"], "total": [5.0, 7.0]}), "retail.csv", "t")
    # "sales" names the other file, but the attached retail file has a "total" column
    p, rows = _answer(reg, retail, "What is the total sales amount?")
    assert _safe_name("retail.csv") in p["tool_calls"][0]["args"]["sql"]
    assert rows == [(12.0,)]


def test_aggregate_narrative_reads_naturally(sales):
    reg, rec = sales
    ctx = _ctx(reg, rec)
    p = plan(ctx, "What is the total revenue?")
    results = [run_tool(ctx, c["name"], c["args"]) for c in p["tool_calls"]]
    out = narrate(ctx, "What is the total revenue?", p, results, None)
    assert "The **total revenue** in `sales.csv` is **180**." in out["answer"]
