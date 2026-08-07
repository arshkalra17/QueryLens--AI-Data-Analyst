"""Tests for the rule-based fallback planner/narrator."""
from __future__ import annotations

import re

import pandas as pd

from app.core.registry import DataFrameRegistry
from app.core.sessions import SessionManager
from app.engine.charts import get_chart_store
from app.engine.fallback import _resolve_dataset, narrate, plan


def _ctx(reg, rec, session_id=None):
    return {
        "registry": reg,
        "sessions": SessionManager(),
        "session_id": session_id,
        "pinned": rec.dataset_id,
        "dataset_ids": [rec.dataset_id],
        "settings": type("S", (), {"allow_pandas_exec": True})(),
        "chart_store": get_chart_store(),
        "metrics": [],
    }


def _reg_with(simple_df):
    reg = DataFrameRegistry()
    rec = reg.register(simple_df, "cities.csv", "test")
    return reg, rec


def test_intent_detection_top_bottom(simple_df):
    reg, rec = _reg_with(simple_df)
    p = plan(_ctx(reg, rec), "What is the top city by value?")
    assert p["narrative"] in ("top", "bottom")
    assert p["tool_calls"][0]["name"] == "run_sql"


def test_intent_detection_anomalies(simple_df):
    reg, rec = _reg_with(simple_df)
    p = plan(_ctx(reg, rec), "Detect anomalies in the value column.")
    assert p["narrative"] == "anomaly"
    assert p["tool_calls"][0]["args"]["columns"] == ["value"]


def test_intent_detection_forecast_periods(sales_df):
    reg = DataFrameRegistry()
    rec = reg.register(sales_df, "sales.csv", "test")
    p = plan(_ctx(reg, rec), "Forecast revenue for the next 6 months.")
    assert p["narrative"] == "forecast"
    assert p["tool_calls"][0]["args"]["periods"] == 6


def test_intent_detection_chart(simple_df):
    reg, rec = _reg_with(simple_df)
    p = plan(_ctx(reg, rec), "Show me a pie chart of category.")
    assert p["narrative"] == "chart"
    args = p["tool_calls"][0]["args"]
    assert args["kind"] == "pie"
    assert args["x"] == "category"


def test_intent_detection_sql(simple_df):
    reg, rec = _reg_with(simple_df)
    p = plan(_ctx(reg, rec), "Generate SQL for this analysis.")
    assert p["narrative"] == "sql"
    assert p["tool_calls"][0]["name"] == "generate_sql"


def test_intent_detection_correlation(simple_df):
    reg, rec = _reg_with(simple_df)
    p = plan(_ctx(reg, rec), "Is there a correlation between value and something?")
    assert p["narrative"] == "correlation"


def test_intent_detection_quality(simple_df):
    reg, rec = _reg_with(simple_df)
    p = plan(_ctx(reg, rec), "Check the data quality of this dataset.")
    assert p["narrative"] == "quality"
    assert p["tool_calls"][0]["name"] == "data_quality"


def test_resolve_dataset_prefers_column_match():
    reg = DataFrameRegistry()
    rec = reg.register(_mk_df([("product", ["a", "b"]), ("qty", [1, 2])]), "products.csv", "t")
    rec2 = reg.register(_mk_df([("name", ["x"]), ("score", [1])]), "scores.csv", "t")
    assert _resolve_dataset("top products", {"registry": reg}) == rec.dataset_id
    assert _resolve_dataset("who has the best score", {"registry": reg}) == rec2.dataset_id


def test_resolve_dataset_no_match_returns_none():
    reg = DataFrameRegistry()
    reg.register(_mk_df([("product", ["a", "b"]), ("qty", [1, 2])]), "products.csv", "t")
    reg.register(_mk_df([("name", ["x"]), ("score", [1])]), "scores.csv", "t")
    assert _resolve_dataset("show the first 10 rows", {"registry": reg}) is None


def test_resolve_dataset_tie_keeps_pinned(simple_df, sales_df):
    reg = DataFrameRegistry()
    reg.register(simple_df, "cities.csv", "t")
    rec2 = reg.register(sales_df, "sales.csv", "t")
    # "what columns does this dataset have" matches both files' "data" token
    # equally, so it must NOT resolve to a specific dataset (pinned stays active).
    assert _resolve_dataset("what columns does this dataset have", {"registry": reg}) is None
    ctx = _ctx(reg, rec2)
    p = plan(ctx, "What columns does this dataset have?")
    assert p["tool_calls"][0]["args"]["dataset_id"] == rec2.dataset_id


def test_resolve_dataset_dedupes_reloaded_duplicates(sales_df):
    reg = DataFrameRegistry()
    # Simulate a persisted reload: the same content re-registered under the
    # mangled persisted filename alongside the original upload.
    rec = reg.register(sales_df, "sales.csv", "t")
    reg.register(sales_df, "sales_csv.csv", "t")
    # Two records, same content hash: scoring must not see a false tie.
    assert _resolve_dataset("which region generated the highest revenue", {"registry": reg}) == rec.dataset_id


def test_plan_preview_keeps_pinned_when_no_dataset_mentioned(simple_df, sales_df):
    reg = DataFrameRegistry()
    reg.register(simple_df, "cities.csv", "t")
    rec2 = reg.register(sales_df, "sales.csv", "t")
    ctx = _ctx(reg, rec2)
    p = plan(ctx, "Show the first 10 rows.")
    assert p["narrative"] == "preview"
    assert p["tool_calls"][0]["args"]["dataset_id"] == rec2.dataset_id


def test_intent_detection_insights(simple_df):
    reg, rec = _reg_with(simple_df)
    p = plan(_ctx(reg, rec), "Give me insights about this dataset.")
    assert p["narrative"] == "insights"
    assert p["tool_calls"][0]["name"] == "generate_insights"


def test_intent_detection_preview_rows(simple_df):
    reg, rec = _reg_with(simple_df)
    p = plan(_ctx(reg, rec), "Show the first 10 rows.")
    assert p["narrative"] == "preview"
    assert p["tool_calls"][0]["name"] == "preview_rows"
    assert p["tool_calls"][0]["args"]["limit"] == 10


def test_intent_detection_preview_default_limit(simple_df):
    reg, rec = _reg_with(simple_df)
    p = plan(_ctx(reg, rec), "Preview the data.")
    assert p["narrative"] == "preview"
    assert p["tool_calls"][0]["args"]["limit"] == 10


def test_narrate_preview_returns_raw_rows(simple_df):
    from app.engine.tools import run_tool

    reg, rec = _reg_with(simple_df)
    ctx = _ctx(reg, rec)
    p = plan(ctx, "Show the first 10 rows.")
    results = [run_tool(ctx, c["name"], c["args"]) for c in p["tool_calls"]]
    out = narrate(ctx, "Show the first 10 rows.", p, results, None)
    assert out["narrative"] == "preview"
    assert "Mumbai" in out["answer"]
    assert "aggregation" not in out["answer"].lower() or "no aggregation" in out["answer"].lower()


def test_count_intent_rows(simple_df):
    reg, rec = _reg_with(simple_df)
    p = plan(_ctx(reg, rec), "How many rows are in the dataset?")
    assert p["narrative"] == "count"
    assert p["tool_calls"][0]["name"] == "run_sql"
    assert "COUNT(*)" in p["tool_calls"][0]["args"]["sql"]

def test_count_intent_distinct_entity(simple_df):
    reg, rec = _reg_with(simple_df)
    p = plan(_ctx(reg, rec), "How many cities are in the dataset?")
    assert p["narrative"] == "count"
    sql = p["tool_calls"][0]["args"]["sql"]
    assert "COUNT(DISTINCT" in sql
    assert '"city"' in sql


def _customer_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Customer Id": [1, 2, 3, 4, 5, 6],
            "full_name": ["a", "b", "c", "d", "e", "f"],
            "country": ["India", "India", "Japan", "India", "Japan", "India"],
            "age": [32, 45, 28, 39, 51, 44],
            "loyalty_score": [88, 71, 95, 60, 82, 66],
        }
    )


def test_count_rank_most_customers_groups_by_country():
    reg, rec = _reg_with(_customer_df())
    p = plan(_ctx(reg, rec), "Which country has the most customers?")
    sql = p["tool_calls"][0]["args"]["sql"]
    assert "COUNT(*)" in sql
    assert '"country"' in sql
    assert "Customer Id" not in sql
    assert 'ORDER BY row_count DESC' in sql


def test_count_rank_most_common_country():
    reg, rec = _reg_with(_customer_df())
    p = plan(_ctx(reg, rec), "What's the most common country?")
    sql = p["tool_calls"][0]["args"]["sql"]
    assert "COUNT(*)" in sql
    assert '"country"' in sql
    assert 'ORDER BY row_count DESC' in sql


def test_count_rank_least_common_ascending():
    reg, rec = _reg_with(_customer_df())
    p = plan(_ctx(reg, rec), "What is the least common country?")
    sql = p["tool_calls"][0]["args"]["sql"]
    assert "COUNT(*)" in sql
    assert 'ORDER BY row_count ASC' in sql


def test_count_rank_metric_column_uses_sum(sales_df):
    reg = DataFrameRegistry()
    rec = reg.register(sales_df, "sales.csv", "test")
    p = plan(_ctx(reg, rec), "Which region generated the highest revenue?")
    sql = p["tool_calls"][0]["args"]["sql"]
    assert 'SUM("revenue")' in sql
    assert '"region"' in sql


def test_pick_metric_ignores_numeric_id_column():
    reg, rec = _reg_with(_customer_df())
    p = plan(_ctx(reg, rec), "What are the top five customers?")
    sql = p["tool_calls"][0]["args"]["sql"]
    assert 'SUM("loyalty_score")' in sql
    assert 'SUM("Customer Id")' not in sql


def test_narrate_count_answers(simple_df):
    from app.engine.tools import run_tool

    reg, rec = _reg_with(simple_df)
    ctx = _ctx(reg, rec)
    p = plan(ctx, "How many cities are in the dataset?")
    results = [run_tool(ctx, c["name"], c["args"]) for c in p["tool_calls"]]
    out = narrate(ctx, "How many cities are in the dataset?", p, results, None)
    assert out["narrative"] == "count"
    assert "3" in out["answer"]
    assert "cities" in out["answer"].lower()


def test_count_rows_uses_count_star_even_with_id_column():
    df = _mk_df([("id", [1, 1, 2, 3]), ("name", ["a", "a", "b", "c"])])
    reg = DataFrameRegistry()
    rec = reg.register(df, "people.csv", "t")
    p = plan(_ctx(reg, rec), "How many rows in total?")
    assert "COUNT(*)" in p["tool_calls"][0]["args"]["sql"]
    assert "COUNT(DISTINCT" not in p["tool_calls"][0]["args"]["sql"]


def test_intent_filter_list_from(simple_df):
    reg, rec = _reg_with(simple_df)
    p = plan(_ctx(reg, rec), "List customers from Mumbai")
    assert p["narrative"] == "filter"
    args = p["tool_calls"][0]["args"]
    assert p["tool_calls"][0]["name"] == "filter_rows"
    assert args["column"] == "city"
    assert args["value"].lower() == "mumbai"


def test_intent_filter_show_in(simple_df):
    reg, rec = _reg_with(simple_df)
    p = plan(_ctx(reg, rec), "Show customers in Delhi")
    assert p["narrative"] == "filter"
    args = p["tool_calls"][0]["args"]
    assert args["column"] == "city"
    assert args["value"].lower() == "delhi"


def test_intent_filter_where_clause(simple_df):
    reg, rec = _reg_with(simple_df)
    p = plan(_ctx(reg, rec), "Show rows where category = A")
    assert p["narrative"] == "filter"
    args = p["tool_calls"][0]["args"]
    assert args["column"] == "category"
    assert args["value"].lower() == "a"


def test_intent_filter_find(simple_df):
    reg, rec = _reg_with(simple_df)
    p = plan(_ctx(reg, rec), "Find the city Pune")
    assert p["narrative"] == "filter"
    args = p["tool_calls"][0]["args"]
    assert args["column"] == "city"
    assert args["value"].lower() == "pune"


def test_intent_filter_named_column_value(simple_df):
    reg, rec = _reg_with(simple_df)
    p = plan(_ctx(reg, rec), "Find the name beta ltd")
    assert p["narrative"] == "filter"
    args = p["tool_calls"][0]["args"]
    assert args["column"] == "name"


def test_narrate_filter_returns_rows(simple_df):
    from app.engine.tools import run_tool

    reg, rec = _reg_with(simple_df)
    ctx = _ctx(reg, rec)
    p = plan(ctx, "List customers from Mumbai")
    results = [run_tool(ctx, c["name"], c["args"]) for c in p["tool_calls"]]
    out = narrate(ctx, "List customers from Mumbai", p, results, None)
    assert out["narrative"] == "filter"
    assert "Mumbai" in out["answer"]
    assert "row_count" not in out["answer"]
    assert "filtered the dataframe" in out["answer"].lower() or "filter" in out["answer"].lower()


def test_filter_intent_not_stronger_intents(simple_df):
    from app.engine.fallback import _filter_intent

    assert not _filter_intent("Show the monthly revenue trend")
    assert not _filter_intent("Show the top 5 customers")
    assert not _filter_intent("Create a bar chart of revenue by region")
    assert not _filter_intent("Detect anomalies in revenue")
    assert not _filter_intent("Summarize the dataset")
    assert not _filter_intent("list datasets")
    assert _filter_intent("List customers from India")
    assert _filter_intent("Show customers in Japan")
    assert _filter_intent("Show rows where Country = India")
    assert _filter_intent("Find customer ID XYZ")


def test_intent_routing_independent(simple_df):
    reg, rec = _reg_with(simple_df)
    ctx = _ctx(reg, rec)

    assert plan(ctx, "Count customers by city")["narrative"] == "count"
    assert plan(ctx, "Create a bar chart of value by city")["narrative"] == "chart"
    assert plan(ctx, "Generate SQL for this analysis")["narrative"] == "sql"
    assert plan(ctx, "Generate pandas code")["narrative"] == "pandas"
    assert plan(ctx, "Detect anomalies in value")["narrative"] == "anomaly"
    assert plan(ctx, "Summarize the dataset")["narrative"] == "insights"
    assert plan(ctx, "List customers from Mumbai")["narrative"] == "filter"


def test_intent_schema_what_columns(simple_df):
    reg, rec = _reg_with(simple_df)
    p = plan(_ctx(reg, rec), "What columns does this dataset have?")
    assert p["narrative"] == "schema"
    assert p["tool_calls"][0]["name"] == "get_schema"


def test_intent_schema_variants(simple_df):
    reg, rec = _reg_with(simple_df)
    ctx = _ctx(reg, rec)
    for q in ("List the columns", "Show the schema", "What fields are in the dataset?"):
        assert plan(ctx, q)["narrative"] == "schema", q


def test_schema_intent_not_stronger(simple_df):
    from app.engine.fallback import _schema_intent

    assert _schema_intent("What columns does this dataset have?")
    assert _schema_intent("list all the columns")
    assert not _schema_intent("Which columns have the highest revenue?")
    assert not _schema_intent("Describe the dataset")
    assert not _schema_intent("Show the monthly trend by region")


def test_narrate_schema_lists_columns(simple_df):
    from app.engine.tools import run_tool

    reg, rec = _reg_with(simple_df)
    ctx = _ctx(reg, rec)
    p = plan(ctx, "What columns does this dataset have?")
    results = [run_tool(ctx, c["name"], c["args"]) for c in p["tool_calls"]]
    out = narrate(ctx, "What columns does this dataset have?", p, results, None)
    assert out["narrative"] == "schema"
    assert "city" in out["answer"]
    assert "value" in out["answer"]
    assert "column" in out["answer"].lower()


def test_count_entity_uses_distinct_id_column():
    df = _mk_df([("id", [1, 1, 2, 3]), ("name", ["a", "a", "b", "c"])])
    reg = DataFrameRegistry()
    rec = reg.register(df, "people.csv", "t")
    p = plan(_ctx(reg, rec), "How many people are in the dataset?")
    assert "COUNT(DISTINCT" in p["tool_calls"][0]["args"]["sql"]


def test_count_unique_dimension_uses_column_not_id():
    df = _mk_df([("Customer Id", [1, 2, 3, 4]), ("country", ["US", "US", "CA", "MX"])])
    reg = DataFrameRegistry()
    rec = reg.register(df, "customers.csv", "t")
    # "unique" must not leak into the entity, and "countries" must not match an
    # unrelated "Customer Id" column.
    p = plan(_ctx(reg, rec), "How many unique countries are there?")
    sql = p["tool_calls"][0]["args"]["sql"]
    assert "COUNT(DISTINCT" in sql
    assert '"country"' in sql
    assert '"Customer Id"' not in sql


def test_count_unknown_dimension_falls_back_to_count_star():
    df = _mk_df([("Customer Id", [1, 2, 3, 4]), ("name", ["a", "b", "c", "d"])])
    reg = DataFrameRegistry()
    rec = reg.register(df, "customers.csv", "t")
    p = plan(_ctx(reg, rec), "How many countries are in the dataset?")
    assert "COUNT(*)" in p["tool_calls"][0]["args"]["sql"]
    assert "COUNT(DISTINCT" not in p["tool_calls"][0]["args"]["sql"]


def test_narrate_produces_markdown(simple_df):
    from app.engine.tools import run_tool

    reg, rec = _reg_with(simple_df)
    ctx = _ctx(reg, rec)
    p = plan(ctx, "What is the top city by value?")
    results = []
    for call in p["tool_calls"]:
        results.append(run_tool(ctx, call["name"], call["args"]))
    out = narrate(ctx, "What is the top city by value?", p, results, None)
    assert out["answer"]
    assert "Mumbai" in out["answer"]
    assert "reasoning" in out
    assert out["narrative"] in ("top", "bottom")


def test_narrate_surfaces_tool_errors_instead_of_generic_text(simple_df):
    reg, rec = _reg_with(simple_df)
    plan_with_errors = {
        "narrative": "generic",
        "tool_calls": [
            {"name": "filter_rows", "args": {}},
            {"name": "run_sql", "args": {"sql": "SELECT broken FROM t"}},
        ],
    }
    results = [
        {"error": "filter_rows: Provide both a `column` and a `value` to filter by."},
        {"error": "SQL execution failed: syntax error at or near \"broken\""},
    ]
    out = narrate(_ctx(reg, rec), "which country has the most customers?", plan_with_errors, results, None)
    assert out["narrative"] == "error"
    assert "filter_rows" in out["answer"]
    assert "syntax error" in out["answer"]
    assert "SELECT broken FROM t" in out["answer"]
    assert "quick SQL profile" not in out["answer"]


def test_narrate_generic_is_honest_with_success(simple_df):
    from app.engine.tools import run_tool

    reg, rec = _reg_with(simple_df)
    ctx = _ctx(reg, rec)
    p = {"narrative": "generic", "tool_calls": [{"name": "run_sql", "args": {"sql": 'SELECT COUNT(*) AS n FROM "cities_csv"'}}]}
    results = [run_tool(ctx, "run_sql", p["tool_calls"][0]["args"])]
    out = narrate(ctx, "How many rows?", p, results, None)
    assert "quick SQL profile" not in out["answer"]
    assert "distributions and correlations" not in out["answer"]
    assert out["narrative"] == "generic"

def test_narrate_anomaly_explains(simple_df):
    from app.engine.tools import run_tool

    reg, rec = _reg_with(simple_df)
    ctx = _ctx(reg, rec)
    p = plan(ctx, "Detect anomalies in the value column.")
    results = [run_tool(ctx, c["name"], c["args"]) for c in p["tool_calls"]]
    out = narrate(ctx, "Detect anomalies in the value column.", p, results, None)
    assert "anomal" in out["answer"].lower()


def _mk_df(cols):
    return pd.DataFrame(dict(cols))


# Mirrors the country frequency distribution of the real customers-1000_2.csv
# fixture: {count: number_of_countries}, summing to 240 countries / 1000 rows.
# The key features this exercises: a clear global min (1) that is far outside
# the top-12 chart window, and a 5-way tie at 9 that straddles the top-5 limit.
_CUSTOMER_COUNTRY_DIST = {12: 1, 10: 1, 9: 5, 8: 7, 7: 23, 6: 14, 5: 44, 4: 48, 3: 45, 2: 33, 1: 19}
_CUSTOMER_COUNTRY_NAMES = [
    "Liechtenstein", "Gabon", "China", "Bangladesh", "Reunion", "Nigeria",
    "Luxembourg", "Uganda", "Pitcairn Islands", "New Zealand", "Cyprus",
    "Hong Kong", "Italy",
]


def _customer_country_df() -> pd.DataFrame:
    rows: list[tuple] = []
    ci = 0
    for count, num in _CUSTOMER_COUNTRY_DIST.items():
        for _ in range(num):
            name = _CUSTOMER_COUNTRY_NAMES[ci] if ci < len(_CUSTOMER_COUNTRY_NAMES) else f"Country{ci:03d}"
            ci += 1
            for _ in range(count):
                rows.append((f"C{len(rows) + 1:05d}", name))
    return pd.DataFrame(rows, columns=["Customer Id", "Country"])


def test_count_rank_min_max_match_independent_groupby():
    """The 'highest'/'lowest' claims and displayed top-N must reflect the true
    global min/max over the entire grouped result, never just the rows that
    happen to be rendered, and ties cut by the display limit must be surfaced."""
    from app.engine.tools import run_tool

    df = _customer_country_df()
    expected = df["Country"].value_counts()
    true_min, true_max = int(expected.min()), int(expected.max())
    assert (true_min, true_max) == (1, 12)

    reg, rec = _reg_with(df)
    ctx = _ctx(reg, rec)
    p = plan(ctx, "Which country has the most customers?")
    sql = p["tool_calls"][0]["args"]["sql"]
    # the grouped query must not hard-truncate, or min/tie detection is impossible
    assert "LIMIT" not in sql
    results = [run_tool(ctx, c["name"], c["args"]) for c in p["tool_calls"]]
    out = narrate(ctx, "Which country has the most customers?", p, results, None)
    answer = out["answer"]

    # 1) the caption's min/max equal an independent pandas groupby-count
    lows = re.findall(r"lowest at '[^']+' \(([^)]+)\)", answer)
    highs = re.findall(r"highest at '[^']+' \(([^)]+)\)", answer)
    assert lows and float(lows[0]) == true_min, f"caption lowest {lows!r} != {true_min}"
    assert highs and float(highs[0]) == true_max, f"caption highest {highs!r} != {true_max}"

    # 2) the top-5 tie at 9 is surfaced and the full tied group is shown
    assert "tied" in answer.lower()
    assert "5 countries are tied" in answer
    for country in ("China", "Bangladesh", "Reunion", "Nigeria", "Luxembourg"):
        assert country in answer, f"tied country {country} missing from displayed top-N"

    # 3) the wrong-min regression: Uganda (8) must NOT be reported as the lowest
    assert "lowest at 'Uganda'" not in answer


def test_count_rank_bottom_reports_true_min_with_ties():
    from app.engine.tools import run_tool

    df = _customer_country_df()
    true_min = int(df["Country"].value_counts().min())

    reg, rec = _reg_with(df)
    ctx = _ctx(reg, rec)
    p = plan(ctx, "What is the least common country?")
    results = [run_tool(ctx, c["name"], c["args"]) for c in p["tool_calls"]]
    out = narrate(ctx, "What is the least common country?", p, results, None)
    answer = out["answer"]

    lows = re.findall(r"lowest at '[^']+' \(([^)]+)\)", answer)
    assert lows and float(lows[0]) == true_min, f"bottom caption lowest {lows!r} != {true_min}"
    assert "19 countries are tied" in answer


# --- conditions: qualifiers ("in 2021", "where X = Y", "in Japan") must become WHERE ---


def _subscriptions_df() -> pd.DataFrame:
    """A small dataset with an ISO string date column and a country column, so
    year filters and category filters are both checkable."""
    import numpy as np

    rng = np.random.default_rng(7)
    n = 100
    dates = pd.date_range("2020-01-01", "2022-12-31", periods=n)
    countries = rng.choice(["India", "Japan", "China"], size=n)
    return pd.DataFrame(
        {
            "Customer Id": [f"C{i:04d}" for i in range(n)],
            "Subscription Date": dates.strftime("%Y-%m-%d"),
            "Country": countries,
        }
    )


def test_count_condition_year_becomes_where():
    """'How many ... subscribed in 2021?' must count only that year, not the
    whole dataset (regression: it previously returned the full row count)."""
    from app.engine.tools import run_tool

    df = _subscriptions_df()
    reg, rec = _reg_with(df)
    ctx = _ctx(reg, rec)

    true_2021 = int(((df["Subscription Date"] >= "2021-01-01") & (df["Subscription Date"] < "2022-01-01")).sum())

    p = plan(ctx, "How many customers subscribed in 2021?")
    assert p["narrative"] == "count"
    sql = p["tool_calls"][0]["args"]["sql"]
    assert "WHERE" in sql and "2021" in sql, sql
    assert "COUNT(*)" not in sql.upper(), "bare COUNT(*) must not be used when a target column exists"

    results = [run_tool(ctx, c["name"], c["args"]) for c in p["tool_calls"]]
    out = narrate(ctx, "How many customers subscribed in 2021?", p, results, None)
    assert f"**{true_2021}**" in out["answer"], out["answer"]
    assert "2021" in out["answer"]


def test_count_condition_category_becomes_where():
    """'How many customers are in India?' must ground to Country = 'India'."""
    from app.engine.tools import run_tool

    df = _subscriptions_df()
    reg, rec = _reg_with(df)
    ctx = _ctx(reg, rec)
    true_india = int(df["Country"].eq("India").sum())

    p = plan(ctx, "How many customers are in India?")
    assert p["narrative"] == "count"
    sql = p["tool_calls"][0]["args"]["sql"]
    assert "WHERE" in sql and "Country" in sql and "India" in sql, sql

    results = [run_tool(ctx, c["name"], c["args"]) for c in p["tool_calls"]]
    out = narrate(ctx, "How many customers are in India?", p, results, None)
    assert f"**{true_india}**" in out["answer"], out["answer"]


def test_count_condition_explicit_where_column_value():
    from app.engine.tools import run_tool

    df = _subscriptions_df()
    reg, rec = _reg_with(df)
    ctx = _ctx(reg, rec)
    true_japan = int(df["Country"].eq("Japan").sum())

    p = plan(ctx, "How many customers where Country = Japan?")
    assert p["narrative"] == "count", p["narrative"]
    sql = p["tool_calls"][0]["args"]["sql"]
    assert "WHERE" in sql and "Country" in sql and "Japan" in sql, sql

    results = [run_tool(ctx, c["name"], c["args"]) for c in p["tool_calls"]]
    out = narrate(ctx, "How many customers where Country = Japan?", p, results, None)
    assert f"**{true_japan}**" in out["answer"], out["answer"]


def test_count_without_condition_unchanged():
    """A plain count question with no qualifier must not gain a WHERE clause."""
    from app.engine.tools import run_tool

    df = _subscriptions_df()
    reg, rec = _reg_with(df)
    ctx = _ctx(reg, rec)

    p = plan(ctx, "How many customers are in the dataset?")
    assert p["narrative"] == "count"
    sql = p["tool_calls"][0]["args"]["sql"]
    assert "WHERE" not in sql, sql

    results = [run_tool(ctx, c["name"], c["args"]) for c in p["tool_calls"]]
    out = narrate(ctx, "How many customers are in the dataset?", p, results, None)
    assert f"**{len(df)}**" in out["answer"], out["answer"]


def test_generate_sql_injects_condition():
    """The generate_sql tool must carry a stated qualifier into the WHERE clause."""
    from app.engine.tools import run_tool

    df = _subscriptions_df()
    reg, rec = _reg_with(df)
    ctx = _ctx(reg, rec)

    res = run_tool(ctx, "generate_sql", {"dataset_id": rec.dataset_id, "analysis": "how many customers subscribed in 2021?"})
    sql = res["sql"]
    assert "WHERE" in sql and "2021-01-01" in sql, sql

    out = run_tool(ctx, "run_sql", {"sql": sql})
    true_2021 = int(((df["Subscription Date"] >= "2021-01-01") & (df["Subscription Date"] < "2022-01-01")).sum())
    assert out["rows"][0]["count"] == true_2021, out["rows"]


def test_audit_sql_warns_on_dropped_condition():
    """audit_sql must flag SQL that drops a stated condition, and stay silent
    when the question has no condition."""
    from app.engine.conditions import audit_sql

    df = _subscriptions_df()
    reg, rec = _reg_with(df)

    warn = audit_sql("How many customers subscribed in 2021?", rec, 'SELECT COUNT(DISTINCT "Customer Id") AS count FROM "subscriptions"')
    assert warn is not None and "2021" in warn, warn

    assert audit_sql("How many customers subscribed in 2021?", rec, 'SELECT COUNT(*) AS c FROM "subscriptions" WHERE "Subscription Date" >= \'2021-01-01\'') is None

    assert audit_sql("How many customers are in the dataset?", rec, 'SELECT COUNT(*) AS c FROM "subscriptions"') is None


# --- charts: "customers by country" must count rows, never plot a row index -----


def test_chart_plan_customers_by_country_uses_count_agg():
    """'bar chart of customers by country' must aggregate a COUNT per country,
    not pick a numeric row-index column (Index) as the y-axis."""
    df = _customer_country_df()
    reg, rec = _reg_with(df)
    ctx = _ctx(reg, rec)

    p = plan(ctx, "Show a bar chart of customers by country")
    assert p["narrative"] == "chart"
    args = p["tool_calls"][0]["args"]
    assert args["kind"] == "bar"
    assert args["x"] == "Country"
    assert args.get("agg") == "count", args
    assert args.get("y") != "Index", f"must not plot a row index: {args}"


def test_chart_plan_explicit_numeric_metric_keeps_y():
    df = pd.DataFrame(
        {
            "Country": ["India", "India", "Japan", "Japan"],
            "revenue": [10.0, 20.0, 5.0, 7.0],
        }
    )
    reg, rec = _reg_with(df)
    ctx = _ctx(reg, rec)
    p = plan(ctx, "Show a bar chart of revenue by country")
    args = p["tool_calls"][0]["args"]
    assert args["y"] == "revenue" and "agg" not in args or args.get("agg") == "sum", args


def test_narrate_derives_chart_narrative_from_tool_calls():
    """An LLM plan may omit/rename the narrative; a successful generate_chart
    must still be narrated as a chart (image + caption), never as the generic
    'no usable result' text that previously contradicted the rendered chart."""
    from app.engine.tools import run_tool

    df = _customer_country_df()
    reg, rec = _reg_with(df)
    ctx = _ctx(reg, rec)
    for plan_narrative in (None, "chart_request", "generic"):
        plan_dict = {
            "tool_calls": [{"name": "generate_chart", "args": {"dataset_id": rec.dataset_id, "kind": "bar", "x": "Country", "agg": "count"}}],
            "narrative": plan_narrative,
        }
        results = [run_tool(ctx, c["name"], c["args"]) for c in plan_dict["tool_calls"]]
        out = narrate(ctx, "Show a bar chart of customers by country", plan_dict, results, rec)
        answer = out["answer"]
        assert "no usable result" not in answer, answer
        assert "How to read it" in answer
        assert out["chart_ids"], out
        # caption must cite the true max (Liechtenstein=12), never a row-index number
        assert "Liechtenstein" in answer and "12" in answer


# --- duplicate detection: "are there any duplicate customer ids?" ----------


def _with_ctx(reg, rec, history=None):
    ctx = _ctx(reg, rec)
    ctx["history"] = history or []
    return ctx


def test_duplicates_plan_routes_to_find_duplicates():
    df = _customer_country_df()
    reg, rec = _reg_with(df)
    ctx = _with_ctx(reg, rec)
    p = plan(ctx, "Are there any duplicate customer ids?")
    assert p["narrative"] == "duplicates", p["narrative"]
    assert p["tool_calls"][0]["name"] == "find_duplicates", p["tool_calls"]
    assert p["tool_calls"][0]["args"].get("columns") == ["Customer Id"], p["tool_calls"][0]["args"]


def test_duplicates_zero_duplicates_reports_clean_not_all_rows():
    """The 1000-row customers fixture has all-unique Customer Ids. Asking about
    duplicates must answer 'no duplicates' — never return 1000 rows or plot the
    whole dataset (the old .shift() bug treated every row as its own duplicate)."""
    from app.engine.tools import run_tool

    df = _customer_country_df()
    assert df["Customer Id"].is_unique
    reg, rec = _reg_with(df)
    ctx = _with_ctx(reg, rec)
    p = plan(ctx, "Are there any duplicate customer ids?")
    results = [run_tool(ctx, c["name"], c["args"]) for c in p["tool_calls"]]
    assert results[0]["duplicate_count"] == 0, results[0]
    out = narrate(ctx, "Are there any duplicate customer ids?", p, results, rec)
    answer = out["answer"]
    assert "No duplicate rows found" in answer, answer
    assert "unique" in answer.lower(), answer
    assert "1000" not in answer  # must not claim 1000 rows are duplicates


def test_duplicates_with_real_dupes_returns_exactly_those_rows():
    from app.engine.tools import run_tool

    df = pd.DataFrame(
        {
            "Customer Id": ["C001", "C001", "C002", "C003", "C003", "C004"],
            "Country": ["India", "Japan", "China", "India", "India", "Japan"],
        }
    )
    reg, rec = _reg_with(df)
    ctx = _with_ctx(reg, rec)
    p = plan(ctx, "Are there any duplicate customer ids?")
    assert p["tool_calls"][0]["args"]["columns"] == ["Customer Id"]
    results = [run_tool(ctx, c["name"], c["args"]) for c in p["tool_calls"]]
    dup = results[0]
    assert dup["duplicate_count"] == 4, dup  # C001 x2 + C003 x2
    assert dup["group_count"] == 2, dup
    ids = {r["Customer Id"] for r in dup["rows"]}
    assert ids == {"C001", "C003"}, dup
    out = narrate(ctx, "Are there any duplicate customer ids?", p, results, rec)
    assert "**2** duplicated value group(s)" in out["answer"], out["answer"]
    assert "C001" in out["answer"] and "C003" in out["answer"]


def test_duplicates_question_does_not_route_to_quality_or_filter():
    df = _customer_country_df()
    reg, rec = _reg_with(df)
    ctx = _with_ctx(reg, rec)
    p = plan(ctx, "Check for duplicate email addresses in the data")
    assert p["narrative"] == "duplicates", p["narrative"]
    assert p["tool_calls"][0]["name"] == "find_duplicates"
    # "how clean is the data, any duplicates?" stays a quality report
    p2 = plan(ctx, "How clean is the data? any duplicates?")
    assert p2["narrative"] == "quality", p2["narrative"]


# --- anaphoric follow-up: "now show that as a percentage of total" ---------


def test_percentage_followup_reuses_prior_grouping():
    """After 'Show a bar chart of customers by country', asking 'now show that
    as a percentage of total' must reuse the SAME Country grouping and express
    each country as % of total — NOT fall through to the generic profile that
    averaged the row-index column (avg_index=500.5, monthly trend on Index)."""
    from app.engine.tools import run_tool

    df = _customer_country_df()
    reg, rec = _reg_with(df)
    total = len(df)
    ctx = _with_ctx(
        reg,
        rec,
        history=[
            {"role": "user", "content": "Show a bar chart of customers by country", "kind": "question"},
            {"role": "assistant", "content": "Here is the count of customers by country.", "kind": "answer"},
        ],
    )

    p = plan(ctx, "now show that as a percentage of total")
    assert p["narrative"] == "percentage", p["narrative"]
    assert p["dimension"] == "Country", p
    sql = p["tool_calls"][0]["args"]["sql"]
    assert '"Country"' in sql and "COUNT(*)" in sql, sql
    assert "100.0 * COUNT(*) / SUM(COUNT(*)) OVER ()" in sql, sql
    assert "avg_" not in sql and "Index" not in sql, sql

    results = [run_tool(ctx, c["name"], c["args"]) for c in p["tool_calls"]]
    rows = results[0]["rows"]
    # every percentage is count / total * 100 (within rounding), Liechtenstein is max
    by_name = {r["dimension"]: r for r in rows}
    liechtenstein = by_name["Liechtenstein"]
    assert abs(liechtenstein["percentage"] - (12 / total * 100)) <= 0.1, liechtenstein
    assert sum(r["count"] for r in rows) == total
    assert abs(sum(r["percentage"] for r in rows) - 100.0) <= 1.0

    # the chart aggregates per Country with agg='percent'
    chart_args = p["tool_calls"][1]["args"]
    assert chart_args["x"] == "Country" and chart_args["agg"] == "percent", chart_args

    out = narrate(ctx, "now show that as a percentage of total", p, results, rec)
    answer = out["answer"]
    assert "as a percentage of the total" in answer, answer
    assert "Liechtenstein" in answer and "1.2" in answer, answer
    assert "avg_index" not in answer and "Index" not in answer, answer
    assert out["chart_ids"], out
    assert out["narrative"] == "percentage"


def test_percentage_followup_without_history_still_groups_by_dimension():
    """Even with no prior turn in history, an explicit 'of total' percentage
    request must produce a grouped percentage (not the generic avg profile)."""
    from app.engine.tools import run_tool

    df = _customer_country_df()
    reg, rec = _reg_with(df)
    ctx = _with_ctx(reg, rec)
    p = plan(ctx, "Show each country as a percentage of total customers")
    assert p["narrative"] == "percentage", p["narrative"]
    results = [run_tool(ctx, c["name"], c["args"]) for c in p["tool_calls"]]
    assert results[0]["rows"], results[0]
    out = narrate(ctx, "Show each country as a percentage of total customers", p, results, rec)
    assert "as a percentage of the total" in out["answer"]
    assert "avg_index" not in out["answer"]


# --- shared SQL generator: direct answers and "generate SQL for X" must agree ----


def test_generate_sql_for_question_groups_by_country_not_customer_id():
    """The explicit show-SQL path must use the SAME generator as the direct
    answer path: 'Generate SQL for which country has the most customers?' must
    GROUP BY "Country" (never "Customer Id", which made every row_count 1)."""
    from app.engine.fallback import sql_for_question
    from app.engine.tools import run_tool

    df = _customer_country_df()
    reg, rec = _reg_with(df)
    ctx = _ctx(reg, rec)

    direct = plan(ctx, "which country has the most customers?")
    direct_sql = direct["tool_calls"][0]["args"]["sql"]
    assert '"Country"' in direct_sql and '"Customer Id"' not in direct_sql, direct_sql

    # same question, wrapped in "Generate SQL for ..." framing
    generated = sql_for_question(rec, "Generate SQL for which country has the most customers?")
    assert generated == direct_sql, f"\ngenerate-sql: {generated}\ndirect:       {direct_sql}"

    # the tool itself must produce and run it, returning Liechtenstein=12
    out = run_tool(ctx, "generate_sql", {"dataset_id": rec.dataset_id, "analysis": "Generate SQL for which country has the most customers?"})
    assert '"Country"' in out["sql"] and '"Customer Id"' not in out["sql"], out["sql"]
    by_country = {r["dimension"]: r["row_count"] for r in out["rows"]}
    assert by_country.get("Liechtenstein") == 12, out
    assert "Customer Id" not in str(by_country), out


def test_generate_sql_framing_stripped_for_other_intents():
    """Framing words ('Generate SQL for ...') must not leak into intent
    detection — e.g. they must not flip a count into a grouping or a trend into
    a generic profile."""
    from app.engine.fallback import sql_for_question

    df = _subscriptions_df()
    reg, rec = _reg_with(df)

    assert sql_for_question(rec, "How many customers are in the dataset?").lstrip().upper().startswith("SELECT COUNT")
    assert sql_for_question(rec, "Generate SQL for how many customers are in the dataset?").lstrip().upper().startswith("SELECT COUNT")
    sql = sql_for_question(rec, "Show me the distribution of customers by country")
    assert '"Country"' in sql and "COUNT(*)" in sql, sql
