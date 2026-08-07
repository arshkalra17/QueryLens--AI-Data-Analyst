"""Tests for the tool registry and execution."""
from __future__ import annotations

from app.core.registry import DataFrameRegistry
from app.core.sessions import SessionManager
from app.engine.charts import get_chart_store
from app.engine.tools import ALL_TOOLS, TOOL_MAP, run_tool


def _ctx(simple_df):
    reg = DataFrameRegistry()
    rec = reg.register(simple_df, "cities.csv", "test")
    return {
        "registry": reg,
        "sessions": SessionManager(),
        "session_id": None,
        "pinned": rec.dataset_id,
        "dataset_ids": [rec.dataset_id],
        "settings": type("S", (), {"allow_pandas_exec": True})(),
        "chart_store": get_chart_store(),
        "metrics": [],
    }, rec


def test_tool_registry_complete():
    assert len(ALL_TOOLS) == 14
    for tool in ALL_TOOLS:
        assert tool.name in TOOL_MAP
        assert tool.description
        assert tool.parameters


def test_pick_record_prefers_pinned_dataset(simple_df, sales_df):
    from app.core.sessions import SessionManager
    from app.engine.tools import _pick_record

    reg = DataFrameRegistry()
    rec_old = reg.register(simple_df, "cities.csv", "test")
    rec_new = reg.register(sales_df, "sales.csv", "test")
    sessions = SessionManager()
    sess = sessions.create([rec_old.dataset_id, rec_new.dataset_id])
    sess.pinned = rec_new.dataset_id
    ctx = {
        "registry": reg,
        "sessions": sessions,
        "session_id": sess.session_id,
        "settings": type("S", (), {"allow_pandas_exec": True})(),
        "chart_store": get_chart_store(),
        "metrics": [],
    }
    picked = _pick_record(ctx, None, sess.session_id)
    assert picked.dataset_id == rec_new.dataset_id


def test_list_datasets(simple_df):
    ctx, rec = _ctx(simple_df)
    out = run_tool(ctx, "list_datasets", {})
    assert out["message"]
    ids = [d["dataset_id"] for d in out["datasets"]]
    assert rec.dataset_id in ids


def test_get_schema(simple_df):
    ctx, rec = _ctx(simple_df)
    out = run_tool(ctx, "get_schema", {"dataset_id": rec.dataset_id})
    assert out["schema"]["name"] == "cities.csv"
    assert "city" in out["snippet"]


def test_run_sql(simple_df):
    ctx, rec = _ctx(simple_df)
    out = run_tool(ctx, "run_sql", {"dataset_id": rec.dataset_id, "sql": 'SELECT SUM("value") AS total FROM "cities_csv"'})
    assert out["row_count"] == 1
    assert out["rows"][0]["total"] == 1175.0
    assert "|" in out["markdown_table"]


def test_generate_sql(simple_df):
    ctx, rec = _ctx(simple_df)
    out = run_tool(ctx, "generate_sql", {"dataset_id": rec.dataset_id, "analysis": "top cities by value"})
    assert out["sql"]
    assert "value" in out["sql"]


def test_run_pandas_guarded(simple_df):
    ctx, rec = _ctx(simple_df)
    out = run_tool(ctx, "run_pandas", {"dataset_id": rec.dataset_id, "code": 'result = df["value"].mean()'})
    assert out["type"] in ("float", "float64")
    assert abs(out["result"] - 195.8333) < 0.01


def test_run_pandas_blocks_dangerous_code(simple_df):
    from app.engine.tools import ToolError

    ctx, rec = _ctx(simple_df)
    try:
        run_tool(ctx, "run_pandas", {"dataset_id": rec.dataset_id, "code": '__import__("os").system("echo hi")'})
        raised = False
    except ToolError:
        raised = True
    assert raised


def test_search_data(simple_df):
    ctx, rec = _ctx(simple_df)
    out = run_tool(ctx, "search_data", {"dataset_id": rec.dataset_id, "query": "Mumbai"})
    assert out["total_matches"] > 0


def test_filter_rows_equal_case_insensitive(simple_df):
    ctx, rec = _ctx(simple_df)
    out = run_tool(ctx, "filter_rows", {"dataset_id": rec.dataset_id, "column": "city", "value": "mumbai"})
    assert out["column"] == "city"
    assert out["matched"] == 3
    assert all(r["city"] == "Mumbai" for r in out["rows"])


def test_filter_rows_numeric_operator(simple_df):
    ctx, rec = _ctx(simple_df)
    out = run_tool(ctx, "filter_rows", {"dataset_id": rec.dataset_id, "column": "value", "value": "200", "operator": ">="})
    assert out["matched"] == 3


def test_filter_rows_contains(simple_df):
    ctx, rec = _ctx(simple_df)
    out = run_tool(ctx, "filter_rows", {"dataset_id": rec.dataset_id, "column": "name", "value": "acme", "operator": "contains"})
    assert out["matched"] == 3


def test_filter_rows_unknown_column_errors(simple_df):
    from app.engine.tools import ToolError

    ctx, rec = _ctx(simple_df)
    try:
        run_tool(ctx, "filter_rows", {"dataset_id": rec.dataset_id, "column": "nope", "value": "x"})
        raised = False
    except ToolError:
        raised = True
    assert raised


def test_filter_rows_missing_args_errors(simple_df):
    from app.engine.tools import ToolError

    ctx, rec = _ctx(simple_df)
    try:
        run_tool(ctx, "filter_rows", {"dataset_id": rec.dataset_id, "column": "city"})
        raised = False
    except ToolError:
        raised = True
    assert raised


def test_filter_rows_accepts_condition_string(simple_df):
    ctx, rec = _ctx(simple_df)
    out = run_tool(ctx, "filter_rows", {"dataset_id": rec.dataset_id, "condition": "city = 'Mumbai'"})
    assert out["column"] == "city"
    assert out["matched"] == 3
    assert all(r["city"] == "Mumbai" for r in out["rows"])


def test_filter_rows_condition_numeric_operator(simple_df):
    ctx, rec = _ctx(simple_df)
    out = run_tool(ctx, "filter_rows", {"dataset_id": rec.dataset_id, "condition": "value >= 200"})
    assert out["column"] == "value"
    assert out["operator"] == ">="
    assert out["matched"] == 3


def test_detect_anomalies_tool(simple_df):
    ctx, rec = _ctx(simple_df)
    out = run_tool(ctx, "detect_anomalies", {"dataset_id": rec.dataset_id, "columns": ["value"]})
    assert "by_column" in out


def test_unknown_tool(simple_df):
    from app.engine.tools import ToolError

    ctx, rec = _ctx(simple_df)
    try:
        run_tool(ctx, "nope", {})
        raised = False
    except ToolError:
        raised = True
    assert raised
