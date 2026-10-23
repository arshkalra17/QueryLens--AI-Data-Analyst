"""Rule-based analyst engine used when no LLM is configured.

It reuses the same tool layer as the LLM agent, so the app is fully functional
without any API key. Intent detection covers the assignment's example questions:
highest/lowest revenue, top N, trends, underperformers, SQL generation, anomaly
detection, insights, quality, forecasting, charts and summaries.
"""
from __future__ import annotations

import logging
import re
from typing import Any

_KNOWN_NARRATIVES = {
    "count", "schema", "chart", "trend_chart", "dashboard", "anomaly", "quality",
    "forecast", "top", "bottom", "preview", "filter", "duplicates", "percentage", "sql", "sql_anomaly",
    "pandas", "correlation", "compare", "generic", "error", "aggregate",
}

import pandas as pd

from ..utils import guess_date_column, is_string_series
from . import conditions
from .tools import (
    TOOL_MAP,
    ToolError,
    _pick_record,
    _safe_name,
    run_tool,
)

logger = logging.getLogger(__name__)

TOP_MARKERS = ["top", "highest", "best", "max", "largest", "most"]
BOTTOM_MARKERS = ["bottom", "lowest", "worst", "min", "least", "fewest", "underperform", "poor"]
TREND_MARKERS = ["trend", "monthly", "weekly", "daily", "over time", "timeseries", "time series", "seasonality", "forecast", "predict", "growth"]
CHART_MARKERS = ["chart", "plot", "graph", "visuali", "bar", "pie", "scatter", "histogram", "heatmap", "dashboard"]
ANOMALY_MARKERS = ["anomal", "outlier", "spike", "suspicious", "unusual", "abnormal", "exception", "flag"]
INSIGHT_MARKERS = ["insight", "summary", "summarize", "summarise", "overview", "describe", "understand"]
QUALITY_MARKERS = ["quality", "clean", "dirty", "missing", "duplicate", "consistent", "valid"]
DUPLICATE_MARKERS = ["duplicate", "duplicates", "repeated", "repeating", "occurs more than once", "occurring more than once", "appears more than once", "appearing more than once", "more than one row", "same value twice", "have the same"]
SQL_MARKERS = ["sql", "query", "select"]
PANDAS_MARKERS = ["pandas", "python code"]
SEARCH_MARKERS = ["search", "find row", "look up"]
COMPARE_MARKERS = ["compare", "versus", " vs ", "difference"]
CORRELATION_MARKERS = ["correlation", "correlat", "relationship", "related"]
COUNT_MARKERS = ["how many", "count of", "count the", "count all", "count rows", "number of", "total number of", "count how", "total rows"]
FILTER_MARKERS = ["filter", "where", "rows where", "filtered"]

# Nouns where "how many <X>" means distinct row entities -> count its id column
# (e.g. "how many customers" -> COUNT(DISTINCT customer_id)).
_ROW_ENTITY_WORDS = {
    "person", "people", "user", "customer", "client", "member", "employee", "vendor",
    "order", "transaction", "product", "item", "sale", "visit", "visitor", "attendee",
    "student", "patient", "driver", "account", "campaign", "lead", "shipment", "invoice",
}
# Generic primary-key column names that identify the row itself.
_ROW_KEY_NAMES = {"id", "uuid", "guid", "key", "rowid", "recordid"}


def plan(ctx: dict, question: str) -> dict:
    """Return an intent plan with tool calls + the kind of narrative to write."""
    question = question.strip()
    low = question.lower()

    # --- dataset selection hints -------------------------------------------------
    registry = ctx["registry"]
    session_id = ctx.get("session_id")
    target = _resolve_dataset(low, ctx)
    if target:
        ctx["pinned"] = target

    pinned = ctx.get("pinned")
    if pinned and registry.get(pinned) is None:
        pinned = None
    record = _pick_record(ctx, pinned, session_id)

    # --- SQL generation ----------------------------------------------------------
    if _has_any(low, SQL_MARKERS):
        calls = [{"name": "generate_sql", "args": {"analysis": question, "dataset_id": record.dataset_id}}]
        narrative = "sql"
        if _has_any(low, ANOMALY_MARKERS):
            calls.append({"name": "detect_anomalies", "args": {"dataset_id": record.dataset_id, "method": "auto"}})
            narrative = "sql_anomaly"
        return {"tool_calls": calls, "narrative": narrative}

    if _has_any(low, PANDAS_MARKERS):
        code = _draft_pandas(record, low)
        calls = [{"name": "run_pandas", "args": {"code": code}}]
        return {"tool_calls": calls, "narrative": "pandas", "pandas": code}

    # --- filter / list rows -------------------------------------------------------
    if _filter_intent(low):
        f = _parse_filter(low, record)
        if f:
            return {
                "tool_calls": [{"name": "filter_rows", "args": {"dataset_id": record.dataset_id, "column": f["column"], "value": f["value"], "operator": f["operator"], "limit": _extract_number(low, default=50)}}],
                "narrative": "filter",
                "filter": f,
            }

    # --- show / preview raw rows -----------------------------------------------
    preview_limit = _preview_limit(low)
    if preview_limit is not None:
        return {"tool_calls": [{"name": "preview_rows", "args": {"dataset_id": record.dataset_id, "limit": preview_limit}}], "narrative": "preview"}

    # --- count / how many -------------------------------------------------------
    if _count_intent(low):
        entity = _count_entity(low) or "rows"
        target = _count_target(record, entity)
        # shared generator: also translates explicit qualifiers ("in 2021",
        # "where X = Y", "in Japan") into a WHERE clause so "how many customers
        # subscribed in 2021?" does not silently count the whole dataset.
        sql = sql_for_question(record, question)
        where = conditions.extract_where(question, record)
        measure = target if target and _is_measure(record, target) else None
        return {"tool_calls": [{"name": "run_sql", "args": {"sql": sql}}], "narrative": "count", "count_entity": entity, "count_target": target, "count_measure": measure, "count_where": where}

    # --- schema / columns ---------------------------------------------------------
    if _schema_intent(low):
        return {"tool_calls": [{"name": "get_schema", "args": {"dataset_id": record.dataset_id}}], "narrative": "schema"}

    # --- percentage of the previous result (anaphoric follow-up) -----------------
    # "now show that as a percentage of total" after "customers by country" must
    # reuse the SAME dimension grouping from the prior turn, not fall through to
    # the generic profile (avg of the row-index column).
    if _percentage_followup(low, ctx):
        prev_q, _ = _followup_context(ctx)
        dim = _pick_dimension((prev_q or question).lower(), record) or _pick_dimension(low, record)
        if dim:
            table = _safe_name(record.name)
            sql = (
                f'SELECT "{dim}" AS dimension, COUNT(*) AS count, '
                f'ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS percentage '
                f'FROM "{table}" GROUP BY "{dim}" ORDER BY count DESC'
            )
            calls = [
                {"name": "run_sql", "args": {"sql": sql}},
                {"name": "generate_chart", "args": {"dataset_id": record.dataset_id, "kind": "bar", "x": dim, "agg": "percent", "title": f"{dim} share of total (%)"}},
            ]
            return {"tool_calls": calls, "narrative": "percentage", "dimension": dim, "followup_question": prev_q}

    # --- charts / dashboard ------------------------------------------------------
    chart_kind, chart_args = _detect_chart(low)
    if chart_kind:
        dim = _pick_dimension(low, record)
        metric = _pick_metric(low, record)
        if chart_kind == "line" and _guess_date(record.df):
            chart_args = {**chart_args, "x": _guess_date(record.df)}
        elif dim:
            chart_args = {**chart_args, "x": dim}
        if metric:
            chart_args = {**chart_args, "y": metric}
        else:
            # no usable numeric metric (e.g. "chart of customers by country"):
            # plot a count per dimension instead of a meaningless row index.
            if chart_kind in ("bar", "barh", "pie", "line", "area") and dim:
                chart_args = {**chart_args, "agg": "count"}
        calls = [{"name": "generate_chart", "args": {"dataset_id": record.dataset_id, "kind": chart_kind, **chart_args}}]
        narrative = "chart"
        if _has_any(low, TREND_MARKERS):
            narrative = "trend_chart"
        return {"tool_calls": calls, "narrative": narrative}

    if "dashboard" in low:
        calls = [
            {"name": "generate_insights", "args": {"dataset_id": record.dataset_id, "limit": 5}},
            {"name": "data_quality", "args": {"dataset_id": record.dataset_id}},
        ]
        return {"tool_calls": calls, "narrative": "dashboard"}

    # --- duplicate detection ------------------------------------------------------
    if _duplicate_intent(low):
        dup_col = _duplicate_column(low, record)
        args: dict = {"dataset_id": record.dataset_id}
        if dup_col:
            args["columns"] = [dup_col]
        return {"tool_calls": [{"name": "find_duplicates", "args": args}], "narrative": "duplicates", "dup_column": dup_col}

    # --- anomaly detection ---------------------------------------------------------
    if _has_any(low, ANOMALY_MARKERS):
        method = "rolling" if _has_any(low, TREND_MARKERS) or _guess_date(record.df) else "zscore"
        args: dict = {"dataset_id": record.dataset_id, "method": method}
        metric = _pick_metric(low, record)
        if any(re.findall(r"[a-z]+", m.lower()) for m in [metric]) and metric and _metric_in_question(low, record):
            args["columns"] = [metric]
        calls = [{"name": "detect_anomalies", "args": args}]
        return {"tool_calls": calls, "narrative": "anomaly"}

    # --- quality --------------------------------------------------------------------
    if _has_any(low, QUALITY_MARKERS):
        return {"tool_calls": [{"name": "data_quality", "args": {"dataset_id": record.dataset_id}}], "narrative": "quality"}

    # --- forecast -------------------------------------------------------------------
    if _has_any(low, TREND_MARKERS) or "forecast" in low:
        date_col = _guess_date(record.df)
        num_col = _pick_metric(low, record) or _first_numeric(record.df)
        if date_col and num_col:
            periods = _extract_number(low, default=12)
            return {
                "tool_calls": [{"name": "forecast", "args": {"dataset_id": record.dataset_id, "date_column": date_col, "value_column": num_col, "periods": periods}}],
                "narrative": "forecast",
            }

    # --- insights / summary ----------------------------------------------------------
    if _has_any(low, INSIGHT_MARKERS) or not _has_any(low, COMPARE_MARKERS + CORRELATION_MARKERS + TOP_MARKERS + BOTTOM_MARKERS):
        if "insight" in low or "summar" in low or "overview" in low or "describe" in low:
            return {"tool_calls": [{"name": "generate_insights", "args": {"dataset_id": record.dataset_id, "limit": 8}}], "narrative": "insights"}

    # --- correlation -------------------------------------------------------------------
    if _has_any(low, CORRELATION_MARKERS):
        return {"tool_calls": [{"name": "generate_chart", "args": {"dataset_id": record.dataset_id, "kind": "heatmap"}}], "narrative": "correlation"}

    # --- compare ------------------------------------------------------------------------
    if _has_any(low, COMPARE_MARKERS):
        return {"tool_calls": [{"name": "generate_insights", "args": {"dataset_id": record.dataset_id, "limit": 5}}], "narrative": "compare"}

    # --- single extreme value ("what is the highest revenue value?") --------------------
    # A max/min over a metric with no grouping dimension is one number, not a
    # ranking of groups; route it before the top/bottom branch claims "highest".
    extreme = _extreme_intent(low, record)
    if extreme:
        return {"tool_calls": [{"name": "run_sql", "args": {"sql": _aggregate_sql(record, question, extreme)}}], "narrative": "aggregate", "aggregate": extreme}

    # --- top / bottom N ---------------------------------------------------------------
    if _has_any(low, TOP_MARKERS + BOTTOM_MARKERS):
        n = _extract_number(low, default=5)
        dimension = _pick_dimension(low, record)
        ascending = "false"
        narrative = "top"
        if _has_any(low, BOTTOM_MARKERS):
            ascending = "true"
            narrative = "bottom"
        # "which country has the most customers" / "most common country" rank by
        # COUNT(*) per group; only SUM a numeric column when the question names one.
        if dimension and _count_rank_intent(low, record):
            chart_y = "row_count"
            count_rank = True
        else:
            metric = _pick_metric(low, record)
            chart_y = metric
            count_rank = False
        # shared generator produces the same SQL used by the "generate SQL for X"
        # tool, so a fix here (e.g. GROUP BY "Country", never "Customer Id") holds
        # in both paths.
        sql = sql_for_question(record, question)
        calls = [
            {"name": "run_sql", "args": {"sql": sql}},
            {"name": "generate_chart", "args": {"dataset_id": record.dataset_id, "kind": "bar", "x": dimension, "y": chart_y, "agg": "count" if chart_y == "row_count" else "sum", "title": f"{'Bottom' if narrative=='bottom' else 'Top'} {n} by {chart_y}"}},
        ]
        return {"tool_calls": calls, "narrative": narrative, "n": n, "dimension": dimension, "count_rank": count_rank, "unit": _rank_unit(low, count_rank, chart_y)}

    # --- explicit aggregate ("total revenue by region", "average rating") --------------
    agg = _aggregate_intent(low, record)
    if agg:
        return {"tool_calls": [{"name": "run_sql", "args": {"sql": _aggregate_sql(record, question, agg)}}], "narrative": "aggregate", "aggregate": agg}

    # --- generic answer: run SQL aggregation + insights -------------------------------
    return {"tool_calls": [{"name": "run_sql", "args": {"sql": _generic_sql(record)}}, {"name": "generate_insights", "args": {"dataset_id": record.dataset_id, "limit": 4}}], "narrative": "generic"}


def narrate(ctx: dict, question: str, plan: dict, results: list[dict], pinned_dataset: dict | None) -> dict:
    """Build the final markdown answer + reasoning from the tool results."""
    narrative = plan.get("narrative")
    # LLM-produced plans often omit the narrative (or label it "generic"). Derive
    # it from the actual tool calls so a successful chart is narrated as a chart
    # (with its image/caption) instead of falling into the generic "no usable
    # result" text that contradicted the rendered chart. Known non-generic
    # narratives (top/bottom/correlation/...) are preserved.
    names = [c.get("name") for c in plan.get("tool_calls", [])]
    if narrative in (None, "", "generic") or narrative not in _KNOWN_NARRATIVES:
        if "generate_chart" in names:
            narrative = "chart"
        elif "detect_anomalies" in names:
            narrative = "anomaly"
        elif "data_quality" in names:
            narrative = "quality"
        elif "forecast" in names:
            narrative = "forecast"
        elif "get_schema" in names:
            narrative = "schema"
        elif "filter_rows" in names:
            narrative = "filter"
        elif "find_duplicates" in names:
            narrative = "duplicates"
        elif "preview_rows" in names:
            narrative = "preview"
        elif "generate_sql" in names:
            narrative = "sql"
        else:
            narrative = narrative or "generic"
    question = question.strip()
    registry = ctx["registry"]
    session_id = ctx.get("session_id")
    record = pinned_dataset
    if record is None:
        pinned = ctx.get("pinned")
        if pinned and registry.get(pinned) is None:
            pinned = None
        record = _pick_record(ctx, pinned, session_id)

    def res(name: str) -> dict | None:
        for call, r in zip(plan["tool_calls"], results):
            if call["name"] == name:
                return r
        return None

    sql_res = res("run_sql") or res("generate_sql")
    chart_res = res("generate_chart")
    anomaly_res = res("detect_anomalies")
    insight_res = res("generate_insights")
    quality_res = res("data_quality")
    forecast_res = res("forecast")
    pandas_res = res("run_pandas")
    heatmap_res = res("generate_chart")

    # Collect tool failures explicitly. These must be surfaced to the user and
    # logged, never silently absorbed by the generic narrative below.
    failures: list[dict] = []
    for call, r in zip(plan.get("tool_calls", []), results):
        if r and r.get("error"):
            args = call.get("args", {}) or {}
            code = args.get("sql") or args.get("code") or ""
            failures.append({"tool": call.get("name", "?"), "error": str(r["error"]), "code": str(code)})
    if failures:
        for f in failures:
            logger.warning("Tool %s failed for question %r: %s", f["tool"], question, f["error"])
        return {
            "answer": _error_answer(failures),
            "reasoning": "One or more tool calls failed; no reliable answer could be produced.",
            "sql": _sql_for_result(sql_res) or (res("generate_sql") or {}).get("sql"),
            "pandas": plan.get("pandas"),
            "chart_ids": [],
            "narrative": "error",
        }

    lines: list[str] = []
    chart_uri = chart_res.get("url") if chart_res else None
    chart_id = chart_res.get("chart_id") if chart_res else None
    chart_caption = chart_res.get("caption") if chart_res else None

    def md_table(columns, rows, max_rows=20):
        if not columns:
            return "_No rows returned._"
        out = ["| " + " | ".join(str(c) for c in columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
        for r in rows[:max_rows]:
            out.append("| " + " | ".join(_fmt_cell(v) for v in r.values()) + " |")
        return "\n".join(out)

    # --- narrative builders --------------------------------------------------------
    if narrative == "count":
        count_res = res("run_sql")
        rows = (count_res or {}).get("rows", []) or []
        entity = plan.get("count_entity") or "rows"
        target = plan.get("count_target")
        measure = plan.get("count_measure")
        count_where = plan.get("count_where")
        if rows:
            val = list(rows[0].values())[0]
            lines.append(f"There are **{_fmt_number(val)}** {entity} in `{record.name}`.")
            if measure:
                scope = f" over rows where `{count_where}`" if count_where else ""
                lines.append(f"\n> **Reasoning:** `{measure}` already holds a count per row, so I summed it{scope} (`SUM(\"{measure}\")`).")
            elif count_where:
                lines.append(f"\n> **Reasoning:** I counted rows where `{count_where}` — the number reflects only the subset you asked about.")
            elif target:
                lines.append(f"\n> **Reasoning:** I ran `COUNT(DISTINCT \"{target}\")` over the dataset, so the number reflects distinct `{target}` values.")
            else:
                lines.append("\n> **Reasoning:** I ran `COUNT(*)` over the dataset to count its rows directly.")
        else:
            lines.append(f"Could not determine the count from `{record.name}`.")
    elif narrative == "preview":
        preview_res = res("preview_rows")
        if preview_res and preview_res.get("rows"):
            lines.append(f"Here are the first **{preview_res.get('row_count')}** row(s) of **{preview_res.get('dataset', record.name)}**:\n")
            lines.append(md_table(preview_res.get("columns", []), preview_res.get("rows", []), 25))
        else:
            lines.append(f"No rows could be previewed from `{record.name}`.")
        lines.append("\n> **Reasoning:** these are the raw rows read directly from the dataset, in file order, with no aggregation or transformation.")
    elif narrative == "filter":
        filter_res = res("filter_rows")
        f = plan.get("filter", {})
        col = f.get("column") or (filter_res or {}).get("column")
        val = f.get("value") or (filter_res or {}).get("value")
        rows = (filter_res or {}).get("rows", []) or []
        matched = (filter_res or {}).get("matched")
        if rows:
            lines.append(f"Here are the rows in **{record.name}** where **{col}** matches **{val}**:\n")
            lines.append(md_table((filter_res or {}).get("columns") or [], rows, 25))
            if matched is not None and matched > len(rows):
                lines.append(f"\nShowing the first {len(rows)} of **{matched}** matching rows.")
        else:
            lines.append(f"No rows in `{record.name}` match **{col} = {val}**.")
        lines.append("\n> **Reasoning:** I filtered the dataframe on the matching column and returned the raw rows where the value matches (case-insensitive).")
    elif narrative == "duplicates":
        dup_res = res("find_duplicates")
        dup_cols = (dup_res or {}).get("columns") or [plan.get("dup_column")] or []
        col_label = ", ".join(str(c) for c in dup_cols) or "all columns"
        total = (dup_res or {}).get("duplicate_count")
        groups = (dup_res or {}).get("group_count")
        rows = (dup_res or {}).get("rows", []) or []
        if total:
            lines.append(f"Found **{groups}** duplicated value group(s) totaling **{total}** row(s) where **{col_label}** appears more than once:\n")
            lines.append(md_table((dup_res or {}).get("columns_out") or [], rows, 25))
            if (dup_res or {}).get("row_count", 0) < total:
                lines.append(f"\nShowing the first {(dup_res or {}).get('row_count', 0)} of **{total}** duplicate rows.")
        else:
            lines.append(f"No duplicate rows found in `{record.name}`: every value of **{col_label}** is unique.")
        lines.append("\n> **Reasoning:** I grouped the data by the checked column(s) and kept only groups with a count greater than 1 (exact match), equivalent to `GROUP BY ... HAVING COUNT(*) > 1`.")
    elif narrative == "percentage":
        dim = plan.get("dimension") or ""
        rows = (sql_res or {}).get("rows", []) or []
        if rows:
            total = sum(r.get("count", 0) or 0 for r in rows) or len(record.df)
            lines.append(f"Here is **{dim}** as a percentage of the total ({total} rows):\n")
            lines.append(md_table((sql_res or {}).get("columns") or [], rows, 25))
            if chart_res and chart_res.get("chart_id"):
                lines.append(f"\nChart rendered below: **{dim} share of total (%)**.")
        else:
            lines.append(f"I could not compute a percentage breakdown by **{dim}**.")
        lines.append("\n> **Reasoning:** this follow-up reuses the grouping from your previous question; each group's row count is divided by the total row count (`COUNT(*) / SUM(COUNT(*)) OVER ()`) and expressed as a percentage.")
    elif narrative == "schema":
        schema_res = res("get_schema")
        meta = (schema_res or {}).get("schema") or {}
        cols = meta.get("columns") or []
        if cols:
            names = ", ".join(str(c["name"]) for c in cols)
            lines.append(f"**{meta.get('name', record.name)}** has {len(cols)} column(s):\n")
            lines.append(f"`{names}`\n")
            rows_out = [
                {"column": c["name"], "type": c.get("dtype", ""), "nulls": c.get("null_count", 0), "unique": c.get("unique_count", "")}
                for c in cols
            ]
            lines.append(md_table(["column", "type", "nulls", "unique"], rows_out, 60))
        else:
            names = ", ".join(str(c) for c in record.df.columns)
            lines.append(f"Column names in `{record.name}`: {names}")
        lines.append("\n> **Reasoning:** read directly from the dataset schema (column names, types, null and unique counts) with no data transformation.")
    elif narrative == "sql":
        sql = (res("generate_sql") or {}).get("sql") or _sql_for_result(sql_res)
        lines.append(f"**SQL for your analysis**\n\n```sql\n{sql}\n```")
        if sql_res and sql_res.get("rows"):
            lines.append("\n**Result:**\n\n" + md_table(sql_res.get("columns", []), sql_res.get("rows", [])))
        lines.append(
            "\n> **Why this SQL:** " +
            _explain_sql(question, sql_res)
        )
    elif narrative == "sql_anomaly":
        lines.extend(_anomaly_section(anomaly_res, record))
        sql = (res("generate_sql") or {}).get("sql")
        if sql:
            lines.append("\n\n**SQL used:**\n\n```sql\n" + sql + "```")
    elif narrative == "pandas":
        lines.append("**Pandas code**\n\n```python\n" + plan.get("pandas", "") + "```")
        if pandas_res:
            lines.append("\n**Output:**\n\n```python\n" + str(pandas_res)[:1200] + "```")
        lines.append("\n> **Reasoning:** the snippet uses the pinned dataframe `df` and assigns the result to `result`.")
    elif narrative in ("chart", "trend_chart"):
        if chart_uri:
            lines.append(f"![{chart_caption or 'chart'}]({chart_uri})")
        if narrative == "trend_chart":
            lines.append(f"\n**Trend analysis:** {_trend_note(record)}")
        if chart_caption:
            lines.append(f"\n{chart_caption}")
        lines.append("\n> **How to read it:** " + _chart_howto(chart_res, record))
    elif narrative == "dashboard":
        lines.append("I prepared a quick dashboard view. Here are the highlights:")
        if insight_res:
            for ins in insight_res.get("insights", [])[:5]:
                lines.append(f"- **{ins['title']}**: {ins['detail']}")
        if quality_res:
            lines.append(f"\n**Data quality score: {quality_res.get('score')}/100**")
            for issue in quality_res.get("issues", [])[:3]:
                lines.append(f"- ⚠ {issue['message']}")
    elif narrative == "anomaly":
        anomaly_cols = None
        for call in plan["tool_calls"]:
            if call.get("name") == "detect_anomalies":
                anomaly_cols = call.get("args", {}).get("columns")
                break
        lines.extend(_anomaly_section(anomaly_res, record, columns=anomaly_cols))
    elif narrative == "quality":
        lines.append(f"**Data quality score: {quality_res.get('score')}/100**\n")
        if quality_res and quality_res.get("issues"):
            for issue in quality_res["issues"]:
                icon = "🔴" if issue["severity"] == "high" else ("🟠" if issue["severity"] == "medium" else "🟡")
                lines.append(f"- {icon} **{issue['kind']}**: {issue['message']}")
        lines.append("\n> **Recommendations:** fix high-severity issues first (missing values, duplicates); re-upload a cleaned file to see the score improve.")
    elif narrative == "forecast":
        if forecast_res:
            lines.append(f"**Forecast summary**\n\n{forecast_res.get('summary', '')}")
            lines.append("\n**Forecast table (next periods):**\n\n" + md_table(["date", "type", "value", "lower", "upper"], forecast_res.get("table", [])[-forecast_res.get("periods", 12) :]))
            if forecast_res.get("chart_url"):
                lines.append(f"\n![forecast]({forecast_res['chart_url']})")
            model = forecast_res.get("model", {})
            lines.append(
                f"\n> **Why this forecast:** linear trend (slope {model.get('slope')}) plus {forecast_res.get('seasonality')} seasonality; "
                f"fit quality MAE={model.get('mae')}, RMSE={model.get('rmse')}."
            )
    elif narrative in ("top", "bottom"):
        label = "Top" if narrative == "top" else "Bottom"
        lines.append(f"**{label} performers**\n")
        rows = (sql_res or {}).get("rows", []) or []
        columns = (sql_res or {}).get("columns", []) or []
        if rows:
            n = int(plan.get("n") or 5)
            display = rows
            note = ""
            value_col = _value_col(columns)
            if value_col and len(rows) > n:
                val = lambda r: _num(r.get(value_col))
                boundary = val(rows[n - 1])
                if boundary is not None:
                    tied = [r for r in rows[n:] if val(r) == boundary]
                    if tied:
                        total_tied = sum(1 for r in rows if val(r) == boundary)
                        display = rows[:n] + tied
                        noun = _plural(_rank_noun(question.lower(), plan.get("dimension")))
                        unit = plan.get("unit") or _rank_unit(question.lower(), plan.get("count_rank", False), value_col)
                        b = str(int(boundary)) if boundary == int(boundary) else f"{boundary:g}"
                        unit = _singular(unit) if b == "1" else unit
                        note = f"\n\n> **Note:** {total_tied} {noun} are tied at **{b} {unit}**; showing the full tied group."
            lines.append(md_table(columns, display))
            if note:
                lines.append(note)
        if chart_uri:
            lines.append(f"\n![{chart_caption or 'chart'}]({chart_uri})")
        if sql_res and sql_res.get("rows"):
            lines.append("\n> **Reasoning:** I aggregated by the relevant dimension and sorted by the metric; the table and chart above show the " + label.lower() + " values.")
    elif narrative == "aggregate":
        agg = plan.get("aggregate") or {}
        rows = (sql_res or {}).get("rows", []) or []
        columns = (sql_res or {}).get("columns", []) or []
        label = f"{agg.get('label', 'value')} {agg.get('metric', '')}".strip()
        where = agg.get("where")
        scope = f" where `{where}`" if where else ""
        if not rows:
            lines.append(f"No rows in `{record.name}` matched{scope}.")
        elif agg.get("dimension"):
            lines.append(f"**{label.capitalize()} by {agg['dimension']}** in `{record.name}`{scope}:\n")
            lines.append(md_table(columns, rows, 25))
            lines.append(
                f"\n> **Reasoning:** I grouped rows by `{agg['dimension']}` and computed "
                f"`{agg.get('func')}(\"{agg.get('metric')}\")` per group, sorted from highest to lowest."
            )
        else:
            val = list(rows[0].values())[0]
            lines.append(f"The **{label}** in `{record.name}`{scope} is **{_fmt_number(val)}**.")
            lines.append(f"\n> **Reasoning:** I ran `{agg.get('func')}(\"{agg.get('metric')}\")` over the matching rows.")
    elif narrative == "correlation":
        if heatmap_res and heatmap_res.get("url"):
            lines.append(f"![correlation heatmap]({heatmap_res['url']})")
        lines.append("\n> **Reasoning:** a correlation heatmap shows how strongly numeric columns move together (r near +1 strong positive, −1 strong negative, 0 none).")
    elif narrative == "compare":
        lines.append("**Comparison**\n")
        if insight_res:
            for ins in insight_res.get("insights", []):
                lines.append(f"- {ins['detail']}")
    else:  # generic
        lines.append(f"Here's what I found in **{record.name}**.\n")
        if sql_res and sql_res.get("rows"):
            lines.append(md_table(sql_res.get("columns", []), sql_res.get("rows", []), 15))
        if insight_res:
            lines.append("\n**Key insights:**")
            for ins in insight_res.get("insights", []):
                lines.append(f"- **{ins['title']}**: {ins['detail']}")
        if sql_res and sql_res.get("rows"):
            lines.append(f"\n> **Reasoning:** I ran the query below against `{record.name}`; the table shows its result.")
        elif insight_res:
            lines.append(f"\n> **Reasoning:** I generated these insights from the columns of `{record.name}`.")
        else:
            logger.warning("Generic narrative produced no output for question %r", question)
            lines.append("\n> I couldn't produce an analysis for that question. The query returned no usable result.")

    answer = "\n".join(lines)
    return {
        "answer": answer,
        "reasoning": _reasoning_text(plan, results, record),
        "sql": _sql_for_result(sql_res) or (res("generate_sql") or {}).get("sql"),
        "pandas": plan.get("pandas"),
        "chart_ids": [c["chart_id"] for c in [chart_res] if c and c.get("chart_id")] or ([forecast_res["chart_id"]] if forecast_res and forecast_res.get("chart_id") else []),
        "narrative": narrative,
    }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _has_any(text: str, markers) -> bool:
    return any(m in text for m in markers)


def _metric_in_question(low: str, record) -> bool:
    """True when a specific numeric column name is mentioned in the question."""
    for c in record.columns:
        if c.numeric:
            for token in re.findall(r"[a-z]+", c.name.lower()):
                if len(token) > 3 and token in low:
                    return True
    return False


def _extract_number(low: str, default: int) -> int:
    m = re.search(r"(\d{1,3})", low)
    return min(50, max(1, int(m.group(1)))) if m else default


def _preview_limit(low: str) -> int | None:
    """Return the number of rows requested when the question asks to *show*
    raw data (e.g. "show the first 10 rows", "preview", "display records").
    Returns None when the question is not a preview request."""
    if any(w in low for w in ("preview", "peek", "raw data", "raw rows", "first rows", "some rows")):
        return _extract_number(low, default=10)
    m = re.search(r"\bfirst\s+(\d{1,3})\s+rows?\b", low)
    if m:
        return min(200, max(1, int(m.group(1))))
    m = re.search(r"\b(?:show|display|list|view|print)\s+(?:me\s+)?(?:the\s+)?(\d{1,3})\s+rows?\b", low)
    if m:
        return min(200, max(1, int(m.group(1))))
    if re.search(r"\b(?:show|display|list|view|print)\b.*\b(?:rows?|records?|samples?|entries?)\b", low):
        return _extract_number(low, default=10)
    return None


_FILTER_STRONGER_MARKERS = (
    TREND_MARKERS
    + CHART_MARKERS
    + ANOMALY_MARKERS
    + QUALITY_MARKERS
    + ["top", "highest", "lowest", "best", "worst", "max", "min",
       "sum", "total", "average", "mean", "aggregate", "correlat",
       "compare", "rank", "sort", "distribution", "share", "breakdown"]
)


def _filter_intent(low: str) -> bool:
    """True for list/filter/show-rows requests (e.g. 'list customers from India').

    Stronger intents (trends, charts, anomalies, top-N, ...) always win, so a
    phrase like "show the monthly trend" is NOT treated as a row filter.
    """
    low = str(low).lower()
    if _has_any(low, _FILTER_STRONGER_MARKERS):
        return False
    # a "how many ..." question is a count, even when it says "where X = Y"
    if _count_intent(low):
        return False
    if _has_any(low, FILTER_MARKERS):
        return True
    if re.search(r"\bwhere\b", low):
        return True
    if re.search(r"\b(?:list|show|display|view|get|print)\b.*\b(?:from|in|for)\b", low):
        return True
    return bool(re.search(r"\bfind\b", low))


def _duplicate_intent(low: str) -> bool:
    """True for duplicate questions like 'are there any duplicate customer ids?'."""
    low = str(low).lower()
    # "how clean is the data? any duplicates?" stays a quality report
    if re.search(r"\b(?:quality|clean(?:liness)?|dirty)\b", low):
        return False
    return _has_any(low, DUPLICATE_MARKERS)


def _duplicate_column(low: str, record) -> str | None:
    """Resolve the column a duplicate question is about, or None (all columns)."""
    low = str(low).lower()
    phrase = None
    m = re.search(r"(?:duplicate[s]?|dup|repeated|repeating)\s+(?:rows?\s+in\s+)?([a-z][a-z0-9 _-]{0,40}?)(?:\s|$|\?|\.)", low)
    if m:
        phrase = m.group(1).strip()
    else:
        m = re.search(r"\brows?\s+in\s+([a-z][a-z0-9 _-]{0,40}?)(?:\s|$|\?|\.)", low)
        if m:
            phrase = m.group(1).strip()
    if not phrase:
        return None
    phrase = re.sub(r"\b(?:column|field|values?|entries?|records?|rows?|key|the|any|all|for|are|of|with|that|in|showing|share|duplicate[s]?)\b.*$", "", phrase)
    phrase = re.sub(r"\s+", " ", phrase).strip()
    if phrase:
        return _match_filter_column(record, phrase)
    return None


# --- conversation-memory / anaphoric follow-up resolution -------------------
# "now show that as a percentage of total" after a "customers by country" turn
# must reuse the previous question's grouping instead of being treated as a
# fully standalone generic analysis (which surfaced avg_index / a trend on the
# row-index column).

_FOLLOWUP_WORDS = ("that", "it", "this", "those", "these", "same", "previous", "prior", "above", "earlier", "again", "earlier")


def _followup_context(ctx: dict) -> tuple[str | None, str | None]:
    """Return the most recent prior (user question, assistant answer) pair."""
    last_user = last_assistant = None
    for m in ctx.get("history") or []:
        if m.get("role") == "user":
            last_user = m.get("content")
        elif m.get("role") == "assistant":
            last_assistant = m.get("content")
    return last_user, last_assistant


def _percentage_followup(low: str, ctx: dict) -> bool:
    """True when the question asks to show the previous result as a percentage.

    Matches "show that as a percentage of total", "express it as a % of total",
    "what share of the total" etc. — i.e. a percentage request that clearly
    builds on the prior turn (anaphoric reference and/or "of total").
    """
    low = str(low).lower()
    if not re.search(r"\b(?:percentage|percent|pct|share|proportion|fraction|%)\b", low):
        return False
    if "of total" in low or "of the total" in low:
        return True
    return _has_any(low, _FOLLOWUP_WORDS)


def _schema_intent(low: str) -> bool:
    """True when asking about a dataset's columns/schema (e.g. 'what columns does this dataset have')."""
    low = str(low).lower()
    if not re.search(r"\b(?:columns?|schema|fields?|attributes?|headers?|column\s*names?|column\s*list)\b", low):
        return False
    if _has_any(low, _FILTER_STRONGER_MARKERS):
        return False
    return bool(
        re.search(r"\b(?:what|which|list|show|display|get|describe|tell|name|all|the)\b", low)
        or low.startswith("columns")
    )


def _parse_filter(low: str, record) -> dict | None:
    """Extract {column, value, operator} from a filter request, or None.

    Handles: "where <col> = <value>", "find <col> <value>", and
    "list/show <thing> from|in <value>" (column inferred from the data).
    """
    low = str(low).lower()
    col, value, op = None, None, "=="

    m = re.search(r"\b(?:where|filter)\s+([a-z_][a-z0-9_ ]*?)\s*(==|!=|<>|>=|<=|>|<|=)\s*['\"]?([^'\"=]+?)['\"]?(?:\s*$|\s+and\b|\s*,)", low)
    if m:
        col, op, value = m.group(1).strip(), m.group(2), m.group(3).strip()
    else:
        m = re.search(r"\bfind\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?([a-z_][a-z0-9_ ]*?)\s+([a-z0-9][a-z0-9_.-]{1,40})\s*$", low)
        if m:
            col, value = m.group(1).strip(), m.group(2).strip()
        else:
            m = re.search(r"\b(?:list|show|display|find|view|get|print)\b.*?\b(?:from|in|for)\s+([a-z0-9][a-z0-9_ .'\"-]{1,50}?)\s*$", low)
            if m:
                value = m.group(1).strip().strip("\"'")
                value = re.sub(r"^(?:the|all|any)\s+", "", value)
                value = re.sub(
                    r"\s+(?:region|state|country|city|province|market|area|zone|district|dataset|table|data|file)\s*$",
                    "",
                    value,
                ).strip()
                m3 = re.search(r"\b(?:list|show|display|find|view|get|print)\b\s*(?:me\s+)?(?:all\s+)?(?:the\s+)?([a-z][a-z0-9_ ]{0,40}?)\s*\b(?:from|in|for)\b", low)
                if m3:
                    col = m3.group(1).strip()

    if not value:
        return None

    if col:
        col = _match_filter_column(record, col)
    if col is None:
        col = _find_value_column(record, value)
    if col is None:
        return None

    return {"column": col, "value": value, "operator": op}


def _match_filter_column(record, name: str | None) -> str | None:
    if not name:
        return None
    want = re.sub(r"[^a-z0-9]", "", name.lower())
    want_sing = want[:-1] if want.endswith("s") else want
    cols = [str(c) for c in record.df.columns]
    for c in cols:
        n = re.sub(r"[^a-z0-9]", "", c.lower())
        if n == want or n == want_sing:
            return c
    for c in cols:
        n = re.sub(r"[^a-z0-9]", "", c.lower())
        if want and (want in n or want_sing in n):
            return c
    return None


def _find_value_column(record, value: str) -> str | None:
    low = str(value).strip().lower()
    exact, sub = [], []
    for c in record.columns:
        s = record.df[c.name]
        if not hasattr(s, "astype"):
            continue
        if pd.api.types.is_numeric_dtype(s):
            try:
                if low in {str(v) for v in s.dropna().unique().tolist()}:
                    exact.append(c)
            except Exception:
                pass
            continue
        vals = s.astype(str).str.lower()
        if (vals == low).any():
            exact.append(c)
        elif is_string_series(s) and vals.str.contains(re.escape(low), regex=True).any():
            sub.append(c)
    pool = exact or sub
    if not pool:
        return None
    # prefer low-cardinality (categorical) columns so "India" matches Country,
    # not a free-text column that happens to contain the word.
    pool.sort(key=lambda c: (0 if c.unique_count else 1, c.unique_count or len(record.df), c.name))
    return pool[0].name


_AGG_OF_COUNT_RE = re.compile(r"\b(?:maximum|minimum|max|min|average|avg|mean|highest|lowest|largest|smallest)\s+number of\b")


def _count_intent(low: str) -> bool:
    """True when the question asks how many / a count (e.g. "how many customers")."""
    if _AGG_OF_COUNT_RE.search(low):
        return False  # "maximum number of sessions" -> MAX(sessions)
    if _has_any(low, COUNT_MARKERS):
        return True
    return bool(re.search(r"\bcount\b", low))


def _count_entity(low: str) -> str | None:
    """Extract the thing being counted, e.g. 'customers' from 'how many customers'."""
    m = re.search(r"(?:how many|number of|total number of|count of|count the|count all|count how|count)\s+([a-z][a-z0-9 _-]{1,30})", low)
    if not m:
        return None
    ent = re.split(r"\s+(?:are|is|in|where|do|does|have|has|were|was|of|for|the|this|that|dataset|table|data|subscribed|enrolled|registered|joined|signed|bought|purchased|listed|made|placed|did|ordered|spent|visited|sold|used|came|happened|had)\b", m.group(1))[0]
    # strip adjectives/quantifiers that carry no column signal ("unique",
    # "distinct", "different", "all") so "how many unique countries" targets
    # the country column rather than leaking into the match.
    ent = re.sub(r"^(?:unique|distinct|different|all|every|each)\s+", "", ent.strip())
    return ent.strip() or None


def _count_target(record, entity: str) -> str | None:
    """Pick an identifier column to count distinctly, or None for a plain COUNT(*)."""
    word = _singular((entity or "").lower())
    if word in ("row", "record", "entry", "line", "case"):
        return None
    ent = re.sub(r"[^a-z0-9]", "", word)
    ent_raw = re.sub(r"[^a-z0-9]", "", (entity or "").lower())
    # 1) the entity names a real column ("countries" -> "Country"). Whole-name
    #    match only: a substring match made "how many sales" count distinct
    #    "sales_rep" values instead of sales rows.
    for c in record.columns:
        cnorm = re.sub(r"[^a-z0-9]", "", c.name.lower())
        if ent and cnorm in (ent, ent_raw):
            return c.name
    # 1b) the entity is the leading word of a quantity column ("how many units
    #     were sold" -> "units_sold"), which is summed rather than counted
    for c in record.columns:
        first = (re.findall(r"[a-z]+", c.name.lower()) or [""])[0]
        if ent and first in (ent, ent_raw) and c.numeric and not _is_idish(c.name):
            return c.name
    # 2) the entity names a row-like noun ("customers" -> "Customer Id"). The
    #    identifier column must relate to the entity, or be the row's own bare
    #    key ("people" -> "id"): "how many sales" with only a customer_id column
    #    counts sales rows, not distinct customers, and "countries" can never be
    #    counted via an unrelated "Customer Id".
    for c in record.columns:
        cnorm = re.sub(r"[^a-z0-9]", "", c.name.lower())
        if re.search(r"\bid\b|_id$|id$|code|email|phone", cnorm):
            related = bool(ent) and (ent in cnorm or cnorm in ent)
            own_key = cnorm in _ROW_KEY_NAMES and ent in _ROW_ENTITY_WORDS
            if related or own_key:
                return c.name
    return None


def _is_measure(record, column: str) -> bool:
    """True for a numeric, non-identifier column (e.g. "sessions" per day): its
    values are quantities to add up, so "how many sessions" means SUM, not a
    count of distinct values."""
    for c in record.columns:
        if c.name == column:
            return bool(c.numeric) and not _is_idish(c.name)
    return False


def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.endswith(("ses", "xes", "zes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _plural(word: str) -> str:
    if not word:
        return word
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


_RANK_STOP = {"by", "of", "the", "a", "an", "for", "in", "at", "with", "has", "have", "is", "are", "that", "common", "frequent", "popular"}


def _rank_word(low: str) -> str | None:
    """Grab the noun right after a top/bottom/most/least qualifier, skipping any
    number word, so "top five customers" and "most common country" both resolve."""
    m = re.search(
        r"\b(?:top|bottom|most|least|more|fewer|fewest)\s+(?:(?:one|two|three|four|five|six|seven|eight|nine|ten|\d{1,3})\s+)?([a-z]+)",
        low,
    )
    if m and m.group(1) not in _RANK_STOP:
        return m.group(1)
    return None


def _rank_noun(low: str, dimension: str | None) -> str:
    """Singular noun describing the grouped entries (dimension column), falling
    back to the question's own noun for identifier dimensions ("Customer Id")."""
    if dimension and not _is_idish(dimension):
        return str(dimension).lower()
    return _rank_word(low) or (str(dimension).lower() if dimension else "item")


def _rank_unit(low: str, count_rank: bool, metric: str) -> str:
    """The thing being counted/aggregated: "customers" for count ranks, else the metric."""
    if count_rank:
        m = _rank_word(low)
        if m and m.endswith("s"):
            return m
        return "rows"
    return metric or "total"


def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _value_col(columns: list) -> str | None:
    """Pick the aggregate value column out of a top/bottom result's columns."""
    for c in reversed(columns):
        if str(c).lower() in ("row_count", "total", "count", "value", "score", "amount", "revenue", "units"):
            return str(c)
    return str(columns[-1]) if columns else None


_SWITCH_MARGIN = 3  # extra relevance another dataset needs to override the pinned one


def _resolve_dataset(low: str, ctx: dict) -> str | None:
    """Pick the most relevant dataset by matching its name/columns against the question."""
    registry = ctx["registry"]
    datasets = registry.list()
    if len(datasets) <= 1:
        return datasets[0].dataset_id if datasets else None

    # Generic tokens ("data", "csv", ...) appear in every file name and carry no
    # signal; ignoring them keeps generic questions ("what columns does this
    # dataset have") from matching all files equally and clobbering the pinned one.
    name_stop = {"data", "dataset", "csv", "file", "table", "sample", "sheet", "info", "list", "upload", "report"}

    def score(rec) -> int:
        s = 0
        for token in re.findall(r"[a-z]+", rec.name.lower()):
            if token in name_stop:
                continue
            if len(token) > 2 and token in low:
                s += 2
        for col in rec.columns:
            for token in re.findall(r"[a-z]+", col.name.lower()):
                if len(token) > 3 and token in low:
                    s += 1
        return s

    # Persisted files reloaded across restarts duplicate the same logical
    # dataset under a mangled name ("sales_data.csv" -> "sales_data_csv.csv");
    # dedupe by content hash so duplicates never create a false tie.
    uniq: dict[str, Any] = {}
    for rec in datasets:
        key = rec.sha256 or rec.name
        uniq.setdefault(key, rec)
    unique = list(uniq.values())

    best, best_score = None, 0
    for rec in unique:
        sc = score(rec)
        if sc > best_score:
            best, best_score = rec.dataset_id, sc
    if best_score == 0:
        return None
    # The dataset the user attached/selected is sticky: only switch away when
    # the question can't be answered from it (no column/name overlap) or points
    # much more clearly at another file. Otherwise one shared word ("total
    # sales amount" vs a file called sales_data.csv) silently queried the
    # wrong dataset.
    pinned = ctx.get("pinned")
    pinned_rec = registry.get(pinned) if pinned else None
    if pinned_rec is not None and best != pinned:
        pinned_score = score(pinned_rec)
        if pinned_score > 0 and best_score - pinned_score < _SWITCH_MARGIN:
            return None
    # A tie means the question names nothing distinctive. Return None so the
    # caller keeps the session's active (pinned) dataset instead of silently
    # switching to whichever dataset was registered first.
    if sum(1 for rec in unique if score(rec) == best_score) > 1:
        return None
    return best


def _detect_chart(low: str) -> tuple[str | None, dict]:
    if "scatter" in low:
        return "scatter", {}
    if "pie" in low:
        return "pie", {}
    if "hist" in low:
        return "hist", {}
    if "heatmap" in low:
        return "heatmap", {}
    if "barh" in low or "horizontal" in low:
        return "barh", {}
    if "box" in low:
        return "box", {}
    if "area" in low:
        return "area", {}
    if "line" in low or "trend" in low or "monthly" in low or "over time" in low:
        return "line", {}
    if "bar" in low or "chart" in low or "plot" in low or "graph" in low or "visuali" in low:
        return "bar", {}
    return None, {}


def _guess_date(df: pd.DataFrame) -> str | None:
    return guess_date_column(df)


def _first_numeric(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            return str(col)
    return None


def _is_idish(name: str) -> bool:
    n = str(name).lower().strip()
    if re.search(r"\bid\b|_id$|id$|code|email|phone", n):
        return True
    # row-index/sequence columns are never metrics or meaningful dimensions
    return bool(
        re.match(
            r"^(index|idx|row(\s*(no|number|num|id)\.?)?|serial(\s*no\.?)?|"
            r"s\.?no\.?|position|seq(\.|uence)?|unnamed|#|rec(\s*no\.?)?)$",
            n,
        )
    )


def _pick_metric(low: str, record) -> str:
    # identifier columns are never metrics (don't SUM an id because the question
    # mentions "customers" and the column is a numeric customer_id).
    numeric = [c.name for c in record.columns if c.numeric and not _is_idish(c.name)]
    if not numeric:
        return ""
    # 1) columns whose name words appear in the question
    best, best_score = None, 0
    for c in numeric:
        tokens = re.findall(r"[a-z]+", str(c).lower())
        score = sum(1 for t in tokens if len(t) > 2 and t in low)
        if score > best_score:
            best, best_score = str(c), score
    if best:
        return best
    # 2) business-meaning priority fallback
    priority = ["revenue", "sales", "amount", "value", "profit", "total", "price", "units", "score", "count"]
    for kw in priority:
        for c in numeric:
            if kw in str(c).lower():
                return str(c)
    return str(numeric[0])


def _pick_dimension(low: str, record) -> str | None:
    categorical = [c.name for c in record.columns if c.categorical]
    if not categorical:
        return None
    # 1) columns whose name words appear in the question
    best, best_score = None, 0
    for c in categorical:
        tokens = re.findall(r"[a-z]+", str(c).lower())
        score = sum(1 for t in tokens if len(t) > 2 and t in low)
        if score > best_score:
            best, best_score = str(c), score
    if best:
        # prefer a real categorical over an identifier on a tie: "which COUNTRY
        # has the most customers" must group by "country", not "Customer Id".
        if best_score > 0 and _is_idish(best):
            for c in categorical:
                tokens = re.findall(r"[a-z]+", str(c).lower())
                if not _is_idish(c) and sum(1 for t in tokens if len(t) > 2 and t in low) == best_score:
                    return str(c)
        return best
    # 2) business-meaning priority fallback
    priority = ["region", "product", "customer", "category", "channel", "city", "country", "segment", "name", "month"]
    for kw in priority:
        for c in categorical:
            if kw in str(c).lower():
                return str(c)
    return str(categorical[0])


def _where_sql(*conds: str | None) -> str:
    parts = [c for c in conds if c]
    return f" WHERE {' AND '.join(f'({p})' if ' OR ' in p.upper() else p for p in parts)}" if parts else ""


def _top_bottom_sql(record, dimension, metric, n, ascending, where: str | None = None) -> str:
    table = _safe_name(record.name)
    order = "ASC" if ascending == "true" else "DESC"
    if not dimension or not metric:
        return f'SELECT * FROM "{table}"{_where_sql(where)} ORDER BY "{metric}" {order}'
    # NULL groups are excluded: a blank sales_rep must never be reported as
    # the "lowest performer".
    return (
        f'SELECT "{dimension}" AS dimension, SUM("{metric}") AS total FROM "{table}"'
        f'{_where_sql(where, f"{_q(dimension)} IS NOT NULL")} '
        f'GROUP BY "{dimension}" ORDER BY total {order}'
    )


def _count_rank_sql(record, dimension, n, ascending, where: str | None = None) -> str:
    table = _safe_name(record.name)
    order = "ASC" if ascending == "true" else "DESC"
    return (
        f'SELECT "{dimension}" AS dimension, COUNT(*) AS row_count FROM "{table}"'
        f'{_where_sql(where, f"{_q(dimension)} IS NOT NULL")} '
        f'GROUP BY "{dimension}" ORDER BY row_count {order}'
    )


def _q(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _token_hits(name: str, low: str) -> int:
    """How many of a column name's words appear as words in the question."""
    return sum(1 for t in re.findall(r"[a-z]+", str(name).lower()) if len(t) > 2 and re.search(rf"\b{t}", low))


def _mentioned_metric(low: str, record) -> str | None:
    """A numeric, non-identifier column the question names explicitly (no
    priority fallback, unlike ``_pick_metric``)."""
    best, best_score = None, 0
    for c in record.columns:
        if c.numeric and not _is_idish(c.name):
            score = _token_hits(c.name, low)
            if score > best_score:
                best, best_score = c.name, score
    return best


_GROUP_RE = re.compile(r"\b(?:by|per|for each|for every|each|across)\s+(?:the\s+)?(?:different\s+)?([a-z][a-z0-9 _-]*)")
_GROUP_STOP = re.compile(r"\s+(?:in|where|for|during|with|and|sorted|ordered|order|from|since|over|that|which)\b.*$")


def _group_dimension(low: str, record) -> str | None:
    """The column after "by / per / for each" ("total revenue by region")."""
    m = _GROUP_RE.search(low)
    if not m:
        return None
    phrase = _GROUP_STOP.sub("", m.group(1)).strip()
    if re.match(r"(?:all|every|the entire|the whole|everything)\b", phrase):
        return None  # "across all sales" is a scope, not a grouping
    n_rows = len(record.df)
    best, best_score = None, 0
    for c in record.columns:
        if c.numeric and not _is_idish(c.name):
            continue
        if _is_idish(c.name) and c.unique_count and c.unique_count >= n_rows:
            continue  # "per transaction" over a unique id = per row, not a group
        score = _token_hits(c.name, phrase)
        if score > best_score:
            best, best_score = c.name, score
    return best


_AGG_FUNCS = (
    ("AVG", "average", re.compile(r"\b(?:average|avg|mean)\b")),
    ("SUM", "total", re.compile(r"\b(?:total|sum)\b")),
)
_EXTREME_RE = re.compile(
    r"\b(?:what(?:'s| is| was| are)?|show(?: me)?|give(?: me)?|find|get)\s+(?:the\s+)?"
    r"(highest|lowest|maximum|minimum|max|min|largest|smallest|biggest)\b"
)


def _aggregate_intent(low: str, record) -> dict | None:
    """SUM/AVG of a named metric, optionally grouped ("by region")."""
    metric = _mentioned_metric(low, record)
    if not metric:
        return None
    dimension = _group_dimension(low, record)
    for func, label, pattern in _AGG_FUNCS:
        if pattern.search(low):
            return {"func": func, "label": label, "metric": metric, "dimension": dimension}
    if dimension:
        # "revenue per sales rep": amounts add up, rates/prices are averaged
        if re.search(r"rate|price|ratio|pct|percent|score|rating|avg|mean", metric.lower()):
            return {"func": "AVG", "label": "average", "metric": metric, "dimension": dimension}
        return {"func": "SUM", "label": "total", "metric": metric, "dimension": dimension}
    return None


def _extreme_intent(low: str, record) -> dict | None:
    """"What is the highest revenue value?" -> MAX(revenue), one number. Only
    when no grouping dimension is named; "which region has the highest
    revenue" stays a ranking."""
    m = _EXTREME_RE.search(low)
    if not m:
        return None
    metric = _mentioned_metric(low, record)
    if not metric or _group_dimension(low, record):
        return None
    for c in record.columns:
        if c.categorical and not c.numeric and not _is_idish(c.name) and _token_hits(c.name, low):
            return None
    is_max = m.group(1) in ("highest", "maximum", "max", "largest", "biggest")
    return {"func": "MAX" if is_max else "MIN", "label": "highest" if is_max else "lowest", "metric": metric, "dimension": None}


def _aggregate_sql(record, question: str, agg: dict) -> str:
    table = _safe_name(record.name)
    metric, dim, func = agg["metric"], agg.get("dimension"), agg["func"]
    alias = f"{agg['label']}_{_ident(metric)}"
    where = conditions.extract_where(question, record)
    agg["where"] = where
    if dim:
        # unlike rankings, a breakdown keeps the NULL group so no rows go missing
        return (
            f'SELECT {_q(dim)}, {func}({_q(metric)}) AS {alias} FROM "{table}"'
            f'{_where_sql(where)} GROUP BY {_q(dim)} ORDER BY {alias} DESC NULLS LAST'
        )
    return f'SELECT {func}({_q(metric)}) AS {alias} FROM "{table}"{_where_sql(where)}'


def _count_rank_intent(low: str, record) -> bool:
    """True when a top/bottom request means *frequency/count* rather than a SUM.

    "most common country", "which country has the most customers?" rank by
    COUNT(*) per group. "highest revenue" / "most units sold" name a numeric
    column and rank by SUM of it.
    """
    if any(w in low for w in ("common", "frequent", "popular", "frequency")):
        return True
    if re.search(r"\b(most|least|fewest)\s+[a-z]+\b", low):
        # a numeric column explicitly named in the question -> aggregate it
        # (id columns don't count: "most customers" must not be read as a SUM).
        for c in record.columns:
            if c.numeric and not _is_idish(c.name):
                tokens = re.findall(r"[a-z]+", str(c.name).lower())
                if any(len(t) > 2 and t in low for t in tokens):
                    return False
        return True
    return False


def _generic_sql(record) -> str:
    numeric = [c.name for c in record.columns if c.numeric]
    table = _safe_name(record.name)
    if numeric:
        cols = ", ".join(f'ROUND(AVG("{c}"), 2) AS avg_{_ident(c)}' for c in numeric[:4])
        return f'SELECT COUNT(*) AS row_count, {cols} FROM "{table}"'
    return f'SELECT * FROM "{table}" LIMIT 10'


_SQL_FRAMING_RE = re.compile(
    r"^(?:please\s+)?(?:generate|write|create|produce|draft|give|get|show|make)\s+"
    r"(?:a\s+)?(?:sql|query|select)\s*(?:statement|query|for|to|that|which|of)?\s*",
    re.IGNORECASE,
)


def _strip_sql_framing(question: str) -> str:
    """Remove "generate SQL for ..." wrapping so the planner reads the real
    question ("which country has the most customers?") instead of re-routing to
    the SQL tool or picking the wrong intent words."""
    out = _SQL_FRAMING_RE.sub("", question.strip()).strip()
    return out or question.strip()


def sql_for_question(record, question: str) -> str:
    """Turn a question into SQL — the SINGLE shared generator.

    Every SQL-producing path must go through here: the direct-answer branches in
    ``plan`` (count, top/bottom) and the "generate SQL for X" tool. Keeping one
    implementation means a fix to intent/column resolution (e.g. 'which country
    has the most customers?' must GROUP BY "Country", not "Customer Id") applies
    everywhere instead of drifting apart into duplicate logic.
    """
    low = _strip_sql_framing(question)
    lowl = low.lower()
    table = f'"{_safe_name(record.name)}"'

    if not lowl or "raw" in lowl or "preview" in lowl:
        return f"SELECT * FROM {table} LIMIT 10"

    if _count_intent(lowl):
        entity = _count_entity(lowl) or "rows"
        target = _count_target(record, entity)
        if target and _is_measure(record, target):
            base = f'SELECT SUM("{target}") AS total_{_ident(target)} FROM {table}'
        elif target:
            base = f'SELECT COUNT(DISTINCT "{target}") AS count FROM {table}'
        else:
            base = f"SELECT COUNT(*) AS count FROM {table}"
        where = conditions.extract_where(question, record)
        return base + (f" WHERE {where}" if where else "")

    extreme = _extreme_intent(lowl, record)
    if extreme:
        return _aggregate_sql(record, question, extreme)

    # top / bottom N — includes "which country has the most customers?"
    if _has_any(lowl, TOP_MARKERS + BOTTOM_MARKERS):
        n = _extract_number(lowl, default=5)
        dimension = _pick_dimension(lowl, record)
        ascending = "true" if _has_any(lowl, BOTTOM_MARKERS) else "false"
        where = conditions.extract_where(question, record)
        if dimension and _count_rank_intent(lowl, record):
            return _count_rank_sql(record, dimension, n, ascending, where)
        metric = _pick_metric(lowl, record)
        return _top_bottom_sql(record, dimension, metric, n, ascending, where)

    agg = _aggregate_intent(lowl, record)
    if agg:
        return _aggregate_sql(record, question, agg)

    # temporal aggregation (monthly/daily/weekly trend, forecast)
    if any(w in lowl for w in ("trend", "monthly", "daily", "weekly", "over time", "forecast")):
        date_col = _guess_date(record.df)
        metric = _pick_metric(lowl, record) or _first_numeric(record.df)
        if date_col and metric:
            grain = "day" if "daily" in lowl else ("week" if "weekly" in lowl else "month")
            return (
                f"SELECT date_trunc('{grain}', \"{date_col}\") AS period, "
                f'SUM("{metric}") AS total_{_ident(metric)} FROM {table} GROUP BY period ORDER BY period'
            )

    if any(w in lowl for w in ("average", "mean")):
        numeric = [c.name for c in record.columns if c.numeric]
        if numeric:
            cols = ", ".join(f'AVG("{c}") AS avg_{_ident(c)}' for c in numeric[:4])
            return f"SELECT {cols} FROM {table}"

    # distribution / count by dimension ("customers by country", "breakdown by ...")
    if any(w in lowl for w in ("distribution", "breakdown", "by ", "per ", "group by")) or "count" in lowl:
        dim = _pick_dimension(lowl, record)
        if dim:
            return f'SELECT "{dim}" AS dimension, COUNT(*) AS row_count FROM {table} GROUP BY "{dim}" ORDER BY row_count DESC'

    return _generic_sql(record)


def _draft_pandas(record, low: str) -> str:
    numeric = [c.name for c in record.columns if c.numeric]
    if "correl" in low and len(numeric) >= 2:
        cols = ", ".join(f'"{c}"' for c in numeric[:4])
        return f"result = df[{cols}].corr()"
    if numeric:
        metric = _pick_metric(low, record)
        dim = _pick_dimension(low, record)
        if dim:
            return f'result = df.groupby("{dim}")["{metric}"].agg(["sum", "mean", "count"]).sort_values("sum", ascending=False)'
        return f'result = df["{metric}"].describe()'
    return "result = df.describe(include='all')"


def _anomaly_section(anomaly_res, record, columns=None) -> list[str]:
    lines = [f"**Anomaly detection on `{record.name}`**\n"]
    checked = anomaly_res.get("by_column", {}) if anomaly_res else {}
    checked_txt = ", ".join(checked) if checked else (", ".join(columns) if columns else "all numeric columns")
    if not anomaly_res or not anomaly_res.get("rows"):
        lines.append(
            f"No significant anomalies detected in `{checked_txt}` "
            f"with the default threshold ({anomaly_res.get('method', 'z-score')} method)."
        )
        return lines
    rows = anomaly_res["rows"]
    lines.append(f"Checked `{checked_txt}` using the **{anomaly_res.get('method')}** method and flagged **"
                 f"{anomaly_res['anomaly_count']}** anomalous value(s). Most affected column: "
                 f"{max(checked, key=checked.get) if checked else 'n/a'}.\n")
    lines.append("| # | Column | Value | Severity | Explanation |")
    lines.append("|---|--------|-------|----------|-------------|")
    for i, r in enumerate(rows[:15], 1):
        lines.append(f"| {i} | {r['column']} | {r['value']} | {r['severity']} | {r['explanation']} |")
    if len(rows) > 15:
        lines.append(f"| ... | ({len(rows)-15} more) | | | |")
    lines.append("\n> **Why these are anomalies:** each flag compares the value against a statistical baseline "
                 "(z-score vs mean/std, IQR fences, or a rolling median for time series) and only values beyond the "
                 "threshold are reported, with the exact rule shown per row.")
    return lines


def _trend_note(record) -> str:
    date_col = _guess_date(record.df)
    numeric = [c.name for c in record.columns if c.numeric]
    if not date_col or not numeric:
        return "I detected a time column; trends are visualised above."
    try:
        tmp = record.df.copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp["_m"] = tmp[date_col].dt.to_period("M")
        vals = tmp.groupby("_m")[numeric[0]].mean()
        if len(vals) >= 2:
            growth = (vals.iloc[-1] - vals.iloc[0]) / abs(vals.iloc[0]) * 100 if vals.iloc[0] else 0
            return f"average `{numeric[0]}` moved from {vals.iloc[0]:,.1f} to {vals.iloc[-1]:,.1f} across the period ({growth:+.1f}%)."
    except Exception:
        pass
    return "trend line shown above reflects the values over time."


def _chart_howto(chart_res, record) -> str:
    kind = (chart_res or {}).get("kind", "chart")
    if kind == "pie":
        return "the pie shows each category's share; the largest slice dominates the total."
    if kind == "scatter":
        return "each point is a row; a clear diagonal pattern implies a strong relationship."
    if kind == "hist":
        return "the histogram shows how values are distributed (skew, spread, gaps)."
    if kind == "heatmap":
        return "cell colors show pairwise correlation strength."
    if kind == "line":
        return "the line traces the metric over the x axis; rising/falling stretches show trends."
    return "bar heights show the aggregated value per category; taller = larger."


def _explain_sql(question: str, sql_res) -> str:
    cols = (sql_res or {}).get("columns", [])
    n = (sql_res or {}).get("row_count", 0)
    if cols:
        return f"the query groups by `{cols[0]}` and aggregates the metric, then sorts/limits; it returned {n} row(s)."
    return "the query was generated from your request and executed against the uploaded data."


def _sql_for_result(sql_res) -> str | None:
    if sql_res and sql_res.get("sql"):
        return sql_res["sql"]
    return None


def _error_answer(failures: list[dict]) -> str:
    """Build a clear error-state answer listing every failing tool call."""
    lines = ["**I couldn't complete this analysis — the query failed.**\n"]
    for f in failures:
        lines.append(f"- **`{f['tool']}` failed:** {f['error']}")
        if f["code"]:
            lines.append(f"  ```sql\n  {f['code']}\n  ```")
    lines.append("\n> No reliable answer can be produced from this result. Check the query above or rephrase your question.")
    return "\n".join(lines)


def _reasoning_text(plan: dict, results: list, record) -> str:
    calls = ", ".join(c["name"] for c in plan["tool_calls"]) or "no tools"
    return (
        f"I analysed `{record.name}` using the following tools: {calls}. "
        f"Each tool returned structured results (tables/charts) that the answer above cites directly."
    )


def _fmt_number(v) -> str:
    """Readable scalar for answer text: 2,400 / 5,173,670.58 / 3.01."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return _fmt_cell(v)
    if float(v).is_integer():
        return f"{int(v):,}"
    return f"{v:,.2f}"


def _fmt_cell(v) -> str:
    if isinstance(v, float) and not v.is_integer():
        v = round(v, 2)
    s = "None" if v is None else str(v)
    s = s.replace("|", "\\|")
    return s[:40]


def _ident(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(name).lower())[:30]
