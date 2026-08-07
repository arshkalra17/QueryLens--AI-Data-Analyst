"""Tool registry for the analyst agent.

Tools are plain functions annotated with :class:`Tool`. Each tool knows how to
render a compact result for the LLM and can emit artifacts (charts) that the
frontend displays.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from ..core.registry import DataFrameRegistry
from ..utils import guess_date_column, is_string_series
from . import anomalies as anomaly_mod
from . import forecast as forecast_mod
from . import insights as insights_mod
from . import quality as quality_mod
from .charts import make_chart
from .sql_exec import generate_select_sql, run_sql

logger = logging.getLogger(__name__)


class ToolError(RuntimeError):
    pass


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable
    render: Callable | None = None
    examples: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _pick_record(ctx, dataset_id: str | None, session_id: str | None):
    registry: DataFrameRegistry = ctx["registry"]
    if dataset_id:
        rec = registry.get(dataset_id)
        if rec is None:
            raise ToolError(f"Unknown dataset id '{dataset_id}'. Upload it or pass a valid id.")
        return rec
    # fall back to the pinned dataset in the session
    if session_id:
        session = ctx["sessions"].get(session_id)
        if session:
            if session.pinned:
                rec = registry.get(session.pinned)
                if rec is not None:
                    return rec
            if session.dataset_ids:
                rec = registry.get(session.dataset_ids[0])
                if rec is not None:
                    return rec
    datasets = registry.list()
    if not datasets:
        raise ToolError("No datasets are loaded. Upload a CSV file first.")
    return datasets[0]


def _resolve_column(rec, name: str) -> str | None:
    """Map a possibly-misspelled/cased column name onto a real column.

    Exact matches win; otherwise a case-insensitive unique match is accepted
    (LLMs routinely pass 'country' for a column stored as 'Country'). Empty
    strings mean "unset" (None); a non-empty name that matches nothing is kept
    as-is so downstream validation can report it instead of silently plotting
    a different column."""
    name = str(name).strip()
    if not name:
        return None
    if name in rec.df.columns:
        return name
    lowered = name.lower()
    matches = [c for c in rec.df.columns if str(c).lower() == lowered]
    if len(matches) == 1:
        return matches[0]
    return name


def _frames(ctx, session_id: str | None) -> dict[str, pd.DataFrame]:
    registry: DataFrameRegistry = ctx["registry"]
    out: dict[str, pd.DataFrame] = {}
    datasets = registry.list()
    if session_id:
        session = ctx["sessions"].get(session_id)
        if session:
            ids = [session.pinned] if session.pinned else []
            ids += [d for d in session.dataset_ids if d not in ids]
            ordered = [r for d in ids for r in datasets if r.dataset_id == d]
            if ordered:
                datasets = ordered
    for rec in datasets:
        out[rec.dataset_id] = rec.df
        out[_safe_name(rec.name)] = rec.df
        # friendly aliases so LLMs can reference a file like "sales_data" or "sales_data.csv"
        base = _safe_name(rec.name).rsplit("_", 1)[0] if rec.name.lower().endswith((".csv", ".tsv", ".txt")) else _safe_name(rec.name)
        out.setdefault(base, rec.df)
        out.setdefault(_safe_name(rec.name).replace("_", ""), rec.df)
    return out


def _safe_name(name: str) -> str:
    import re

    return re.sub(r"[^a-zA-Z0-9_]", "_", name.lower())[:60]


def _resolve_ref(ctx, dataset_id: str | None, names: tuple = ()) -> str | None:
    """Resolve a dataset reference from an id or filename-ish aliases."""
    if dataset_id:
        return dataset_id
    registry = ctx["registry"]
    for n in names:
        if not n:
            continue
        wanted = str(n)
        for rec in registry.list():
            if (
                rec.name == wanted
                or _safe_name(rec.name) == _safe_name(wanted)
                or rec.name.rsplit(".", 1)[0] == wanted.rsplit(".", 1)[0]
                or rec.dataset_id == wanted
            ):
                return rec.dataset_id
    return None


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------

def tool_list_datasets(ctx, **kwargs):
    registry: DataFrameRegistry = ctx["registry"]
    datasets = registry.list()
    if not datasets:
        return {"message": "No datasets loaded yet.", "datasets": []}
    return {
        "datasets": [r.to_meta() for r in datasets],
        "message": f"{len(datasets)} dataset(s) available.",
    }


def tool_get_schema(ctx, dataset_id=None, session_id=None, dataset=None, dataset_name=None, **kwargs):
    dataset_id = _resolve_ref(ctx, dataset_id, (dataset, dataset_name))
    rec = _pick_record(ctx, dataset_id, session_id)
    meta = rec.to_meta()
    return {"schema": meta, "snippet": rec.schema_snippet()}


def tool_preview_rows(ctx, dataset_id=None, session_id=None, limit=10, rows=None, n=None, dataset=None, dataset_name=None, **kwargs):
    dataset_id = _resolve_ref(ctx, dataset_id, (dataset, dataset_name))
    rec = _pick_record(ctx, dataset_id, session_id)
    take = rows or n or limit
    take = max(1, min(200, int(take)))
    df = rec.df.head(take)
    columns = [str(c) for c in df.columns]
    row_dicts = [_clean_row(row) for _, row in df.iterrows()]
    return {
        "dataset": rec.name,
        "columns": columns,
        "rows": row_dicts,
        "row_count": len(row_dicts),
        "requested": take,
        "markdown_table": _table_md(columns, row_dicts),
    }


def tool_filter_rows(ctx, dataset_id=None, session_id=None, column=None, value=None, operator="==", limit=50, dataset=None, dataset_name=None, condition=None, filter=None, where=None, query=None, **kwargs):
    """Return the rows where `column` matches `value` (dataframe filtering)."""
    dataset_id = _resolve_ref(ctx, dataset_id, (dataset, dataset_name))
    rec = _pick_record(ctx, dataset_id, session_id)
    # LLMs often hand over a natural-language condition ("country = 'India'")
    # instead of structured column/value args; parse it when present. The key
    # varies between models (condition/filter/where/query), so accept them all.
    cond = condition or filter or where or query
    if (column is None or value is None or str(value) == "") and cond:
        parsed = _parse_condition(str(cond))
        if parsed:
            column, operator, value = parsed
    if column is None or value is None or str(value) == "":
        raise ToolError("Provide both a `column` and a `value` to filter by.")
    col = _match_column_name(rec, column)
    if col is None:
        raise ToolError(f"Column '{column}' not found in {rec.name}. Columns: {', '.join(str(c) for c in rec.df.columns)}")
    mask = _build_filter_mask(rec.df[col], value, operator)
    matched = rec.df[mask]
    take = max(1, min(200, int(limit or 50)))
    df = matched.head(take)
    columns = [str(c) for c in df.columns]
    row_dicts = [_clean_row(row) for _, row in df.iterrows()]
    return {
        "dataset": rec.name,
        "column": col,
        "value": str(value),
        "operator": operator or "==",
        "columns": columns,
        "rows": row_dicts,
        "row_count": len(row_dicts),
        "matched": int(matched.shape[0]),
        "markdown_table": _table_md(columns, row_dicts),
    }


def tool_find_duplicates(ctx, dataset_id=None, session_id=None, columns=None, column=None, limit=50, dataset=None, dataset_name=None, **kwargs):
    """Return rows whose key (one or more columns) is shared with at least one
    other row — equivalent to ``GROUP BY <cols> HAVING COUNT(*) > 1`` / pandas
    ``df[df.duplicated(subset=<cols>, keep=False)]``.

    Uses exact (case-sensitive) matching, which is correct for identifiers like
    Customer Id. Returns duplicate rows with a per-group occurrence count, never
    the whole dataset.
    """
    dataset_id = _resolve_ref(ctx, dataset_id, (dataset, dataset_name))
    rec = _pick_record(ctx, dataset_id, session_id)
    cols = columns or ([column] if column else None)
    if not cols:
        cols = [str(c) for c in rec.df.columns]
    if isinstance(cols, str):
        cols = [cols]
    resolved = []
    for c in cols:
        m = _match_column_name(rec, c)
        if m is None:
            raise ToolError(f"Column '{c}' not found in {rec.name}. Columns: {', '.join(str(c) for c in rec.df.columns)}")
        if m not in resolved:
            resolved.append(m)
    if not resolved:
        raise ToolError("Provide at least one `column` to check for duplicates.")

    df = rec.df
    counts = df.groupby(resolved, dropna=False)[resolved[0]].transform("count")
    dup_mask = (counts > 1).values
    dup = df[dup_mask].copy()
    dup["_occurrences"] = counts[dup_mask].values
    total = int(dup_mask.sum())
    groups = int(df[dup_mask][resolved].drop_duplicates().shape[0]) if total else 0

    take = max(0, min(200, int(limit or 50)))
    shown = dup.head(take)
    row_dicts = [_clean_row(row) for _, row in shown.iterrows()]
    return {
        "dataset": rec.name,
        "columns": resolved,
        "duplicate_count": total,
        "group_count": groups,
        "columns_out": [str(c) for c in shown.columns],
        "rows": row_dicts,
        "row_count": len(row_dicts),
        "matched": total,
        "markdown_table": _table_md([str(c) for c in shown.columns], row_dicts),
    }


def tool_run_sql(ctx, sql=None, dataset_id=None, session_id=None, max_rows=None, query=None, **kwargs):
    sql = sql or query
    if not sql:
        raise ToolError("Provide the `sql` (or `query`) argument with a SELECT statement.")
    frames = _frames(ctx, session_id)
    result = run_sql(sql, frames, max_rows=max_rows)
    if result.error:
        raise ToolError(result.error)
    payload = {
        "sql": result.sql,
        "columns": result.columns,
        "rows": result.rows,
        "row_count": result.row_count,
        "truncated": result.truncated,
        "elapsed_ms": round(result.elapsed_ms, 2),
        "markdown_table": _table_md(result.columns, result.rows),
    }
    return payload


def tool_run_pandas(ctx, code, session_id=None, dataset_id=None, dataset=None, dataset_name=None, **kwargs):
    settings = ctx["settings"]
    if not settings.allow_pandas_exec:
        raise ToolError("Pandas code execution is disabled (ALLOW_PANDAS_EXEC=false).")
    dataset_id = _resolve_ref(ctx, dataset_id, (dataset, dataset_name))
    rec = _pick_record(ctx, dataset_id, session_id)
    df = rec.df
    code = _strip_fences(code)
    _guard_pandas(code)
    namespace: dict[str, Any] = {
        "df": df,
        "pd": pd,
        "np": __import__("numpy", fromlist=["np"]),
        "__builtins__": {"len": len, "range": range, "str": str, "int": int, "float": float, "list": list, "dict": dict, "tuple": tuple, "min": min, "max": max, "sum": sum, "print": print, "abs": abs, "round": round, "enumerate": enumerate, "zip": zip},
    }
    exec_globals: dict[str, Any] = {}
    exec(code, namespace, exec_globals)
    if "result" in exec_globals:
        result = exec_globals["result"]
    elif "result" in namespace:
        result = namespace["result"]
    else:
        result = None
    return _describe_result(result)


def tool_generate_chart(ctx, dataset_id=None, session_id=None, kind="bar", x=None, y=None, title="", agg=None, group_by=None, dataset=None, dataset_name=None, type=None, chart_type=None, **kwargs):
    dataset_id = _resolve_ref(ctx, dataset_id, (dataset, dataset_name))
    rec = _pick_record(ctx, dataset_id, session_id)
    # LLMs routinely use aliases/sloppy casing: accept 'type'/'chart_type' as
    # the kind, treat empty strings as "unset", and resolve columns
    # case-insensitively so x='country' still finds 'Country'.
    kind = kind or type or chart_type or "bar"
    y = y or None
    if x is not None:
        x = _resolve_column(rec, x)
    if y is not None:
        y = _resolve_column(rec, y)
    if group_by is not None:
        group_by = _resolve_column(rec, group_by)
    record = make_chart(rec.df, kind=kind, x=x, y=y, title=title, agg=agg, group_by=group_by)
    record.dataset_ids = [rec.dataset_id]
    ctx["chart_store"].put(record)
    return {
        "chart_id": record.chart_id,
        "kind": record.kind,
        "title": record.title,
        "caption": record.caption,
        "url": f"/api/charts/{record.chart_id}/image",
        "html_url": f"/api/charts/{record.chart_id}/html",
        "plotly_url": f"/api/charts/{record.chart_id}/plotly",
    }


def tool_detect_anomalies(ctx, dataset_id=None, session_id=None, columns=None, group_by=None, method="auto", max_results=50, column=None, dataset=None, dataset_name=None, **kwargs):
    dataset_id = _resolve_ref(ctx, dataset_id, (dataset, dataset_name))
    if columns is None and column:
        columns = [column]
    rec = _pick_record(ctx, dataset_id, session_id)
    flags = anomaly_mod.detect_anomalies(
        rec.df, columns=columns, group_by=group_by, method=method, max_results=max_results
    )
    summary: dict[str, int] = {}
    for f in flags:
        summary[f.column] = summary.get(f.column, 0) + 1
    rows = [
        {
            "row_index": f.row_index,
            "column": f.column,
            "group": f.group,
            "value": f.value,
            "method": f.method,
            "score": round(f.score, 3),
            "severity": f.severity,
            "explanation": f.explanation,
        }
        for f in flags
    ]
    return {
        "dataset": rec.name,
        "method": flags[0].method if flags else method,
        "anomaly_count": len(rows),
        "by_column": summary,
        "rows": rows,
    }


def tool_generate_insights(ctx, dataset_id=None, session_id=None, limit=10, dataset=None, dataset_name=None, **kwargs):
    dataset_id = _resolve_ref(ctx, dataset_id, (dataset, dataset_name))
    rec = _pick_record(ctx, dataset_id, session_id)
    insights = insights_mod.generate_insights(rec.df, rec.name, limit=limit)
    return {"dataset": rec.name, "insights": [i.to_dict() for i in insights]}


def tool_data_quality(ctx, dataset_id=None, session_id=None, dataset=None, dataset_name=None, **kwargs):
    dataset_id = _resolve_ref(ctx, dataset_id, (dataset, dataset_name))
    rec = _pick_record(ctx, dataset_id, session_id)
    report = quality_mod.quality_report(rec.df)
    return {"dataset": rec.name, **report.to_dict()}


def tool_forecast(ctx, dataset_id=None, session_id=None, date_column=None, value_column=None, periods=12, seasonality="month", dataset=None, dataset_name=None, **kwargs):
    dataset_id = _resolve_ref(ctx, dataset_id, (dataset, dataset_name))
    rec = _pick_record(ctx, dataset_id, session_id)
    if date_column is None or date_column not in rec.df.columns:
        date_column = _guess_date(rec.df)
    if value_column is None:
        value_column = _pick_numeric(rec.df)
    if date_column is None or value_column is None:
        raise ToolError("Forecasting needs a date column and a numeric value column.")
    result = forecast_mod.forecast(rec.df, date_column, value_column, periods=periods, seasonality=seasonality)
    chart_record = _make_forecast_chart(result)
    ctx["chart_store"].put(chart_record)
    result["chart_id"] = chart_record.chart_id
    result["chart_url"] = f"/api/charts/{chart_record.chart_id}/image"
    result["chart_caption"] = chart_record.caption
    return result


def tool_data_quality_score(ctx, **kwargs):
    return tool_data_quality(ctx, **kwargs)


def tool_search_data(ctx, dataset_id=None, session_id=None, query=None, column=None, limit=10, dataset=None, dataset_name=None, **kwargs):
    """Lightweight semantic/keyword search over a dataset's text columns."""
    dataset_id = _resolve_ref(ctx, dataset_id, (dataset, dataset_name))
    rec = _pick_record(ctx, dataset_id, session_id)
    if not query:
        raise ToolError("Provide a search `query`.")
    q = str(query).lower()
    candidates = []
    text_cols = [str(c) for c in rec.df.columns if is_string_series(rec.df[c])]
    if column and column in rec.df.columns:
        text_cols = [column]
    if not text_cols:
        raise ToolError("No text columns to search.")
    for i, row in rec.df.iterrows():
        score = 0.0
        for col in text_cols:
            val = row[col]
            if pd.isna(val):
                continue
            s = str(val).lower()
            if q in s:
                score += 3.0
            score += _token_similarity(q, s)
        if score > 0:
            candidates.append({"row_index": int(i), "score": round(score, 3), "record": _clean_row(row)})
    candidates.sort(key=lambda c: -c["score"])
    return {"query": query, "results": candidates[:limit], "total_matches": len(candidates)}


def tool_generate_sql(ctx, dataset_id=None, session_id=None, analysis=None, dataset=None, dataset_name=None, **kwargs):
    """Produce (and optionally run) a SQL query for the requested analysis."""
    dataset_id = _resolve_ref(ctx, dataset_id, (dataset, dataset_name))
    rec = _pick_record(ctx, dataset_id, session_id)
    intent = (analysis or "").strip()
    # Single shared generator (same one `fallback.plan` uses for direct answers),
    # so "which country has the most customers?" groups by Country here too.
    from . import fallback

    sql = fallback.sql_for_question(rec, intent)
    # keep explicit conditions ("in 2021", "for the North region") as a WHERE clause
    from . import conditions

    cond = conditions.extract_where(intent, rec)
    if cond and not conditions.sql_has_where(sql):
        sql = conditions.inject_where(sql, cond) or sql
    frames = _frames(ctx, session_id)
    result = run_sql(sql, frames, max_rows=20)
    return {
        "sql": sql,
        "sql_pretty": sql,
        "columns": result.columns,
        "rows": result.rows[:10],
        "row_count": result.row_count,
        "error": result.error,
        "markdown_table": _table_md(result.columns, result.rows[:10]),
    }


def tool_describe(ctx, **kwargs):
    return {"message": "Available tools: " + ", ".join(t.name for t in ALL_TOOLS)}


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def _strip_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        lines = code.splitlines()
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    return code


def _guard_pandas(code: str) -> None:
    lowered = code.lower()
    for bad in ("os.", "subprocess", "import os", "import sys", "open(", "eval(", "exec(", "__import__", "socket", "requests.", "urllib"):
        if bad in lowered:
            raise ToolError(f"Pandas snippet blocked: contains '{bad}'")


def _describe_result(result) -> dict:
    if result is None:
        return {"result": None, "type": "None"}
    if isinstance(result, pd.DataFrame):
        return {
            "type": "dataframe",
            "shape": list(result.shape),
            "columns": [str(c) for c in result.columns],
            "rows": [_clean_row(row) for _, row in result.head(20).iterrows()],
        }
    if isinstance(result, pd.Series):
        return {"type": "series", "values": result.head(50).to_dict()}
    if isinstance(result, (int, float, str, bool)) or result is None:
        return {"type": type(result).__name__, "result": result}
    try:
        import numpy as np

        if isinstance(result, np.ndarray):
            return {"type": "array", "shape": list(result.shape), "values": result.tolist()[:50]}
    except ImportError:
        pass
    return {"type": type(result).__name__, "result": str(result)[:2000]}


def _table_md(columns, rows, max_rows=20) -> str:
    if not columns:
        return "(no rows)"
    header = "| " + " | ".join(str(c) for c in columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, sep]
    for r in rows[:max_rows]:
        lines.append("| " + " | ".join(_short(v) for v in r.values()) + " |")
    return "\n".join(lines)


def _short(v, n=28) -> str:
    s = "None" if v is None else str(v)
    return s if len(s) <= n else s[: n - 3] + "..."


def _clean_row(row) -> dict:
    out = {}
    for k, v in row.items():
        if pd.isna(v):
            out[str(k)] = None
        elif hasattr(v, "isoformat"):
            out[str(k)] = v.isoformat()
        else:
            out[str(k)] = v
    return out


def _match_column_name(rec, name: str) -> str | None:
    want = re.sub(r"[^a-z0-9]", "", str(name).lower())
    for c in rec.df.columns:
        if re.sub(r"[^a-z0-9]", "", str(c).lower()) == want:
            return str(c)
    for c in rec.df.columns:
        if want and want in re.sub(r"[^a-z0-9]", "", str(c).lower()):
            return str(c)
    return None


def _parse_condition(cond: str) -> tuple[str, str, Any] | None:
    """Parse a natural-language condition ("country = 'India'", "age > 18").

    Returns (column, operator, value) or None if it doesn't look like a
    `column OP value` expression.
    """
    m = re.match(r"^\s*([A-Za-z_][\w .-]*?)\s*(<=|>=|!=|==|=|<|>|like|contains)\s*(.+?)\s*$", cond, re.IGNORECASE)
    if not m:
        return None
    column = m.group(1).strip().strip('"').strip("`")
    op = m.group(2).lower()
    value = m.group(3).strip()
    if len(value) >= 2 and value[0] in ("'", '"') and value[-1] == value[0]:
        value = value[1:-1]
    op_map = {"=": "==", "==": "==", "!=": "!=", ">": ">", "<": "<", ">=": ">=", "<=": "<=", "like": "contains", "contains": "contains"}
    return column, op_map[op], value


def _coerce_value(series, value):
    if pd.api.types.is_bool_dtype(series):
        return str(value).lower() in ("true", "1", "yes", "y")
    if pd.api.types.is_numeric_dtype(series):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    if pd.api.types.is_datetime64_any_dtype(series):
        try:
            return pd.Timestamp(value)
        except (TypeError, ValueError):
            return value
    return str(value)


def _build_filter_mask(series, value, operator):
    op = (operator or "==").lower()
    if op in ("contains", "like"):
        return series.astype(str).str.contains(str(value), case=False, regex=False, na=False)
    string_col = is_string_series(series) or pd.api.types.is_string_dtype(series)
    if not string_col:
        val = _coerce_value(series, value)
        try:
            if op == "!=":
                return series != val
            if op == ">":
                return series > val
            if op == ">=":
                return series >= val
            if op == "<":
                return series < val
            if op == "<=":
                return series <= val
            return series == val
        except (TypeError, ValueError):
            pass
    v = str(value).lower()
    sv = series.astype(str).str.lower()
    if op == "!=":
        return sv != v
    return sv == v


def _guess_date(df: pd.DataFrame) -> str | None:
    return guess_date_column(df)


def _pick_numeric(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            return str(col)
    return None


def _make_forecast_chart(result: dict):
    fdf = pd.DataFrame(result["table"])
    fdf["date"] = pd.to_datetime(fdf["date"])
    # only forecast points plus last few actuals
    tail = fdf[fdf["type"] == "actual"].tail(6)
    fc = fdf[fdf["type"] == "forecast"]
    combined = pd.concat([tail, fc])
    return make_chart(
        combined, kind="line", x="date", y="value",
        title=f"Forecast: {result['value_column']} ({result['periods']} periods)",
    )


def _token_similarity(q: str, s: str) -> float:
    """Token-overlap cosine-ish score between query and text."""
    import re

    qt = set(re.findall(r"[a-z0-9]+", q))
    st = set(re.findall(r"[a-z0-9]+", s))
    if not qt or not st:
        return 0.0
    return len(qt & st) / (len(qt) ** 0.5 * len(st) ** 0.5)


# ---------------------------------------------------------------------------
# registry construction
# ---------------------------------------------------------------------------

def _param(*required, properties: dict | None = None) -> dict:
    props = {**({"properties": properties} if properties else {})}
    return {"type": "object", "required": list(required), **props}


ALL_TOOLS: list[Tool] = [
    Tool(
        "list_datasets",
        "List all uploaded datasets with row/column counts.",
        _param(),
        tool_list_datasets,
        examples=["what datasets do I have", "list datasets"],
    ),
    Tool(
        "get_schema",
        "Return the schema (columns, dtypes, sample values) of a dataset.",
        _param("dataset_id", properties={"dataset_id": {"type": "string", "description": "dataset id (optional; defaults to the pinned dataset)"}}),
        tool_get_schema,
        examples=["describe the sales dataset"],
    ),
    Tool(
        "preview_rows",
        "Show the first N raw rows of a dataset exactly as they appear in the CSV (no aggregation).",
        _param("limit", properties={"limit": {"type": "integer", "description": "number of rows to show (default 10)"}}),
        tool_preview_rows,
        examples=["show the first 10 rows", "preview the data", "display some records"],
    ),
    Tool(
        "filter_rows",
        "Return the rows of a dataset that match a filter condition, e.g. one column equals a value. Use for list/show/find row requests such as 'list customers from India' or 'show rows where country = India'.",
        _param("column", "value", properties={"column": {"type": "string", "description": "column name to filter on"}, "value": {"type": "string", "description": "value to match (case-insensitive)"}, "condition": {"type": "string", "description": "optional shorthand like \"country = 'India'\" or \"age > 18\" (alternative to column+value)"}, "operator": {"type": "string", "enum": ["==", "!=", ">", ">=", "<", "<=", "contains"]}, "limit": {"type": "integer", "description": "max rows to return (default 50)"}}),
        tool_filter_rows,
        examples=["list customers from India", "show customers in Japan", "find customer ID xyz", "show rows where country = india"],
    ),
    Tool(
        "find_duplicates",
        "Find rows whose key column(s) are repeated (shared by at least two rows) — equivalent to GROUP BY <cols> HAVING COUNT(*) > 1. "
        "Use for duplicate questions like 'are there duplicate customer IDs?' or 'show rows that appear more than once'. "
        "Matching is exact/case-sensitive, which is correct for identifier columns.",
        _param("columns", properties={"columns": {"type": "array", "items": {"type": "string"}, "description": "column(s) to check for duplicates (default: all columns)"}, "column": {"type": "string", "description": "single column shorthand"}, "limit": {"type": "integer", "description": "max duplicate rows to return (default 50)"}}),
        tool_find_duplicates,
        examples=["are there any duplicate customer ids?", "show rows that appear more than once", "find duplicate emails"],
    ),
    Tool(
        "run_sql",
        "Execute a read-only SQL (SELECT/WITH) query against the loaded datasets and return the result rows. Table names are dataset ids or dataset filenames.",
        _param("sql", properties={"sql": {"type": "string", "description": "SQL SELECT query (also accepts `query`)"}, "max_rows": {"type": "integer", "description": "max rows to return"}}),
        tool_run_sql,
        examples=["which region generated the highest revenue", "monthly sales trend", "top 5 customers"],
    ),
    Tool(
        "generate_sql",
        "Generate (and run) a SQL query for a natural-language analysis request, returning the SQL plus its output.",
        _param("analysis", properties={"analysis": {"type": "string", "description": "the analysis request in plain words"}}),
        tool_generate_sql,
        examples=["generate sql for this analysis", "write sql for top products by revenue"],
    ),
    Tool(
        "run_pandas",
        "Execute a short pandas snippet. `df` is the pinned dataset. Assign the result to `result`.",
        _param("code", properties={"code": {"type": "string", "description": "pandas code that assigns `result`"}}),
        tool_run_pandas,
        examples=["compute monthly average", "correlation between columns using df.corr()"],
    ),
    Tool(
        "generate_chart",
        "Create a chart (bar, barh, line, pie, scatter, hist, box, area, heatmap). Returns a chart image. "
        "For a count question like 'customers by country' set agg='count' and do NOT set a y column "
        "(do not pass SQL expressions like y='count(*)' — they are not valid column names).",
        _param("kind", properties={"kind": {"type": "string", "enum": sorted(["bar", "barh", "line", "pie", "scatter", "hist", "box", "area", "heatmap"])}, "x": {"type": "string"}, "y": {"type": "string"}, "title": {"type": "string"}, "agg": {"type": "string", "description": "sum/mean/median/min/max/count to aggregate y by x; use 'count' for 'number of rows per x'"}, "group_by": {"type": "string"}}),
        tool_generate_chart,
        examples=["show a bar chart of revenue by region", "line chart of monthly sales", "pie chart of category share", "scatter of units vs price", "bar chart of customers by country (agg='count')"],
    ),
    Tool(
        "detect_anomalies",
        "Detect anomalies/outliers in a dataset and explain why each was flagged.",
        _param("method", properties={"method": {"type": "string", "enum": ["auto", "zscore", "iqr", "rolling"]}, "columns": {"type": "array", "items": {"type": "string"}}, "group_by": {"type": "string"}, "max_results": {"type": "integer"}}),
        tool_detect_anomalies,
        examples=["detect anomalies in the dataset", "find outliers in revenue"],
    ),
    Tool(
        "generate_insights",
        "Generate automatic business insights and a summary for a dataset.",
        _param("limit", properties={"limit": {"type": "integer"}}),
        tool_generate_insights,
        examples=["generate insights", "summarize the data", "give me a business summary"],
    ),
    Tool(
        "data_quality",
        "Run a data-quality check returning a 0-100 score, issues and recommendations.",
        _param(),
        tool_data_quality,
        examples=["check data quality", "how clean is the data"],
    ),
    Tool(
        "forecast",
        "Forecast a numeric column over time (linear trend + seasonality) for the next N periods.",
        _param("periods", properties={"date_column": {"type": "string"}, "value_column": {"type": "string"}, "periods": {"type": "integer"}, "seasonality": {"type": "string", "enum": ["month", "dayofweek", "none"]}}),
        tool_forecast,
        examples=["forecast next 12 months of revenue", "predict sales"],
    ),
    Tool(
        "search_data",
        "Search text columns in a dataset by query term (keyword + token similarity).",
        _param("query", properties={"query": {"type": "string"}, "column": {"type": "string"}, "limit": {"type": "integer"}}),
        tool_search_data,
        examples=["search for the customer named Acme", "find rows mentioning 'premium'"],
    ),
]


TOOL_MAP = {t.name: t for t in ALL_TOOLS}


def run_tool(ctx, name: str, args: dict) -> dict:
    tool = TOOL_MAP.get(name)
    if tool is None:
        raise ToolError(f"Unknown tool '{name}'. Available: {', '.join(TOOL_MAP)}")
    logger.info("running tool %s args=%s", name, {k: v for k, v in args.items() if k != "image"})
    started = time.perf_counter()
    try:
        result = tool.func(ctx, **args)
    except ToolError:
        raise
    except Exception as exc:  # surface any engine error cleanly
        logger.exception("tool %s failed", name)
        raise ToolError(f"Tool '{name}' failed: {exc}") from exc
    finally:
        ctx.setdefault("metrics", []).append(
            {"tool": name, "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}
        )
    if tool.render:
        result["_rendered"] = tool.render(result)
    return result
