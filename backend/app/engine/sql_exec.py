"""SQL generation + execution against in-memory pandas DataFrames via DuckDB.

`run_sql` is the workhorse used by the agent's `run_sql` tool and by the SQL
"Generate SQL for this analysis" feature. Tables are registered as their
dataset_id so multi-file analysis (JOINs across CSVs) just works.
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Any

import duckdb
import pandas as pd

from ..config import get_settings

logger = logging.getLogger(__name__)

# Only read-only statements are allowed.
_ALLOWED_PREFIXES = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
_FORBIDDEN = re.compile(
    r"\b(copy|install|load|attach|detach|pragma|vacuum|export|import|insert|update|delete|create|drop|alter|grant|call|prepare|execute)\b",
    re.IGNORECASE,
)

SELECT_PATTERN = re.compile(r"^\s*SELECT\b", re.IGNORECASE)


@dataclass
class SQLResult:
    sql: str
    columns: list[str]
    rows: list[dict]
    row_count: int
    truncated: bool
    elapsed_ms: float
    error: str | None = None


def validate_sql(sql: str) -> None:
    settings = get_settings()
    if not sql or not sql.strip():
        raise ValueError("SQL statement is empty")
    if len(sql) > settings.max_sql_chars:
        raise ValueError(f"SQL exceeds {settings.max_sql_chars} characters")
    if not _ALLOWED_PREFIXES.match(sql):
        raise ValueError("Only SELECT / WITH statements are allowed")
    if _FORBIDDEN.search(sql):
        raise ValueError("Statement contains forbidden keywords")


def _sandboxed_connection() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB connection that cannot touch the filesystem or network.

    The keyword blocklist in ``validate_sql`` cannot catch table functions such
    as ``read_text('/app/.env')`` or ``read_csv('https://...')``, so isolation is
    enforced by DuckDB itself. Registered pandas DataFrames remain queryable.
    ``lock_configuration`` stops generated SQL from re-enabling access.
    Fails closed: if any setting cannot be applied, no query runs.
    """
    con = duckdb.connect()
    try:
        con.execute("SET threads TO 1")
        con.execute("SET enable_external_access = false")
        con.execute("SET autoinstall_known_extensions = false")
        con.execute("SET autoload_known_extensions = false")
        con.execute("SET lock_configuration = true")
    except Exception:
        con.close()
        raise
    return con


def table_names(sql: str) -> list[str]:
    """Best-effort list of referenced identifiers that look like tables."""
    cleaned = re.sub(r"['\"`\[\]]", "", sql)
    return re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{1,}\b", cleaned)


def run_sql(sql: str, frames: dict[str, pd.DataFrame], max_rows: int | None = None) -> SQLResult:
    """Execute `sql` against a dict of named DataFrames.

    Args:
        sql: read-only SQL (SELECT/WITH).
        frames: mapping of table name -> DataFrame (e.g. {"sales": df}).
        max_rows: cap on the number of rows returned to callers.
    """
    import time

    settings = get_settings()
    max_rows = max_rows or settings.max_sql_rows
    validate_sql(sql)
    if not frames:
        raise ValueError("No data registered; upload a dataset first")

    start = time.perf_counter()
    con = _sandboxed_connection()
    # SQL_TIMEOUT_SECONDS: interrupt runaway queries (e.g. an accidental cross
    # join from generated SQL) instead of pinning a worker indefinitely.
    timeout = settings.sql_timeout_seconds
    timer = threading.Timer(timeout, con.interrupt) if timeout and timeout > 0 else None
    try:
        for name, df in frames.items():
            con.register(name, df)
        if timer:
            timer.start()
        result = con.execute(sql)
        df_out = result.fetchdf()
    except duckdb.InterruptException:
        return SQLResult(
            sql=sql, columns=[], rows=[], row_count=0, truncated=False,
            elapsed_ms=(time.perf_counter() - start) * 1000,
            error=f"SQL execution failed: query exceeded the {timeout:g}s time limit",
        )
    except duckdb.Error as exc:
        return SQLResult(
            sql=sql, columns=[], rows=[], row_count=0, truncated=False,
            elapsed_ms=(time.perf_counter() - start) * 1000,
            error=f"SQL execution failed: {exc}",
        )
    finally:
        if timer:
            timer.cancel()
        con.close()

    elapsed = (time.perf_counter() - start) * 1000
    total = int(len(df_out))
    truncated = total > max_rows
    df_out = df_out.head(max_rows)
    columns = [str(c) for c in df_out.columns]
    rows = []
    for _, row in df_out.iterrows():
        rows.append(_clean_row(row))
    return SQLResult(
        sql=sql, columns=columns, rows=rows, row_count=total,
        truncated=truncated, elapsed_ms=elapsed,
    )


def _clean_row(row: pd.Series) -> dict:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            out[str(key)] = None
        elif hasattr(value, "item") and isinstance(value, (pd.Timestamp,)):
            out[str(key)] = value.isoformat()
        elif hasattr(value, "item"):
            try:
                out[str(key)] = value.item()
            except Exception:
                out[str(key)] = str(value)
        elif pd.isna(value):
            out[str(key)] = None
        else:
            out[str(key)] = value
    return out


def generate_select_sql(dataset_name: str, columns: list[str] | None = None, limit: int = 100) -> str:
    """Produce a simple, safe SELECT for a dataset (used when generating SQL)."""
    if columns:
        cols = ", ".join(_quote_ident(c) for c in columns)
    else:
        cols = "*"
    return f"SELECT {cols} FROM {_quote_ident(dataset_name)} LIMIT {limit}"


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'
