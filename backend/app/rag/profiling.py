"""Table profiling: build a compact, semantically rich description of a
dataset that gets embedded and stored in the vector store.

The profile captures *structure* (columns, types, stats, semantics, samples)
rather than the raw dataset body, so retrieval is cheap and the LLM is never
flooded with raw rows.
"""
from __future__ import annotations

import pandas as pd

from ..core.registry import DatasetRecord
from ..utils import is_string_series, numeric_columns


def build_table_profile(record: DatasetRecord) -> str:
    """Human-readable profile used for embedding + retrieval."""
    df = record.df
    lines = [
        f"Dataset: {record.name} ({record.row_count} rows, {record.col_count} columns).",
        f"Columns: {', '.join(c.name for c in record.columns)}",
    ]
    numeric_cols = numeric_columns(df)
    for c in record.columns:
        stats = _column_stats(df, c.name)
        lines.append(
            f"  - {c.name} [{c.dtype}]: "
            f"{'numeric' if c.numeric else 'text' if c.categorical else 'other'}, "
            f"{stats}"
        )
    # high-cardinality / id columns are mentioned once so retrieval can match them
    ids = [c.name for c in record.columns if c.categorical and c.unique_count > 100]
    if ids:
        lines.append(f"Identifier-like columns: {', '.join(ids)} (high cardinality, e.g. IDs).")
    # sample rows, compact
    sample = _sample_text(df, max_chars=400)
    if sample:
        lines.append("Sample rows:")
        lines.append(sample)
    # numeric summary for quick grounding
    if numeric_cols:
        summary = df[numeric_cols].describe().T.round(2)
        lines.append("Numeric summary:")
        for col in numeric_cols[:8]:
            row = summary.loc[col]
            lines.append(f"  - {col}: mean={row['mean']}, median={_median(df, col)}, std={row['std']}")
    return "\n".join(lines)


def profile_text_for_record(record: DatasetRecord) -> str:
    """Compatibility alias used when a DatasetRecord is available."""
    return build_table_profile(record)


def _column_stats(df: pd.DataFrame, col: str) -> str:
    s = df[col]
    nulls = int(s.isna().sum())
    uniques = int(s.nunique(dropna=True))
    base = f"{nulls} nulls, {uniques} unique"
    if pd.api.types.is_numeric_dtype(s):
        return f"{base}; min={_fmt(s.min())}, max={_fmt(s.max())}, mean={_fmt(s.mean())}"
    if is_string_series(s) and uniques <= 40:
        vals = ", ".join(str(v) for v in s.dropna().unique()[:12])
        return f"{base}; values: {vals}"
    if is_string_series(s):
        sample = ", ".join(str(v) for v in s.dropna().head(5))
        return f"{base}; e.g. {sample}"
    return base


def _median(df: pd.DataFrame, col: str):
    try:
        return _fmt(df[col].median())
    except Exception:
        return "n/a"


def _fmt(v) -> str:
    try:
        if pd.isna(v):
            return "n/a"
        f = float(v)
        return f"{f:,.2f}"
    except (TypeError, ValueError):
        return str(v)


def _sample_text(df: pd.DataFrame, max_chars: int = 400) -> str:
    """Render up to 4 sample rows as compact text."""
    head = df.head(4)
    lines = []
    for _, row in head.iterrows():
        cells = [f"{str(k)}={str(v)[:24]}" for k, v in row.items()][:8]
        lines.append("  | " + ", ".join(cells))
    text = "\n".join(lines)
    return text[:max_chars]
