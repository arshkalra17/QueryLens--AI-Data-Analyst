"""Shared pandas helpers (pandas 2 & 3 compatible string handling)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def is_string_series(series: pd.Series) -> bool:
    """True for text columns regardless of 'object' or the pandas 3 'str' dtype."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return False
    try:
        if pd.api.types.is_string_dtype(series) and not pd.api.types.is_numeric_dtype(series):
            return True
    except TypeError:
        pass
    return pd.api.types.is_object_dtype(series)


def string_columns(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.columns if is_string_series(df[c])]


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def guess_date_column(df: pd.DataFrame) -> str | None:
    """Find a column that looks like a datetime (parses cleanly)."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return str(col)
        if is_string_series(df[col]) and df[col].nunique(dropna=True) > 10:
            try:
                if pd.to_datetime(df[col], errors="coerce").notna().mean() > 0.9:
                    return str(col)
            except Exception:
                continue
    return None


def to_json_safe(value):
    """Convert a scalar (possibly numpy/pandas typed) to a JSON-safe python value."""
    try:
        if value is None:
            return None
        if pd.isna(value):
            return None
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            f = float(value)
            return None if f != f else f
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (pd.Timestamp, np.datetime64)):
            return pd.Timestamp(value).isoformat()
        if isinstance(value, (pd.Timedelta,)):
            return str(value)
        if isinstance(value, (np.ndarray, pd.Series)):
            return [to_json_safe(v) for v in value.tolist()]
        if isinstance(value, float):
            return None if value != value else value
        return value
    except Exception:
        return str(value)
