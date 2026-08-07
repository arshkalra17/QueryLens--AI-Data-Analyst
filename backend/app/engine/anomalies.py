"""Anomaly detection with human-readable explanations.

Methods:
- ``zscore``  : |z| > threshold (default 3) using mean/std
- ``iqr``     : outside Q1 - 1.5*IQR .. Q3 + 1.5*IQR
- ``rolling`` : time-series spike detection, |value - rolling_median| > 3*rolling_mad
               (or 3*rolling_std), requires a date column
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..utils import guess_date_column

logger = logging.getLogger(__name__)

METHODS = {"zscore", "iqr", "rolling"}


@dataclass
class AnomalyRow:
    column: str
    row_index: int
    group: str | None
    value: Any
    method: str
    score: float
    bounds: tuple[Any, Any] | None
    severity: str
    explanation: str


def detect_anomalies(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    group_by: str | None = None,
    date_column: str | None = None,
    method: str = "auto",
    z_threshold: float = 3.0,
    iqr_factor: float = 1.5,
    max_results: int = 100,
) -> list[AnomalyRow]:
    """Return flagged anomalies with per-row explanations.

    ``method``: "zscore" | "iqr" | "rolling" | "auto".
    When ``group_by`` is given, statistics are computed within each group so a
    value is only anomalous relative to its own group.
    """
    if method not in METHODS and method != "auto":
        raise ValueError(f"Unknown method '{method}'. Use {sorted(METHODS)} or 'auto'")
    if df.empty:
        return []

    numeric_cols = [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if columns:
        numeric_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return []

    used_method = method
    if method == "auto":
        used_method = "rolling" if (date_column or guess_date_column(df)) is not None else "zscore"

    groups = [(None, df)]
    if group_by and group_by in df.columns:
        groups = [(str(k), g) for k, g in df.groupby(group_by, dropna=False)]

    results: list[AnomalyRow] = []
    for group_name, gdf in groups:
        for col in numeric_cols:
            if used_method == "rolling":
                results.extend(_rolling_flags(gdf, col, group_name, date_column, z_threshold))
            elif used_method == "iqr":
                results.extend(_iqr_flags(gdf, col, group_name, iqr_factor))
            else:
                results.extend(_zscore_flags(gdf, col, group_name, z_threshold))

    results.sort(key=lambda r: -abs(r.score))
    for r in results:
        r.severity = _severity(abs(r.score))
    return results[:max_results]


def _guess_date_column(df: pd.DataFrame) -> str | None:
    return guess_date_column(df)


def _zscore_flags(gdf: pd.DataFrame, col: str, group: str | None, threshold: float) -> list[AnomalyRow]:
    vals = pd.to_numeric(gdf[col], errors="coerce")
    mean, std = vals.mean(), vals.std()
    out: list[AnomalyRow] = []
    if not np.isfinite(std) or std == 0:
        return out
    z = (vals - mean) / std
    for i in vals.index:
        zz = float(z.loc[i])
        if abs(zz) >= threshold and np.isfinite(zz):
            out.append(
                AnomalyRow(
                    column=col, row_index=int(i), group=group, value=_val(vals.loc[i]),
                    method="zscore", score=zz,
                    bounds=(round(float(mean - threshold * std), 4), round(float(mean + threshold * std), 4)),
                    severity="", explanation=(
                        f"value {_fmt(vals.loc[i])} is {abs(zz):.1f} standard deviations "
                        f"from the {'group' if group else 'overall'} mean {_fmt(mean)} "
                        f"(|z|={abs(zz):.2f} >= {threshold})"
                    ),
                )
            )
    return out


def _iqr_flags(gdf: pd.DataFrame, col: str, group: str | None, factor: float) -> list[AnomalyRow]:
    vals = pd.to_numeric(gdf[col], errors="coerce")
    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
    iqr = q3 - q1
    out: list[AnomalyRow] = []
    if not np.isfinite(iqr) or iqr == 0:
        return out
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    for i in vals.index:
        v = vals.loc[i]
        if pd.isna(v):
            continue
        if v < lower or v > upper:
            outside = "below" if v < lower else "above"
            out.append(
                AnomalyRow(
                    column=col, row_index=int(i), group=group, value=_val(v),
                    method="iqr", score=float(abs(v - (upper if v > upper else lower)) / iqr),
                    bounds=(round(float(lower), 4), round(float(upper), 4)),
                    severity="",
                    explanation=(
                        f"value {_fmt(v)} falls {outside} the IQR fence "
                        f"[{_fmt(lower)}, {_fmt(upper)}] (Q1={_fmt(q1)}, Q3={_fmt(q3)}, "
                        f"IQR={_fmt(iqr)}, factor={factor})"
                    ),
                )
            )
    return out


def _rolling_flags(
    gdf: pd.DataFrame, col: str, group: str | None, date_column: str | None, threshold: float
) -> list[AnomalyRow]:
    work = gdf.copy()
    dc = date_column or _guess_date_column(work)
    if dc is None:
        return _zscore_flags(gdf, col, group, threshold)
    work[dc] = pd.to_datetime(work[dc], errors="coerce")
    work = work.dropna(subset=[dc]).sort_values(dc)
    vals = pd.to_numeric(work[col], errors="coerce")
    window = max(7, min(30, int(len(vals) / 5)))
    roll_med = vals.rolling(window, min_periods=3, center=True).median()
    roll_std = vals.rolling(window, min_periods=3, center=True).std()
    out: list[AnomalyRow] = []
    for i in vals.index:
        v = vals.loc[i]
        med, std = roll_med.get(i), roll_std.get(i)
        if pd.isna(v) or pd.isna(med) or pd.isna(std) or std == 0:
            continue
        z = abs(v - med) / std
        if z >= threshold:
            ts = work.loc[i, dc]
            out.append(
                AnomalyRow(
                    column=col, row_index=int(i), group=group, value=_val(v),
                    method="rolling", score=float(z),
                    bounds=(round(float(med - threshold * std), 4), round(float(med + threshold * std), 4)),
                    severity="",
                    explanation=(
                        f"at {pd.Timestamp(ts).strftime('%Y-%m-%d')} value {_fmt(v)} deviates "
                        f"{z:.1f}x the rolling std from the {window}-period rolling median {_fmt(med)}"
                    ),
                )
            )
    return out


def _severity(abs_score: float) -> str:
    if abs_score >= 5:
        return "critical"
    if abs_score >= 3:
        return "high"
    return "moderate"


def _fmt(v: Any) -> str:
    try:
        if isinstance(v, float):
            return f"{v:,.2f}"
        return str(v)
    except Exception:
        return str(v)


def _val(v: Any) -> Any:
    try:
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
    except Exception:
        pass
    return v
