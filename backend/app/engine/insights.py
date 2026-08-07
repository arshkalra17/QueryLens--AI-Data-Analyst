"""Automatic business insight + summary generation.

Every insight is a statement paired with the evidence used to derive it so the
agent can "show its work" when explaining reasoning.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..utils import guess_date_column, string_columns


@dataclass
class Insight:
    title: str
    detail: str
    category: str
    evidence: str
    importance: float  # 0..1

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "detail": self.detail,
            "category": self.category,
            "evidence": self.evidence,
            "importance": self.importance,
        }


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:,.2f}"
    return str(v)


def generate_insights(df: pd.DataFrame, dataset_name: str = "dataset", limit: int = 12) -> list[Insight]:
    """Return ranked insights for a dataframe."""
    insights: list[Insight] = []
    n_rows, n_cols = df.shape

    insights.append(
        Insight(
            title=f"Dataset overview",
            detail=f"{dataset_name} contains {n_rows:,} rows and {n_cols} columns.",
            category="overview", evidence=f"shape=({n_rows}, {n_cols})", importance=0.5,
        )
    )

    # Missing values
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing):
        worst = missing.idxmax()
        insights.append(
            Insight(
                title="Missing values present",
                detail=f"Column '{worst}' has {int(missing.max()):,} missing values ({missing.max()/n_rows*100:.1f}% of rows).",
                category="quality", evidence=f"missing={dict(missing.head(3))}", importance=0.7,
            )
        )

    # Categorical dominance
    for col in string_columns(df):
        vc = df[col].value_counts()
        if vc.empty or n_rows == 0:
            continue
        top, top_count = vc.index[0], vc.iloc[0]
        share = top_count / n_rows * 100
        if share >= 40:
            insights.append(
                Insight(
                    title=f"'{col}' is dominated by one category",
                    detail=f"'{top}' accounts for {share:.0f}% of rows ({top_count:,} of {n_rows:,}).",
                    category="distribution", evidence=f"top='{top}', share={share:.1f}%", importance=0.6,
                )
            )
            if len(insights) >= limit:
                break

    # Numeric columns: spread + extreme records
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols[:6]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        mean, std = s.mean(), s.std()
        spread = std / mean if mean not in (0, None) and mean == mean else None
        if spread is not None and spread > 1.0:
            insights.append(
                Insight(
                    title=f"High variability in '{col}'",
                    detail=(
                        f"'{col}' ranges from {_fmt(s.min())} to {_fmt(s.max())} "
                        f"(mean {_fmt(mean)}, std {_fmt(std)}), indicating outliers or extreme skew."
                    ),
                    category="distribution",
                    evidence=f"min={_fmt(s.min())}, max={_fmt(s.max())}, std={_fmt(std)}",
                    importance=0.65,
                )
            )

    # Correlations
    if len(num_cols) >= 2:
        corr = df[num_cols].corr().abs()
        pairs = []
        for i, a in enumerate(corr.columns):
            for b in corr.columns[i + 1 :]:
                v = corr.loc[a, b]
                if v > 0.7:
                    pairs.append((a, b, float(v)))
        pairs.sort(key=lambda t: -t[2])
        for a, b, v in pairs[:3]:
            insights.append(
                Insight(
                    title=f"Strong correlation: {a} ↔ {b}",
                    detail=f"'{a}' and '{b}' are correlated with r = {v:.2f}.",
                    category="correlation", evidence=f"pearson_r={v:.3f}", importance=0.7,
                )
            )

    # Time trend if a date column exists
    date_col = guess_date_column(df)
    if date_col is not None:
        try:
            tmp = df.copy()
            tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
            tmp = tmp.dropna(subset=[date_col])
            if len(tmp) and any(pd.api.types.is_numeric_dtype(tmp[c]) for c in num_cols):
                value_col = next(c for c in num_cols if pd.api.types.is_numeric_dtype(tmp[c]))
                tmp["_period"] = tmp[date_col].dt.to_period("M")
                monthly = tmp.groupby("_period")[value_col].mean()
                if len(monthly) >= 3:
                    first, last = monthly.iloc[0], monthly.iloc[-1]
                    growth = (last - first) / abs(first) * 100 if first != 0 else 0
                    best = monthly.idxmax()
                    insights.append(
                        Insight(
                            title="Monthly trend detected",
                            detail=(
                                f"'{value_col}' went from {_fmt(first)} ({monthly.index[0]}) to {_fmt(last)} "
                                f"({monthly.index[-1]}), a {growth:+.1f}% change; best period was {best} ({_fmt(monthly.max())})."
                            ),
                            category="trend",
                            evidence=f"first={_fmt(first)}, last={_fmt(last)}, growth={growth:.1f}%",
                            importance=0.8,
                        )
                    )
        except Exception:
            pass

    # Outliers summary
    try:
        from .anomalies import detect_anomalies

        flagged = detect_anomalies(df, max_results=20)
        if flagged:
            by_col: dict[str, int] = {}
            for f in flagged:
                by_col[f.column] = by_col.get(f.column, 0) + 1
            top_col = max(by_col, key=by_col.get)
            insights.append(
                Insight(
                    title="Outliers detected",
                    detail=(
                        f"Found {len(flagged)} anomalous records; '{top_col}' has the most flags "
                        f"({by_col[top_col]}). Use the anomaly detection tool for details."
                    ),
                    category="anomalies", evidence=f"count={len(flagged)}, by_column={by_col}", importance=0.75,
                )
            )
    except Exception:
        pass

    insights.sort(key=lambda i: -i.importance)
    return insights[:limit]


def summarize_dataset(df: pd.DataFrame, name: str) -> str:
    """Concise natural-language overview used in chat and reports."""
    insights = generate_insights(df, name, limit=6)
    lines = [f"**{name}**: {df.shape[0]:,} rows × {df.shape[1]:,} columns."]
    for ins in insights[1:]:
        lines.append(f"- {ins.detail}")
    return "\n".join(lines)
