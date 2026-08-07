"""Chart generation (matplotlib) + an in-memory chart store with PNG endpoints."""
from __future__ import annotations

import base64
import io
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

CHART_KINDS = {"bar", "barh", "line", "pie", "scatter", "hist", "box", "area", "heatmap"}

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["font.size"] = 10


@dataclass
class ChartRecord:
    chart_id: str
    kind: str
    title: str
    dataset_ids: list[str]
    created_at: float
    caption: str
    png: bytes = field(repr=False)
    html: str = field(repr=False)
    plotly_json: dict | None = field(default=None, repr=False)

    def png_data_uri(self) -> str:
        return "data:image/png;base64," + base64.b64encode(self.png).decode()


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:,.2f}"
    return str(v)


class ChartStore:
    def __init__(self, max_charts: int = 200) -> None:
        self._charts: dict[str, ChartRecord] = {}
        self._lock = threading.Lock()
        self._max_charts = max_charts

    def put(self, record: ChartRecord) -> ChartRecord:
        with self._lock:
            if len(self._charts) >= self._max_charts:
                self._charts.pop(next(iter(self._charts)))
            self._charts[record.chart_id] = record
        return record

    def get(self, chart_id: str) -> ChartRecord | None:
        with self._lock:
            return self._charts.get(chart_id)


_chart_store: ChartStore | None = None


def get_chart_store() -> ChartStore:
    global _chart_store
    if _chart_store is None:
        _chart_store = ChartStore()
    return _chart_store


def reset_chart_store() -> None:
    global _chart_store
    _chart_store = None


def make_chart(
    df: pd.DataFrame,
    kind: str,
    x: str | None = None,
    y: str | None = None,
    title: str = "",
    agg: str | None = None,
    group_by: str | None = None,
    max_categories: int = 12,
) -> ChartRecord:
    """Build a chart from a dataframe. Returns a ChartRecord with PNG + HTML.

    ``agg`` can be "count" or an aggregation function name ("sum","mean","median","min","max").
    ``group_by`` splits numeric data by a categorical column (used for bar/line grouped views).
    """
    kind = kind.lower()
    if kind not in CHART_KINDS:
        raise ValueError(f"Unsupported chart kind '{kind}'. Choose from {sorted(CHART_KINDS)}")
    if df is None or len(df) == 0:
        raise ValueError("No data to chart: the dataset (or filtered result) is empty.")

    # SQL-ish y values like 'count(*)' / 'count(1)' mean "count rows per x category";
    # translate them to a count aggregation instead of crashing on a missing column.
    y_name = str(y).lower().strip() if y else None
    if y_name and re.search(r"^count\s*\(\s*(\*|1)\s*\)$", y_name):
        y = None
        agg = "count" if not agg else agg

    work = df.copy()

    # categorical x is chosen before aggregation so count/pivot charts can run
    # without needing any pre-existing numeric column.
    if x is None:
        x = _pick_categorical_col(work)
        if x is None and kind in ("bar", "barh", "pie"):
            # a "bar chart of X by country" may not name the dimension; prefer a
            # dimension-like column even when it is high cardinality (the chart
            # is truncated to max_categories anyway).
            x = _pick_dimension_col(work)

    # aggregate dense datetime x-axes (e.g. daily -> monthly) for trend charts
    if x and x in work.columns and pd.api.types.is_datetime64_any_dtype(work[x]):
        if kind in ("line", "area", "bar") and work[x].nunique() > 60:
            try:
                tmp = work.set_index(x)[y]
                tmp = pd.to_numeric(tmp, errors="coerce").resample("ME").mean().dropna()
                work = tmp.reset_index(name=y)
            except Exception:
                pass

    # "count rows per x category" applies when agg='count' was requested, OR when
    # there is no numeric column to plot at all (e.g. "customers by country" on a
    # CSV whose only numeric column is a row index). Counting is always a valid,
    # truthful aggregation; plotting a row index is not.
    count_requested = bool(agg and agg == "count")
    count_fallback = (
        not count_requested
        and y is None
        and x and x in work.columns
        and kind in ("bar", "barh", "pie", "line", "area")
        and not group_by
        and _pick_numeric_col(work) is None
        and not (agg and str(agg).lower() == "percent")
    )
    if count_requested or count_fallback:
        if x in work.columns:
            counts = work[x].astype(str).value_counts()
            work = counts.rename("count").reset_index()
            y = "count"
            agg = None

    # "percent" of total rows per x category: count per x / total * 100 (used by
    # "now show that as a percentage of total" follow-ups).
    if agg and str(agg).lower() == "percent":
        if x in work.columns:
            counts = work[x].astype(str).value_counts()
            pct = (counts / counts.sum() * 100).round(1).rename("percentage")
            work = pct.reset_index()
            y = "percentage"
            agg = None

    if y is None:
        y = _pick_numeric_col(work)
    if y is None:
        raise ValueError("No numeric column found to plot")

    if agg and x and agg != "count":
        numeric = _is_numeric(work[y])
        if numeric:
            try:
                work = work.groupby(work[x].astype(str), as_index=False)[y].agg(agg).sort_values(y, ascending=False)
            except Exception:
                agg = None

    # Caption statistics are computed over the FULL aggregated result (before
    # the max_categories truncation) so claims like "highest at ... / lowest at
    # ..." stay globally true even when the rendered chart shows only a top-N.
    caption_work = work.copy()

    if len(work) == 0:
        raise ValueError("No data to chart: the query produced no usable rows for this chart.")
    if y not in work.columns:
        raise ValueError(f"No data to chart: column {y!r} does not exist (got columns: {list(work.columns)}).")
    if not _is_numeric(work[y]):
        raise ValueError(f"No data to chart: column {y!r} is not numeric and cannot be plotted.")

    if x and work[x].nunique() > max_categories:
        top = work[x].value_counts().head(max_categories).index
        work = work[work[x].isin(top)]

    if kind == "scatter":
        x2 = x if _is_numeric(work[x]) else None
        y2 = y
    else:
        x2 = x
        y2 = y

    fig, ax = plt.subplots(figsize=(9, 5), dpi=110)

    if kind == "bar":
        if group_by:
            _grouped_bar(ax, work, x, y, group_by)
        else:
            ax.bar(work[x].astype(str).tolist(), pd.to_numeric(work[y], errors="coerce").fillna(0).tolist())
            ax.tick_params(axis="x", rotation=30 if work[x].nunique() > 6 else 0)
    elif kind == "barh":
        data = work.sort_values(y, ascending=True)
        ax.barh(data[x].astype(str).tolist(), pd.to_numeric(data[y], errors="coerce").fillna(0).tolist())
    elif kind == "line":
        for label, grp in work.groupby(group_by) if group_by else [("", work)]:
            if x is None:
                ax.plot(pd.to_numeric(grp[y], errors="coerce").reset_index(drop=True), label=str(label))
            else:
                xs = pd.to_numeric(grp[x], errors="coerce") if _is_numeric(grp[x]) else grp[x].astype(str).tolist()
                ax.plot(xs, pd.to_numeric(grp[y], errors="coerce"))
                if not _is_numeric(grp[x]):
                    ax.tick_params(axis="x", rotation=30)
                if group_by:
                    ax.legend(title=group_by, fontsize=8)
    elif kind == "area":
        vals = pd.to_numeric(work[y], errors="coerce").fillna(0)
        ax.fill_between(range(len(vals)), vals.values)
        if x:
            labels = work[x].astype(str).tolist()
            if len(labels) <= 20:
                ax.set_xticks(range(len(labels)), labels, rotation=30, fontsize=7)
    elif kind == "pie":
        counts = work[x].value_counts().head(10)
        ax.pie(counts.values, labels=counts.index.astype(str), autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
    elif kind == "scatter":
        if x2 is None:
            raise ValueError("Scatter chart needs a numeric x column")
        ax.scatter(pd.to_numeric(work[x2], errors="coerce"), pd.to_numeric(work[y2], errors="coerce"), s=28, alpha=0.65)
        ax.set_xlabel(str(x2))
        ax.set_ylabel(str(y))
    elif kind == "hist":
        values = pd.to_numeric(work[y], errors="coerce").dropna()
        ax.hist(values.values, bins=min(40, max(10, int(np.sqrt(len(values))))), edgecolor="white")
        ax.set_ylabel("Frequency")
    elif kind == "box":
        if group_by and group_by in work.columns:
            groups = [pd.to_numeric(g[y], errors="coerce").dropna().values for _, g in work.groupby(group_by)]
            labels = [str(k) for k, _ in work.groupby(group_by)]
            ax.boxplot([g.tolist() for g in groups], tick_labels=labels[:20])
            ax.tick_params(axis="x", rotation=30)
        else:
            values = pd.to_numeric(work[y], errors="coerce").dropna().values
            ax.boxplot(values)
            ax.set_xticks([1], [y])
    elif kind == "heatmap":
        numeric = work.select_dtypes(include=np.number)
        if numeric.shape[1] < 2:
            raise ValueError("Heatmap needs at least 2 numeric columns")
        corr = numeric.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", mask=mask, cbar_kws={"shrink": 0.8})
        ax.set_title(title or "Correlation heatmap")

    caption = _make_caption(caption_work, kind, x, y, group_by)
    if not title:
        title = caption or f"{kind.title()} of {y}"
    ax.set_title(title)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    png = buf.getvalue()
    html = _embed_png(png, title, caption)
    plotly_json = _plotly_spec(work, kind, x, y, group_by, title, caption)

    return ChartRecord(
        chart_id=uuid.uuid4().hex[:12],
        kind=kind,
        title=title,
        dataset_ids=[],
        created_at=time.time(),
        caption=caption,
        png=png,
        html=html,
        plotly_json=plotly_json,
    )


def _plotly_spec(work, kind, x, y, group_by, title, caption) -> dict | None:
    """Build an interactive Plotly figure spec (dict) for the given chart."""
    try:
        import plotly.graph_objects as go

        xv = work[x].astype(str).tolist() if x else None
        yv = pd.to_numeric(work[y], errors="coerce").fillna(0).tolist()
        fig = go.Figure()
        if kind == "pie":
            counts = work[x].value_counts().head(10)
            fig.add_trace(go.Pie(labels=counts.index.astype(str).tolist(), values=counts.values.tolist()))
        elif kind == "scatter":
            fig.add_trace(go.Scatter(x=xv, y=yv, mode="markers"))
        elif kind == "line" or kind == "area":
            if group_by:
                for label, grp in work.groupby(group_by):
                    fig.add_trace(go.Scatter(x=grp[x].astype(str).tolist(), y=grp[y].tolist(), mode="lines", name=str(label)))
            else:
                fig.add_trace(go.Scatter(x=xv, y=yv, mode="lines", fill="tozeroy" if kind == "area" else None))
        elif kind == "hist":
            fig.add_trace(go.Histogram(x=work[y].tolist()))
        elif kind == "box":
            if group_by:
                for label, grp in work.groupby(group_by):
                    fig.add_trace(go.Box(y=grp[y].tolist(), name=str(label)))
            else:
                fig.add_trace(go.Box(y=work[y].tolist(), name=y))
        elif kind == "bar" and group_by:
            pivot = work.pivot_table(index=x, columns=group_by, values=y, aggfunc="sum", fill_value=0)
            for col in pivot.columns[:8]:
                fig.add_trace(go.Bar(x=pivot.index.astype(str).tolist(), y=pivot[col].tolist(), name=str(col)))
        else:
            fig.add_trace(go.Bar(x=xv, y=yv, orientation="h" if kind == "barh" else None))
        fig.update_layout(title=title or caption, template="plotly_white", height=420, margin=dict(l=40, r=20, t=50, b=40))
        return fig.to_dict()
    except Exception:
        return None


def _grouped_bar(ax, work, x, y, group_by):
    categories = work[x].astype(str)
    groups = work[group_by].astype(str)
    frame = pd.DataFrame({"x": categories, "g": groups, "v": pd.to_numeric(work[y], errors="coerce").fillna(0)})
    pivot = frame.pivot_table(index="x", columns="g", values="v", aggfunc="sum", fill_value=0)
    pivot.head(12).plot(kind="bar", ax=ax, legend=len(pivot.columns) <= 8)
    ax.set_xlabel(str(x))
    ax.set_ylabel(str(y))
    if len(pivot.columns) <= 8:
        ax.legend(title=group_by, fontsize=8)
    ax.tick_params(axis="x", rotation=30)


def _make_caption(work, kind, x, y, group_by) -> str:
    try:
        if kind == "pie":
            counts = work[x].value_counts()
            top = counts.index[0]
            return f"Distribution of {x}: '{top}' is the largest slice ({_fmt(counts.iloc[0] / counts.sum() * 100)}%)."
        if kind == "scatter":
            return f"Scatter of {y} vs {x} showing the relationship between the two numeric columns."
        if kind == "hist":
            values = pd.to_numeric(work[y], errors="coerce").dropna()
            return f"Histogram of {y}: range [{_fmt(values.min())}, {_fmt(values.max())}], mean {_fmt(values.mean())}."
        if kind == "heatmap":
            return f"Correlation heatmap of numeric columns in the dataset."
        vals = pd.to_numeric(work[y], errors="coerce")
        if len(vals):
            if kind in ("bar", "barh", "line", "area"):
                top = vals.idxmax()
                top_label = work.loc[top, x] if x in work.columns else top
                return f"{y} by {x}: highest at '{top_label}' ({_fmt(vals.max())}), lowest at '{work.loc[vals.idxmin(), x] if x in work.columns else vals.idxmin()}' ({_fmt(vals.min())})."
            if kind == "box":
                return f"Box plot of {y} across {group_by or 'the dataset'} highlighting spread and outliers."
    except Exception:
        pass
    return f"{kind.title()} chart of {y}."


def _embed_png(png: bytes, title: str, caption: str) -> str:
    data_uri = "data:image/png;base64," + base64.b64encode(png).decode()
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Chart</title>"
        "<style>body{font-family:sans-serif;margin:0;padding:16px;background:#fff;color:#111}"
        "img{max-width:100%;height:auto;border:1px solid #e2e8f0;border-radius:8px}"
        ".cap{font-size:13px;color:#334155;margin-top:8px}</style></head><body>"
        f"<img src='{data_uri}' alt='{title}'/><div class='cap'>{caption}</div></body></html>"
    )


def _is_row_index(name: str) -> bool:
    n = str(name).lower().strip()
    return bool(
        re.match(
            r"^(index|idx|row(\s*(no|number|num|id)\.?)?|serial(\s*no\.?)?|"
            r"s\.?no\.?|position|seq(\.|uence)?|unnamed|#|rec(\s*no\.?)?)$",
            n,
        )
        or re.search(r"\bid$|_id$", n)
    )


def _pick_numeric_col(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if _is_numeric(df[col]) and not _is_row_index(col):
            return str(col)
    return None


def _pick_categorical_col(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if not _is_numeric(df[col]) and df[col].nunique(dropna=True) <= 50:
            return str(col)
    return None


_DIMENSION_KEYWORDS = ("country", "region", "product", "category", "city", "segment", "name", "status", "channel", "month", "state", "department")


def _pick_dimension_col(df: pd.DataFrame) -> str | None:
    """Prefer a natural grouping column for 'bar chart of X by Y' requests even
    when it has high cardinality (e.g. 240 countries), excluding row indices."""
    cands = [c for c in df.columns if not _is_numeric(df[c]) and not _is_row_index(c)]
    if not cands:
        return None
    for kw in _DIMENSION_KEYWORDS:
        for c in cands:
            if kw in str(c).lower():
                return str(c)
    return min(cands, key=lambda c: df[c].nunique(dropna=True))


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)
