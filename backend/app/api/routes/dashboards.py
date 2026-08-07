"""Dashboard generation: a self-contained HTML page combining charts, tables,
insights and quality checks for one or more datasets."""
from __future__ import annotations

import base64
import html
import io
import threading
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from ...core.registry import get_registry
from ...engine.anomalies import detect_anomalies
from ...engine.charts import get_chart_store, make_chart
from ...engine.insights import generate_insights
from ...engine.quality import quality_report
from ..schemas import DashboardRequest

router = APIRouter(prefix="/api/dashboards", tags=["dashboards"])

_store: dict[str, str] = {}
_lock = threading.Lock()


@router.post("")
def create_dashboard(payload: DashboardRequest):
    registry = get_registry()
    if not payload.dataset_ids:
        raise HTTPException(status_code=422, detail="Select at least one dataset")
    sections: list[str] = []
    for dataset_id in payload.dataset_ids:
        record = registry.get(dataset_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Unknown dataset id: {dataset_id}")
        sections.append(_dataset_section(record))
    dashboard_id = uuid.uuid4().hex[:12]
    dashboard_html = _wrap(payload.title, "".join(sections))
    with _lock:
        if len(_store) >= 50:
            _store.pop(next(iter(_store)))
        _store[dashboard_id] = dashboard_html
    return {
        "dashboard_id": dashboard_id,
        "url": f"/api/dashboards/{dashboard_id}/html",
        "download_url": f"/api/dashboards/{dashboard_id}/html?download=1",
    }


@router.get("/{dashboard_id}/html", response_class=HTMLResponse)
def dashboard_html(dashboard_id: str, download: int = 0):
    with _lock:
        content = _store.get(dashboard_id)
    if content is None:
        raise HTTPException(status_code=404, detail="Dashboard not found")
    if download:
        return HTMLResponse(
            content,
            headers={"Content-Disposition": f'attachment; filename="dashboard-{dashboard_id}.html"'},
        )
    return HTMLResponse(content)


def _dataset_section(record) -> str:
    df = record.df
    num_cols = [c.name for c in record.columns if c.numeric]
    cat_cols = [c.name for c in record.columns if c.categorical]
    date_cols = [c.name for c in record.columns if c.datetime]

    blocks: list[str] = []
    blocks.append(f'<div class="ds-title">📊 {html.escape(record.name)} <span class="muted">{df.shape[0]:,} rows × {df.shape[1]} cols</span></div>')

    # key metric cards
    cards = []
    if num_cols:
        for c in num_cols[:4]:
            s = df[c]
            cards.append(_card(c, f"{s.mean():,.2f}", f"mean · min {s.min():,.2f} · max {s.max():,.2f}"))
    if date_cols:
        cards.append(_card("Date range", f"{df[date_cols[0]].min()}".split(" ")[0], f"→ {df[date_cols[0]].max()}".split(" ")[0]))
    if cards:
        blocks.append('<div class="cards">' + "".join(cards) + "</div>")

    # charts row 1
    chart_imgs = []
    if cat_cols and num_cols:
        try:
            chart_imgs.append(make_chart(df, "bar", x=cat_cols[0], y=num_cols[0], agg="sum", title=f"Top {cat_cols[0]} by {num_cols[0]}"))
        except Exception:
            pass
    if date_cols and num_cols:
        try:
            chart_imgs.append(make_chart(df, "line", x=date_cols[0], y=num_cols[0], title=f"{num_cols[0]} over time"))
        except Exception:
            pass
    if cat_cols:
        try:
            chart_imgs.append(make_chart(df, "pie", x=cat_cols[0], title=f"{cat_cols[0]} distribution"))
        except Exception:
            pass
    if len(num_cols) >= 2:
        try:
            chart_imgs.append(make_chart(df, "heatmap", title="Correlations"))
        except Exception:
            pass
    for ch in chart_imgs[:4]:
        get_chart_store().put(ch)
        blocks.append(f'<div class="chart"><img src="data:image/png;base64,{base64.b64encode(ch.png).decode()}"/><div class="cap">{html.escape(ch.caption)}</div></div>')

    # insights + quality + anomalies
    quality = quality_report(df)
    blocks.append(
        f'<div class="qbox">Data quality score: <b>{quality.score}/100</b>'
        + (f"<br>⚠ {html.escape(quality.issues[0].message)}" if quality.issues else "")
        + "</div>"
    )
    insights = generate_insights(df, record.name, limit=6)
    if insights:
        items = "".join(f"<li><b>{html.escape(i.title)}</b> — {html.escape(i.detail)}</li>" for i in insights)
        blocks.append(f"<div class='panel'><h4>Insights</h4><ul>{items}</ul></div>")

    flags = detect_anomalies(df, max_results=8)
    if flags:
        rows = "".join(
            f"<tr><td>{html.escape(f.column)}</td><td>{html.escape(str(f.value))}</td><td>{f.severity}</td>"
            f"<td>{html.escape(f.explanation)}</td></tr>"
            for f in flags
        )
        blocks.append(f"<div class='panel'><h4>⚠ Anomalies ({len(flags)} flagged)</h4><table><tr><th>Column</th><th>Value</th><th>Severity</th><th>Why</th></tr>{rows}</table></div>")

    # preview table
    preview = df.head(8).to_html(classes="preview", border=0, index=False)
    blocks.append(f"<div class='panel'><h4>Preview</h4>{preview}</div>")
    return '<div class="ds">' + "".join(blocks) + "</div>"


def _card(label: str, value: str, sub: str) -> str:
    return (
        f'<div class="card"><div class="clabel">{html.escape(label)}</div>'
        f'<div class="cvalue">{html.escape(value)}</div><div class="csub">{html.escape(sub)}</div></div>'
    )


def _wrap(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
  body{{font-family:-apple-system,'Segoe UI',Roboto,sans-serif;background:#f1f5f9;color:#0f172a;margin:0;padding:24px}}
  h1{{font-size:22px;margin:0 0 4px}} .muted{{color:#64748b;font-weight:400;font-size:13px}}
  .ds{{background:#fff;border-radius:12px;padding:20px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
  .ds-title{{font-size:17px;font-weight:700;margin-bottom:12px}}
  .cards{{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}}
  .card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:10px 14px;min-width:150px}}
  .clabel{{font-size:12px;color:#64748b}} .cvalue{{font-size:20px;font-weight:700}} .csub{{font-size:11px;color:#94a3b8}}
  .chart{{margin:10px 0}} .chart img{{max-width:100%;border:1px solid #e2e8f0;border-radius:8px}}
  .cap{{font-size:12px;color:#334155;margin-top:4px}}
  .qbox{{background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:10px;margin:8px 0}}
  .panel{{margin:10px 0}} .panel h4{{margin:6px 0}} .panel li{{font-size:13px;margin:3px 0}}
  table{{border-collapse:collapse;width:100%;font-size:12px}} th,td{{border:1px solid #e2e8f0;padding:5px 8px;text-align:left}}
  th{{background:#f1f5f9}} .preview{{width:100%}}
  @media print{{body{{background:#fff}} .ds{{box-shadow:none;border:1px solid #ddd}}}}
</style></head><body><h1>{html.escape(title)}</h1>
<div class="muted">Generated by AI Data Analyst · {time.strftime('%Y-%m-%d %H:%M')}</div>
{body}
</body></html>"""
