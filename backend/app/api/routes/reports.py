"""Export a dataset as a human-readable Markdown/HTML report (insights,
quality checks, anomalies and charts inlined)."""
from __future__ import annotations

import base64
import html

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse

from ...core.registry import get_registry
from ...engine.anomalies import detect_anomalies
from ...engine.charts import make_chart
from ...engine.insights import generate_insights
from ...engine.quality import quality_report
from ..schemas import ReportRequest

router = APIRouter(prefix="/api/reports", tags=["reports"])


def _build_report(payload: ReportRequest) -> dict:
    registry = get_registry()
    record = registry.get(payload.dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown dataset id: {payload.dataset_id}")
    sections: dict[str, str] = {}
    charts: list[dict] = []

    overview = (
        f"# Data Report: {record.name}\n\n"
        f"- **Rows:** {record.row_count:,}\n- **Columns:** {record.col_count}\n"
        f"- **Columns:** {', '.join(c.name for c in record.columns)}\n"
    )
    sections["overview"] = overview

    if payload.include_quality:
        report = quality_report(record.df)
        lines = [f"## Data Quality: {report.score}/100", ""]
        for issue in report.issues:
            lines.append(f"- [{issue.severity}] {issue.message}")
        sections["quality"] = "\n".join(lines)

    if payload.include_insights:
        insights = generate_insights(record.df, record.name, limit=10)
        lines = ["## Insights", ""]
        for i in insights:
            lines.append(f"### {i.title}")
            lines.append(f"{i.detail}")
            lines.append(f"*Evidence: {i.evidence}*")
            lines.append("")
        sections["insights"] = "\n".join(lines)

    flags = detect_anomalies(record.df, max_results=20)
    if flags:
        lines = ["## Anomalies", "", "| Column | Value | Severity | Why |", "|---|---|---|---|"]
        for f in flags:
            lines.append(f"| {f.column} | {f.value} | {f.severity} | {f.explanation} |")
        sections["anomalies"] = "\n".join(lines)

    if payload.include_charts:
        num_cols = [c.name for c in record.columns if c.numeric]
        cat_cols = [c.name for c in record.columns if c.categorical]
        date_cols = [c.name for c in record.columns if c.datetime]
        attempts = []
        if cat_cols and num_cols:
            attempts.append(("bar", cat_cols[0], num_cols[0], None, f"Top {cat_cols[0]} by {num_cols[0]}"))
        if date_cols and num_cols:
            attempts.append(("line", date_cols[0], num_cols[0], None, f"{num_cols[0]} over time"))
        if len(num_cols) >= 2:
            attempts.append(("heatmap", None, None, None, "Correlation heatmap"))
        if cat_cols:
            attempts.append(("pie", cat_cols[0], None, None, f"{cat_cols[0]} distribution"))
        for kind, x, y, g, title in attempts[:4]:
            try:
                chart = make_chart(record.df, kind, x=x, y=y, title=title)
                charts.append({"kind": kind, "title": title, "png": chart.png, "caption": chart.caption})
            except Exception:
                continue

    sections["preview"] = record.df.head(10).to_string()
    return {"record": record, "sections": sections, "charts": charts}


@router.post("/{dataset_id}/markdown", response_class=PlainTextResponse)
def report_markdown(dataset_id: str):
    payload = ReportRequest(dataset_id=dataset_id)
    data = _build_report(payload)
    body = "\n\n".join(data["sections"].values())
    for ch in data["charts"]:
        body += f"\n\n![{ch['title']}](chart_{ch['kind']})"
    return body


@router.post("/{dataset_id}/html", response_class=HTMLResponse)
def report_html(dataset_id: str):
    payload = ReportRequest(dataset_id=dataset_id)
    data = _build_report(payload)
    chunks = []
    for name, md in data["sections"].items():
        chunks.append(f"<h2>{html.escape(name.title())}</h2><pre style='white-space:pre-wrap'>{html.escape(md)}</pre>")
    for ch in data["charts"]:
        chunks.append(
            f"<figure><img src='data:image/png;base64,{base64.b64encode(ch['png']).decode()}'/>"
            f"<figcaption>{html.escape(ch['caption'])}</figcaption></figure>"
        )
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Report</title>"
        "<style>body{font-family:sans-serif;max-width:900px;margin:24px auto;color:#0f172a}"
        "pre{background:#f8fafc;border:1px solid #e2e8f0;padding:12px;border-radius:8px;font-size:13px}"
        "img{max-width:100%}</style></head><body>"
        + "".join(chunks)
        + "</body></html>"
    )
