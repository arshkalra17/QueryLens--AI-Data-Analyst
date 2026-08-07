"""Chart creation + image/html delivery."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, Response

from ...core.registry import get_registry
from ...engine.charts import CHART_KINDS, get_chart_store, make_chart
from ..schemas import ChartRequest, ChartResponse

router = APIRouter(prefix="/api/charts", tags=["charts"])


@router.post("", response_model=ChartResponse)
def create_chart(payload: ChartRequest):
    registry = get_registry()
    record = registry.get(payload.dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown dataset id: {payload.dataset_id}")
    if payload.kind not in CHART_KINDS:
        raise HTTPException(status_code=422, detail=f"Unsupported chart kind '{payload.kind}'")
    try:
        chart = make_chart(
            record.df,
            kind=payload.kind,
            x=payload.x,
            y=payload.y,
            title=payload.title,
            agg=payload.agg,
            group_by=payload.group_by,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not build chart: {exc}")
    chart.dataset_ids = [record.dataset_id]
    get_chart_store().put(chart)
    return ChartResponse(
        chart_id=chart.chart_id,
        kind=chart.kind,
        title=chart.title,
        caption=chart.caption,
        url=f"/api/charts/{chart.chart_id}/image",
        html_url=f"/api/charts/{chart.chart_id}/html",
        image=chart.png_data_uri(),
    )


@router.get("/{chart_id}/image")
def chart_image(chart_id: str):
    chart = get_chart_store().get(chart_id)
    if chart is None:
        raise HTTPException(status_code=404, detail="Chart not found (charts are ephemeral per session)")
    return Response(content=chart.png, media_type="image/png", headers={"Cache-Control": "no-cache"})


@router.get("/{chart_id}/html", response_class=HTMLResponse)
def chart_html(chart_id: str):
    chart = get_chart_store().get(chart_id)
    if chart is None:
        raise HTTPException(status_code=404, detail="Chart not found")
    return HTMLResponse(chart.html)


@router.get("/{chart_id}/plotly")
def chart_plotly(chart_id: str):
    """Interactive Plotly spec for the chart (used by the Streamlit UI)."""
    chart = get_chart_store().get(chart_id)
    if chart is None:
        raise HTTPException(status_code=404, detail="Chart not found")
    return {"chart_id": chart.chart_id, "kind": chart.kind, "spec": chart.plotly_json, "caption": chart.caption}


@router.get("/kinds")
def chart_kinds():
    return {"kinds": sorted(CHART_KINDS)}
