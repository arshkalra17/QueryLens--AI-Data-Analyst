"""Auto-insight endpoint."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...core.registry import get_registry
from ...engine.insights import generate_insights
from ..schemas import InsightRequest, InsightResponse

router = APIRouter(prefix="/api/insights", tags=["insights"])


@router.post("", response_model=InsightResponse)
def insights(payload: InsightRequest):
    registry = get_registry()
    record = registry.get(payload.dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown dataset id: {payload.dataset_id}")
    return InsightResponse(
        dataset=record.name,
        insights=[i.to_dict() for i in generate_insights(record.df, record.name, limit=payload.limit)],
    )
