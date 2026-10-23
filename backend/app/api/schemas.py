"""Pydantic request/response models."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(..., min_length=1, max_length=4000)
    dataset_ids: list[str] = Field(default_factory=list, description="datasets to attach for this turn")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    reasoning: str
    sql: str | None = None
    pandas: str | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    chart_ids: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    provider: str
    citations: list[dict[str, Any]] = Field(default_factory=list, description="documents/definitions the answer relied on")
    route: str = Field("data", description="'data' (SQL/tools) or 'documents' (document RAG)")


class SessionResponse(BaseModel):
    session_id: str


class InsightRequest(BaseModel):
    dataset_id: str
    limit: int = 10


class InsightResponse(BaseModel):
    dataset: str
    insights: list[dict[str, Any]]


class ChartRequest(BaseModel):
    dataset_id: str
    kind: Literal["bar", "barh", "line", "pie", "scatter", "hist", "box", "area", "heatmap"]
    x: str | None = None
    y: str | None = None
    title: str = ""
    agg: str | None = None
    group_by: str | None = None


class ChartResponse(BaseModel):
    chart_id: str
    kind: str
    title: str
    caption: str
    url: str
    html_url: str
    image: str  # data uri


class QualityRequest(BaseModel):
    dataset_id: str


class ForecastRequest(BaseModel):
    dataset_id: str
    date_column: str | None = None
    value_column: str | None = None
    periods: int = 12
    seasonality: Literal["month", "dayofweek", "none"] = "month"


class DashboardRequest(BaseModel):
    dataset_ids: list[str]
    title: str = "Data Dashboard"


class ReportRequest(BaseModel):
    dataset_id: str
    include_insights: bool = True
    include_quality: bool = True
    include_charts: bool = True


class DeleteResponse(BaseModel):
    deleted: bool
    dataset_id: str
