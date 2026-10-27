"""Health + observability endpoints."""
from __future__ import annotations

import time

from fastapi import APIRouter

from ...config import get_settings
from ...core.caching import TTLCache
from ...core.llm import get_llm
from ...core.logging_setup import metrics
from ...core.registry import get_registry

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health():
    started_at = getattr(health, "_started", time.time())
    registry = get_registry()
    return {
        "status": "ok",
        "app": get_settings().app_name,
        "version": get_settings().app_version,
        "uptime_seconds": round(time.time() - started_at, 1),
        "llm": get_llm().describe(),
        "datasets_loaded": len(registry.list()),
        "cache": {"enabled": get_settings().enable_cache},
        "demo": {"enabled": get_settings().demo_mode, "rate_limit_per_hour": get_settings().rate_limit_per_hour},
    }


@router.get("/metrics")
def get_metrics():
    return metrics.snapshot()


@router.get("/cache/stats")
def cache_stats():
    return {"cache": {"enabled": get_settings().enable_cache, "ttl": get_settings().cache_ttl_seconds}}
