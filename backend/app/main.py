"""FastAPI application entrypoint.

Run with:  uvicorn app.main:app --reload
or:        python -m app.main
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import charts, chat, dashboards, datasets, insights, metrics, reports
from .config import PROJECT_ROOT, get_settings
from .core import logging_setup
from .core.llm import get_llm, reset_llm
from .core.logging_setup import request_context, timed, _current_request_id
from .core.registry import get_registry, reset_registry
from .core.security import check_auth, is_auth_enabled

logger = logging.getLogger(__name__)

FRONTEND_DIR = PROJECT_ROOT / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging_setup.setup_logging()
    loaded = get_registry().startup()
    if loaded:
        logger.info("Reloaded %d persisted dataset(s) from disk", loaded)
    llm = get_llm()
    if llm.available:
        desc = llm.describe()
        logger.info("LLM configured: %s", desc)
    else:
        logger.warning(
            "No LLM configured (provider=%s). Using the built-in rule engine. "
            "Set OPENAI_API_KEY or OPENAI_BASE_URL in .env to enable an LLM.",
            get_settings().llm_provider,
        )
    if get_settings().enable_rag:
        from .rag import get_rag

        try:
            indexed = get_rag().index_all()
            logger.info("RAG vector store ready (%d table profile(s) indexed)", indexed)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("RAG index_all failed: %s", exc)
    yield
    reset_registry()
    reset_llm()


app = FastAPI(
    title=get_settings().app_name,
    version=get_settings().app_version,
    lifespan=lifespan,
    description="AI-powered Data Analyst: upload CSVs and analyse them in natural language.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def auth_and_observability(request: Request, call_next):
    rid = request.headers.get("x-request-id")
    with request_context(rid):
        if is_auth_enabled() and request.url.path.startswith("/api"):
            try:
                check_auth(request)
            except HTTPException as exc:
                return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        with timed("http.request"):
            response = await call_next(request)
        response.headers["X-Request-Id"] = str(_current_request_id)
        return response


# API routes
app.include_router(datasets.router)
app.include_router(chat.router)
app.include_router(insights.router)
app.include_router(charts.router)
app.include_router(dashboards.router)
app.include_router(reports.router)
app.include_router(metrics.router)

# static sample data (used by the "load samples" button)
app.mount("/data", StaticFiles(directory=str(PROJECT_ROOT / "data")), name="data")


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/app/{path:path}")
def static_app(path: str):
    file_path = (FRONTEND_DIR / path).resolve()
    if file_path.is_file() and FRONTEND_DIR in file_path.parents:
        return FileResponse(file_path)
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=settings.debug)
