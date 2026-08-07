"""Chat endpoints: blocking + streaming (SSE)."""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import StreamingResponse

from ...core.registry import get_registry
from ...core.sessions import get_sessions
from ...engine.agent import Agent
from ..schemas import ChatRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _get_agent() -> Agent:
    return Agent(get_registry(), get_sessions())


def _resolve_session(session_id: str | None, dataset_ids: list[str] | None = None) -> str:
    sessions = get_sessions()
    if session_id:
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Unknown session id: {session_id}")
        return session_id
    session = sessions.create(dataset_ids)
    return session.session_id


@router.post("")
def chat(payload: ChatRequest):
    """Blocking chat turn returning a structured answer."""
    session_id = _resolve_session(payload.session_id, payload.dataset_ids)
    sessions = get_sessions()
    for dataset_id in payload.dataset_ids:
        try:
            sessions.attach_dataset(session_id, dataset_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown dataset id: {dataset_id}")
    try:
        response = _get_agent().run(session_id, payload.message)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("chat failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")
    return response


@router.post("/stream")
async def chat_stream(payload: ChatRequest):
    """SSE endpoint streaming agent events: meta, plan, tool_*, token, final."""
    try:
        session_id = _resolve_session(payload.session_id, payload.dataset_ids)
    except HTTPException:
        raise
    sessions = get_sessions()
    for dataset_id in payload.dataset_ids:
        try:
            sessions.attach_dataset(session_id, dataset_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown dataset id: {dataset_id}")

    agent = _get_agent()

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            for event in agent.stream_run(session_id, payload.message):
                yield _sse(event)
        except Exception as exc:
            logger.exception("stream error")
            yield _sse({"event": "error", "data": {"message": f"Analysis failed: {exc}"}})

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _sse(event: dict) -> str:
    name = event.get("event", "message")
    data = event.get("data", {})
    return f"event: {name}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("/session")
def new_session(
    dataset_ids: str = "",
    body: dict[str, Any] = Body(default={}),
):
    """Create a fresh session (ids as query param or JSON body {dataset_ids: [...]})."""
    ids = body.get("dataset_ids") or [i for i in dataset_ids.split(",") if i]
    session = get_sessions().create(ids)
    return {"session_id": session.session_id, "dataset_ids": ids}


@router.post("/session/{session_id}/reset")
def reset_session(session_id: str):
    try:
        session = get_sessions().reset(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown session id: {session_id}")
    return {"session_id": session.session_id, "reset": True}


@router.get("/session/{session_id}/history")
def session_history(session_id: str):
    sessions = get_sessions()
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session id: {session_id}")
    return {"session_id": session_id, "messages": session.messages, "dataset_ids": session.dataset_ids, "pinned": session.pinned}
