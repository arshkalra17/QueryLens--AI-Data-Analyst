"""Conversation / session context management.

A session stores the message history plus the set of datasets the user has
"attached" for analysis. Context (recent messages + schema summaries) is
rebuilt and injected into every prompt so the LLM/agent maintains state.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

from ..config import get_settings
from .registry import DatasetRecord


@dataclass
class Session:
    session_id: str
    created_at: float
    messages: list[dict] = field(default_factory=list)  # {"role","content","kind"}
    dataset_ids: list[str] = field(default_factory=list)
    pinned: str | None = None  # currently active dataset id (frontend selection)


class SessionManager:
    def __init__(self, max_sessions: int = 200) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._max_sessions = max_sessions

    def create(self, dataset_ids: list[str] | None = None) -> Session:
        session = Session(
            session_id=uuid.uuid4().hex[:12],
            created_at=time.time(),
            dataset_ids=[d for d in (dataset_ids or []) if d],
        )
        with self._lock:
            if len(self._sessions) >= self._max_sessions:
                self._sessions.pop(next(iter(self._sessions)))
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def require(self, session_id: str) -> Session:
        session = self.get(session_id)
        if session is None:
            raise KeyError(f"Unknown session id: {session_id}")
        return session

    def reset(self, session_id: str) -> Session:
        session = self.require(session_id)
        session.messages = []
        return session

    def attach_dataset(self, session_id: str, dataset_id: str) -> Session:
        session = self.require(session_id)
        if dataset_id not in session.dataset_ids:
            session.dataset_ids.append(dataset_id)
        session.pinned = dataset_id
        return session

    def add_message(self, session_id: str, role: str, content: str, kind: str = "text") -> Session:
        session = self.require(session_id)
        session.messages.append({"role": role, "content": content, "kind": kind, "ts": time.time()})
        return session

    def context(self, session_id: str, registry, max_window: int | None = None) -> dict:
        """Assemble the context handed to the agent: recent history + schemas."""
        settings = get_settings()
        session = self.require(session_id)
        window = max_window or settings.conversation_window
        history = session.messages[-window:]

        records: list[DatasetRecord] = []
        seen: set[str] = set()
        ids = []
        if session.pinned:
            ids.append(session.pinned)
        ids += [d for d in session.dataset_ids if d != session.pinned]
        for dataset_id in ids:
            rec = registry.get(dataset_id)
            if rec is not None and rec.dataset_id not in seen:
                seen.add(rec.dataset_id)
                records.append(rec)

        schemas = "\n\n".join(r.schema_snippet() for r in records) or "(no datasets attached)"
        return {
            "session_id": session_id,
            "history": history,
            "records": records,
            "schemas": schemas,
            "dataset_ids": [r.dataset_id for r in records],
        }


_sessions: SessionManager | None = None


def get_sessions() -> SessionManager:
    global _sessions
    if _sessions is None:
        _sessions = SessionManager()
    return _sessions


def reset_sessions() -> None:
    global _sessions
    _sessions = None
