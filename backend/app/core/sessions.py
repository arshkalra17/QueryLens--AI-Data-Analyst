"""Conversation / session context management.

A session stores the message history plus the set of datasets the user has
"attached" for analysis. Context (recent messages + schema summaries) is
rebuilt and injected into every prompt so the LLM/agent maintains state.

Sessions live in memory by default. With a ``db_path`` they are persisted to
SQLite and every read goes to the database, so conversations survive restarts
and are shared by multiple API workers (``SESSION_STORE=sqlite``, the app
default).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

from ..config import get_settings
from .registry import DatasetRecord


@dataclass
class Session:
    session_id: str
    created_at: float
    messages: list[dict] = field(default_factory=list)  # {"role","content","kind"}
    dataset_ids: list[str] = field(default_factory=list)
    pinned: str | None = None  # currently active dataset id (frontend selection)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    pinned      TEXT,
    dataset_ids TEXT NOT NULL,
    messages    TEXT NOT NULL
)
"""


class SessionManager:
    def __init__(
        self,
        max_sessions: int = 200,
        db_path: Path | str | None = None,
        max_messages: int = 200,
    ) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._max_sessions = max_sessions
        self._max_messages = max_messages
        self._db_path = Path(db_path) if db_path else None
        if self._db_path:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            with closing(self._connect()) as con, con:
                con.execute("PRAGMA journal_mode=WAL")
                con.execute(_SCHEMA)

    @property
    def persistent(self) -> bool:
        return self._db_path is not None

    def create(self, dataset_ids: list[str] | None = None) -> Session:
        session = Session(
            session_id=uuid.uuid4().hex[:12],
            created_at=time.time(),
            dataset_ids=[d for d in (dataset_ids or []) if d],
        )
        if self._db_path:
            self.save(session)
            self._evict_db()
            return session
        with self._lock:
            if len(self._sessions) >= self._max_sessions:
                self._sessions.pop(next(iter(self._sessions)))
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        if self._db_path:
            return self._load(session_id)
        with self._lock:
            return self._sessions.get(session_id)

    def save(self, session: Session) -> None:
        """Persist ``session`` (no-op for the in-memory store, where callers
        already hold the live object)."""
        if not self._db_path:
            return
        session.messages = session.messages[-self._max_messages:]
        with closing(self._connect()) as con, con:
            con.execute(
                "INSERT INTO sessions (session_id, created_at, updated_at, pinned, dataset_ids, messages)"
                " VALUES (?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(session_id) DO UPDATE SET updated_at=excluded.updated_at,"
                " pinned=excluded.pinned, dataset_ids=excluded.dataset_ids, messages=excluded.messages",
                (
                    session.session_id,
                    session.created_at,
                    time.time(),
                    session.pinned,
                    json.dumps(session.dataset_ids),
                    json.dumps(session.messages),
                ),
            )

    def require(self, session_id: str) -> Session:
        session = self.get(session_id)
        if session is None:
            raise KeyError(f"Unknown session id: {session_id}")
        return session

    def reset(self, session_id: str) -> Session:
        session = self.require(session_id)
        session.messages = []
        self.save(session)
        return session

    def attach_dataset(self, session_id: str, dataset_id: str) -> Session:
        session = self.require(session_id)
        if dataset_id not in session.dataset_ids:
            session.dataset_ids.append(dataset_id)
        session.pinned = dataset_id
        self.save(session)
        return session

    def add_message(self, session_id: str, role: str, content: str, kind: str = "text") -> Session:
        session = self.require(session_id)
        session.messages.append({"role": role, "content": content, "kind": kind, "ts": time.time()})
        self.save(session)
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


    # ------------------------------------------------------------------ sqlite
    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, timeout=5.0)

    def _load(self, session_id: str) -> Session | None:
        with closing(self._connect()) as con:
            row = con.execute(
                "SELECT session_id, created_at, pinned, dataset_ids, messages FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return Session(
            session_id=row[0],
            created_at=row[1],
            pinned=row[2],
            dataset_ids=json.loads(row[3]),
            messages=json.loads(row[4]),
        )

    def _evict_db(self) -> None:
        """Keep at most ``max_sessions`` rows, dropping the least recently used."""
        with closing(self._connect()) as con, con:
            con.execute(
                "DELETE FROM sessions WHERE session_id NOT IN"
                " (SELECT session_id FROM sessions ORDER BY updated_at DESC LIMIT ?)",
                (self._max_sessions,),
            )


_sessions: SessionManager | None = None


def get_sessions() -> SessionManager:
    global _sessions
    if _sessions is None:
        settings = get_settings()
        if settings.session_store == "sqlite":
            _sessions = SessionManager(max_sessions=settings.max_sessions, db_path=settings.session_db_path)
        else:
            _sessions = SessionManager(max_sessions=settings.max_sessions)
    return _sessions


def reset_sessions() -> None:
    global _sessions
    _sessions = None
