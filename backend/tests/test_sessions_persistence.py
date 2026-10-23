"""Tests for the SQLite-backed session store."""
from __future__ import annotations

from app.core.sessions import SessionManager


def test_session_survives_restart(tmp_path):
    db = tmp_path / "sessions.db"
    first = SessionManager(db_path=db)
    session = first.create(dataset_ids=["ds1"])
    first.attach_dataset(session.session_id, "ds2")
    first.add_message(session.session_id, "user", "top regions?", "question")
    first.add_message(session.session_id, "assistant", "North leads.", "answer")

    restarted = SessionManager(db_path=db)  # simulates a process restart
    loaded = restarted.require(session.session_id)

    assert loaded.dataset_ids == ["ds1", "ds2"]
    assert loaded.pinned == "ds2"
    assert [m["content"] for m in loaded.messages] == ["top regions?", "North leads."]


def test_workers_share_sessions(tmp_path):
    db = tmp_path / "sessions.db"
    worker_a, worker_b = SessionManager(db_path=db), SessionManager(db_path=db)

    session = worker_a.create()
    worker_b.add_message(session.session_id, "user", "hello")

    assert worker_a.require(session.session_id).messages[0]["content"] == "hello"


def test_reset_is_persisted(tmp_path):
    db = tmp_path / "sessions.db"
    mgr = SessionManager(db_path=db)
    session = mgr.create()
    mgr.add_message(session.session_id, "user", "hi")
    mgr.reset(session.session_id)

    assert SessionManager(db_path=db).require(session.session_id).messages == []


def test_save_persists_direct_mutation(tmp_path):
    mgr = SessionManager(db_path=tmp_path / "sessions.db")
    session = mgr.create()
    session.dataset_ids.append("ds9")
    mgr.save(session)

    assert mgr.require(session.session_id).dataset_ids == ["ds9"]


def test_oldest_sessions_evicted(tmp_path):
    mgr = SessionManager(max_sessions=3, db_path=tmp_path / "sessions.db")
    ids = [mgr.create().session_id for _ in range(5)]

    assert mgr.get(ids[0]) is None and mgr.get(ids[1]) is None
    assert all(mgr.get(i) is not None for i in ids[2:])


def test_message_history_is_capped(tmp_path):
    mgr = SessionManager(db_path=tmp_path / "sessions.db", max_messages=3)
    session = mgr.create()
    for i in range(5):
        mgr.add_message(session.session_id, "user", f"m{i}")

    assert [m["content"] for m in mgr.require(session.session_id).messages] == ["m2", "m3", "m4"]


def test_unknown_session_returns_none(tmp_path):
    assert SessionManager(db_path=tmp_path / "s.db").get("missing") is None


def test_memory_store_is_default():
    assert SessionManager().persistent is False
