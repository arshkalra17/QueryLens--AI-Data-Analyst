"""Tests for the agent's SQL self-correction loop (LLM path, faked offline)."""
from __future__ import annotations

import pytest

from app.engine.agent import Agent
from app.engine.tools import ToolError


class FakeLLM:
    """Returns queued JSON replies; records the prompts it was sent."""

    available = True

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def extract_json(self, messages, max_tokens=None):
        self.prompts.append(messages[-1]["content"])
        return self.replies.pop(0)

    def describe(self):
        return {"active": "fake"}


def _drive(gen):
    """Run a generator to completion, returning (events, return value)."""
    events = []
    try:
        while True:
            events.append(next(gen))
    except StopIteration as stop:
        return events, stop.value


@pytest.fixture()
def agent_ctx(registry, sessions, simple_df):
    rec = registry.register(simple_df, "cities", "test")
    session = sessions.create(dataset_ids=[rec.dataset_id])
    agent = Agent(registry, sessions)
    return agent, agent._build_ctx(session.session_id)


def test_failed_sql_is_repaired_with_error_feedback(agent_ctx):
    agent, ctx = agent_ctx
    agent.llm = FakeLLM([{"sql": 'SELECT "city", SUM("value") AS total FROM "cities" GROUP BY "city"'}])

    events, result = _drive(agent._run_sql_with_repair(ctx, "total value per city", {"sql": 'SELECT SUM("vale") FROM "cities"'}))

    assert result["row_count"] == 3
    assert [e["event"] for e in events] == ["note"]
    # the DuckDB error and the failed query are both fed back to the model
    assert '"vale"' in agent.llm.prompts[0]
    assert "vale" in agent.llm.prompts[0].split("## DuckDB error")[1]


def test_repair_gives_up_after_configured_attempts(agent_ctx):
    agent, ctx = agent_ctx
    bad = [{"sql": f'SELECT "nope{i}" FROM "cities"'} for i in range(5)]
    agent.llm = FakeLLM(bad)

    with pytest.raises(ToolError):
        _drive(agent._run_sql_with_repair(ctx, "q", {"sql": 'SELECT "missing" FROM "cities"'}))
    assert len(agent.llm.prompts) == ctx["settings"].sql_repair_attempts


def test_repair_stops_when_model_returns_same_sql(agent_ctx):
    agent, ctx = agent_ctx
    sql = 'SELECT "missing" FROM "cities"'
    agent.llm = FakeLLM([{"sql": sql}])

    with pytest.raises(ToolError):
        _drive(agent._run_sql_with_repair(ctx, "q", {"sql": sql}))
    assert len(agent.llm.prompts) == 1


def test_valid_sql_never_calls_llm(agent_ctx):
    agent, ctx = agent_ctx
    agent.llm = FakeLLM([])

    events, result = _drive(agent._run_sql_with_repair(ctx, "q", {"sql": 'SELECT COUNT(*) AS n FROM "cities"'}))

    assert result["rows"][0]["n"] == 6
    assert events == [] and agent.llm.prompts == []


class _BrokenLLM:
    """Provider that is up but returns unusable plans (e.g. malformed JSON)."""

    available = True

    def extract_json(self, messages, max_tokens=None):
        from app.core.llm import LLMError

        raise LLMError("fake returned invalid JSON: Expecting ',' delimiter")

    def stream(self, messages, max_tokens=None, temperature=None):
        from app.core.llm import LLMError

        raise LLMError("fake stream failed")
        yield  # pragma: no cover

    def describe(self):
        return {"active": "fake"}


def test_agent_degrades_to_rule_engine_when_llm_plan_is_unusable(agent_ctx):
    agent, ctx = agent_ctx
    agent.llm = _BrokenLLM()
    events = list(agent.stream_run(ctx["session_id"], "Give me an analysis of value by city"))
    names = [e["event"] for e in events]
    assert "error" not in names and names[-1] == "final"
    assert any(e["event"] == "note" and "rule engine" in e["data"]["text"] for e in events)
    assert events[-1]["data"]["response"]["answer"]


def test_llm_usage_counters():
    from app.core.llm import LLMUsage

    u = LLMUsage()
    u.record(100.0, prompt_tokens=50, completion_tokens=10)
    u.record(300.0, prompt_tokens=20, completion_tokens=5)
    u.bump("rate_limit_retries")
    snap = u.snapshot()
    assert snap["requests"] == 2 and snap["total_tokens"] == 85 and snap["avg_latency_ms"] == 200.0
    assert snap["rate_limit_retries"] == 1 and set(snap) >= {"failures", "cache_hits"}


def test_chat_note_names_a_missing_model(agent_ctx):
    from app.core.llm import LLMError

    class _NoModel(_BrokenLLM):
        def extract_json(self, messages, max_tokens=None):
            raise LLMError("groq: model 'llama-3.1-8b-instant' not found on groq (404)")

    agent, ctx = agent_ctx
    agent.llm = _NoModel()
    notes = [e["data"]["text"] for e in agent.stream_run(ctx["session_id"], "Give me an analysis of value by city")
             if e["event"] == "note"]
    assert any("llama-3.1-8b-instant' not found" in n and "GROQ_MODEL" in n for n in notes)
