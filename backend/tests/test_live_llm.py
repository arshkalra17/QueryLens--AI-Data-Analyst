"""Live smoke tests against the real LLM provider configured in .env or the
environment (e.g. the GROQ_API_KEY repository secret in CI).

Excluded from the default run; execute with:

    pytest -m live_llm

Each test skips when no key is configured. They are deliberately small (a few
short requests in total) to stay within free-tier rate limits, and they never
print or log the key.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.config import get_settings
from app.core.llm import get_llm, reset_llm, usage

pytestmark = pytest.mark.live_llm


@pytest.fixture()
def live_llm(monkeypatch):
    from tests.conftest import LIVE_LLM_ENV

    for key, value in LIVE_LLM_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    get_settings.cache_clear()
    reset_llm()
    llm = get_llm()
    if not llm.available:
        pytest.skip("no LLM API key configured (set GROQ_API_KEY in .env or the environment)")
    yield llm
    monkeypatch.setenv("LLM_PROVIDER", "none")
    get_settings.cache_clear()
    reset_llm()


@pytest.fixture()
def live_agent(live_llm, registry, sessions):
    from app.engine.agent import Agent

    df = pd.DataFrame({
        "region": ["North", "South", "North", "East"],
        "revenue": [100.0, 250.0, 175.0, 90.0],
        "units": [1, 3, 2, 1],
    })
    rec = registry.register(df, "sales.csv", "test")
    session = sessions.create(dataset_ids=[rec.dataset_id])
    agent = Agent(registry, sessions)
    agent.llm = live_llm
    return agent, agent._build_ctx(session.session_id)


def test_live_plan_is_valid_tool_json(live_agent):
    from app.engine.tools import TOOL_MAP

    agent, ctx = live_agent
    plan = agent._plan_with_llm(ctx, "What is the total revenue by region?")
    assert isinstance(plan.get("tool_calls"), list) and plan["tool_calls"]
    assert all(call["name"] in TOOL_MAP for call in plan["tool_calls"])


def test_live_sql_repair_fixes_a_real_error(live_agent):
    agent, ctx = live_agent
    gen = agent._run_sql_with_repair(ctx, "total revenue by region",
                                     {"sql": 'SELECT "regon", SUM("revenu") FROM "sales_csv" GROUP BY 1'})
    events = []
    try:
        while True:
            events.append(next(gen))
    except StopIteration as stop:
        result = stop.value
    assert any(e["event"] == "note" for e in events), "the repair loop should have been used"
    assert result["row_count"] == 3


def test_live_document_answer_is_cited_and_grounded(live_llm, docs_store):
    from app.knowledge.answer import answer
    from app.knowledge.retrieval import search

    q = "What makes a retail purchase count as high value?"
    out = answer(q, search(q), llm=live_llm)
    assert out["found"] and out["mode"] == "llm"
    assert any(c["doc_name"] == "metric_definitions.md" for c in out["citations"])
    assert "10,000" in out["answer"] or "10000" in out["answer"]


def test_live_repeated_prompt_hits_cache(live_llm):
    before = usage.snapshot()
    messages = [{"role": "user", "content": "Reply with the single word: ok"}]
    first = live_llm.complete(messages, max_tokens=5, temperature=0.0)
    second = live_llm.complete(messages, max_tokens=5, temperature=0.0)
    after = usage.snapshot()
    assert first == second
    assert after["requests"] - before["requests"] == 1 and after["cache_hits"] - before["cache_hits"] == 1
