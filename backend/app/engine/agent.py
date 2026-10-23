"""Analyst agent: plans tool calls, executes them, then produces an answer.

Two backends:
- LLM path: OpenAI-compatible chat completions. First pass asks the model for a
  JSON plan (tool calls); tools run server-side; the final answer is streamed
  token-by-token from the model.
- Fallback path: the rule-based engine (`fallback.py`), used when no LLM is
  configured or the LLM call fails. Fully functional without an API key.

``_turn`` is a generator of SSE-style event dicts. ``run`` consumes it for
blocking use; ``stream_run`` exposes it directly for a streaming endpoint.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Iterator

from ..config import get_settings
from ..core.llm import LLMError, get_llm
from ..core.logging_setup import metrics, timed
from ..core.sessions import SessionManager, get_sessions
from ..engine.charts import get_chart_store
from ..prompts import answer_final_prompt, sql_repair_prompt, system_prompt, tool_results_block
from . import conditions
from .tools import TOOL_MAP, ToolError, run_tool

logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, registry, sessions: SessionManager | None = None) -> None:
        self.registry = registry
        self.sessions = sessions or get_sessions()
        self.llm = get_llm()

    # ------------------------------------------------------------------
    def run(self, session_id: str, question: str) -> dict:
        """Blocking turn; returns the structured response dict."""
        response: dict | None = None
        for event in self._turn(session_id, question):
            if event.get("event") == "final":
                response = event.get("data", {}).get("response", {})
        return response or {"answer": "No response produced.", "tools": [], "chart_ids": [], "datasets": [], "errors": []}

    def stream_run(self, session_id: str, question: str) -> Iterator[dict]:
        """Generator of SSE event dicts (meta/note/plan/tool_*/token/final/error)."""
        yield from self._turn(session_id, question)

    # ------------------------------------------------------------------
    def _turn(self, session_id: str, question: str) -> Iterator[dict]:
        if not question.strip():
            yield _ev("error", {"message": "Question is empty"})
            return

        records = self.registry.list()
        if not records:
            yield _ev("error", {"message": "No datasets are loaded. Upload a CSV file first."})
            return

        ctx = self._build_ctx(session_id)
        yield _ev("meta", {"provider": self.llm.describe().get("active", "fallback") if self.llm.available else "fallback", "dataset_ids": ctx["dataset_ids"]})

        with timed("agent.total"):
            from . import fallback

            # 1) plan tool calls
            plan: dict | None = None
            llm_failed = False
            # Deterministic intents: "show the first N rows", "how many X?" and
            # list/filter requests ("list customers from India") must always
            # produce a correct, complete answer regardless of model quality or
            # rate limits. Bypass LLM planning/narration for them.
            forced_deterministic = (
                fallback._preview_limit(question.lower()) is not None
                or fallback._count_intent(question.lower())
                or fallback._filter_intent(question.lower())
                or fallback._schema_intent(question.lower())
                or fallback._has_any(question.lower(), fallback.TOP_MARKERS + fallback.BOTTOM_MARKERS)
                or fallback._duplicate_intent(question.lower())
                or fallback._percentage_followup(question.lower(), ctx)
            )
            if forced_deterministic:
                plan = fallback.plan(ctx, question)
            elif self.llm.available:
                try:
                    with timed("agent.plan"):
                        plan = self._plan_with_llm(ctx, question)
                except LLMError as exc:
                    logger.warning("LLM planning failed, using rule engine: %s", exc)
                    llm_failed = True
            if plan is None:
                if llm_failed:
                    yield _ev("note", {"text": "LLM call failed — falling back to the built-in rule engine."})
                plan = fallback.plan(ctx, question)

            yield _ev("plan", {"tools": [c["name"] for c in plan.get("tool_calls", [])]})

            # 1.5) audit generated SQL: a question with an explicit condition
            # ("in 2021", "where Country = India") must not run as a bare
            # aggregate over the whole table. Inject the missing WHERE when we
            # can ground it, otherwise flag it so it is surfaced, not silent.
            condition_warnings: list[str] = []
            pinned_record = ctx.get("pinned") and self.registry.get(ctx["pinned"]) or None
            if pinned_record is None:
                records = self.registry.list()
                pinned_record = records[0] if records else None
            for call in plan.get("tool_calls", []):
                if call.get("name") not in ("run_sql", "generate_sql"):
                    continue
                args = call.get("args") or {}
                sql = args.get("sql") or args.get("query")
                if not sql or conditions.sql_has_where(sql) or not conditions.has_condition(question.lower()):
                    continue
                cond = conditions.extract_where(question, pinned_record)
                if cond:
                    fixed = conditions.inject_where(sql, cond)
                    if fixed:
                        call["args"]["sql"] = fixed
                        call["args"].pop("query", None)
                        logger.warning("Question condition missing from generated SQL; injected WHERE %s", cond)
                        yield _ev("note", {"text": f"Detected a condition in your question and added it to the query (`WHERE {cond}`)."})
                        continue
                condition_warnings.append(conditions.audit_sql(question, pinned_record, sql) or conditions.has_condition(question.lower()) and "Your question included an explicit condition, but the generated query had no WHERE clause.")
            if condition_warnings:
                yield _ev("note", {"text": condition_warnings[0]})

            # 2) execute tools
            results: list[dict] = []
            errors: list[str] = []
            for call in plan.get("tool_calls", []):
                name, args = call.get("name"), call.get("args", {}) or {}
                yield _ev("tool_start", {"name": name})
                try:
                    if name == "run_sql" and not forced_deterministic and self.llm.available:
                        result = yield from self._run_sql_with_repair(ctx, question, args)
                    else:
                        result = run_tool(ctx, name, args)
                    results.append(result)
                    yield _ev("tool_done", {"name": name, "summary": _tool_summary(result)})
                except ToolError as exc:
                    results.append({"error": str(exc)})
                    errors.append(f"{name}: {exc}")
                    logger.warning("tool %s failed (args=%s): %s", name, args, exc)
                    yield _ev("tool_error", {"name": name, "error": str(exc)})
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("tool %s crashed", name)
                    results.append({"error": str(exc)})
                    errors.append(f"{name}: {exc}")
                    yield _ev("tool_error", {"name": name, "error": str(exc)})

            # 3) final answer
            answer_payload: dict = {}
            if forced_deterministic:
                answer_payload = self._answer_with_fallback(ctx, question, plan, results)
                for token in _stream_words(answer_payload.get("answer", "")):
                    yield _ev("token", {"text": token})
            elif self.llm.available and not errors:
                try:
                    with timed("agent.answer"):
                        for ev in self._answer_with_llm(ctx, question, plan, results):
                            yield ev
                        answer_payload = self._last_answer_payload
                except LLMError as exc:
                    logger.warning("LLM answer failed, using rule engine: %s", exc)
                    answer_payload = self._answer_with_fallback(ctx, question, plan, results)
            else:
                answer_payload = self._answer_with_fallback(ctx, question, plan, results)
                for token in _stream_words(answer_payload.get("answer", "")):
                    yield _ev("token", {"text": token})

            answer = answer_payload.get("answer") or "I could not produce an answer for that request."
            reasoning = answer_payload.get("reasoning") or ""
            sql = answer_payload.get("sql")
            pandas_code = answer_payload.get("pandas")

            # surface any condition-dropping warnings so an unfiltered answer is
            # never returned silently
            if condition_warnings:
                answer = answer + "\n\n" + "\n\n".join(f"> ⚠ {w}" for w in condition_warnings)
                reasoning = (reasoning + " " if reasoning else "") + " ".join(condition_warnings)

            # 4) persist conversation context
            session = self.sessions.require(session_id)
            for d in ctx.get("dataset_ids", []):
                if d not in session.dataset_ids:
                    session.dataset_ids.append(d)
            self.sessions.add_message(session_id, "user", question, "question")
            self.sessions.add_message(session_id, "assistant", answer, "answer")
            if ctx.get("rag"):
                try:
                    ctx["rag"].remember(session_id, question, sql, answer[:400])
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("RAG remember failed: %s", exc)

            response = {
                "session_id": session_id,
                "answer": answer,
                "reasoning": reasoning,
                "sql": sql,
                "pandas": pandas_code,
                "tools": [{"name": c["name"], "status": "ok"} for c in plan.get("tool_calls", [])],
                "chart_ids": answer_payload.get("chart_ids", []),
                "datasets": [r.dataset_id for r in records],
                "errors": errors,
                "provider": self.llm.describe().get("active", "fallback") if self.llm.available else "fallback",
            }
            metrics.inc("agent.turns")
            yield _ev("final", {"response": response})

    # ------------------------------------------------------------------
    def _build_ctx(self, session_id: str) -> dict:
        session = self.sessions.require(session_id)
        context = self.sessions.context(session_id, self.registry)
        return {
            "registry": self.registry,
            "sessions": self.sessions,
            "session_id": session_id,
            "dataset_ids": context["dataset_ids"],
            "schemas": context["schemas"],
            "history": context["history"],
            "settings": get_settings(),
            "chart_store": get_chart_store(),
            "rag": _get_rag_or_none(),
            "pinned": session.pinned,
            "metrics": [],
        }

    # ------------------------------------------------------------------
    def _run_sql_with_repair(self, ctx: dict, question: str, args: dict) -> Iterator[dict]:
        """Run LLM-generated SQL; on a DuckDB error, feed the error back to the
        model and retry up to ``sql_repair_attempts`` times. Generator that
        yields ``note`` events and returns the tool result (raises the last
        ToolError if every attempt fails)."""
        attempts = max(0, int(ctx["settings"].sql_repair_attempts))
        for attempt in range(attempts + 1):
            try:
                result = run_tool(ctx, "run_sql", args)
                if attempt:
                    metrics.inc("agent.sql_repaired")
                return result
            except ToolError as exc:
                sql = args.get("sql") or args.get("query") or ""
                if attempt == attempts or not sql:
                    raise
                try:
                    reply = self.llm.extract_json([
                        {"role": "user", "content": sql_repair_prompt(ctx["schemas"], question, sql, str(exc))},
                    ])
                except LLMError:
                    raise exc
                fixed = reply.get("sql") if isinstance(reply, dict) else None
                if not isinstance(fixed, str) or not fixed.strip() or fixed.strip() == sql.strip():
                    raise
                metrics.inc("agent.sql_repair_attempts")
                logger.info("SQL repair attempt %d: %s", attempt + 1, exc)
                yield _ev("note", {"text": f"Query failed ({exc}); retrying with a corrected query."})
                args = {**args, "sql": fixed}
                args.pop("query", None)
        raise ToolError("SQL repair exhausted")  # pragma: no cover - loop always returns/raises

    # ------------------------------------------------------------------
    def _plan_with_llm(self, ctx: dict, question: str) -> dict:
        retrieved = ""
        if ctx.get("rag"):
            retrieved = ctx["rag"].context(question, ctx.get("session_id"), dataset_ids=ctx.get("dataset_ids"))
        messages = [
            {"role": "system", "content": system_prompt(ctx["schemas"], retrieved_context=retrieved)},
            *self._history_messages(ctx),
            {"role": "user", "content": question},
        ]
        reply = self.llm.extract_json(messages)
        if not isinstance(reply.get("tool_calls"), list):
            reply["tool_calls"] = []
        suggested = _suggest_dataset_for_question(ctx, question)
        clean = []
        for call in reply["tool_calls"]:
            if not isinstance(call, dict) or call.get("name") not in TOOL_MAP:
                continue
            args = call.get("args", {}) or {}
            if not isinstance(args, dict):
                args = {}
            # safety: if the question names a column that only one dataset has,
            # steer the tool to that dataset so we don't analyse the wrong file.
            if suggested and not _args_has_matching_column(ctx, args, question):
                args["dataset_id"] = suggested
            clean.append({"name": call["name"], "args": args})
        reply["tool_calls"] = clean
        return reply

    def _answer_with_llm(self, ctx, question, plan, results) -> Iterator[dict]:
        if plan.get("answer") and not plan.get("tool_calls"):
            # model answered directly without tools
            self._last_answer_payload = {
                "answer": plan["answer"],
                "reasoning": "Plan: " + plan.get("thought", ""),
                "sql": plan.get("sql_generated"),
                "pandas": plan.get("pandas_generated"),
                "chart_ids": [],
            }
            yield _ev("token", {"text": plan["answer"]})
            return
        retrieved = ""
        if ctx.get("rag"):
            retrieved = ctx["rag"].context(question, ctx.get("session_id"), dataset_ids=ctx.get("dataset_ids"))
        messages = [
            {"role": "system", "content": system_prompt(ctx["schemas"], retrieved_context=retrieved)},
            *self._history_messages(ctx),
            {"role": "user", "content": question},
            {"role": "assistant", "content": "Plan: " + plan.get("thought", "")},
            {"role": "user", "content": tool_results_block(plan["tool_calls"], results)},
            {"role": "user", "content": answer_final_prompt(ctx["schemas"])},
        ]
        buffer = []
        for delta in self.llm.stream(messages):
            buffer.append(delta)
            yield _ev("token", {"text": delta})
        answer = "".join(buffer)
        # safety: if the model returned the JSON envelope again, unwrap the "answer" field
        try:
            parsed = json.loads(answer)
            if isinstance(parsed, dict) and parsed.get("answer"):
                answer = parsed["answer"]
        except Exception:
            pass
        self._last_answer_payload = {
            "answer": answer,
            "reasoning": "Plan: " + plan.get("thought", "") + ". The answer is grounded in the tool outputs above.",
            "sql": plan.get("sql_generated") or _sql_from_results(results),
            "pandas": plan.get("pandas_generated"),
            "chart_ids": _chart_ids_from_answer(answer),
        }

    def _answer_with_fallback(self, ctx, question, plan, results) -> dict:
        from . import fallback

        pinned = ctx.get("pinned")
        record = self.registry.get(pinned) if pinned else None
        if record is None:
            datasets = self.registry.list()
            record = datasets[0] if datasets else None
        payload = fallback.narrate(ctx, question, plan, results, record)
        # stream the fallback answer token-by-token as an SSE nicety
        payload["chart_ids"] = payload.get("chart_ids", [])
        return payload

    def _history_messages(self, ctx: dict) -> list[dict]:
        out = []
        for m in ctx["history"]:
            role = m.get("role")
            if role in ("user", "assistant"):
                out.append({"role": role, "content": m.get("content", "")[:1200]})
        return out


def _ev(kind: str, data: dict) -> dict:
    return {"event": kind, "data": data}


def _tool_summary(result: dict) -> dict:
    if "error" in result:
        return {"error": result["error"]}
    if "anomaly_count" in result:
        return {"anomaly_count": result["anomaly_count"], "by_column": result.get("by_column")}
    if "row_count" in result:
        return {"row_count": result["row_count"], "columns": result.get("columns", [])[:6]}
    if "score" in result and "issues" in result:
        return {"quality_score": result["score"], "issues": len(result["issues"])}
    if "insights" in result:
        return {"insights": [i["title"] for i in result["insights"][:4]]}
    if "chart_id" in result:
        return {"chart_id": result["chart_id"], "caption": result.get("caption")}
    if "datasets" in result:
        return {"datasets": len(result["datasets"])}
    return {"keys": list(result.keys())[:6]}


def _chart_ids_from_answer(answer: str) -> list[str]:
    return re.findall(r"chart:([a-f0-9]{12})", answer)


def _sql_from_results(results: list[dict]) -> str | None:
    for res in results:
        if isinstance(res, dict) and res.get("sql"):
            return res["sql"]
    return None


def _get_rag_or_none():
    from ..rag import get_rag

    try:
        return get_rag()
    except Exception:  # pragma: no cover - defensive
        return None


def _suggest_dataset_for_question(ctx: dict, question: str) -> str | None:
    """If the question names columns that only one dataset possesses, return it.

    Guards against the LLM analysing the wrong uploaded file when multiple
    datasets are attached (e.g. "anomalies in revenue" must use the dataset
    that has a revenue column).
    """
    low = question.lower()
    hits: dict[str, set[str]] = {}
    for rec in ctx["registry"].list():
        for col in rec.columns:
            name = col.name.lower()
            if len(name) >= 4 and name in low:
                hits.setdefault(rec.dataset_id, set()).add(name)
    return next(iter(hits)) if len(hits) == 1 else None


def _args_has_matching_column(ctx: dict, args: dict, question: str) -> bool:
    did = args.get("dataset_id")
    if not did:
        return False
    rec = ctx["registry"].get(did)
    if rec is None:
        return True  # leave unknown ids alone; they'll error and be handled
    low = question.lower()
    return any(col.name.lower() in low for col in rec.columns)


def _stream_words(text: str) -> Iterator[str]:
    """Split text into small chunks for simulated streaming."""
    import time

    words = re.split(r"(?<=\s)", text)
    for w in words:
        if w.strip():
            time.sleep(0.004)
        yield w
