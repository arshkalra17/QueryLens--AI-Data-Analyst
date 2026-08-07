# Architecture

## Overview

The application is a FastAPI backend serving a no-build static frontend. A chat
agent orchestrates a registry of analytical *tools*. When an LLM API key is
configured the agent plans tool calls with OpenAI function calling; otherwise a
rule-based engine (`fallback.py`) plans and narrates the same tools, so the app
is fully functional offline.

```
Browser (frontend SPA)
   │  fetch / EventSource-style SSE
   ▼
FastAPI (app/main.py)
   │  auth middleware (optional bearer key)  • request-id + metrics
   ├─ /api/datasets  ──► DataFrameRegistry (core/registry.py)
   ├─ /api/chat      ──► Agent (engine/agent.py)
   │                       ├─ LLM plan?  (core/llm.py -> prompts.py)
   │                       ├─ rule plan? (engine/fallback.py)
   │                       ├─ run tools  (engine/tools.py  +  engine/sql_exec.py)
   │                       └─ narrate    (LLM stream or fallback narrator)
   ├─ /api/charts  ──► engine/charts.py + ChartStore
   ├─ /api/insights ─► engine/insights.py
   ├─ /api/dashboards / reports ─► HTML/Markdown assembly
   └─ /api/metrics ─► core/logging_setup.py (counters + timings)
```

## Key components

### core
- `registry.py` — `DataFrameRegistry` holds `DatasetRecord`s (schema, preview,
  dataframe). Uploads are persisted to `data/uploads/*.csv` and re-loaded on
  startup. Column classification: numeric / categorical / datetime.
- `sessions.py` — per-browser conversation state (message history, pinned +
  attached dataset ids). Enables follow-up questions.
- `llm.py` — thin OpenAI-compatible client: `complete`, `stream`, and
  `extract_json` (repairs trailing commas) for structured tool-call plans.
- `caching.py` — TTL cache keyed by a stable hash (question + schemas).
- `security.py` — optional bearer-token auth; constant-time key compare.
- `logging_setup.py` — request-context logging with request ids and an
  in-memory metrics store (`MetricsStore`).

### engine
- `sql_exec.py` — DuckDB over pandas DataFrames (registered by dataset id and
  safe filename, enabling cross-dataset joins). `validate_sql` enforces
  read-only SELECT/WITH and rejects dangerous statements; per-query timeout.
- `charts.py` — matplotlib/seaborn renders for 9 chart kinds; dense datetime
  axes are resampled to monthly for readable trend charts; each chart carries a
  caption and a standalone HTML embed.
- `anomalies.py` — z-score, IQR, and rolling-window detection; optional
  `group_by`; every flag gets a human explanation + severity.
- `insights.py` — deterministic, evidence-backed insights (overview, missing
  values, categorical dominance, variability, correlations, time trend,
  outliers) ranked by importance.
- `quality.py` — 0–100 score with per-issue severity and recommendations
  (duplicates, missing, constant columns, type issues, hygiene, outliers,
  dates).
- `forecast.py` — linear trend + seasonal (month/day-of-week) decomposition with
  a confidence band and model diagnostics (MAE/RMSE/MAPE).
- `tools.py` — the 11-tool registry shared by LLM and fallback paths:
  `list_datasets`, `get_schema`, `run_sql`, `generate_sql`, `run_pandas`
  (guarded), `generate_chart`, `detect_anomalies`, `generate_insights`,
  `data_quality`, `forecast`, `search_data`.
- `fallback.py` — intent detection via keyword markers + token scoring;
  resolves the relevant dataset by matching question tokens against names and
  column names; picks metrics/dimensions that appear in the question.
- `agent.py` — orchestrates a turn: build context → plan → execute tools →
  narrate; emits SSE events (`meta`, `note`, `plan`, `tool_start`,
  `tool_done`, `tool_error`, `token`, `final`, `error`).

## Data flow for a chat turn

1. `POST /api/chat` resolves/creates a session and attaches the requested
   datasets.
2. `Agent.run()` builds `ctx` (registry, session, schemas, history, settings,
   chart store).
3. Plan: LLM `extract_json` (or `fallback.plan`) → ordered `tool_calls`.
4. Execute: each tool runs sandboxed; failures are captured as events, not
   exceptions.
5. Narrate: LLM streams the answer from `system_prompt + history + tool results`
   (or `fallback.narrate` builds markdown); answer, reasoning, sql/pandas and
   chart ids are persisted to the session and returned.

## Frontend

Single-page app (`frontend/index.html`, `styles.css`, `app.js`) — no build
step. It reads SSE events from `/api/chat/stream`, renders markdown, copies
code blocks, lazy-loads chart images, pins datasets, and opens a dashboard
modal via `/api/dashboards`. Sample datasets are served from `/data/samples`.

## Persistence & caches

- Datasets: `data/uploads/*.csv` (survives restarts).
- Sessions: in-memory (reset on server restart).
- Charts/insights: in-memory stores with LRU/TTL limits.

## Testing

`backend/tests/` covers the registry, SQL validation/execution, charts, anomaly
detection, quality, insights, forecasting, the tool registry, the fallback
planner/narrator, and end-to-end API flows (upload → chat → stream → dashboard
→ report → metrics → auth). Run `cd backend && ../.venv/bin/python -m pytest -q`.
