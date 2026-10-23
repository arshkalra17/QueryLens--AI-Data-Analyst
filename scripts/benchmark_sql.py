"""Execution-accuracy benchmark for QueryLens' text-to-SQL path.

Each case in ``benchmarks/sql_cases.json`` pairs a natural-language question
with hand-written gold SQL over the bundled sample datasets. The question is
sent through the real chat API (exactly like the UI); the SQL the app returns
and the gold SQL are both executed in the same DuckDB sandbox, and the
*results* are compared, not the SQL text.

Two scores are reported:

- **execution match**: every gold column appears in the predicted result with
  the same values row-for-row (extra predicted columns are allowed, so
  ``SELECT region, SUM(revenue)`` matches a gold ``SELECT region ...``). For
  top-k gold queries (``ORDER BY ... LIMIT k``) the first k predicted rows must
  match, so returning a full ranking still counts as answering "top 5".
- **strict match**: additionally the same number of rows and columns.

Numbers are compared at 2 decimal places; row order is ignored unless the gold
SQL has ``ORDER BY`` + ``LIMIT``. Cases marked ``any_of`` list every tied
correct answer and accept a predicted top row matching any of them.

Usage (from the repo root):

    python scripts/benchmark_sql.py                     # provider from .env
    LLM_PROVIDER=none python scripts/benchmark_sql.py   # offline rule engine
    python scripts/benchmark_sql.py --fail-below 0.8 --json benchmarks/results/run.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "benchmarks" / "sql_cases.json"
SAMPLES = ROOT / "sample_data"

# Isolate all runtime state so a benchmark never touches real uploads/sessions.
_tmp = Path(tempfile.mkdtemp(prefix="querylens-bench-"))
os.environ.setdefault("ENABLE_RAG", "false")
os.environ["SESSION_STORE"] = "memory"
os.environ["UPLOAD_DIR"] = str(_tmp / "uploads")
os.environ["DOCS_DIR"] = str(_tmp / "docs")
# flags that change app settings must take effect before settings are read
os.environ["ENABLE_DOCS"] = "false" if "--no-docs" in sys.argv else "true"
if "--llm-only" in sys.argv:
    os.environ["DETERMINISTIC_INTENTS"] = "false"
os.environ["RAG_STORE_DIR"] = str(_tmp / "rag_store")
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from fastapi.testclient import TestClient  # noqa: E402

from app.core.llm import get_llm, usage  # noqa: E402
from app.core.registry import get_registry  # noqa: E402
from app.engine.sql_exec import run_sql  # noqa: E402
from app.engine.tools import _frames  # noqa: E402
from app.main import app  # noqa: E402
from benchmark_scoring import compare  # noqa: E402

_TOPK = re.compile(r"\border\s+by\b.*\blimit\s+\d+", re.IGNORECASE | re.DOTALL)


# ------------------------------------------------------------------ running
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--fail-below", type=float, default=0.0, help="exit 1 if execution match is below this")
    parser.add_argument("--json", type=Path, help="write per-case results to this file")
    parser.add_argument("--only", help="run only case ids containing this substring")
    parser.add_argument("-v", "--verbose", action="store_true", help="print SQL for every failure")
    parser.add_argument("--no-docs", action="store_true",
                        help="ablation: don't index the sample business documents (definitions unavailable)")
    parser.add_argument("--llm-only", action="store_true",
                        help="route every question through the LLM planner (no deterministic intents)")
    parser.add_argument("--split", nargs="+", help="only these splits (dev, holdout-1, holdout-2, docs)")
    parser.add_argument("--limit", type=int, help="at most this many cases (keeps live runs inside free-tier quotas)")
    parser.add_argument("--sleep", type=float, default=0.0, help="seconds to wait between questions (rate limits)")
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text())
    if args.only:
        cases = [c for c in cases if args.only in c["id"]]
    if args.split:
        cases = [c for c in cases if c.get("split", "dev") in args.split]
    if args.limit:
        cases = cases[: args.limit]

    client = TestClient(app)
    for csv in sorted(SAMPLES.glob("*.csv")):
        with csv.open("rb") as fh:
            client.post("/api/datasets", files={"files": (csv.name, fh, "text/csv")})
    datasets = {d["name"]: d["dataset_id"] for d in client.get("/api/datasets").json()["datasets"]}
    if not args.no_docs:
        docs = sorted((ROOT / "sample_data" / "docs").iterdir())
        client.post("/api/documents", files=[("files", (f.name, f.read_bytes())) for f in docs])
    registry = get_registry()
    frames = _frames({"registry": registry, "sessions": None}, None)
    provider = None

    results = []
    usage.reset()
    for i, case in enumerate(cases):
        if i and args.sleep:
            time.sleep(args.sleep)
        gold = run_sql(case["gold_sql"], frames, max_rows=10_000)
        if gold.error:
            raise SystemExit(f"gold SQL for {case['id']} is invalid: {gold.error}")
        started = time.perf_counter()
        with redirect_stderr(io.StringIO()):
            resp = client.post(
                "/api/chat",
                json={"message": case["question"], "dataset_ids": [datasets[case["dataset"]]]},
            ).json()
        latency_ms = (time.perf_counter() - started) * 1000
        provider = provider or resp.get("provider")
        pred_sql = resp.get("sql")
        outcome = {"id": case["id"], "split": case.get("split", "dev"), "category": case["category"], "question": case["question"],
                   "gold_sql": case["gold_sql"], "pred_sql": pred_sql, "latency_ms": round(latency_ms, 1)}
        if not pred_sql:
            outcome.update(match=False, strict=False, reason="no SQL produced")
        else:
            pred = run_sql(pred_sql, frames, max_rows=10_000)
            if pred.error:
                outcome.update(match=False, strict=False, reason=pred.error.splitlines()[0][:120])
            else:
                ordered = bool(_TOPK.search(case["gold_sql"]))
                match, strict = compare(pred, gold, ordered, any_of=case.get("any_of", False))
                outcome.update(match=match, strict=strict, reason="" if match else "result differs")
        results.append(outcome)

    llm_info = get_llm().describe()
    model = llm_info.get("model") if provider not in (None, "fallback") else None
    run_usage = usage.snapshot()
    _report(results, f"{provider or 'unknown'}{f' / {model}' if model else ''}", args.verbose)
    if model:
        print(f"\nLLM usage: {run_usage['requests']} requests · {run_usage['total_tokens']:,} tokens "
              f"({run_usage['prompt_tokens']:,} prompt) · avg {run_usage['avg_latency_ms']:.0f} ms · "
              f"{run_usage['rate_limit_retries']} rate-limit retries · {run_usage['failures']} failures · "
              f"{run_usage['cache_hits']} cache hits")
    accuracy = sum(r["match"] for r in results) / max(len(results), 1)
    out = args.json
    if out is None and model:  # live runs are always saved
        slug = re.sub(r"[^a-z0-9]+", "-", f"{provider}_{model}".lower()).strip("-")
        suffix = ("_llm-only" if args.llm_only else "") + ("_no-docs" if args.no_docs else "")
        out = ROOT / "benchmarks" / "results" / f"sql_{slug}{suffix}.json"
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "provider": provider, "model": model, "llm_only": args.llm_only, "documents": not args.no_docs,
            "splits": args.split, "execution_match": accuracy, "usage": run_usage if model else None,
            "cases": results,
        }, indent=2))
        print(f"results saved to {out.relative_to(ROOT) if out.is_absolute() and ROOT in out.parents else out}")
    if accuracy < args.fail_below:
        print(f"\nFAIL: execution match {accuracy:.1%} is below --fail-below {args.fail_below:.0%}")
        return 1
    return 0


def _report(results: list[dict], provider: str, verbose: bool) -> None:
    total = len(results)
    match = sum(r["match"] for r in results)
    strict = sum(r["strict"] for r in results)
    lat = sorted(r["latency_ms"] for r in results)
    p50 = lat[len(lat) // 2] if lat else 0
    docs = "off" if os.environ.get("ENABLE_DOCS") == "false" else "on"
    print(f"\nQueryLens SQL benchmark  ·  provider: {provider}  ·  documents: {docs}  ·  {total} cases")
    print(f"execution match: {match}/{total} = {match / total:.1%}   strict: {strict}/{total} = {strict / total:.1%}   p50 latency: {p50:.0f} ms\n")

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    print("| category | cases | execution match |")
    print("|---|---:|---:|")
    for cat, rows in sorted(by_cat.items()):
        m = sum(r["match"] for r in rows)
        print(f"| {cat} | {len(rows)} | {m}/{len(rows)} |")

    splits: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        splits[r["split"]].append(r)
    if len(splits) > 1:
        print("\n| split | cases | execution match |")
        print("|---|---:|---:|")
        for name, rows in splits.items():
            m = sum(r["match"] for r in rows)
            print(f"| {name} | {len(rows)} | {m}/{len(rows)} |")

    failures = [r for r in results if not r["match"]]
    if failures:
        print("\nFailures:")
        for r in failures:
            print(f"  ✗ {r['id']}: {r['question']}  [{r['reason']}]")
            if verbose:
                print(f"      gold: {r['gold_sql']}\n      pred: {r['pred_sql']}")


if __name__ == "__main__":
    sys.exit(main())
