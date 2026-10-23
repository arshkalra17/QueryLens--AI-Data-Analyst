"""Retrieval + answer-quality benchmark for QueryLens' document RAG.

Indexes ``sample_data/docs`` into a throwaway store, then for each labelled
question in ``benchmarks/rag_cases.json`` and each retrieval mode
(dense | bm25 | hybrid | hybrid_rerank) measures:

Retrieval (in-scope questions; gold = the section/page that answers it)
  Recall@1, Recall@5   gold passage ranked first / in the top 5
  MRR                  mean reciprocal rank of the first gold passage
Abstention
  out-of-scope rejected   share of unanswerable questions marked "not covered"
  in-scope accepted       share of answerable questions marked "covered"
Answers
  answer correctness   in-scope: required facts present; out-of-scope: "not found"
  citation accuracy    answered questions citing at least one gold passage
  faithfulness         share of answer sentences supported by their cited passages
                       (token recall >= 0.7 against the cited text)

Answers use the configured LLM when one is available, otherwise the offline
extractive answerer. Usage:

    LLM_PROVIDER=none python scripts/benchmark_rag.py
    python scripts/benchmark_rag.py --modes hybrid_rerank --json benchmarks/results/rag_llm.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_tmp = Path(tempfile.mkdtemp(prefix="querylens-rag-bench-"))
os.environ["ENABLE_DOCS"] = "true"
os.environ["DOCS_DIR"] = str(_tmp / "docs")
os.environ.setdefault("ENABLE_RAG", "false")
sys.path.insert(0, str(ROOT / "backend"))

from app.core.llm import get_llm  # noqa: E402
from app.knowledge import get_document_store  # noqa: E402
from app.knowledge.answer import answer  # noqa: E402
from app.knowledge.retrieval import MODES, search  # noqa: E402
from app.knowledge.store import tokenize  # noqa: E402

CITATION = re.compile(r"\[[^\[\]]*\.(?:md|pdf|docx|txt|markdown)[^\[\]]*\]")


def is_gold(chunk, gold: list[dict]) -> bool:
    for g in gold:
        if chunk.doc_name != g["doc"]:
            continue
        if "page" in g and chunk.page == g["page"]:
            return True
        if "section" in g and chunk.heading.split(" > ")[-1].strip().lower() == g["section"].lower():
            return True
    return False


def faithfulness(text: str, cited_texts: list[str]) -> float:
    """Share of answer sentences whose content words appear in the cited passages."""
    support = set(tokenize(" ".join(cited_texts)))
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", CITATION.sub("", text)) if tokenize(s)]
    if not sentences:
        return 0.0
    ok = sum(1 for s in sentences if len(set(tokenize(s)) & support) / len(set(tokenize(s))) >= 0.7)
    return ok / len(sentences)


def run(cases: list[dict], modes: list[str], llm, answer_modes: list[str]) -> dict:
    report = {}
    for mode in modes:
        rows = []
        for case in cases:
            t0 = time.perf_counter()
            result = search(case["question"], k=5, mode=mode)
            latency = (time.perf_counter() - t0) * 1000
            ranks = [i + 1 for i, h in enumerate(result.hits) if is_gold(h.chunk, case["gold"])]
            row = {"id": case["id"], "type": case["type"], "covered": result.covered,
                   "first_gold_rank": ranks[0] if ranks else None, "retrieval_ms": round(latency, 1)}
            if mode in answer_modes:
                t1 = time.perf_counter()
                out = answer(case["question"], result, llm=llm)
                row["answer_ms"] = round((time.perf_counter() - t1) * 1000, 1)
                row["answer"] = out["answer"]
                row["found"] = out["found"]
                row["answer_mode"] = out["mode"]
                cited = [h for h in result.hits if any(c["chunk_id"] == h.chunk.chunk_id for c in out["citations"])]
                row["citation_correct"] = any(is_gold(h.chunk, case["gold"]) for h in cited) if case["gold"] else None
                row["faithfulness"] = faithfulness(out["answer"], [h.chunk.text for h in cited]) if out["found"] else None
                if case["gold"]:
                    low = out["answer"].lower()
                    row["correct"] = out["found"] and all(k.lower() in low for k in case["answer_keywords"])
                else:
                    row["correct"] = not out["found"]
            rows.append(row)
        report[mode] = {"summary": summarize(rows, mode in answer_modes), "cases": rows}
    return report


def summarize(rows: list[dict], with_answers: bool) -> dict:
    ins = [r for r in rows if r["type"] != "out_of_scope"]
    oos = [r for r in rows if r["type"] == "out_of_scope"]
    s = {
        "recall@1": sum(1 for r in ins if r["first_gold_rank"] == 1) / len(ins),
        "recall@5": sum(1 for r in ins if r["first_gold_rank"]) / len(ins),
        "mrr": sum(1 / r["first_gold_rank"] for r in ins if r["first_gold_rank"]) / len(ins),
        "oos_rejected": sum(1 for r in oos if not r["covered"]) / len(oos) if oos else None,
        "in_scope_accepted": sum(1 for r in ins if r["covered"]) / len(ins),
        "retrieval_p50_ms": statistics.median(r["retrieval_ms"] for r in rows),
    }
    if with_answers:
        answered = [r for r in ins if r.get("found")]
        faith = [r["faithfulness"] for r in rows if r.get("faithfulness") is not None]
        s.update({
            "answer_correct": sum(1 for r in rows if r["correct"]) / len(rows),
            "citation_accuracy": sum(1 for r in answered if r["citation_correct"]) / len(answered) if answered else 0.0,
            "faithfulness": sum(faith) / len(faith) if faith else 0.0,
        })
    return s


def print_report(report: dict, provider: str, n: int) -> None:
    print(f"\nQueryLens document-RAG benchmark  ·  answers: {provider}  ·  {n} questions\n")
    print("| retrieval | Recall@1 | Recall@5 | MRR | out-of-scope rejected | in-scope accepted | answer correct | citation accuracy | faithfulness | p50 retrieval |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    pct = lambda v: "–" if v is None else f"{v:.0%}"  # noqa: E731
    for mode, data in report.items():
        s = data["summary"]
        print(f"| {mode} | {pct(s['recall@1'])} | {pct(s['recall@5'])} | {s['mrr']:.3f} | {pct(s['oos_rejected'])} | "
              f"{pct(s['in_scope_accepted'])} | {pct(s.get('answer_correct'))} | {pct(s.get('citation_accuracy'))} | "
              f"{pct(s.get('faithfulness'))} | {s['retrieval_p50_ms']:.0f} ms |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", type=Path, default=ROOT / "benchmarks" / "rag_cases.json")
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--answer-modes", nargs="+", choices=MODES,
                        help="modes to generate answers for (default: all offline, hybrid_rerank only with an LLM)")
    parser.add_argument("--json", type=Path, help="write the full report here")
    parser.add_argument("--fail-below", type=float, default=0.0,
                        help="exit 1 if hybrid_rerank Recall@5 or answer correctness is below this")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    store = get_document_store()
    for f in sorted((ROOT / "sample_data" / "docs").iterdir()):
        store.add(f.name, f.read_bytes())
    cases = json.loads(args.cases.read_text())

    llm = get_llm()
    llm = llm if llm.available else None
    answer_modes = args.answer_modes or (["hybrid_rerank"] if llm else args.modes)
    provider = (llm.describe().get("active") or "llm") if llm else "offline extractive"
    report = run(cases, args.modes, llm, answer_modes)
    print_report(report, provider, len(cases))
    if args.verbose:
        for mode, data in report.items():
            for r in data["cases"]:
                bad = (r["type"] != "out_of_scope" and r["first_gold_rank"] != 1) or r.get("correct") is False
                if bad:
                    print(f"  [{mode}] {r['id']}: gold rank {r['first_gold_rank']}, covered={r['covered']}, "
                          f"correct={r.get('correct')} :: {str(r.get('answer', ''))[:120]}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"answers": provider, "report": report}, indent=2))
    best = report.get("hybrid_rerank", {}).get("summary", {})
    if best and min(best["recall@5"], best.get("answer_correct", 1.0)) < args.fail_below:
        print(f"\nFAIL: hybrid_rerank below --fail-below {args.fail_below:.0%}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
