"""Evaluation harness for the AI Data Analyst.

Uploads datasets and runs the assignment's scenario prompts against the chat
API, scoring each answer for expected tokens and behaviour. Prints a results
table. Exits non-zero when the overall score is below ``--fail-below``.

Usage:
    ../.venv/bin/python scripts/evaluate.py \\
        --csv data/samples/sales_data.csv data/samples/retail_transactions.csv \\
        --fail-below 0.6
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.core.registry import reset_registry  # noqa: E402
from app.main import app  # noqa: E402


SCENARIOS = [
    {
        "name": "top region by revenue",
        "question": "Which region generated the highest revenue?",
        "expected": ["north", "south", "east", "west"],
        "require_chart": True,
        "weight": 1.0,
    },
    {
        "name": "top 5 customers",
        "question": "What are the top five customers?",
        "expected": ["cust"],
        "require_chart": True,
        "weight": 1.0,
    },
    {
        "name": "underperformers",
        "question": "Which products are underperforming?",
        "expected": ["charger", "monitor", "headset"],
        "weight": 1.0,
    },
    {
        "name": "monthly trend chart",
        "question": "Show monthly sales trends.",
        "expected": ["trend"],
        "require_chart": True,
        "weight": 1.0,
    },
    {
        "name": "anomaly in named column",
        "question": "Detect anomalies in revenue.",
        "expected": ["anomal", "revenue"],
        "weight": 1.0,
    },
    {
        "name": "forecast",
        "question": "Forecast revenue for the next 6 months.",
        "expected": ["forecast", "month"],
        "weight": 1.0,
    },
    {
        "name": "SQL generation",
        "question": "Generate SQL for this analysis.",
        "expected": ["select"],
        "weight": 0.8,
    },
    {
        "name": "pie chart",
        "question": "Show a pie chart of product categories.",
        "expected": [],
        "require_chart": True,
        "weight": 0.8,
    },
    {
        "name": "data quality",
        "question": "Check the data quality of this dataset.",
        "expected": ["score", "/100"],
        "weight": 0.8,
    },
    {
        "name": "insights",
        "question": "Give me business insights about this data.",
        "expected": ["insight"],
        "weight": 0.8,
    },
]


def load_dataset(client, path: Path) -> str:
    with open(path, "rb") as fh:
        resp = client.post("/api/datasets", files={"files": (path.name, io.BytesIO(fh.read()), "text/csv")})
    if resp.status_code != 200:
        raise SystemExit(f"Failed to upload {path.name}: {resp.text}")
    payload = resp.json()
    if payload["errors"]:
        raise SystemExit(f"Upload errors for {path.name}: {payload['errors']}")
    return payload["imported"][0]["dataset_id"]


def score_answer(answer: str, scenario: dict) -> dict:
    text = answer.lower()
    hits = sum(1 for tok in scenario["expected"] if tok in text)
    pct = hits / len(scenario["expected"]) if scenario["expected"] else 1.0
    # chart check: answers reference a chart URL
    chart = scenario.get("require_chart", False)
    has_chart = "chart" in text or "image" in text
    chart_ok = (has_chart == chart) if chart else True
    return {"hit_ratio": pct, "chart_ok": chart_ok, "pass": pct >= 0.99 and chart_ok}


def run() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True, help="CSV paths to load before evaluation")
    parser.add_argument("--fail-below", type=float, default=0.6, help="overall pass rate to require (exit 1 below)")
    args = parser.parse_args()

    reset_registry()
    for stale in get_settings().upload_dir_resolved.glob("*.csv"):
        stale.unlink()
    with TestClient(app) as client:
        dataset_ids = [load_dataset(client, Path(p)) for p in args.csv]
        rows = []
        for scenario in SCENARIOS:
            resp = client.post(
                "/api/chat",
                json={"message": scenario["question"], "dataset_ids": dataset_ids},
            )
            payload = resp.json()
            answer = payload.get("answer", "")
            result = score_answer(answer, scenario)
            rows.append({**scenario, **result, "status": resp.status_code})
            print(
                f"[{'PASS' if result['pass'] else 'FAIL'}] {scenario['name']:<30} "
                f"ratio={result['hit_ratio']:.2f} chart={result['chart_ok']} "
                f"(chars={len(answer)})"
            )

    overall = sum(1 for r in rows if r["pass"]) / len(rows)
    print(f"\noverall: {overall:.0%} ({sum(1 for r in rows if r['pass'])}/{len(rows)} scenarios passed)")
    if overall < args.fail_below:
        print(f"below threshold {args.fail_below:.0%}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
