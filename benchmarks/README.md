# Text-to-SQL accuracy benchmark

`sql_cases.json` holds 80 natural-language questions over the four bundled sample
datasets, each paired with hand-written **gold SQL**. `scripts/benchmark_sql.py`
asks every question through the real chat API (the same path the UI uses), then
**executes** both the SQL the app returned and the gold SQL in the DuckDB sandbox
and compares the *results*, not the SQL text.

```bash
LLM_PROVIDER=none python scripts/benchmark_sql.py -v          # offline rule engine
python scripts/benchmark_sql.py --json benchmarks/results/llm.json   # provider from .env
```

## Scoring

| Metric | Passes when |
|---|---|
| **Execution match** | every gold column appears in the predicted result with the same values, row for row. Extra predicted columns are allowed (`SELECT region, SUM(revenue)` answers "which region…"). For top-k gold queries (`ORDER BY … LIMIT k`) the first k predicted rows must match, so a full ranking still answers "top 5". |
| **Strict match** | execution match **and** the same number of rows and columns. |

Numbers are compared at 2 decimal places. Row order is ignored unless the gold SQL
has `ORDER BY … LIMIT`. Cases marked `any_of` (tied winners, e.g. two cities with
229 transactions each) accept any of the tied answers. Scoring is unit-tested in
`backend/tests/test_benchmark_scoring.py`.

## Splits and how the numbers were produced

Tuning a rule engine against its own test set inflates the score, so the cases
were written in three rounds:

| Split | Cases | Role |
|---|---:|---|
| `dev` | 40 | Written first; failures here drove the engine fixes. |
| `holdout-1` | 20 | Written after the first round of fixes and scored blind: **15/20 (75%)**. Its general failures were then fixed and it joined the dev pool. |
| `holdout-2` | 20 | Written after round two and scored blind **once**: **16/20 (80%)**. This is the best estimate of accuracy on unseen phrasings. |

## Results: offline rule engine (`LLM_PROVIDER=none`)

| | Before | After |
|---|---:|---:|
| Execution match, all 80 cases | 36/80 = **45.0%** | 79/80 = **98.8%** |
| Strict match | 16/80 = 20.0% | 62/80 = 77.5% |
| Blind held-out (holdout-2, first run) | 7/20 = 35% | **16/20 = 80%** |
| p50 latency per question | 476 ms | 147 ms |

Raw per-case output: [`results/rule_engine_baseline.json`](results/rule_engine_baseline.json)
(before) and [`results/rule_engine.json`](results/rule_engine.json) (after).

**Known miss:** "Which city generated the most *sales*?" on a table whose money
column is called `total`. Mapping the business word "sales" to that column needs
semantic understanding, which is what the LLM path is for; the rule engine ranks
by transaction count instead.

**What the fixes were** (each pinned by `backend/tests/test_fallback_aggregates.py`):
the attached dataset is now sticky instead of being overridden by one shared word;
explicit `SUM/AVG/MAX/MIN` intents with optional `GROUP BY` replace a generic
averages table; numeric conditions ("rating below 3") become `WHERE` clauses;
"how many sales" counts rows instead of distinct `sales_rep`; quantity columns
("how many units") are summed; rankings exclude `NULL` groups; "fewest" ranks
ascending.

## Limits

80 questions over 4 small CSVs is a regression suite, not a research benchmark like
Spider or BIRD: the questions are single-table, the schemas are clean, and the same
person wrote the questions and the engine. The held-out score is the honest number;
treat the 98.8% as "no known regressions on these phrasings".
