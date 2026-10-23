# Benchmarks

Two benchmarks, both run offline in CI:

- **[Document RAG](#document-rag-benchmark)**: retrieval quality, abstention and answer
  quality over the sample business documents (`rag_cases.json`, 40 questions).
- **[Text-to-SQL](#text-to-sql-benchmark)**: execution accuracy over the sample datasets
  (`sql_cases.json`, 94 questions), including a split that only documentation can answer.

```bash
LLM_PROVIDER=none python scripts/benchmark_rag.py -v      # document RAG, offline answers
LLM_PROVIDER=none python scripts/benchmark_sql.py -v      # text-to-SQL, offline engine
LLM_PROVIDER=none python scripts/benchmark_sql.py --no-docs   # ablation: without documents
```

Without `LLM_PROVIDER=none`, both use the LLM configured in `.env`.

---

## Document RAG benchmark

The four sample documents in `sample_data/docs/` cover all supported formats: metric
definitions (Markdown), a data dictionary (Markdown), a business glossary (DOCX) and a
reporting policy (PDF). `rag_cases.json` has 40 questions:

- **34 answerable** ones, labelled with the section or page that answers them. Most are
  paraphrased so they don't reuse the heading's words ("Who keeps the original pricing
  plan?" for *Early subscriber*), which is what separates keyword and vector retrieval.
- **6 unanswerable** ones (weather, CEO, share price…), where the right behaviour is to
  say the documents don't cover it.

### Metrics

| Metric | Definition |
|---|---|
| Recall@1 / Recall@5 | Gold passage ranked first / in the top 5 (answerable questions). |
| MRR | Mean reciprocal rank of the first gold passage. |
| Out-of-scope rejected | Unanswerable questions marked "not covered". |
| In-scope accepted | Answerable questions marked "covered". |
| Answer correct | Answerable: the required facts appear in the answer. Unanswerable: "not found". |
| Citation accuracy | Answered questions that cite at least one gold passage. |
| Faithfulness | Share of answer sentences whose content words (≥ 70%) appear in the passages they cite. |

### Results: retrieval ablation (offline extractive answers)

| Retrieval | Recall@1 | Recall@5 | MRR | Out-of-scope rejected | In-scope accepted | Answer correct | Citation accuracy | p50 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense (MiniLM) | 85% | 97% | 0.902 | 83% | 97% | 82% | 91% | 1 ms |
| BM25 | 79% | 100% | 0.892 | 0% | 100% | 78% | 94% | <1 ms |
| Hybrid (RRF) | 91% | 100% | 0.946 | 17% | 100% | 80% | 94% | 1 ms |
| **Hybrid + rerank** | **91%** | **100%** | **0.949** | **100%** | **100%** | **88%** | 91% | 165 ms |

What the ablation shows:

- **Fusion beats either retriever alone.** Dense misses exact terms and BM25 misses
  paraphrases; reciprocal rank fusion gets the best ranking (Recall@1 79–85% → 91%).
- **The cross-encoder is what makes abstention possible.** BM25 always finds *some*
  keyword overlap, so on its own it never says "I don't know" (0% rejected). The
  reranker's score separates answerable questions (≥ −4.8) from unanswerable ones
  (≤ −7.2), giving 100% on both.
- **The cost is latency**: ~165 ms per question on CPU, against ~1 ms without reranking.

**Calibration caveat:** the relevance threshold (`DOC_MIN_RERANK_SCORE = −6.0`) is the
midpoint of the score gap observed on *this* question set, so the 100%/100% abstention
figures are in-sample. A larger held-out set would give a fairer estimate.

Faithfulness is 100% offline by construction (extractive answers quote the cited passages).
It becomes a meaningful metric with an LLM generating the answer; see
[Live LLM evaluation](#live-llm-evaluation).

Raw output: [`results/rag_offline.json`](results/rag_offline.json).

---

## Text-to-SQL benchmark

`sql_cases.json` holds 94 natural-language questions over the four bundled sample
datasets, each paired with hand-written **gold SQL**. `scripts/benchmark_sql.py` asks
every question through the real chat API (the same path the UI uses), then **executes**
both the SQL the app returned and the gold SQL in the DuckDB sandbox and compares the
*results*, not the SQL text.

### Scoring

| Metric | Passes when |
|---|---|
| **Execution match** | every gold column appears in the predicted result with the same values, row for row. Extra predicted columns are allowed (`SELECT region, SUM(revenue)` answers "which region…"). For top-k gold queries (`ORDER BY … LIMIT k`) the first k predicted rows must match, so a full ranking still answers "top 5". |
| **Strict match** | execution match **and** the same number of rows and columns. |

Numbers are compared at 2 decimal places. Row order is ignored unless the gold SQL
has `ORDER BY … LIMIT`. Cases marked `any_of` (tied winners, e.g. two cities with
229 transactions each) accept any of the tied answers. Scoring is unit-tested in
`backend/tests/test_benchmark_scoring.py`.

### Splits and how the numbers were produced

Tuning a rule engine against its own test set inflates the score, so the cases
were written in rounds:

| Split | Cases | Role |
|---|---:|---|
| `dev` | 40 | Written first; failures here drove the engine fixes. |
| `holdout-1` | 20 | Written after the first round of fixes and scored blind: **15/20 (75%)**. Its general failures were then fixed and it joined the dev pool. |
| `holdout-2` | 20 | Written after round two and scored blind **once**: **16/20 (80%)**. This is the best estimate of accuracy on unseen phrasings. |
| `docs` | 14 | Questions that use a business term only the documentation defines ("high-value transactions in Mumbai", "gross margin by region"). Gold SQL follows `metric_definitions.md`. |

### Results: offline rule engine (`LLM_PROVIDER=none`)

| | Before | After |
|---|---:|---:|
| Execution match, the 80 data-only cases | 36/80 = **45.0%** | 79/80 = **98.8%** |
| Strict match, same 80 | 16/80 = 20.0% | 62/80 = 77.5% |
| Blind held-out (holdout-2, first run) | 7/20 = 35% | **16/20 = 80%** |
| p50 latency per question | 476 ms | ~170 ms |

### Results: does documentation help? (`docs` split)

| | Without documents (`--no-docs`) | With documents |
|---|---:|---:|
| Definition-dependent questions | **0/14** | **14/14** |
| All 94 cases | 79/94 = 84.0% | 93/94 = **98.9%** |

Without the documentation the engine can't know that a *high-value transaction* means
`"total" > 10000` or that *gross margin* is `SUM(revenue) - SUM(cost)`, so it answers a
different question. With it, the definition is retrieved, validated against the
dataset's real columns, applied, and cited in the answer. The `docs` split was written
alongside the feature, so treat 14/14 as "the mechanism works", not as a held-out estimate.

Raw output: [`results/rule_engine_baseline.json`](results/rule_engine_baseline.json) (before),
[`results/rule_engine.json`](results/rule_engine.json) (after),
[`results/rule_engine_no_docs.json`](results/rule_engine_no_docs.json) (ablation).

**Known miss:** "Which city generated the most *sales*?" on a table whose money
column is called `total`. Mapping the business word "sales" to that column needs
semantic understanding, which is what the LLM path is for; the rule engine ranks
by transaction count instead.

**What the rule-engine fixes were** (each pinned by `backend/tests/test_fallback_aggregates.py`):
the attached dataset is now sticky instead of being overridden by one shared word;
explicit `SUM/AVG/MAX/MIN` intents with optional `GROUP BY` replace a generic
averages table; numeric conditions ("rating below 3") become `WHERE` clauses;
"how many sales" counts rows instead of distinct `sales_rep`; quantity columns
("how many units") are summed; rankings exclude `NULL` groups; "fewest" ranks
ascending.

---

## Live LLM evaluation

Both scripts use the LLM from `.env` when `LLM_PROVIDER` isn't `none`.

**Run 1 (2026-10-24), system result.** `--llm-only --split holdout-2 docs`, with Groq
`openai/gpt-oss-20b` as primary and Gemini `3.5-flash-lite` as failover: **33/34 (97.1%)**,
holdout-2 19/20, docs 14/14. Groq's free tier rate-limited 34 requests (51 retries) and the
failover answered them, so this measures the deployed configuration, not gpt-oss-20b alone.
The miss was "Revenue per sales rep" (the LLM's SQL returned a different result). Usage:
35 successful requests, ~98k tokens, median 6.1 s per question. To score one model, disable
the other provider (`GEMINI_API_KEY=` or `GROQ_API_KEY=`) and use `--sleep 20` to stay
under Groq's per-minute token limit.

## Limits

These are regression suites, not research benchmarks like Spider, BIRD or BEIR: four
small documents, 94 single-table questions, clean schemas, and the same person wrote
the questions and the system. The held-out SQL score is the honest generalisation
number; treat the in-sample figures as "no known regressions".
