---
title: QueryLens
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: Ask CSVs and business docs questions; cited RAG + text-to-SQL
---

# QueryLens: AI data analyst (public demo)

Ask questions about CSV data in plain English, grounded in your business documents.
Answers come from SQL that runs in a sandboxed DuckDB, and document answers cite their sources.

Sample datasets and documents are preloaded. Try:

- *What counts as a high-value transaction?* (document RAG, cited)
- *How many high-value transactions were made in Mumbai?* (documents + SQL)
- *Which region generated the highest revenue?*
- *Gross margin by region*

This is a shared demo: uploads are visible to other visitors and are wiped on restart,
and requests are rate-limited per visitor.

Source code, benchmarks and methodology: https://github.com/arshkalra17/QueryLens--AI-Data-Analyst
