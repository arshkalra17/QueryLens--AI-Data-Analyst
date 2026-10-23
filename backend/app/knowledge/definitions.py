"""Business definitions from documentation, applied to text-to-SQL.

Metric/term definitions in the uploaded docs that carry an implementation line

    Applies to: retail_transactions.csv
    SQL filter: `"total" > 10000`              (a row condition)
    SQL expression: `SUM("revenue") - SUM("cost")`   (a metric)

are parsed into :class:`Definition` objects. When a question uses a defined
term ("how many high-value transactions in Mumbai?"), the definition's filter
is applied as a WHERE condition, and the answer cites the document it came
from. Expressions validate against the dataset's real columns before use, so
a stale or malicious document can't inject arbitrary SQL.
Prose-only definitions are not executed; they reach the LLM path as retrieved
context instead.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ..config import get_settings

logger = logging.getLogger(__name__)

_APPLIES = re.compile(r"Applies to:\s*([^\s`]+)", re.IGNORECASE)
_FILTER = re.compile(r"SQL filter:\s*`([^`]+)`", re.IGNORECASE)
_EXPR = re.compile(r"SQL expression:\s*`([^`]+)`", re.IGNORECASE)
_FORBIDDEN = re.compile(
    r";|--|/\*|\b(select|from|join|union|copy|install|load|attach|pragma|insert|update|delete|create|drop|alter|"
    r"read_\w+|glob|getenv)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Definition:
    term: str
    dataset: str          # file name it applies to ("" = any)
    kind: str             # "filter" | "expression"
    sql: str
    citation: str
    doc_name: str
    chunk_id: str

    def to_api(self) -> dict:
        return {"term": self.term, "dataset": self.dataset, "kind": self.kind, "sql": self.sql,
                "citation": self.citation, "doc_name": self.doc_name}


_cache: tuple[int, list[Definition]] | None = None


def all_definitions() -> list[Definition]:
    """Definitions parsed from every indexed chunk (cached per corpus)."""
    global _cache
    if not get_settings().enable_docs:
        return []
    from .store import get_document_store

    chunks = get_document_store().chunks()
    key = hash(frozenset(chunks))
    if _cache is not None and _cache[0] == key:
        return _cache[1]
    found: list[Definition] = []
    for c in chunks.values():
        term = (c.heading.split(" > ")[-1] if c.heading else "").strip()
        applies = _APPLIES.search(c.text)
        for kind, rx in (("filter", _FILTER), ("expression", _EXPR)):
            m = rx.search(c.text)
            if term and m:
                found.append(Definition(term=term, dataset=applies.group(1) if applies else "", kind=kind,
                                        sql=m.group(1).strip(), citation=c.citation(), doc_name=c.doc_name,
                                        chunk_id=c.chunk_id))
    _cache = (key, found)
    return found


def reset_cache() -> None:
    global _cache
    _cache = None
    _valid_cache.clear()


def match(question: str, record, kind: str | None = None) -> list[Definition]:
    """Definitions whose term appears in ``question`` and that apply to and
    validate against ``record``, longest term first."""
    if record is None:
        return []
    low = _norm(question)
    out = []
    for d in all_definitions():
        if kind and d.kind != kind:
            continue
        if d.dataset and d.dataset.lower() != str(record.name).lower():
            continue
        if not _mentions(low, d.term) or not _valid(d.sql, record):
            continue
        out.append(d)
    out.sort(key=lambda d: len(d.term), reverse=True)
    # a longer term wins over its own sub-term ("margin rate" over "margin")
    result: list[Definition] = []
    for d in out:
        if not any(_norm(d.term) in _norm(r.term) for r in result):
            result.append(d)
    return result


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _mentions(low_question: str, term: str) -> bool:
    words = _norm(term).split()
    if not words:
        return False
    # tolerate plurals on the head noun: "bulk orders", "high value transactions"
    head = re.escape(words[-1]) + r"(?:s|es)?"
    pattern = r"\b" + r"\s+".join([re.escape(w) for w in words[:-1]] + [head]) + r"\b"
    return re.search(pattern, low_question) is not None


_valid_cache: dict[tuple[str, str], bool] = {}


def _valid(sql: str, record) -> bool:
    key = (sql, getattr(record, "dataset_id", record.name))
    if key not in _valid_cache:
        _valid_cache[key] = _check(sql, record)
    return _valid_cache[key]


def _check(sql: str, record) -> bool:
    """Reject anything that isn't a plain expression over the dataset's columns."""
    if _FORBIDDEN.search(sql):
        logger.warning("Ignoring unsafe definition SQL: %s", sql)
        return False
    cols = {c.name for c in record.columns}
    quoted = re.findall(r'"((?:[^"]|"")+)"', sql)
    if not quoted or any(q.replace('""', '"') not in cols for q in quoted):
        return False
    from ..engine.sql_exec import run_sql
    from ..engine.tools import _safe_name

    table = _safe_name(record.name)
    probe = f'SELECT {sql} AS v FROM "{table}" LIMIT 1' if "(" in sql and re.search(r"\b(sum|avg|count|min|max)\s*\(", sql, re.I) \
        else f'SELECT COUNT(*) FROM "{table}" WHERE {sql}'
    result = run_sql(probe, {table: record.df}, max_rows=1)
    return result.error is None
