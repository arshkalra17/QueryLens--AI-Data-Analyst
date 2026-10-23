"""Result comparison for the SQL benchmark (pure functions, no app imports).

See ``benchmark_sql.py`` for how execution match and strict match are defined.
"""
from __future__ import annotations

import itertools
import math


def _norm(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return round(float(value), 2)
    text = str(value).strip()
    return text[:-9] if text.endswith(" 00:00:00") or text.endswith("T00:00:00") else text


def _columns(result) -> list[list]:
    return [[_norm(row[c]) for row in result.rows] for c in result.columns]


def _value_key(v):
    return (v is None, str(v))


def _sort_key(row):
    return tuple(_value_key(v) for v in row)


def compare(pred, gold, ordered: bool, any_of: bool = False) -> tuple[bool, bool]:
    """Return (execution_match, strict_match).

    ``any_of``: gold lists every acceptable answer (e.g. tied winners for
    "which city has the most ..."); the predicted top row must be one of them.
    """
    g_cols, p_cols = _columns(gold), _columns(pred)
    n = len(gold.rows)
    if any_of:
        if not pred.rows:
            return False, False
        top = [col[0] for col in p_cols]
        accepted = set(zip(*g_cols))
        for mapping in itertools.permutations(range(len(top)), len(g_cols)):
            if tuple(top[i] for i in mapping) in accepted:
                return True, len(pred.rows) == 1 and len(pred.columns) == len(gold.columns)
        return False, False
    if ordered:
        if len(pred.rows) < n:
            return False, False
        p_view = [col[:n] for col in p_cols]
    else:
        if len(pred.rows) != n:
            return False, False
        p_view = p_cols

    def same(a, b):
        return a == b if ordered else sorted(a, key=_value_key) == sorted(b, key=_value_key)

    candidates = [[i for i, p in enumerate(p_view) if same(p, g)] for g in g_cols]
    if any(not c for c in candidates):
        return False, False
    gold_rows = list(zip(*g_cols)) if g_cols else []
    for mapping in itertools.islice(itertools.product(*candidates), 256):
        pred_rows = list(zip(*(p_view[i] for i in mapping)))
        if (pred_rows == gold_rows) if ordered else (sorted(pred_rows, key=_sort_key) == sorted(gold_rows, key=_sort_key)):
            strict = len(pred.rows) == n and len(pred.columns) == len(gold.columns)
            return True, strict
    return False, False
