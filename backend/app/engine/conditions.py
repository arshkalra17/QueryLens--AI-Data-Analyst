"""Grounded WHERE-clause extraction from natural-language conditions.

Conditions such as "in 2021", "where Country = India", "for region North" must
be translated into SQL WHERE clauses instead of being silently dropped by a bare
aggregate ("how many customers subscribed in 2021?" -> COUNT(*) over everything
is wrong; it must filter the subscription year). This module extracts a grounded
condition from a question, and exposes helpers to detect when a generated SQL
statement is missing one, so callers can inject the filter or raise a warning.
"""
from __future__ import annotations

import re

import pandas as pd

from ..utils import is_string_series

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
_MONTH_NAMES = "|".join(_MONTHS)

# capitalized words that are grammar, not data values
_QUESTION_WORDS = {
    "what", "which", "how", "who", "where", "when", "why", "whose", "whom",
    "show", "list", "display", "view", "get", "print", "find", "tell", "give",
    "me", "the", "a", "an", "of", "in", "for", "from", "with", "by", "and",
    "or", "to", "than", "as", "at", "on", "this", "that", "these", "those",
    "our", "your", "their", "over", "under", "top", "bottom",
}

# what the user is counting (things that must NOT be mistaken for a filter value)
_ROW_NOUNS = {
    "customers", "customer", "orders", "order", "rows", "row", "records",
    "record", "entries", "entry", "people", "person", "users", "user",
    "clients", "client", "items", "item", "purchases", "purchase", "sales",
    "sales", "transactions", "transaction", "products", "product",
}


def _q(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _lit(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _date_col(record) -> str | None:
    for c in record.columns:
        if c.datetime:
            return c.name
    return None


def _match_column(record, name: str | None) -> str | None:
    want = re.sub(r"[^a-z0-9]", "", (name or "").lower())
    if not want:
        return None
    cols = [(c.name, re.sub(r"[^a-z0-9]", "", c.name.lower())) for c in record.columns]
    for cname, cnorm in cols:
        if cnorm == want:
            return cname
    for cname, cnorm in cols:
        if want in cnorm:
            return cname
    return None


def _find_value_col(record, value: str) -> str | None:
    """Find the column that actually contains ``value``, preferring low-cardinality."""
    low = value.strip().lower()
    exact: list = []
    sub: list = []
    for c in record.columns:
        name = c.name
        if c.datetime or not hasattr(record.df[name], "astype"):
            continue
        s = record.df[name]
        if pd.api.types.is_numeric_dtype(s):
            try:
                if low in {str(v) for v in s.dropna().unique().tolist()}:
                    exact.append((c, name))
            except Exception:
                pass
            continue
        vals = s.astype(str).str.lower()
        if (vals == low).any():
            exact.append((c, name))
        elif is_string_series(s) and vals.str.contains(re.escape(low), regex=True).any():
            sub.append((c, name))
    pool = exact or sub
    if not pool:
        return None
    pool.sort(key=lambda t: (0 if t[0].unique_count else 1, t[0].unique_count or len(record.df), t[1]))
    return pool[0][1]


def _is_numeric_col(record, col: str) -> bool:
    return bool(pd.api.types.is_numeric_dtype(record.df[col]))


def _op_condition(col: str, value: str, op: str, record) -> str:
    qcol = _q(col)
    opmap = {"==": "=", "=": "=", "is": "=", "!=": "!=", "<>": "!=",
             "is not": "!=", ">": ">", ">=": ">=", "<": "<", "<=": "<="}
    o = opmap.get(op.lower(), "=")
    if _is_numeric_col(record, col) and _num(value) is not None:
        return f"{qcol} {o} {value}"
    return f"{qcol} {o} {_lit(value)}"


def _year_condition(low: str, date_col: str | None) -> str | None:
    """Year/date range conditions against the (single) datetime column."""
    if date_col is None:
        return None
    dc = _q(date_col)
    # before / prior to / until / through / up to 2021  ->  < 2021-01-01
    m = re.search(r"\b(?:before|prior to|until|through|up to)\s+(?:the\s+)?(?:year\s+)?(20\d{2})\b", low)
    if m:
        return f"{dc} < {_lit(m.group(1) + '-01-01')}"
    # after / since 2020  ->  >= 2021-01-01 (after the *end* of 2020)
    m = re.search(r"\b(?:after|since)\s+(?:the\s+)?(?:year\s+)?(20\d{2})\b", low)
    if m:
        return f"{dc} >= {_lit(str(int(m.group(1)) + 1) + '-01-01')}"
    # from 2020 (inclusive)  ->  >= 2020-01-01
    m = re.search(r"\bfrom\s+(?:the\s+)?(?:year\s+)?(20\d{2})\b", low)
    if m:
        return f"{dc} >= {_lit(m.group(1) + '-01-01')}"
    # in January 2021  ->  [2021-01-01, 2021-02-01)
    m = re.search(r"\b(?:in|during|for)\s+(" + _MONTH_NAMES + r")\s+(20\d{2})\b", low)
    if m:
        month = _MONTHS[m.group(1)]
        start = f"{m.group(2)}-{month:02d}-01"
        if month == 12:
            end = f"{int(m.group(2)) + 1}-01-01"
        else:
            end = f"{m.group(2)}-{month + 1:02d}-01"
        return f"{dc} >= {_lit(start)} AND {dc} < {_lit(end)}"
    # in / during / for / throughout / calendar year 2021  ->  [2021-01-01, 2022-01-01)
    m = re.search(r"\b(?:in|during|for|throughout|calendar)\s+(?:the\s+)?(?:year\s+)?(20\d{2})\b", low)
    if m:
        return f"{dc} >= {_lit(m.group(1) + '-01-01')} AND {dc} < {_lit(str(int(m.group(1)) + 1) + '-01-01')}"
    return None


def _explicit_condition(low: str, record) -> str | None:
    """'where <col> <op> <value>' style conditions spelled out in the question."""
    m = re.search(
        r"\bwhere\s+([a-zA-Z_][a-zA-Z0-9_ .-]*?)\s*(==|!=|<>|>=|<=|>|<|=|is not|is)\s*['\"]?([^'\"=]+?)['\"]?\s*$",
        low,
    )
    if not m:
        return None
    col = _match_column(record, m.group(1))
    if col is None:
        return None
    raw = m.group(3).strip().rstrip("?!.,;: ")
    value = _canonical_value(record, col, raw)
    return _op_condition(col, value, m.group(2), record)


_COMPARE_OPS = (
    (r"at least|no less than|>=", ">="),
    (r"at most|no more than|<=", "<="),
    (r"above|over|greater than|more than|higher than|exceeding|exceeds|>", ">"),
    (r"below|under|less than|lower than|fewer than|<", "<"),
    (r"equal to|equals|of|=|is", "="),
)
_COMPARE_RE = re.compile(
    r"\b([a-z][a-z0-9_ ]*?)\s+(?:of\s+)?("
    + "|".join(p for p, _ in _COMPARE_OPS)
    + r")\s+(-?\d+(?:\.\d+)?)\b(?!\s*(?:rows|records|items|results|days|customers))"
)


def _comparison_condition(low: str, record) -> str | None:
    """'<numeric column> above/below/of <number>' ("bounce rate above 60",
    "a rating of 5"). The column must be named right before the operator, so
    "top 5 customers" or "first 10 rows" never become filters."""
    for m in _COMPARE_RE.finditer(low):
        words = re.findall(r"[a-z0-9_]+", m.group(1))
        col = None
        for size in (3, 2, 1):  # longest trailing phrase that names a column
            if len(words) < size:
                continue
            want = re.sub(r"[^a-z0-9]", "", "".join(words[-size:]))
            for c in record.columns:
                if re.sub(r"[^a-z0-9]", "", c.name.lower()) == want and _is_numeric_col(record, c.name):
                    col = c.name
                    break
            if col:
                break
        if col is None:
            continue
        op = next(sym for pat, sym in _COMPARE_OPS if re.fullmatch(pat, m.group(2)))
        return f"{_q(col)} {op} {m.group(3)}"
    return None


def _canonical_value(record, col: str, value: str) -> str:
    """Return the original-cased form of ``value`` as it appears in the column
    (case-insensitive match), so 'japan?'/lowercase input still filters on the
    real stored value like 'Japan'."""
    stripped = value.strip().rstrip("?!.,;: ")
    if not stripped:
        return value
    if _is_numeric_col(record, col) and _num(stripped) is not None:
        return stripped
    col_data = record.df[col]
    try:
        match = next((v for v in col_data.dropna().unique() if isinstance(v, str) and v.lower() == stripped.lower()), None)
        if match is not None:
            return match
    except Exception:
        pass
    return stripped


def _category_condition(low: str, record) -> str | None:
    """A capitalized data value mentioned in the question ('in Japan', 'for the North region')."""
    for cand in re.findall(r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)", low):
        value = cand.strip()
        word = value.lower().split()[0]
        if word in _QUESTION_WORDS or word in _ROW_NOUNS or word in _MONTHS:
            continue
        col = _find_value_col(record, value)
        if col:
            return _op_condition(col, value, "=", record)
    return None


def extract_where(low: str, record) -> str | None:
    """Return a SQL WHERE *expression* (no leading WHERE) for an explicit
    condition in the question, or None when nothing can be grounded.

    ``low`` is the original question (case matters: category values are matched
    by their capitalisation); the matchers themselves are case-insensitive.
    """
    question = str(low or "")
    lower = question.lower()
    if not lower or record is None:
        return None
    cond = _year_condition(lower, _date_col(record))
    if cond:
        return cond
    cond = _explicit_condition(lower, record)
    if cond:
        return cond
    cond = _comparison_condition(lower, record)
    if cond:
        return cond
    cond = _category_condition(question, record)
    if cond:
        return cond
    return None


def has_condition(low: str) -> bool:
    """True when the question carries an explicit qualifier that should become a
    WHERE clause (a year, a where-clause, a comparison, or a named category value)."""
    raw = str(low or "")
    low = raw.lower()
    if re.search(r"\bwhere\b", low):
        return True
    if re.search(r"\b(?:in|during|for|since|from|before|after|until|through|throughout|calendar|as of|as at)\s+(?:the\s+)?(?:year\s+)?(20\d{2})\b", low):
        return True
    if re.search(r"\b(?:greater|less|more|lower|higher|equal|equals|at least|at most|between|minimum|maximum)\b", low):
        return True
    if re.search(r"\b(?:above|over|below|under|exceeding|exceeds)\s+-?\d", low):
        return True
    if re.search(r"\b(?:in|from|for)\s+(?:the\s+)?[A-Z]", raw):
        return True
    return False


def sql_has_where(sql: str | None) -> bool:
    return bool(re.search(r"\bwhere\b", sql or "", re.IGNORECASE))


def inject_where(sql: str, condition: str) -> str | None:
    """Insert ``WHERE <condition>`` into a simple single-table SELECT.

    Returns the patched SQL, or None when the statement is too complex to patch
    safely (has a join/subquery/having, or already has a WHERE clause).
    """
    if not condition or sql_has_where(sql):
        return None
    if re.search(r"\b(join|union|having|where)\b", sql, re.IGNORECASE):
        return None
    if re.search(r"\bfrom\s*\(", sql, re.IGNORECASE):
        return None
    m = re.search(r"\b(group by|order by|limit)\b", sql, re.IGNORECASE)
    if m:
        return sql[: m.start()] + f"WHERE {condition} " + sql[m.start():]
    return sql.rstrip().rstrip(";") + f" WHERE {condition}"


def audit_sql(question: str, record, sql: str | None) -> str | None:
    """Return a warning string when the question's explicit condition is missing
    from ``sql`` (i.e. a bare aggregate with no WHERE), else None."""
    if not sql or sql_has_where(sql):
        return None
    low = str(question or "").lower()
    if not has_condition(low):
        return None
    cond = extract_where(str(question or ""), record)
    if cond:
        return (
            f"Your question included a condition that the generated query did not apply "
            f"(expected a WHERE such as `{cond}`); the result above is unfiltered."
        )
    return "Your question included an explicit condition, but the generated query had no WHERE clause and the filter could not be applied."
