"""Tests for the condition->WHERE grounding helpers in app.engine.conditions."""
from __future__ import annotations

import pandas as pd

from app.core.registry import DataFrameRegistry
from app.engine.conditions import (
    extract_where,
    has_condition,
    inject_where,
    sql_has_where,
)


def _record():
    df = pd.DataFrame(
        {
            "Customer Id": [f"C{i:04d}" for i in range(40)],
            "Subscription Date": ["2021-03-15" if i % 2 else "2020-11-02" for i in range(40)],
            "Country": ["India" if i % 3 == 0 else "Japan" for i in range(40)],
        }
    )
    reg = DataFrameRegistry()
    return reg.register(df, "subs.csv", "test")


def test_has_condition_detection():
    assert has_condition("how many customers subscribed in 2021?")
    assert has_condition("how many customers are in India?")
    assert has_condition("customers where Country = Japan")
    assert not has_condition("how many customers are in the dataset?")


def test_extract_where_year():
    rec = _record()
    assert '2021-01-01' in extract_where("how many customers subscribed in 2021?", rec)
    assert extract_where("how many customers are in the dataset?", rec) is None


def test_extract_where_category_and_explicit():
    rec = _record()
    w = extract_where("how many customers are in India?", rec)
    assert w and '"Country" = \'India\'' in w, w
    w = extract_where("how many customers where Country = Japan?", rec)
    assert w and '"Country" = \'Japan\'' in w, w


def test_sql_has_where():
    assert sql_has_where('SELECT COUNT(*) AS c FROM "t" WHERE x = 1')
    assert not sql_has_where('SELECT COUNT(*) AS c FROM "t"')
    assert not sql_has_where('SELECT COUNT(*) AS c FROM "t" GROUP BY y')


def test_inject_where_positions():
    rec = _record()
    cond = extract_where("how many customers subscribed in 2021?", rec)

    sql = 'SELECT COUNT(*) AS c FROM "t"'
    assert inject_where(sql, cond) == f'{sql} WHERE {cond}'

    sql = 'SELECT "Country", COUNT(*) AS c FROM "t" GROUP BY "Country"'
    assert inject_where(sql, cond) == f'SELECT "Country", COUNT(*) AS c FROM "t" WHERE {cond} GROUP BY "Country"'

    sql = 'SELECT "Country", COUNT(*) AS c FROM "t" GROUP BY "Country" ORDER BY c DESC LIMIT 10'
    assert inject_where(sql, cond).endswith(f' WHERE {cond} GROUP BY "Country" ORDER BY c DESC LIMIT 10')

    # already has a WHERE, or is a join/subquery we must not mangle
    assert inject_where('SELECT * FROM "t" WHERE x = 1', cond) is None
    assert inject_where('SELECT * FROM "t" JOIN "u" ON t.id = u.id', cond) is None
    assert inject_where('SELECT * FROM (SELECT 1)', cond) is None


def test_audit_sql_warns_when_condition_dropped():
    from app.engine.conditions import audit_sql

    rec = _record()
    warn = audit_sql("How many customers subscribed in 2021?", rec, 'SELECT COUNT(*) AS c FROM "t"')
    assert warn is not None and "2021" in warn, warn
    assert audit_sql("How many customers subscribed in 2021?", rec, 'SELECT COUNT(*) AS c FROM "t" WHERE 1=1') is None
    assert audit_sql("How many customers are in the dataset?", rec, 'SELECT COUNT(*) AS c FROM "t"') is None
