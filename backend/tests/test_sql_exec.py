"""Tests for SQL generation and execution (DuckDB)."""
from __future__ import annotations

import pandas as pd
import pytest

from app.engine.sql_exec import generate_select_sql, run_sql, validate_sql


def test_validate_rejects_write_statements():
    with pytest.raises(ValueError):
        validate_sql("DROP TABLE foo")
    with pytest.raises(ValueError):
        validate_sql("DELETE FROM t WHERE 1=1")
    with pytest.raises(ValueError):
        validate_sql("COPY t TO 'x.csv'")


def test_validate_rejects_non_select():
    with pytest.raises(ValueError):
        validate_sql("SHOW TABLES")


def test_basic_aggregation(simple_df):
    result = run_sql('SELECT "city", SUM("value") AS total FROM "cities" GROUP BY "city" ORDER BY total DESC', {"cities": simple_df})
    assert result.error is None
    assert result.columns == ["city", "total"]
    assert result.rows[0]["city"] == "Delhi"


def test_join_across_tables(simple_df):
    other = pd.DataFrame({"city": ["Mumbai", "Delhi"], "bonus": [10.0, 20.0]})
    sql = 'SELECT a."city", SUM(a."value") + b."bonus" AS adj FROM "cities" a JOIN "extra" b ON a."city" = b."city" GROUP BY a."city", b."bonus"'
    result = run_sql(sql, {"cities": simple_df, "extra": other})
    assert result.error is None
    assert len(result.rows) >= 1


def test_sql_error_reported_not_raised(simple_df):
    result = run_sql('SELECT * FROM "missing_table"', {"cities": simple_df})
    assert result.error is not None


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT content FROM read_text('{path}')",
        "SELECT * FROM read_csv_auto('{path}')",
        "SELECT * FROM '{path}'",
        "SELECT * FROM glob('{dir}/*')",
    ],
)
def test_sandbox_blocks_filesystem_reads(simple_df, tmp_path, sql):
    secret = tmp_path / "secret.env"
    secret.write_text("GROQ_API_KEY=should-never-leak\n")
    # these pass the keyword validator, so isolation must come from DuckDB itself
    query = sql.format(path=secret, dir=tmp_path)
    validate_sql(query)
    result = run_sql(query, {"cities": simple_df})
    assert result.error is not None
    assert result.rows == []


def test_sandbox_setting_is_locked_off(simple_df):
    result = run_sql(
        "SELECT current_setting('enable_external_access') AS v FROM \"cities\" LIMIT 1",
        {"cities": simple_df},
    )
    assert result.error is None
    assert str(result.rows[0]["v"]).lower() == "false"


def test_generate_select_sql(simple_df):
    sql = generate_select_sql("cities", ["city", "value"], limit=5)
    assert "city" in sql and "LIMIT 5" in sql
