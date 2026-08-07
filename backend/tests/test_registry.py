"""Tests for the dataset registry and sessions."""
from __future__ import annotations

import pandas as pd
import pytest

from app.core.registry import DataFrameRegistry, classify_columns


def test_register_and_get(simple_df):
    reg = DataFrameRegistry()
    rec = reg.register(simple_df, "cities.csv", "test")
    assert rec.dataset_id
    assert rec.row_count == 6
    assert rec.col_count == 4
    fetched = reg.get(rec.dataset_id)
    assert fetched is not None
    assert fetched.name == "cities.csv"


def test_register_duplicate_columns_rejected(simple_df):
    reg = DataFrameRegistry()
    df = simple_df.copy()
    df.columns = ["a", "a", "b", "c"]
    with pytest.raises(ValueError):
        reg.register(df, "bad.csv", "test")


def test_columns_classified(simple_df):
    cols = classify_columns(simple_df)
    by = {c.name: c for c in cols}
    assert by["city"].categorical and not by["city"].numeric
    assert by["value"].numeric
    assert by["value"].unique_count == 6


def test_datetime_coercion():
    df = pd.DataFrame({"date": ["2023-01-01", "2023-01-02", "2023-01-03"], "v": [1, 2, 3]})
    reg = DataFrameRegistry()
    rec = reg.register(df, "ts.csv", "test")
    rec_col = {c.name: c for c in rec.columns}
    assert rec_col["date"].datetime


def test_delete_and_list(simple_df):
    reg = DataFrameRegistry()
    a = reg.register(simple_df, "a.csv", "t")
    b = reg.register(simple_df, "b.csv", "t")
    assert len(reg.list()) == 2
    assert reg.delete(a.dataset_id) is True
    assert reg.get(a.dataset_id) is None
    assert len(reg.list()) == 1
    assert reg.delete("nope") is False


def test_persist_and_reload(tmp_path, simple_df):
    reg = DataFrameRegistry(upload_dir=tmp_path)
    rec = reg.register(simple_df, "cities.csv", "t")
    reg.persist(rec)
    # name keeps its original extension: "cities.csv" persists as "cities.csv"
    # (not the old mangled "cities_csv.csv" which reloaded as "..._csv.csv" and
    # produced double "_csv_csv" table names)
    assert (tmp_path / "cities.csv").exists()
    assert not (tmp_path / "cities_csv.csv").exists()
    reg2 = DataFrameRegistry(upload_dir=tmp_path)
    assert reg2.startup() == 1
    loaded = reg2.list()[0]
    assert loaded.row_count == 6
    assert loaded.name == "cities.csv"


def test_schema_snippet_contains_columns(simple_df):
    reg = DataFrameRegistry()
    rec = reg.register(simple_df, "cities.csv", "t")
    snippet = rec.schema_snippet()
    assert "city" in snippet
    assert "value" in snippet
