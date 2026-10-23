"""Tests for the SQL benchmark's result comparison."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from benchmark_scoring import compare  # noqa: E402


def res(columns, rows):
    return SimpleNamespace(columns=columns, rows=[dict(zip(columns, r)) for r in rows])


def test_extra_columns_and_names_are_ignored():
    gold = res(["region"], [["North"], ["South"]])
    pred = res(["dimension", "total"], [["South", 2.0], ["North", 1.0]])
    assert compare(pred, gold, ordered=False) == (True, False)


def test_exact_result_is_strict():
    gold = res(["count_star()"], [[42]])
    pred = res(["count"], [[42.0]])
    assert compare(pred, gold, ordered=False) == (True, True)


def test_floats_compared_at_two_decimals():
    assert compare(res(["x"], [[1.004]]), res(["s"], [[1.0]]), ordered=False)[0]
    assert not compare(res(["x"], [[1.02]]), res(["s"], [[1.0]]), ordered=False)[0]


def test_rows_must_pair_up_not_just_columns():
    gold = res(["k", "v"], [["a", 1], ["b", 2]])
    swapped = res(["k", "v"], [["a", 2], ["b", 1]])
    assert compare(swapped, gold, ordered=False) == (False, False)


def test_topk_accepts_longer_ranking_with_same_prefix():
    gold = res(["rep"], [["Diana"], ["Fatima"]])
    pred = res(["dimension", "total"], [["Diana", 9], ["Fatima", 8], ["Omar", 1]])
    assert compare(pred, gold, ordered=True) == (True, False)
    wrong_order = res(["dimension"], [["Fatima"], ["Diana"]])
    assert compare(wrong_order, gold, ordered=True)[0] is False


def test_unordered_requires_same_row_count():
    gold = res(["region"], [["North"], ["South"]])
    assert compare(res(["r"], [["North"]]), gold, ordered=False)[0] is False


def test_any_of_accepts_any_tied_winner():
    gold = res(["city"], [["Hyderabad"], ["Pune"]])
    assert compare(res(["dimension", "n"], [["Pune", 229], ["Kolkata", 218]]), gold, ordered=True, any_of=True)[0]
    assert not compare(res(["dimension", "n"], [["Kolkata", 218]]), gold, ordered=True, any_of=True)[0]


def test_null_ranked_first_is_a_mismatch():
    gold = res(["rep"], [["Diana"]])
    pred = res(["dimension", "total"], [[None, 1.0], ["Diana", 2.0]])
    assert compare(pred, gold, ordered=True)[0] is False
