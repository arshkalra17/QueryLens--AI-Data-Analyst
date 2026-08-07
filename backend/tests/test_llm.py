"""Unit tests for LLM response parsing helpers."""
from __future__ import annotations

import pytest

from app.core.llm import _parse_json_object


def test_parse_json_object_tolerates_trailing_prose():
    text = '{"tool_calls": [{"name": "x", "args": {}}]}\nSure, here is the plan.'
    assert _parse_json_object(text) == {"tool_calls": [{"name": "x", "args": {}}]}


def test_parse_json_object_takes_first_complete_value():
    text = '{"a": 1} {"b": 2}'
    assert _parse_json_object(text) == {"a": 1}


def test_parse_json_object_markdown_fence():
    assert _parse_json_object("```json\n{\"a\": 1}\n```") == {"a": 1}


def test_parse_json_object_repairs_trailing_commas():
    assert _parse_json_object('{"x": [1, 2,], "y": 3}') == {"x": [1, 2], "y": 3}


def test_parse_json_object_rejects_non_json():
    with pytest.raises(ValueError):
        _parse_json_object("no json here")
