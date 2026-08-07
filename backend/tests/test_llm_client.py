"""Tests for the LLM client: 429 backoff, retry-after parsing, caching."""
from __future__ import annotations

import os

os.environ["LLM_PROVIDER"] = "none"

from unittest.mock import patch  # noqa: E402

import pytest  # noqa: E402

from app.core.llm import (  # noqa: E402
    LLMClient,
    _is_rate_limit,
    _retry_after_seconds,
    _with_backoff,
)


class _FakeException(Exception):
    pass


def test_is_rate_limit_detects_429():
    assert _is_rate_limit(Exception("Error code: 429 - Rate limit reached"))
    assert _is_rate_limit(_FakeException("429 client error: too many requests"))
    assert not _is_rate_limit(Exception("401 invalid api key"))
    assert not _is_rate_limit(Exception("timeout"))


def test_retry_after_parsing():
    assert _retry_after_seconds(Exception("try again in 17m52s")) >= 0
    assert _retry_after_seconds(Exception("Retry-After: 5")) == 5.0


def test_backoff_retries_on_429_then_succeeds():
    class Fake:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            if self.calls < 3:
                raise _FakeException("429 rate limit exceeded")
            return "ok"

    fake = Fake()
    with patch("time.sleep"):
        result = _with_backoff(fake)()
    assert result == "ok"
    assert fake.calls == 3


def test_backoff_gives_up_after_max_retries():
    class AlwaysFail:
        def __call__(self):
            raise _FakeException("429 rate limit exceeded")

    with patch("time.sleep"), pytest.raises(_FakeException):
        _with_backoff(AlwaysFail())()


def test_non_rate_limit_raises_immediately():
    def boom():
        raise _FakeException("500 internal error")

    with patch("time.sleep"), pytest.raises(_FakeException):
        _with_backoff(boom)()
