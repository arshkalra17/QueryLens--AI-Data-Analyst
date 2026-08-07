"""Structured logging and lightweight observability (counters + latency histograms)."""
from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from ..config import get_settings


def _no_op(*_args, **_kwargs) -> None:
    return


@contextmanager
def request_context(request_id: str | None = None) -> Iterator[str]:
    """Yield a request id bound to the logging context."""
    rid = request_id or uuid.uuid4().hex[:12]
    # Python logging does not have thread-local contexts built in; emulate via
    # a global that downstream helper functions read.
    _current_request_id.value = rid
    try:
        yield rid
    finally:
        _current_request_id.value = None


class _RequestID:
    value: str | None = None

    def __str__(self) -> str:
        return self.value or "-"


_current_request_id = _RequestID()


class RequestIDFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = str(_current_request_id)
        return True


def setup_logging() -> None:
    settings = get_settings()
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(request_id)s | %(message)s"
        )
    )
    handler.addFilter(RequestIDFilter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level.upper())
    # Quiet noisy third-party loggers.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

@dataclass
class MetricsStore:
    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    latencies: dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=200)))
    _lock: object = None

    def __post_init__(self) -> None:
        import threading
        self._lock = threading.Lock()

    def inc(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.counters[name] += n

    def time(self, name: str, seconds: float) -> None:
        with self._lock:
            self.latencies[name].append(seconds)

    def snapshot(self) -> dict:
        with self._lock:
            counters = dict(self.counters)
            latencies = {}
            for k, v in self.latencies.items():
                if v:
                    latencies[k] = {
                        "count": len(v),
                        "mean_ms": round(sum(v) / len(v) * 1000, 2),
                        "p95_ms": round(sorted(v)[int(len(v) * 0.95)] * 1000, 2)
                        if len(v) >= 20
                        else None,
                    }
            return {"counters": counters, "latencies": latencies}


metrics = MetricsStore()


@contextmanager
def timed(name: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        metrics.time(name, time.perf_counter() - start)
