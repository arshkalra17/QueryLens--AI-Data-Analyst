"""Public-demo safeguards (DEMO_MODE=true, used for the Hugging Face Space).

- **Rate limit**: POST requests to /api are limited per client IP over a
  sliding hour, so strangers can't drain the LLM quota. Over the limit the
  API answers 429 and the visitor can retry later.
- **Seeded data**: the sample datasets and business documents are loaded at
  startup, so the first visitor can ask a question immediately.
- **Read-only samples**: DELETE on datasets and documents is refused, so one
  visitor can't empty the demo for everyone else.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque

from ..config import PROJECT_ROOT, get_settings

logger = logging.getLogger(__name__)

_PROTECTED_PREFIXES = ("/api/datasets", "/api/documents")


def is_demo() -> bool:
    return get_settings().demo_mode


def blocks_delete(method: str, path: str) -> bool:
    return is_demo() and method == "DELETE" and path.startswith(_PROTECTED_PREFIXES)


def client_ip(headers, fallback: str | None) -> str:
    """First hop of X-Forwarded-For (set by the Spaces proxy), else the socket peer."""
    forwarded = headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or fallback or "unknown"


class RateLimiter:
    """Sliding-window counter per key. In-memory: one process, resets on restart."""

    def __init__(self, limit: int, window_seconds: float = 3600.0):
        self.limit, self.window = limit, window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> tuple[bool, int]:
        """Record a hit. Returns (allowed, seconds until the oldest hit expires)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False, int(self.window - (now - hits[0])) + 1
            hits.append(now)
            if len(self._hits) > 10_000:  # bound memory: forget idle clients
                for k in [k for k, v in self._hits.items() if not v]:
                    del self._hits[k]
            return True, 0


_limiter: RateLimiter | None = None


def get_limiter() -> RateLimiter | None:
    global _limiter
    limit = get_settings().rate_limit_per_hour
    if limit <= 0:
        return None
    if _limiter is None or _limiter.limit != limit:
        _limiter = RateLimiter(limit)
    return _limiter


def seed_samples() -> None:
    """Load the bundled sample CSVs and documents if they aren't loaded yet."""
    import pandas as pd

    from ..api.routes.datasets import _index_record
    from .registry import get_registry

    registry = get_registry()
    loaded = {r.name for r in registry.list()}
    for folder in (PROJECT_ROOT / "data" / "samples", PROJECT_ROOT / "sample_data"):
        csvs = sorted(folder.glob("*.csv"))
        if not csvs:
            continue
        for path in csvs:
            if path.name in loaded:
                continue
            try:
                record = registry.register(pd.read_csv(path), path.name, f"sample:{path.name}")
                _index_record(record)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Could not seed %s: %s", path.name, exc)
        break
    if get_settings().enable_docs:
        from ..knowledge import definitions
        from ..knowledge.store import get_document_store

        for folder in (PROJECT_ROOT / "data" / "samples" / "docs", PROJECT_ROOT / "sample_data" / "docs"):
            if folder.is_dir():
                store = get_document_store()
                for f in sorted(folder.iterdir()):
                    if f.is_file():
                        store.add(f.name, f.read_bytes())
                definitions.reset_cache()
                break
    logger.info("Demo mode: %d dataset(s) and sample documents ready", len(registry.list()))
