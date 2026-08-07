"""In-memory TTL cache with an LRU flavour, used for tool results and LLM calls."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any

from ..config import get_settings


class TTLCache:
    def __init__(self, ttl_seconds: float | None = None, max_items: int = 512) -> None:
        self._ttl = ttl_seconds if ttl_seconds is not None else get_settings().cache_ttl_seconds
        self._max_items = max_items
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def _now(self) -> float:
        return time.monotonic()

    def _evict(self) -> None:
        if len(self._store) <= self._max_items:
            return
        # drop expired then oldest by insertion order (dict preserves insertion order)
        expired = [k for k, (ts, _) in self._store.items() if self._now() - ts > self._ttl]
        for k in expired:
            del self._store[k]
        while len(self._store) > self._max_items:
            self._store.pop(next(iter(self._store)))

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            ts, value = item
            if self._now() - ts > self._ttl:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any) -> None:
        if not get_settings().enable_cache:
            return
        with self._lock:
            self._store[key] = (self._now(), value)
            self._evict()

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def stats(self) -> dict:
        with self._lock:
            return {"items": len(self._store), "ttl_seconds": self._ttl, "max_items": self._max_items}


def stable_key(*parts: Any) -> str:
    """Deterministic hash key from arbitrary JSON-serialisable parts."""
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
