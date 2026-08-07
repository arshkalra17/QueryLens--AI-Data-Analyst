"""Local embedding service based on Hugging Face sentence-transformers.

Fully local (no external API): the model is downloaded once and cached. An
LRU cache avoids re-embedding identical texts, which also saves ChromaDB
writes and keeps repeated queries cheap.
"""
from __future__ import annotations

import threading

from diskcache import Cache
from sentence_transformers import SentenceTransformer

from ..config import PROJECT_ROOT, get_settings

_cache_obj: Cache | None = None
_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                settings = get_settings()
                _model = SentenceTransformer(settings.embedding_model)
    return _model


def _cache() -> Cache:
    global _cache_obj
    if _cache_obj is None:
        settings = get_settings()
        path = PROJECT_ROOT / "data" / "embed_cache"
        _cache_obj = Cache(str(path))
    return _cache_obj


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, caching exact-duplicate embeddings on disk."""
    if not texts:
        return []
    cache = _cache()
    results: list[list[float] | None] = [None] * len(texts)
    to_compute: list[int] = []
    keys: list[str] = []
    for i, t in enumerate(texts):
        key = f"emb:{hash(t)}"
        cached = cache.get(key)
        if cached is not None:
            results[i] = cached
        else:
            to_compute.append(i)
            keys.append(key)
    if to_compute:
        model = _get_model()
        computed = model.encode([texts[i] for i in to_compute], normalize_embeddings=True)
        for j, i in enumerate(to_compute):
            vec = [float(v) for v in computed[j]]
            results[i] = vec
            cache.set(keys[j], vec, expire=86400 * 30)
    return [r for r in results if r is not None]  # type: ignore[return-value]


def embed_one(text: str) -> list[float]:
    return embed([text])[0]


def reset() -> None:
    global _model
    _model = None
