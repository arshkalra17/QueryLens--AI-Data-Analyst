"""Hybrid retrieval over document chunks.

Four modes, so each stage's contribution can be measured (see benchmarks/):

- ``dense``          cosine similarity of local MiniLM embeddings (Chroma)
- ``bm25``           keyword relevance (Okapi BM25)
- ``hybrid``         both candidate lists fused with Reciprocal Rank Fusion
- ``hybrid_rerank``  hybrid candidates re-scored by a cross-encoder that reads
                     the question and passage together (default)

Dense retrieval catches paraphrases ("customers who bought recently" ~ "active
customer"); BM25 catches exact terms, codes and column names that small
embedding models blur; the cross-encoder fixes the ordering. RRF needs no score
calibration between the two retrievers, only ranks.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from ..config import get_settings
from .chunking import Chunk
from .store import DocumentStore, get_document_store, tokenize

logger = logging.getLogger(__name__)

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")
RRF_K = 60  # standard RRF constant (Cormack et al., 2009)
DENSE_MIN_SIMILARITY = 0.35

_reranker = None
_reranker_lock = threading.Lock()


@dataclass
class Hit:
    chunk: Chunk
    score: float                        # final score used for ordering
    dense_rank: int | None = None       # 1-based ranks in each retriever (None = not retrieved)
    bm25_rank: int | None = None
    rerank_score: float | None = None
    signals: dict = field(default_factory=dict)

    def to_api(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id, "doc_id": self.chunk.doc_id, "doc_name": self.chunk.doc_name,
            "heading": self.chunk.heading, "page": self.chunk.page, "citation": self.chunk.citation(),
            "text": self.chunk.text, "score": round(self.score, 4), "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 3),
        }


@dataclass
class SearchResult:
    hits: list[Hit]
    mode: str
    covered: bool          # False -> the documents don't answer this question

    def to_api(self) -> dict:
        return {"mode": self.mode, "covered": self.covered, "hits": [h.to_api() for h in self.hits]}


def search(query: str, k: int | None = None, mode: str | None = None,
           doc_ids: list[str] | None = None, store: DocumentStore | None = None) -> SearchResult:
    settings = get_settings()
    store = store or get_document_store()
    mode = mode or settings.doc_retrieval_mode
    if mode not in MODES:
        raise ValueError(f"unknown retrieval mode {mode!r}; use one of {MODES}")
    k = k or settings.doc_top_k
    n_cand = max(settings.doc_candidates, k)
    chunks = store.chunks()
    if doc_ids is not None:
        allowed = set(doc_ids)
        chunks = {cid: c for cid, c in chunks.items() if c.doc_id in allowed}
    if not query.strip() or not chunks:
        return SearchResult(hits=[], mode=mode, covered=False)

    dense = _dense(query, store, n_cand, set(chunks)) if mode != "bm25" else []
    keyword = _bm25(query, store, n_cand, set(chunks)) if mode != "dense" else []
    dense_rank = {cid: i + 1 for i, (cid, _) in enumerate(dense)}
    bm25_rank = {cid: i + 1 for i, (cid, _) in enumerate(keyword)}

    if mode == "dense":
        ordered = [(cid, 1.0 - dist) for cid, dist in dense]
    elif mode == "bm25":
        ordered = list(keyword)
    else:
        fused: dict[str, float] = {}
        for ranks in (dense_rank, bm25_rank):
            for cid, r in ranks.items():
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + r)
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

    dense_sim = {cid: 1.0 - dist for cid, dist in dense}
    hits = [Hit(chunk=chunks[cid], score=float(s), dense_rank=dense_rank.get(cid), bm25_rank=bm25_rank.get(cid),
                signals={"dense_similarity": dense_sim.get(cid)})
            for cid, s in ordered[:n_cand] if cid in chunks]

    if mode == "hybrid_rerank" and hits:
        scores = _rerank(query, [h.chunk.index_text for h in hits])
        for h, s in zip(hits, scores):
            h.rerank_score = float(s)
            h.score = float(s)
        hits.sort(key=lambda h: h.score, reverse=True)

    hits = hits[:k]
    return SearchResult(hits=hits, mode=mode, covered=_covered(hits, mode))


def _covered(hits: list[Hit], mode: str) -> bool:
    """Whether the best passage is relevant enough to answer from.

    Only the cross-encoder produces a calibrated relevance score. The other
    modes fall back to weaker heuristics: a keyword (BM25) match, or an
    embedding similarity of at least ``DENSE_MIN_SIMILARITY``. How much worse
    those abstain is part of what the ablation measures.
    """
    if not hits:
        return False
    top = hits[0]
    if top.rerank_score is not None:
        return top.rerank_score >= get_settings().doc_min_rerank_score
    sim = top.signals.get("dense_similarity")
    return top.bm25_rank is not None or (sim is not None and sim >= DENSE_MIN_SIMILARITY)


def _dense(query: str, store: DocumentStore, n: int, allowed: set[str]) -> list[tuple[str, float]]:
    from ..rag import embeddings

    coll = store.collection()
    total = coll.count()
    if total == 0:
        return []
    res = coll.query(query_embeddings=[embeddings.embed_one(query)], n_results=min(total, n * 3), include=["distances"])
    out = [(cid, float(d)) for cid, d in zip(res["ids"][0], res["distances"][0]) if cid in allowed]
    return out[:n]


def _bm25(query: str, store: DocumentStore, n: int, allowed: set[str]) -> list[tuple[str, float]]:
    index, ids = store.bm25()
    q = tokenize(query)
    if index is None or not q:
        return []
    scores = index.get_scores(q)
    ranked = sorted(((cid, float(s)) for cid, s in zip(ids, scores) if s > 0 and cid in allowed),
                    key=lambda kv: kv[1], reverse=True)
    return ranked[:n]


def _rerank(query: str, passages: list[str]) -> list[float]:
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                from sentence_transformers import CrossEncoder

                _reranker = CrossEncoder(get_settings().reranker_model)
    return list(_reranker.predict([(query, p) for p in passages]))
