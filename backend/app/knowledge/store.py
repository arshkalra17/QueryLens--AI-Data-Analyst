"""Persistent document store: raw files, a manifest, Chroma vectors and BM25.

- Raw uploads live in ``DOCS_DIR/files`` and a JSON manifest in
  ``DOCS_DIR/manifest.json``; the Chroma collection ``doc_chunks`` lives in
  ``DOCS_DIR/chroma``.
- Chunk ids are content hashes, so re-uploading an edited document embeds only
  the chunks that changed and deletes the ones that disappeared.
- The BM25 index is rebuilt in memory from the stored chunks (cheap at this
  scale) whenever the corpus changes.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import get_settings
from .chunking import Chunk, chunk_sections
from .parsing import parse_document

logger = logging.getLogger(__name__)

COLLECTION = "doc_chunks"

_STOPWORDS = frozenset(
    "a an the and or of to in on for by with at from as is are was were be been it its this that these "
    "those what which who whom how when where why do does did our your their we you they i me my "
    "there here than then so if not no can could should would will about into per".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens without stopwords, lightly stemmed so "exported",
    "exporting" and "exports" all match "export"."""
    out = []
    for tok in re.findall(r"[a-z0-9]+(?:_[a-z0-9]+)*", text.lower()):
        if tok in _STOPWORDS:
            continue
        out.append(_stem(tok))
    return out


def _stem(tok: str) -> str:
    for suffix in ("ing", "ed"):
        if len(tok) > len(suffix) + 3 and tok.endswith(suffix):
            return tok[: -len(suffix)]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


@dataclass
class DocumentMeta:
    doc_id: str
    name: str
    sha256: str
    size: int
    uploaded_at: float
    chunk_ids: list[str] = field(default_factory=list)
    pages: int | None = None

    def to_api(self) -> dict:
        return {
            "doc_id": self.doc_id, "name": self.name, "size": self.size,
            "uploaded_at": self.uploaded_at, "chunks": len(self.chunk_ids), "pages": self.pages,
        }


class DocumentStore:
    def __init__(self, root: Path | None = None) -> None:
        settings = get_settings()
        self.root = Path(root or settings.docs_dir)
        self.files_dir = self.root / "files"
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root / "manifest.json"
        self._lock = threading.RLock()
        self._docs: dict[str, DocumentMeta] = self._load_manifest()
        self._client: Any = None
        self._chunks: dict[str, Chunk] | None = None   # chunk_id -> Chunk (lazy)
        self._bm25: Any = None
        self._bm25_ids: list[str] = []

    # ------------------------------------------------------------------ public
    def list(self) -> list[DocumentMeta]:
        return sorted(self._docs.values(), key=lambda d: d.uploaded_at)

    def get(self, doc_id: str) -> DocumentMeta | None:
        return self._docs.get(doc_id)

    def add(self, filename: str, data: bytes) -> dict:
        """Parse, chunk, embed (only new chunks) and index a document. A document
        with the same name replaces the previous version."""
        from ..rag import embeddings

        settings = get_settings()
        if len(data) > settings.max_doc_mb * 1024 * 1024:
            raise ValueError(f"document exceeds {settings.max_doc_mb} MB")
        name = Path(filename).name
        sha = hashlib.sha256(data).hexdigest()
        sections = parse_document(name, data)
        doc_id = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
        chunks = chunk_sections(sections, doc_id, name, settings.doc_chunk_chars, settings.doc_chunk_overlap)

        with self._lock:
            coll = self._collection()
            previous = self._docs.get(doc_id)
            new_ids = [c.chunk_id for c in chunks]
            existing = set(coll.get(ids=new_ids, include=[])["ids"]) if new_ids else set()
            to_embed = [c for c in chunks if c.chunk_id not in existing]
            if to_embed:
                vectors = embeddings.embed([c.index_text for c in to_embed])
                coll.upsert(
                    ids=[c.chunk_id for c in to_embed],
                    embeddings=vectors,
                    documents=[c.index_text for c in to_embed],
                    metadatas=[{**c.metadata(), "text": c.text} for c in to_embed],
                )
            stale = [cid for cid in (previous.chunk_ids if previous else []) if cid not in set(new_ids)]
            if stale:
                coll.delete(ids=stale)
            (self.files_dir / f"{doc_id}__{name}").write_bytes(data)
            pages = max((s.page or 0) for s in sections) or None
            self._docs[doc_id] = DocumentMeta(
                doc_id=doc_id, name=name, sha256=sha, size=len(data),
                uploaded_at=time.time(), chunk_ids=new_ids, pages=pages,
            )
            self._save_manifest()
            self._invalidate()
        logger.info("Indexed document %s: %d chunks (%d embedded, %d reused, %d removed)",
                    name, len(chunks), len(to_embed), len(chunks) - len(to_embed), len(stale))
        return {
            **self._docs[doc_id].to_api(),
            "embedded": len(to_embed), "reused": len(chunks) - len(to_embed), "removed": len(stale),
        }

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            meta = self._docs.pop(doc_id, None)
            if meta is None:
                return False
            if meta.chunk_ids:
                self._collection().delete(ids=meta.chunk_ids)
            for f in self.files_dir.glob(f"{doc_id}__*"):
                f.unlink(missing_ok=True)
            self._save_manifest()
            self._invalidate()
            return True

    def chunks(self) -> dict[str, Chunk]:
        """All indexed chunks by id (loaded from Chroma once, then cached)."""
        with self._lock:
            if self._chunks is None:
                self._chunks = {}
                if self._docs:
                    res = self._collection().get(include=["documents", "metadatas"])
                    for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
                        self._chunks[cid] = Chunk(
                            chunk_id=cid, doc_id=meta["doc_id"], doc_name=meta["doc_name"],
                            text=meta.get("text", doc), index_text=doc, heading=meta.get("heading", ""),
                            page=meta.get("page") or None, position=int(meta.get("position", 0)),
                        )
            return self._chunks

    def bm25(self):
        """(BM25Okapi index, ordered chunk ids), rebuilt when the corpus changes."""
        from rank_bm25 import BM25Okapi

        with self._lock:
            if self._bm25 is None:
                chunks = self.chunks()
                self._bm25_ids = list(chunks)
                corpus = [tokenize(chunks[cid].index_text) or ["_"] for cid in self._bm25_ids]
                self._bm25 = BM25Okapi(corpus) if corpus else None
            return self._bm25, self._bm25_ids

    def collection(self):
        return self._collection()

    # ----------------------------------------------------------------- internal
    def _collection(self):
        if self._client is None:
            import chromadb

            (self.root / "chroma").mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.root / "chroma"))
        try:
            return self._client.get_collection(COLLECTION)
        except Exception:
            return self._client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    def _invalidate(self) -> None:
        self._chunks = None
        self._bm25 = None
        self._bm25_ids = []

    def _load_manifest(self) -> dict[str, DocumentMeta]:
        if not self._manifest_path.exists():
            return {}
        try:
            raw = json.loads(self._manifest_path.read_text())
            return {d["doc_id"]: DocumentMeta(**d) for d in raw}
        except Exception as exc:  # corrupt manifest: start empty rather than crash
            logger.warning("Ignoring unreadable document manifest: %s", exc)
            return {}

    def _save_manifest(self) -> None:
        tmp = self._manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps([d.__dict__ for d in self._docs.values()], indent=1))
        tmp.replace(self._manifest_path)


_store: DocumentStore | None = None
_store_lock = threading.Lock()


def get_document_store() -> DocumentStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = DocumentStore()
    return _store


def reset_document_store() -> None:
    global _store
    _store = None
