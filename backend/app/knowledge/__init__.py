"""Document RAG: ingest business documents and answer from them with citations.

Pipeline: parse (PDF/MD/TXT/DOCX) -> heading-aware chunks with overlap ->
content-hash dedup -> local embeddings (Chroma) + BM25 -> reciprocal rank
fusion -> cross-encoder rerank -> relevance gate -> cited answer (LLM or
extractive). Definitions with SQL implementations also ground text-to-SQL.
"""
from .store import get_document_store, reset_document_store  # noqa: F401
