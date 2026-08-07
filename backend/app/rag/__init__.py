"""Retrieval-augmented generation (RAG) layer.

Pipeline: upload -> table profiling -> embedding (sentence-transformers) ->
ChromaDB index -> per-question retrieval (table profiles + past Q&A) ->
grounded prompt construction.

The Groq LLM never sees raw datasets dumped into the prompt; it is grounded by
the retrieved, compact, semantically-relevant context built here.
"""

from .profiling import build_table_profile, profile_text_for_record  # noqa: F401
from .retrieval import (  # noqa: F401
    get_rag,
    reset_rag,
    retrieval_context,
)
