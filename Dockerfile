# syntax=docker/dockerfile:1
# -----------------------------------------------------------------------------
# QueryLens — single runtime image.
#
# The project is one Python application: FastAPI serves the REST API + the
# static web app on :8000, and Streamlit runs as an alternate UI client on
# :8501 reusing the same dependencies. A single image therefore serves both
# services; docker-compose.yml just overrides the entrypoint for the UI.
# -----------------------------------------------------------------------------

FROM python:3.14-slim

LABEL org.opencontainers.image.title="QueryLens" \
      org.opencontainers.image.description="RAG + SQL-powered AI data analyst (Groq, DuckDB, ChromaDB)" \
      org.opencontainers.image.source="https://github.com/arshkalra17/QueryLens--AI-Data-Analyst" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # keep the Hugging Face embedding-model cache inside the named volume
    HF_HOME=/app/data/embed_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/data/embed_cache \
    TRANSFORMERS_CACHE=/app/data/embed_cache

WORKDIR /app

# --- Non-root runtime user (fixed UID so named volumes keep the same owner) ---
RUN groupadd --system --gid 1000 appuser \
    && useradd --system --uid 1000 --gid appuser --home-dir /app --shell /usr/sbin/nologin appuser

# --- Dependencies first (cache layer until requirements.txt changes) ---------
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# --- Application code --------------------------------------------------------
COPY backend ./backend
COPY frontend ./frontend
COPY scripts ./scripts

# --- Seed sample CSVs served by the "load samples" button --------------------
COPY sample_data/*.csv ./data/samples/

# --- Runtime data dirs (uploaded CSVs, RAG vector store, model cache) --------
RUN mkdir -p /app/data/uploads /app/data/rag_store /app/data/embed_cache /app/data/sessions \
    && chown -R appuser:appuser /app

USER appuser

WORKDIR /app/backend

EXPOSE 8000 8501

# Groq is the sole LLM provider (config.py falls back to the deterministic rule
# engine when no API key is configured).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
