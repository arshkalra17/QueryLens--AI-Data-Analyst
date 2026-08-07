# ---- Base image ------------------------------------------------------------
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # keep the Hugging Face embedding-model cache inside the bind-mounted volume
    HF_HOME=/app/data/embed_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/data/embed_cache

WORKDIR /app

# ---- Dependencies ----------------------------------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- App -------------------------------------------------------------------
COPY backend ./backend
COPY frontend ./frontend
COPY scripts ./scripts

# Sample CSVs served by the "load samples" button (data/ is bind-mounted at
# runtime, so only seed the read-only samples here).
COPY sample_data/*.csv ./data/samples/
RUN mkdir -p /app/data/uploads /app/data/rag_store /app/data/embed_cache

WORKDIR /app/backend

EXPOSE 8000 8501

# Groq is the sole LLM provider (failover to the deterministic rule engine).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
