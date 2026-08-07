# ---- Base image ------------------------------------------------------------
FROM python:3.14-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ---- Dependencies ----------------------------------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- App -------------------------------------------------------------------
COPY backend ./backend
COPY frontend ./frontend
COPY data ./data
COPY sample_data ./sample_data
COPY scripts ./scripts

WORKDIR /app/backend

# ---- Runtime ---------------------------------------------------------------
EXPOSE 8000

# Groq is the sole LLM provider (failover to the deterministic rule engine).
ENV LLM_PROVIDER=openai
ENV LLM_MODEL=llama-3.1-8b-instant
ENV APP_API_KEY=

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
