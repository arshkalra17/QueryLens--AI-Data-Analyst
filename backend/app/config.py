"""Application configuration loaded from environment variables / .env file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    app_name: str = "AI Data Analyst"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # --- LLM provider ---
    # "openai"   : use an OpenAI-compatible API (Groq, OpenAI, Azure, LM Studio, ...)
    # "none"     : force the built-in rule-based engine (no API key required)
    llm_provider: str = "openai"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1200

    # --- Groq (primary LLM provider) ---
    # Spec-first configuration: if GROQ_API_KEY is set it wins, pointing the
    # OpenAI-compatible client at Groq's endpoint with the requested model.
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.1-8b-instant"

    # --- LLM failover ---
    # Optional secondary providers (tried in order if primary fails)
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"

    @model_validator(mode="after")
    def _groq_wins(self) -> "Settings":
        if self.groq_api_key:
            self.openai_api_key = self.groq_api_key
            self.openai_base_url = self.groq_base_url
            if self.llm_model == "gpt-4o-mini":
                self.llm_model = self.groq_model
        return self

    # --- Security ---
    # If set, all /api endpoints require `Authorization: Bearer <APP_API_KEY>`.
    app_api_key: str | None = None

    # --- Uploads / storage ---
    max_upload_mb: int = 50
    data_dir: Path = PROJECT_ROOT / "data"
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    max_datasets: int = 25
    max_sql_chars: int = 4000
    sql_timeout_seconds: float = 10.0

    # --- Caching ---
    cache_ttl_seconds: int = 3600
    enable_cache: bool = True

    # --- Agent ---
    conversation_window: int = 8        # recent messages fed to the LLM
    max_sql_rows: int = 500             # rows returned by run_sql to the LLM
    allow_pandas_exec: bool = True      # execute generated pandas snippets

    # --- RAG (retrieval-augmented generation) ---
    enable_rag: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    rag_store_dir: Path = PROJECT_ROOT / "data" / "rag_store"
    rag_k_table: int = 3                # table profiles retrieved per query
    rag_k_history: int = 4              # past Q&A pairs retrieved per query
    max_chunk_chars: int = 800

    @property
    def upload_dir_resolved(self) -> Path:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        return self.upload_dir

    @property
    def llm_available(self) -> bool:
        """True when a real LLM endpoint is configured."""
        if self.llm_provider == "none":
            return False
        if self.llm_provider == "local":
            return bool(self.openai_base_url)
        return bool(self.openai_api_key or self.groq_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
