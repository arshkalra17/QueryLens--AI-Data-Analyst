"""LLM client abstraction.

Supports any OpenAI-compatible chat-completions endpoint (Groq, OpenAI, Azure,
Together, LM Studio, Ollama, Google AI Studio, ...) with automatic failover
between configured providers. A "none" provider uses the built-in rule engine.

Rate-limit behaviour (Groq free tier):
- HTTP 429 responses are retried honouring the ``Retry-After`` header with
  exponential backoff, so the app throttles itself instead of erroring.
- ``LLMClient.complete`` is wrapped in an exact-match LRU cache so repeated
  identical questions do not burn extra requests/tokens.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Iterator

from openai import OpenAI

from ..config import get_settings
from .logging_setup import metrics

logger = logging.getLogger(__name__)

_MAX_429_RETRIES = 3
_CACHE_SIZE = 64


class LLMUsage:
    """Process-wide LLM usage counters for evaluation runs and /api/metrics.

    Records requests, tokens (when the provider reports them), latency,
    rate-limit retries, failures and cache hits. Never records prompts,
    responses or credentials.
    """

    def __init__(self) -> None:
        import threading

        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.requests = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.latency_ms = 0.0
        self.rate_limit_retries = 0
        self.failures = 0
        self.cache_hits = 0

    def record(self, latency_ms: float, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        with self._lock:
            self.requests += 1
            self.latency_ms += latency_ms
            self.prompt_tokens += int(prompt_tokens or 0)
            self.completion_tokens += int(completion_tokens or 0)

    def bump(self, field: str) -> None:
        with self._lock:
            setattr(self, field, getattr(self, field) + 1)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "requests": self.requests,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "avg_latency_ms": round(self.latency_ms / self.requests, 1) if self.requests else 0.0,
                "rate_limit_retries": self.rate_limit_retries,
                "failures": self.failures,
                "cache_hits": self.cache_hits,
            }


usage = LLMUsage()


@contextmanager
def metrics_time(name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        metrics.time(name, time.perf_counter() - start)


class LLMError(RuntimeError):
    pass


def _is_rate_limit(exc: Exception) -> bool:
    """Best-effort detection of an HTTP 429 (rate limit) error."""
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "status_code" in text and "429" in text


def _retry_after_seconds(exc: Exception) -> float:
    text = str(exc)
    import re

    m = re.search(r"retry[-_ ]?after[^\d]*(\d+)", text, re.IGNORECASE)
    if m:
        return min(10.0, max(0.0, float(m.group(1))))
    return 2.0


def _with_backoff(fn):
    """Retry ``fn`` on HTTP 429 up to ``_MAX_429_RETRIES`` times, honouring
    Retry-After. Called inside the provider wrapper so the retry lives next to
    the API call."""

    def wrapper(*args, **kwargs):
        delay = 1.0
        for attempt in range(_MAX_429_RETRIES):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                if not _is_rate_limit(exc) or attempt == _MAX_429_RETRIES - 1:
                    raise
                wait = max(_retry_after_seconds(exc), delay)
                usage.bump("rate_limit_retries")
                logger.warning(
                    "Rate limited (%s); backing off %.1fs (attempt %d/%d)",
                    args[0] if args else "?",
                    wait,
                    attempt + 1,
                    _MAX_429_RETRIES,
                )
                time.sleep(wait)
                delay *= 2
        raise RuntimeError("unreachable")

    return wrapper


class _Provider:
    def __init__(self, name: str, base_url: str | None, api_key: str, model: str, temperature: float, max_tokens: int):
        self.name = name
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: OpenAI | None = None
        self._json_mode = True  # turned off if the endpoint rejects response_format
        self._init_client()

    def _init_client(self) -> None:
        # Bound the openai SDK's own HTTP timeout and internal retries so a
        # slow/stalling endpoint (e.g. Groq rate-limited) fails fast instead of
        # blocking the request loop for many seconds.
        kwargs: dict[str, Any] = {"timeout": 30.0, "max_retries": 0}
        if self.name == "local":
            kwargs.update(base_url=self.base_url or "http://localhost:11434/v1", api_key="local")
            self._client = OpenAI(**kwargs)
        elif self.name in ("openai", "groq") and self.api_key:
            kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        elif self.name == "gemini" and self.api_key:
            # Google AI Studio OpenAI-compatible endpoint
            kwargs.update(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=self.api_key
            )
            self._client = OpenAI(**kwargs)

    @property
    def available(self) -> bool:
        return self._client is not None

    def complete(self, messages: list[dict], max_tokens: int | None = None, temperature: float | None = None,
                 response_format: dict | None = None) -> str:
        if self._client is None:
            raise LLMError(f"Provider {self.name} not configured")
        try:
            started = time.perf_counter()
            extra = {"response_format": response_format} if response_format else {}
            with metrics_time(f"llm.complete.{self.name}"):
                resp = _with_backoff(self._client.chat.completions.create)(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature if temperature is None else temperature,
                    max_tokens=max_tokens or self.max_tokens,
                    **extra,
                )
            u = getattr(resp, "usage", None)
            usage.record((time.perf_counter() - started) * 1000,
                         getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))
            return resp.choices[0].message.content or ""
        except Exception as exc:
            usage.bump("failures")
            logger.warning("LLM completion failed on %s: %s", self.name, exc)
            raise LLMError(f"{self.name} request failed: {exc}") from exc

    def stream(self, messages: list[dict], max_tokens: int | None = None, temperature: float | None = None) -> Iterator[str]:
        if self._client is None:
            raise LLMError(f"Provider {self.name} not configured")
        try:
            stream = _with_backoff(self._client.chat.completions.create)(
                model=self.model,
                messages=messages,
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    metrics.inc("llm.tokens")
                    yield delta.content
        except Exception as exc:
            logger.warning("LLM streaming failed on %s: %s", self.name, exc)
            raise LLMError(f"{self.name} stream failed: {exc}") from exc

    def extract_json(self, messages: list[dict], max_tokens: int | None = None) -> dict:
        text = None
        if self._json_mode:
            # ask the endpoint for guaranteed-JSON output (OpenAI-compatible
            # JSON mode); endpoints that reject the parameter are remembered
            try:
                text = self.complete(messages, max_tokens=max_tokens or 900, temperature=0.0,
                                     response_format={"type": "json_object"})
            except LLMError as exc:
                if "response_format" not in str(exc) and "json" not in str(exc).lower():
                    raise
                logger.info("%s rejected JSON mode; retrying without it", self.name)
                self._json_mode = False
        if text is None:
            text = self.complete(messages, max_tokens=max_tokens or 900, temperature=0.0)
        try:
            obj = _parse_json_object(text)
        except (ValueError, json.JSONDecodeError) as exc:
            # malformed JSON is a provider failure: callers fall back to the
            # rule engine instead of surfacing a 500
            usage.bump("failures")
            raise LLMError(f"{self.name} returned invalid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            usage.bump("failures")
            raise LLMError(f"{self.name} returned JSON that is not an object")
        return obj

    def describe(self) -> dict:
        return {
            "provider": self.name,
            "model": self.model,
            "configured": self.available,
        }


def _build_providers(settings) -> list[_Provider]:
    """Build provider list in priority order from settings."""
    providers: list[_Provider] = []

    # 1. Primary: Groq is the spec'd sole provider. When a Groq key is present
    #    the OpenAI-compatible client points at Groq's endpoint and is labelled
    #    "groq" so health/chat reporting never claims an OpenAI provider.
    if settings.groq_api_key and settings.llm_provider != "none":
        providers.append(_Provider(
            name="groq",
            base_url=settings.groq_base_url,
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        ))
    # 1b. Generic OpenAI-compatible endpoint (OpenAI, Azure, Together, ...) only
    #     used when no Groq key is configured.
    elif settings.llm_provider == "openai" and settings.openai_api_key:
        providers.append(_Provider(
            name="openai",
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        ))

    # 2. Explicit Gemini (if key present and LLM is enabled)
    if settings.gemini_api_key and settings.llm_provider != "none":
        providers.append(_Provider(
            name="gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=settings.gemini_api_key,
            model=settings.gemini_model or "gemini-1.5-flash",
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        ))

    # 3. Local (Ollama/LM Studio) if configured
    if settings.llm_provider == "local":
        providers.append(_Provider(
            name="local",
            base_url=settings.openai_base_url,
            api_key="local",
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        ))

    return providers


class LLMClient:
    """Multi-provider LLM client with automatic failover + exact-match caching."""

    def __init__(self) -> None:
        settings = get_settings()
        self._providers = _build_providers(settings)
        self._cache: OrderedDict[str, str] = OrderedDict()
        if not self._providers:
            logger.warning("No LLM providers configured; running in offline mode")

    def _cache_key(self, messages: list[dict], max_tokens: int | None, temperature: float | None) -> str:
        payload = json.dumps(
            {
                "model": self._providers[0].model if self._providers else "",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _try_providers(self, operation: str, *args, **kwargs) -> Any:
        """Try each provider in order until one succeeds."""
        last_exc: Exception | None = None
        for p in self._providers:
            try:
                return getattr(p, operation)(*args, **kwargs)
            except LLMError as exc:
                last_exc = exc
                logger.info("Falling back from %s to next provider", p.name)
                continue
        # all providers failed
        if last_exc:
            raise last_exc
        raise LLMError("No LLM providers available")

    def complete(self, messages: list[dict], max_tokens: int | None = None, temperature: float | None = None) -> str:
        key = self._cache_key(messages, max_tokens, temperature)
        hit = self._cache.pop(key, None)
        if hit is not None:
            metrics.inc("llm.cache_hits")
            usage.bump("cache_hits")
            self._cache[key] = hit  # refresh LRU order
            return hit
        metrics.inc("llm.cache_misses")
        result = self._try_providers("complete", messages, max_tokens, temperature)
        self._cache[key] = result
        while len(self._cache) > _CACHE_SIZE:
            self._cache.popitem(last=False)
        return result

    def stream(self, messages: list[dict], max_tokens: int | None = None, temperature: float | None = None) -> Iterator[str]:
        yield from self._try_providers("stream", messages, max_tokens, temperature)

    def extract_json(self, messages: list[dict], max_tokens: int | None = None) -> dict:
        return self._try_providers("extract_json", messages, max_tokens)

    def describe(self) -> dict:
        counters = metrics.snapshot().get("counters", {})
        return {
            "providers": [p.describe() for p in self._providers],
            "active": self._providers[0].name if self._providers else "none",
            "model": self._providers[0].model if self._providers else None,
            "usage": usage.snapshot(),
            "cache": {
                "size": len(self._cache),
                "max": _CACHE_SIZE,
                "hits": int(counters.get("llm.cache_hits", 0)),
                "misses": int(counters.get("llm.cache_misses", 0)),
            },
        }

    @property
    def available(self) -> bool:
        return any(p.available for p in self._providers)


def _parse_json_object(text: str) -> dict:
    """Best-effort extraction of a JSON object from an LLM response."""
    text = text.strip()
    # strip markdown fences
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in response: {text[:200]!r}")
    # prefer the first complete JSON value; weak models often append prose after
    # the object, which would otherwise blow up a whole-string json.loads.
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return obj
    except json.JSONDecodeError:
        pass
    import re

    end = text.rfind("}")
    if end > start:
        candidate = text[start : end + 1]
        for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                pass
    # weak models mis-nest closers ('...{"sql": "SELECT 1"}}}]}' has one '}'
    # too many) or stop mid-object; rebalance the whole remainder
    remainder = re.sub(r",\s*([}\]])", r"\1", text[start:])
    return json.loads(_balance_brackets(remainder))


def _balance_brackets(text: str) -> str:
    """Drop closers that don't match the innermost open bracket and append any
    missing ones, ignoring brackets inside JSON strings."""
    pairs = {"}": "{", "]": "["}
    closer = {"{": "}", "[": "]"}
    out: list[str] = []
    stack: list[str] = []
    in_string = escaped = False
    for ch in text:
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack or stack[-1] != pairs[ch]:
                continue  # unmatched closer: drop it
            stack.pop()
        out.append(ch)
    out.extend(closer[c] for c in reversed(stack))
    return "".join(out)


_llm: LLMClient | None = None


def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm


def reset_llm() -> None:
    global _llm
    _llm = None
