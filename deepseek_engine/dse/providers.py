"""Real-LLM providers behind the ``LLM`` protocol (stdlib only, no deps).

An OpenAI-compatible chat-completions client implemented with ``urllib``, so
the engine gains real-model capability without adding dependencies. Works with
any OpenAI-compatible endpoint:

- **DeepSeek API** — `https://api.deepseek.com` (needs ``DSE_PROVIDER_KEY``).
- **Ollama** (local) — `http://localhost:11434/v1` (no key).
- **OpenAI** — `https://api.openai.com/v1` (needs key).
- **GitHub Models** — `https://models.github.ai/inference` (needs a GitHub PAT).

**Copilot note (honest):** VS Code Copilot does not expose a public third-party
API, so this engine cannot call "Copilot" directly. GitHub Models *is*
OpenAI-compatible and serves Copilot-class models (gpt-4o, o4-mini,
deepseek-*, ...) with a GitHub personal access token — that is the supported
path for Copilot-class models here.

Environment variables:

- ``DSE_PROVIDER_URL`` — base URL (defaults per provider).
- ``DSE_PROVIDER_KEY`` — API key / bearer token.
- ``DSE_MODEL_CHEAP`` / ``DSE_MODEL_EXPENSIVE`` — model names for the two tiers.
- ``DSE_TIMEOUT`` — request timeout seconds (default 120).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .config import ModelConfig, provider_models
from .env import load_env
from .llm import Completion

PROVIDER_ENDPOINTS = {
    "deepseek": "https://api.deepseek.com",
    "openai": "https://api.openai.com/v1",
    "ollama": "http://localhost:11434/v1",
    "github": "https://models.github.ai/inference",
}

# DeepSeek V4 Flash (and some other OpenAI-compatible endpoints) intermittently
# return EMPTY ``content`` for a billed request — a documented provider quirk.
# The judge path has its own retry ladder; this makes EVERY generation path
# (agents, chat, verifier) retry blank completions at the single choke point.
_EMPTY_COMPLETION_RETRIES = 3


class OpenAICompatibleLLM:
    """Minimal OpenAI-compatible chat-completions client.

    ``models`` maps tier keys ("cheap"/"expensive") to ``ModelConfig``; the
    actual model name sent to the API is ``ModelConfig.provider_model``.
    """

    def __init__(
        self,
        models: dict[str, ModelConfig],
        base_url: str,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._models = models
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        # --- reliability telemetry (stress-test addition) ------------------
        # ``empty_completions`` = blank completions encountered from the API
        # (all but possibly the last are retried); ``fallback_count`` =
        # empty-content completions salvaged from ``reasoning_content``.
        # Lets ops measure how often the provider's known blank-completion
        # quirk actually fires.
        self.empty_completions = 0
        self.fallback_count = 0

    def _provider_model(self, tier: str) -> str:
        cfg = self._models.get(tier)
        provider_model = cfg.provider_model if cfg else None
        if not provider_model:
            raise ValueError(
                f"tier {tier!r} has no provider_model configured; build the "
                "model dict with dse.config.provider_models(...)"
            )
        return provider_model

    def complete(
        self,
        messages: list[dict],
        *,
        model: str = "cheap",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Completion:
        provider_model = self._provider_model(model)
        payload = {
            "model": provider_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        last: Completion | None = None
        # deepseek-v4-flash is a reasoning model: prompts that trigger long
        # hidden chain-of-thought can exhaust ``max_tokens`` on
        # ``reasoning_content`` and return EMPTY ``content``. We ESCALATE the
        # budget on retry so the model can finish thinking and write the
        # visible answer — raw chain-of-thought is only used as a last resort.
        budget = int(max_tokens)
        for attempt in range(_EMPTY_COMPLETION_RETRIES):
            payload["max_tokens"] = budget
            request = urllib.request.Request(
                f"{self._base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            start = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:  # surface provider errors clearly
                detail = exc.read().decode("utf-8", "replace")[:500]
                raise RuntimeError(
                    f"provider {self._base_url} HTTP {exc.code}: {detail}"
                ) from exc
            latency_s = time.perf_counter() - start
            try:
                message = data["choices"][0]["message"]
                text = message.get("content") or ""
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(
                    f"unexpected provider response shape: {str(data)[:300]}"
                ) from exc
            reasoning = message.get("reasoning_content") or ""
            if text.strip():
                usage = data.get("usage") or {}
                return Completion(
                    text=text,
                    reasoning=reasoning,
                    tokens_in=int(usage.get("prompt_tokens", 0) or 0),
                    tokens_out=int(usage.get("completion_tokens", 0) or 0),
                    latency_s=round(latency_s, 4),
                    model=model,
                )
            # empty content (known provider quirk): count it, escalate the
            # budget when hidden reasoning was present, and keep a last-resort
            # completion (the reasoning text) in case every retry stays blank.
            self.empty_completions += 1
            last = Completion(
                text=reasoning or "",
                reasoning=reasoning,
                tokens_in=int((data.get("usage") or {}).get("prompt_tokens", 0) or 0),
                tokens_out=int((data.get("usage") or {}).get("completion_tokens", 0) or 0),
                latency_s=round(latency_s, 4),
                model=model,
            )
            if reasoning.strip():
                budget = min(budget * 3, 8192)
        # exhausted retries: return the (empty) completion rather than raising,
        # so callers keep their existing fallback behaviour (e.g. the judge's
        # strict/bare retry ladder, agents recording an empty answer).
        if last is not None and last.text.strip():
            self.fallback_count += 1  # salvaged from reasoning_content
        return last


def make_provider(
    name: str,
    models: dict[str, ModelConfig] | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
):
    """Build a provider LLM from ``name`` (deepseek/ollama/openai/github) and env.

    Returns ``None`` for ``"mock"`` (caller builds a MockLLM instead).
    ``models`` defaults to ``provider_models(...)`` populated from
    ``DSE_MODEL_CHEAP`` / ``DSE_MODEL_EXPENSIVE``.  Explicit ``api_key`` /
    ``base_url`` override the environment (used by the chat settings panel so
    users can bring their own key without a restart).
    """
    load_env()  # make sure DSE_* from .env are available before reading them
    name = name.lower()
    if name == "mock":
        return None
    if name not in PROVIDER_ENDPOINTS:
        raise ValueError(
            f"unknown provider {name!r}; choose from {sorted(PROVIDER_ENDPOINTS)} or 'mock'"
        )
    base_url = base_url or os.environ.get("DSE_PROVIDER_URL", PROVIDER_ENDPOINTS[name])
    api_key = api_key if api_key is not None else os.environ.get("DSE_PROVIDER_KEY")
    timeout = float(os.environ.get("DSE_TIMEOUT", "120"))
    if models is None:
        models = provider_models(
            cheap_model=os.environ.get("DSE_MODEL_CHEAP", "deepseek-chat"),
            expensive_model=os.environ.get("DSE_MODEL_EXPENSIVE", "deepseek-reasoner"),
        )
    return OpenAICompatibleLLM(models, base_url, api_key=api_key, timeout=timeout)
