"""Planning LLM behind one small interface, so the provider is a config switch.

Two backends: the Anthropic SDK, and the OpenAI SDK pointed at any
OpenAI-compatible endpoint (OpenAI, OpenRouter, DeepSeek, Ollama, vLLM, LM Studio).
Messages are plain {"role": "user"|"assistant", "content": str}; tools are
{"name", "description", "schema"}. The agent needs nothing richer.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse

from ah.config import AgentSettings

# Server-side refusal fallback ("default" routes by refusal category); only on the Claude API itself.
ANTHROPIC_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
ANTHROPIC_FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TOKENS = 16_000


class LLMError(RuntimeError):
    """Anything that should reach the user as one readable line."""


@dataclass
class ToolCall:
    name: str
    input: dict[str, Any]


@dataclass
class Reply:
    """usage holds token counts as the provider names them, plus prompt_tokens/completion_tokens for both."""

    text: str
    calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    latency_s: float | None = None


def flat_usage(usage: Any) -> dict[str, int]:
    """Integer counts from a usage object; nested details (cached, reasoning tokens) are lifted to the top."""
    flat: dict[str, int] = {}
    for key, value in (usage.model_dump() if usage is not None else {}).items():
        if isinstance(value, dict):
            flat.update({k: v for k, v in value.items() if type(v) is int})
        elif type(value) is int:
            flat[key] = value
    return flat


class LLM(Protocol):
    name: str

    def complete(self, system: str, messages: list[dict[str, str]], tools: list[dict[str, Any]]) -> Reply:
        ...


class AnthropicLLM:
    def __init__(self, cfg: AgentSettings):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=cfg.api_key or None, base_url=cfg.base_url or None, max_retries=3)
        self.model = cfg.model
        self.name = f"anthropic:{cfg.model}"
        self.fallback = not cfg.base_url and cfg.model in ANTHROPIC_FALLBACK_MODELS
        self.key_env = cfg.api_key_env

    def complete(self, system, messages, tools):
        kwargs: dict[str, Any] = {"model": self.model, "max_tokens": MAX_TOKENS, "system": system, "messages": messages}
        if tools:
            kwargs["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["schema"]} for t in tools]
        a = self._anthropic
        started = time.perf_counter()
        try:
            if self.fallback:
                response = self.client.beta.messages.create(**kwargs, betas=[ANTHROPIC_FALLBACK_BETA], fallbacks="default")
            else:
                response = self.client.messages.create(**kwargs)
        except a.AuthenticationError:
            raise LLMError(f"Anthropic не принял ключ: проверь переменную {self.key_env}") from None
        except a.RateLimitError:
            raise LLMError("Anthropic: превышен лимит запросов, попробуй через минуту") from None
        except a.APIStatusError as err:
            raise LLMError(f"Anthropic вернул {err.status_code}: {err.message}") from None
        except a.APIConnectionError:
            raise LLMError("Нет связи с Anthropic API") from None
        if response.stop_reason == "refusal":
            raise LLMError("Модель отказалась выполнять этот запрос")
        if response.stop_reason == "max_tokens":
            raise LLMError("Ответ модели не поместился в лимит токенов")
        usage = flat_usage(response.usage)
        # Anthropic's input_tokens excludes cache reads and writes; prompt_tokens is the whole prompt as elsewhere.
        usage["prompt_tokens"] = sum(usage.get(k, 0) for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        usage["completion_tokens"] = usage.get("output_tokens", 0)
        return Reply(
            text="".join(b.text for b in response.content if b.type == "text").strip(),
            calls=[ToolCall(b.name, dict(b.input)) for b in response.content if b.type == "tool_use"],
            usage=usage,
            latency_s=round(time.perf_counter() - started, 3),
        )


class OpenAICompatibleLLM:
    def __init__(self, cfg: AgentSettings):
        import openai

        self._openai = openai
        # Local servers (Ollama, LM Studio) ignore the key but the SDK insists on one.
        self.client = openai.OpenAI(api_key=cfg.api_key or "not-needed", base_url=cfg.base_url or None, max_retries=3)
        self.model = cfg.model
        self.name = f"openai-compatible:{cfg.model}@{cfg.base_url or 'api.openai.com'}"
        self.key_env = cfg.api_key_env
        # DeepSeek's knobs; sent only when set, so other OpenAI-compatible servers never see them.
        self.extra_body: dict[str, Any] = {}
        if cfg.thinking:
            self.extra_body["thinking"] = {"type": cfg.thinking}
        if cfg.reasoning_effort:
            self.extra_body["reasoning_effort"] = cfg.reasoning_effort

    def complete(self, system, messages, tools):
        kwargs: dict[str, Any] = {"model": self.model, "messages": [{"role": "system", "content": system}, *messages]}
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        if tools:
            kwargs["tools"] = [{"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["schema"]}} for t in tools]
        o = self._openai
        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(**kwargs)
        except o.AuthenticationError:
            raise LLMError(f"Провайдер не принял ключ: проверь переменную {self.key_env}") from None
        except o.RateLimitError:
            raise LLMError("Провайдер: превышен лимит запросов, попробуй через минуту") from None
        except o.APIStatusError as err:
            raise LLMError(f"Провайдер вернул {err.status_code}: {err.message}") from None
        except o.APIConnectionError:
            raise LLMError("Нет связи с провайдером LLM: проверь base_url") from None
        message = response.choices[0].message
        calls = []
        for call in message.tool_calls or []:
            try:
                calls.append(ToolCall(call.function.name, json.loads(call.function.arguments or "{}")))
            except json.JSONDecodeError:
                raise LLMError("Модель вернула вызов инструмента с битым JSON") from None
        return Reply(text=(message.content or "").strip(), calls=calls, usage=flat_usage(response.usage),
                     latency_s=round(time.perf_counter() - started, 3))


def make_llm(cfg: AgentSettings) -> LLM:
    if cfg.provider == "anthropic":
        return AnthropicLLM(cfg)
    if cfg.provider == "openai":
        return OpenAICompatibleLLM(cfg)
    raise LLMError(f"неизвестный провайдер {cfg.provider!r}")


def configured(cfg: AgentSettings) -> bool:
    """A key is required, except for an OpenAI-compatible server on this machine (Ollama, LM Studio)."""
    if cfg.api_key:
        return True
    return cfg.provider == "openai" and (urlparse(cfg.base_url).hostname or "") in ("localhost", "127.0.0.1", "::1")
