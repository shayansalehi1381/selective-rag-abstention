"""Provider-agnostic LLM client layer shared by benchmark generation and the reader.

``LLMClient.complete(system, prompt, *, temperature, max_tokens) -> str`` is the only
interface the rest of the code depends on. Two adapters are provided:

* ``AnthropicClient``: the Anthropic Messages API (default model ``claude-opus-5-5``),
  with the server-side refusal fallback enabled. A refusal that survives the fallback
  raises ``LLMRefusalError``.
* ``OpenAICompatibleClient``: Chat Completions for OpenAI, vLLM, Ollama and similar servers.

SDKs are imported lazily, so importing this module needs neither package.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, prompt: str, *, temperature: float | None, max_tokens: int) -> str: ...


class LLMRefusalError(RuntimeError):
    pass


class AnthropicClient:
    """Anthropic Messages API adapter (``pip install anthropic``).

    Current Claude models reject sampling parameters such as ``temperature`` (HTTP 400)
    and the 1.x SDK no longer exposes them, so ``temperature`` is sent (via
    ``extra_body``) only when ``supports_temperature=True`` is set for an older model.
    Diversity otherwise comes from varied prompts and sampled chunks. The
    server-side refusal fallback is enabled, so a declined request is retried on a
    fallback model in the same call.
    """

    def __init__(self, model: str = "claude-opus-5-5", *, supports_temperature: bool = False,
                 client: Any = None) -> None:
        self.model = model
        self.name = f"anthropic:{model}"
        self.supports_temperature = supports_temperature
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client

    def complete(self, system: str, prompt: str, *, temperature: float | None, max_tokens: int) -> str:
        kwargs: dict[str, Any] = {}
        if self.supports_temperature and temperature is not None:
            # The 1.x SDK no longer exposes sampling parameters; legacy models still
            # accept them on the wire, so send the field through ``extra_body``.
            kwargs["extra_body"] = {"temperature": temperature}
        response = self._client.beta.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            **kwargs,
        )
        if response.stop_reason == "refusal":
            raise LLMRefusalError(f"{self.model} declined the request")
        return "".join(block.text for block in response.content if block.type == "text")


class OpenAICompatibleClient:
    """Chat Completions adapter (``pip install openai``) for OpenAI, vLLM, Ollama and similar servers."""

    def __init__(self, model: str, *, base_url: str | None = None, client: Any = None) -> None:
        self.model = model
        self.name = f"openai:{model}"
        if client is None:
            from openai import OpenAI

            # A local model on CPU can be very slow; fail with a clear timeout instead of hanging for the
            # SDK default of 10 minutes x 3 attempts. Override with SRAG_LLM_TIMEOUT (seconds).
            client = OpenAI(base_url=base_url, timeout=float(os.environ.get("SRAG_LLM_TIMEOUT", "180")), max_retries=1)
        self._client = client

    def complete(self, system: str, prompt: str, *, temperature: float | None, max_tokens: int) -> str:
        kwargs: dict[str, Any] = {} if temperature is None else {"temperature": temperature}
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.choices[0].message.content or ""


def extract_json(text: str) -> dict[str, Any]:
    """Return the first JSON object in ``text`` (models sometimes wrap JSON in prose or fences)."""
    decoder = json.JSONDecoder()
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        try:
            obj, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object found in model output")
