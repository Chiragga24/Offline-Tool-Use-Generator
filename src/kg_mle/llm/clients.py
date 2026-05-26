"""Provider-neutral structured JSON LLM clients.

The generator and judge only need one capability from an LLM provider:
return a JSON object for a system/user prompt pair. This module keeps that
capability behind a small adapter so provider changes do not leak into agent
logic.
"""

from __future__ import annotations

import json
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError

from kg_mle.llm.providers import LLMProviderConfig


class LLMClientError(RuntimeError):
    """Raised when a provider call cannot return usable text."""


class StructuredLLMClient:
    """Small JSON-completion adapter for Gemini, Groq, HF, and compatible APIs."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        provider: str = "gemini",
        base_url: str | None = None,
        extra: dict[str, str] | None = None,
        timeout_seconds: int = 60,
    ) -> None:
        self.model = model
        self.provider = provider.lower()
        self.api_key = api_key
        self.base_url = base_url
        self.extra = extra or {}
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_config(cls, config: LLMProviderConfig) -> "StructuredLLMClient":
        return cls(
            model=config.model,
            api_key=config.api_key,
            provider=config.provider,
            base_url=config.base_url,
            extra=config.extra,
        )

    def complete_json(self, *, system: str, user: str, temperature: float = 0.4) -> str:
        if self.provider == "gemini":
            return self._gemini_complete_json(system=system, user=user, temperature=temperature)
        if self.provider == "groq":
            return self._openai_compatible_complete_json(
                system=system,
                user=user,
                temperature=temperature,
                default_base_url="https://api.groq.com/openai/v1",
            )
        if self.provider in {"openai", "deepseek", "qwen", "together", "xai", "lmstudio", "vllm"}:
            return self._openai_compatible_complete_json(
                system=system,
                user=user,
                temperature=temperature,
                default_base_url=_default_openai_compatible_base_url(self.provider),
            )
        if self.provider == "huggingface":
            return self._huggingface_complete_json(
                system=system,
                user=user,
                temperature=temperature,
            )
        raise LLMClientError(f"Unsupported LLM provider: {self.provider!r}.")

    def _gemini_complete_json(self, *, system: str, user: str, temperature: float) -> str:
        if not self.api_key:
            raise LLMClientError("Gemini provider requires GOOGLE_API_KEY.")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 1500,
                "responseMimeType": "application/json",
            },
        }
        data = _post_json(url, payload, timeout_seconds=self.timeout_seconds)
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError(f"Gemini response did not contain text: {data!r}") from exc

    def _openai_compatible_complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        default_base_url: str | None,
    ) -> str:
        if not self.api_key and self.provider not in {"lmstudio", "vllm"}:
            raise LLMClientError(f"{self.provider} provider requires an API key.")
        base_url = (self.base_url or default_base_url or "").rstrip("/")
        if not base_url:
            raise LLMClientError(f"{self.provider} provider requires KG_MLE_LLM_BASE_URL.")
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": 1500,
            "response_format": {"type": "json_object"},
        }
        data = _post_json(
            f"{base_url}/chat/completions",
            payload,
            headers=headers,
            timeout_seconds=self.timeout_seconds,
        )
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError(
                f"{self.provider} response did not contain chat content: {data!r}"
            ) from exc

    def _huggingface_complete_json(self, *, system: str, user: str, temperature: float) -> str:
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise LLMClientError("Hugging Face provider requires huggingface-hub.") from exc
        client = InferenceClient(
            model=self.model,
            token=self.api_key,
            provider=self.extra.get("hf_provider"),
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if hasattr(client, "chat_completion"):
            try:
                response = client.chat_completion(
                    messages=messages,
                    max_tokens=1500,
                    temperature=temperature,
                    top_p=1.0,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or ""
                if content:
                    return content
            except Exception:
                pass
        response = client.text_generation(
            f"{system}\n\n{user}\n\nReturn only JSON.",
            max_new_tokens=1500,
            temperature=temperature,
            return_full_text=False,
        )
        return str(response)


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout_seconds: int,
) -> dict[str, Any]:
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers=request_headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise LLMClientError(f"Provider HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise LLMClientError(f"Provider request failed: {exc}") from exc
    return json.loads(raw)


def _default_openai_compatible_base_url(provider: str) -> str | None:
    return {
        "deepseek": "https://api.deepseek.com/v1",
        "openai": "https://api.openai.com/v1",
        "qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "together": "https://api.together.xyz/v1",
        "xai": "https://api.x.ai/v1",
    }.get(provider)
