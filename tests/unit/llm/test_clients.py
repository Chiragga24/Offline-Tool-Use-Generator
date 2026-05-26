import json

import pytest

from kg_mle.llm.clients import LLMClientError, StructuredLLMClient
from kg_mle.llm.providers import LLMProviderConfig


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_structured_client_builds_from_provider_config(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    config = LLMProviderConfig(
        provider="gemini",
        model="gemini-2.0-flash-lite-001",
        api_key_env="GOOGLE_API_KEY",
    )

    client = StructuredLLMClient.from_config(config)

    assert client.provider == "gemini"
    assert client.model == "gemini-2.0-flash-lite-001"
    assert client.api_key == "test-google-key"


def test_gemini_complete_json_uses_json_response_mode(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(
            {
                "candidates": [
                    {"content": {"parts": [{"text": '{"ok": true}'}]}}
                ]
            }
        )

    monkeypatch.setattr("kg_mle.llm.clients.request.urlopen", fake_urlopen)
    client = StructuredLLMClient(
        provider="gemini",
        model="gemini-2.0-flash-lite-001",
        api_key="test-key",
    )

    result = client.complete_json(system="Return JSON.", user="Ping.")

    assert result == '{"ok": true}'
    assert "generativelanguage.googleapis.com" in captured["url"]
    assert captured["payload"]["generationConfig"]["responseMimeType"] == "application/json"
    assert captured["payload"]["systemInstruction"]["parts"][0]["text"] == "Return JSON."


def test_groq_complete_json_uses_openai_compatible_json_mode(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(
            {"choices": [{"message": {"content": '{"provider": "groq"}'}}]}
        )

    monkeypatch.setattr("kg_mle.llm.clients.request.urlopen", fake_urlopen)
    client = StructuredLLMClient(
        provider="groq",
        model="llama-3.1-8b-instant",
        api_key="test-groq-key",
    )

    result = client.complete_json(system="Return JSON.", user="Ping.")

    assert result == '{"provider": "groq"}'
    assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-groq-key"
    assert captured["payload"]["response_format"] == {"type": "json_object"}


def test_structured_client_raises_without_gemini_key():
    client = StructuredLLMClient(
        provider="gemini",
        model="gemini-2.0-flash-lite-001",
        api_key=None,
    )

    with pytest.raises(LLMClientError, match="GOOGLE_API_KEY"):
        client.complete_json(system="Return JSON.", user="Ping.")


def test_anthropic_complete_json_uses_messages_api_with_prefilled_brace(monkeypatch):
    """Anthropic has no native JSON mode. We prepend a JSON-only instruction
    to the system prompt and prefill the assistant turn with `{`, then
    restore the leading brace on the returned content."""
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(
            {
                "content": [
                    {"type": "text", "text": '"provider": "anthropic"}'}
                ],
                "stop_reason": "end_turn",
            }
        )

    monkeypatch.setattr("kg_mle.llm.clients.request.urlopen", fake_urlopen)
    client = StructuredLLMClient(
        provider="anthropic",
        model="claude-3-5-haiku-latest",
        api_key="test-anthropic-key",
    )

    result = client.complete_json(system="Return JSON.", user="Ping.")

    assert result == '{"provider": "anthropic"}'
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["X-api-key"] == "test-anthropic-key"
    assert captured["headers"]["Anthropic-version"] == "2023-06-01"
    assert "Return ONLY a single valid JSON object" in captured["payload"]["system"]
    # Last message is the prefilled assistant turn forcing JSON.
    assert captured["payload"]["messages"][-1] == {"role": "assistant", "content": "{"}


def test_anthropic_raises_without_api_key():
    client = StructuredLLMClient(
        provider="anthropic",
        model="claude-3-5-haiku-latest",
        api_key=None,
    )

    with pytest.raises(LLMClientError, match="ANTHROPIC_API_KEY"):
        client.complete_json(system="Return JSON.", user="Ping.")


def test_ollama_uses_openai_compatible_localhost_by_default(monkeypatch):
    """Ollama's `/v1/chat/completions` is OpenAI-compatible and runs locally;
    no API key is required by default."""
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(
            {"choices": [{"message": {"content": '{"provider": "ollama"}'}}]}
        )

    monkeypatch.setattr("kg_mle.llm.clients.request.urlopen", fake_urlopen)
    client = StructuredLLMClient(
        provider="ollama",
        model="gemma4",
        api_key=None,
    )

    result = client.complete_json(system="Return JSON.", user="Ping.")

    assert result == '{"provider": "ollama"}'
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
    # No Authorization header — ollama doesn't need a bearer token by default.
    assert "Authorization" not in captured["headers"]
    assert captured["payload"]["response_format"] == {"type": "json_object"}


def test_ollama_honors_custom_base_url(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        return _FakeResponse(
            {"choices": [{"message": {"content": "{}"}}]}
        )

    monkeypatch.setattr("kg_mle.llm.clients.request.urlopen", fake_urlopen)
    client = StructuredLLMClient(
        provider="ollama",
        model="gemma4",
        api_key=None,
        base_url="http://remote-ollama:11434/v1",
    )

    client.complete_json(system="Return JSON.", user="Ping.")
    assert captured["url"] == "http://remote-ollama:11434/v1/chat/completions"
