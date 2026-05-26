import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LLMProviderConfig:
    provider: str
    model: str
    api_key_env: str | None = None
    base_url: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        return os.getenv(self.api_key_env)


PROVIDER_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "huggingface": "HF_TOKEN",
    "openai": "OPENAI_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "together": "TOGETHER_API_KEY",
    "xai": "XAI_API_KEY",
}


DEFAULT_PROVIDER_MODELS = {
    "anthropic": "claude-3-5-haiku-latest",
    "deepseek": "deepseek-chat",
    "gemini": "gemini-2.0-flash-lite-001",
    "groq": "llama-3.1-8b-instant",
    "huggingface": "google/gemma-4-E2B-it",
    "lmstudio": "local-model",
    "ollama": "gemma4",
    "openai": "gpt-4.1-mini",
    "qwen": "qwen-plus",
    "together": "google/gemma-2-9b-it",
    "vllm": "google/gemma-4-E2B-it",
    "xai": "grok-3-mini",
}


def load_llm_provider_config(
    *,
    provider_env: str = "KG_MLE_LLM_PROVIDER",
    model_env: str = "KG_MLE_LLM_MODEL",
    base_url_env: str = "KG_MLE_LLM_BASE_URL",
) -> LLMProviderConfig:
    provider = os.getenv(provider_env, "huggingface").strip().lower()
    model = os.getenv(model_env, DEFAULT_PROVIDER_MODELS.get(provider, "local-model"))
    api_key_env = PROVIDER_API_KEY_ENV.get(provider)
    base_url = os.getenv(base_url_env)
    extra = {}
    if provider == "huggingface":
        hf_provider = os.getenv("KG_MLE_HF_PROVIDER")
        if hf_provider:
            extra["hf_provider"] = hf_provider

    return LLMProviderConfig(
        provider=provider,
        model=model,
        api_key_env=api_key_env,
        base_url=base_url,
        extra=extra,
    )


def load_mem0_llm_provider_config() -> LLMProviderConfig:
    return load_llm_provider_config(
        provider_env="KG_MLE_MEM0_LLM_PROVIDER",
        model_env="KG_MLE_MEM0_LLM_MODEL",
        base_url_env="KG_MLE_MEM0_LLM_BASE_URL",
    )
