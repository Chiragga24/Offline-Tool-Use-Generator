from kg_mle.llm.providers import load_llm_provider_config, load_mem0_llm_provider_config


def test_load_llm_provider_config_infers_api_key_env(monkeypatch):
    monkeypatch.setenv("KG_MLE_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("KG_MLE_LLM_MODEL", "gemini-2.0-flash-lite-001")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    config = load_llm_provider_config()

    assert config.provider == "gemini"
    assert config.model == "gemini-2.0-flash-lite-001"
    assert config.api_key_env == "GOOGLE_API_KEY"
    assert config.api_key == "test-key"


def test_load_llm_provider_config_supports_hf_provider_hint(monkeypatch):
    monkeypatch.setenv("KG_MLE_LLM_PROVIDER", "huggingface")
    monkeypatch.setenv("KG_MLE_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    monkeypatch.setenv("KG_MLE_HF_PROVIDER", "featherless-ai")

    config = load_llm_provider_config()

    assert config.provider == "huggingface"
    assert config.extra["hf_provider"] == "featherless-ai"


def test_load_llm_provider_config_supports_local_base_url(monkeypatch):
    monkeypatch.setenv("KG_MLE_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("KG_MLE_LLM_MODEL", "gemma4")
    monkeypatch.setenv("KG_MLE_LLM_BASE_URL", "http://localhost:11434")

    config = load_llm_provider_config()

    assert config.provider == "ollama"
    assert config.model == "gemma4"
    assert config.api_key_env is None
    assert config.base_url == "http://localhost:11434"


def test_load_mem0_llm_provider_config_uses_mem0_env_prefix(monkeypatch):
    monkeypatch.setenv("KG_MLE_MEM0_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("KG_MLE_MEM0_LLM_MODEL", "deepseek-chat")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    config = load_mem0_llm_provider_config()

    assert config.provider == "deepseek"
    assert config.model == "deepseek-chat"
    assert config.api_key_env == "DEEPSEEK_API_KEY"
    assert config.api_key == "test-key"
