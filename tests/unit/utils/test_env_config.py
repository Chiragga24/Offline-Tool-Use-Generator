import importlib

from kg_mle import config


def test_env_defaults_are_present_without_required_secret_values():
    assert config.DEFAULT_LLM_PROVIDER == "huggingface"
    assert config.DEFAULT_LLM_MODEL
    assert config.DEFAULT_EMBEDDING_PROVIDER == "huggingface"
    assert config.DEFAULT_EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.DEFAULT_SEMANTIC_THRESHOLD == 0.78
    assert config.DEFAULT_SEMANTIC_BACKEND == "local"
    assert config.DEFAULT_SEMANTIC_TOP_K >= 1
    assert config.DEFAULT_LLM_CONFIG.provider
    assert config.DEFAULT_LLM_CONFIG.model
    assert config.DEFAULT_MEM0_LLM_CONFIG.provider
    assert config.DEFAULT_MEM0_LLM_CONFIG.model


def test_env_config_can_be_overridden(monkeypatch):
    monkeypatch.setenv("KG_MLE_SEMANTIC_THRESHOLD", "0.83")
    monkeypatch.setenv("KG_MLE_SEMANTIC_TOP_K", "3")

    reloaded = importlib.reload(config)

    assert reloaded.DEFAULT_SEMANTIC_THRESHOLD == 0.83
    assert reloaded.DEFAULT_SEMANTIC_TOP_K == 3

    monkeypatch.delenv("KG_MLE_SEMANTIC_THRESHOLD")
    monkeypatch.delenv("KG_MLE_SEMANTIC_TOP_K")
    importlib.reload(config)
