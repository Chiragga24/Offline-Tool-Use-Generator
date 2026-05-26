from pathlib import Path
import os

from dotenv import load_dotenv

from kg_mle.llm.providers import load_llm_provider_config, load_mem0_llm_provider_config


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "sample_toolbench" / "tools.json"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"
DEFAULT_DATASET_PATH = DEFAULT_OUTPUTS_DIR / "conversations.jsonl"
DEFAULT_EVALUATION_PATH = DEFAULT_OUTPUTS_DIR / "evaluation_metrics.json"
DEFAULT_REGISTRY_PATH = DEFAULT_ARTIFACTS_DIR / "registry.json"
DEFAULT_GRAPH_PATH = DEFAULT_ARTIFACTS_DIR / "tool_graph.json"

DEFAULT_LLM_PROVIDER = os.getenv("KG_MLE_LLM_PROVIDER", "huggingface")
DEFAULT_LLM_MODEL = os.getenv("KG_MLE_LLM_MODEL", "google/gemma-4-E2B-it")
DEFAULT_SEMANTIC_BACKEND = os.getenv("KG_MLE_SEMANTIC_BACKEND", "local")
DEFAULT_EMBEDDING_PROVIDER = os.getenv("KG_MLE_EMBEDDING_PROVIDER", "huggingface")
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "KG_MLE_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
DEFAULT_SEMANTIC_THRESHOLD = float(os.getenv("KG_MLE_SEMANTIC_THRESHOLD", "0.78"))
DEFAULT_SEMANTIC_TOP_K = int(os.getenv("KG_MLE_SEMANTIC_TOP_K", "5"))
DEFAULT_LLM_CONFIG = load_llm_provider_config()
DEFAULT_MEM0_LLM_CONFIG = load_mem0_llm_provider_config()
