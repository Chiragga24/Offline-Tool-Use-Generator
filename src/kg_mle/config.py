from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "sample_toolbench" / "tools.json"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"
DEFAULT_DATASET_PATH = DEFAULT_OUTPUTS_DIR / "conversations.jsonl"
DEFAULT_EVALUATION_PATH = DEFAULT_OUTPUTS_DIR / "evaluation_metrics.json"
DEFAULT_REGISTRY_PATH = DEFAULT_ARTIFACTS_DIR / "registry.json"
DEFAULT_GRAPH_PATH = DEFAULT_ARTIFACTS_DIR / "tool_graph.json"

