# KG MLE Offline Tool-Use Generator

This repository contains a production-oriented MVP for the SAP offline KG + MLE exercise.

## Quickstart

```powershell
uv venv .venv
uv pip install -e ".[dev]"
kgmle --help
```

The implementation uses a curated ToolBench-style fixture in `data/sample_toolbench/tools.json`.

Common workflow:

```powershell
kgmle build
kgmle generate --count 10 --seed 42
kgmle evaluate --input data/outputs/conversations.jsonl --output data/outputs/evaluation_metrics.json
```

`kgmle evaluate` writes the metrics JSON and a scored JSONL next to it by
default. The scored JSONL contains the original conversations with
`metadata.evaluation` populated.

Optional hosted judge:

```powershell
kgmle --use-llm evaluate --max-llm-judge-records 10
```

`--use-llm` is the global switch for optional hosted-model features. The
older `evaluate --llm-judge` flag remains supported for judge-only runs.

For a full 100-record scored dataset:

```powershell
kgmle build --semantic-graph --semantic-backend local
kgmle generate --count 100 --seed 42 --allow-semantic-edges
kgmle --use-llm evaluate --repair --max-llm-judge-records 10
```

The raw generated JSONL is useful for debugging. The scored JSONL produced by
`evaluate` is the training/evaluation-ready dataset because it includes
`metadata.evaluation` with deterministic metrics, optional LLM judge scores,
quality band, and repair metadata.

Optional bounded repair pass:

```powershell
kgmle evaluate --repair --repair-threshold 8.0
kgmle --use-llm evaluate --repair --repair-threshold 8.0
```

Diversity experiment:

```powershell
kgmle diversity --count 100 --seed 42
kgmle --use-llm diversity --count 100 --seed 42 --max-llm-judge-records 10
```

Tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -m "not live"
```

The E2E test builds artifacts, generates 100 conversations, evaluates them
through the LLM-judge interface with a deterministic fake provider, and asserts
the mean judge score exceeds the documented threshold.
