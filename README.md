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
kgmle evaluate --llm-judge --max-llm-judge-records 10
```
