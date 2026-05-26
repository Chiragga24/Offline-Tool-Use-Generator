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
