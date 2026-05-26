# Design Notes

This document is intentionally written as a living design record. The implementation is scoped as a production-oriented MVP: every required assignment capability should exist end-to-end, but the submitted run uses a curated ToolBench-style subset so reviewers can reproduce it offline.

## Current Scope

The assignment asks for an offline synthetic data generation system that produces multi-turn, multi-tool conversations grounded in ToolBench-style tool schemas.

The submitted workflow uses a representative fixture at `data/sample_toolbench/tools.json` with:

- 9 categories: Finance, Sports, AI / ML, Entertainment, Travel, Gaming, Events, Food, Weather.
- 45 total endpoints.
- Intentional raw-schema inconsistencies that mimic ToolBench-style data quality issues.

The fixture is not meant to replace ToolBench. It is a reproducible subset used to exercise the pipeline locally. The loader is written against generic ToolBench-style JSON so the same normalization approach can be extended to larger ToolBench exports.

## ToolBench Interpretation

ToolBench provides raw API/tool definitions: categories, tool names, endpoint names, parameter lists, descriptions, methods, paths, and sometimes response schemas. In this project, those definitions are used as the source of truth for what tools exist.

The generated dataset is synthetic. We are not reusing ToolBench conversations as training data. Instead, the system turns tool definitions into new conversations and tool traces.

Pipeline shape:

```text
raw ToolBench-style JSON
-> normalized registry
-> tool graph
-> constrained sampler
-> offline executor
-> conversation generator
-> evaluator / repair loop
-> JSONL dataset
```

## Packaging And CLI

The project is packaged with `pyproject.toml` and exposes a console command:

```text
kgmle
```

Current CLI surface:

```text
kgmle build
kgmle generate
kgmle evaluate
```

Design decision:

- Use `typer` for the CLI because it gives a clear command structure, validation, and help output with little code.
- Use `rich` for human-readable logs and command output.
- Use `pydantic` for normalized data models so invalid internal objects fail early.
- Use `hatchling` as a lightweight modern build backend.

Alternative considered:

- Plain `argparse` plus dictionaries.

Reason rejected:

- It would work, but the code would be less self-documenting and weaker as a production-style submission. The assignment is reviewed as a system, so a polished CLI and typed internal contracts are worth the small dependency cost.

## Configuration And Secrets

Default paths are centralized in `src/kg_mle/config.py`:

- raw fixture: `data/sample_toolbench/tools.json`
- derived artifacts: `artifacts/`
- generated datasets: `data/outputs/`

The current implementation does not require secrets. Future optional LLM-backed generation or judging should read credentials from environment variables, with `.env.example` documenting expected names. Real keys should never be committed.

Design decision:

- Keep default filesystem config in Python constants for reproducibility.
- Keep secrets outside the repo through environment variables.
- Add `.env.example` as documentation only.

Alternative considered:

- Add a full YAML/TOML config system immediately.

Reason rejected:

- It adds another file format and validation layer before the pipeline needs it. CLI flags plus centralized defaults are enough for the current end-to-end workflow.

## Artifact And Logging Strategy

Each pipeline stage writes explicit artifacts:

- `artifacts/registry.json`
- later: `artifacts/tool_graph.json`
- later: generated JSONL and evaluation metrics under `data/outputs/`

Directory creation is handled by small helpers in `src/kg_mle/utils/paths.py`.

Logging is configured through `src/kg_mle/utils/logging.py` using standard Python logging with a Rich handler.

Design decision:

- Use standard logging rather than print statements inside pipeline modules.
- Reserve console output for concise command results.

Reason:

- This keeps the implementation inspectable during local runs and makes later debugging easier without coupling core logic to CLI rendering.

## Tool Registry

The registry is the normalized source of truth for tools and endpoints.

`tools.json` tells us what tools exist, but it is raw input. The registry turns inconsistent raw definitions into a stable internal model:

- `ToolRegistry`
- `Tool`
- `Endpoint`
- `Parameter`
- `ResponseField`

The registry is built by:

```text
kgmle build --input data/sample_toolbench/tools.json --artifacts-dir artifacts
```

and persisted to:

```text
artifacts/registry.json
```

### Normalization Behavior

The loader handles common messy cases:

- lowercase or missing HTTP methods
- `path` instead of `url`
- `response` instead of `response_schema`
- empty or null response schemas
- `required` instead of `required_parameters`
- `optional_params` or `optionalParameters` instead of `optional_parameters`
- missing parameter descriptions
- missing or unknown parameter types
- long category names such as `Artificial_Intelligence_Machine_Learning`

Unknown parameter types default to `string`.

Design decision:

- Normalize tolerantly and preserve endpoints whenever possible.

Alternative considered:

- Strict schema validation that rejects incomplete endpoints.

Reason rejected:

- ToolBench-style data is known to be inconsistent. Rejecting incomplete endpoints would reduce coverage and make the pipeline brittle. The graph and executor can still use endpoints with partial metadata if the registry records a clean fallback.

Alternative considered:

- Use raw JSON dictionaries directly in graph/sampler/executor.

Reason rejected:

- Every downstream module would need repeated defensive parsing. Centralizing normalization keeps later components simpler and easier to test.

### Why The Registry Is Not The Graph

The registry stores cleaned facts:

```text
travel/search_hotels requires city, check_in
travel/search_hotels returns hotel_id, hotel_name, nightly_price
travel/book_itinerary requires flight_id, hotel_id, traveler_name
```

The graph will derive relationships:

```text
search_hotels -> book_itinerary because hotel_id can flow forward
```

So:

```text
tools.json = raw catalog
registry.json = normalized catalog
tool_graph.json = endpoint relationships
sampler = chain selection from graph
```

## Testing Strategy

Tests are organized by purpose:

- `tests/fixtures/`: raw fixture shape and intentional messiness.
- `tests/unit/`: isolated module behavior.
- `tests/integration/`: CLI and multi-module behavior.
- `tests/e2e/`: reserved for the required full build/generate/evaluate workflow.

Current tests cover:

- raw fixture validity and intentional inconsistencies
- default path helpers
- CLI smoke behavior
- registry normalization for clean and messy input
- registry JSON persistence

Design decision:

- Add tests immediately after each component.

Reason:

- The assignment explicitly asks for unit, integration, and end-to-end tests. Keeping tests close to each implementation step prevents a late testing scramble and makes design claims easier to justify.

## Open Design Areas

The next components still need full design and implementation:

- Tool graph schema and edge semantics.
- Sampler constraints and cross-conversation steering.
- Offline execution state model.
- Multi-agent generation protocol.
- Judge dimensions, deterministic fallback, and optional LLM judge.
- Retry/repair policy.
- Diversity experiment metrics and results.

