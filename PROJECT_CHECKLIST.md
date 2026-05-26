# SAP Offline KG + MLE Exercise Checklist

## Current Situation

- Assignment deadline from email: Tuesday, May 26 at 11:59 PM.
- Current working assumption from Chirag: start time is Sunday, May 24 around 7 PM.
- Target strategy: build a production-looking, end-to-end MVP that implements every required capability, while keeping scope bounded and well documented.
- Primary grading signal: a strong `DESIGN.md` plus a runnable pipeline, clear tests, and honest tradeoff analysis.
- Scope decision: use a curated representative ToolBench-style subset with 9 categories and 4-5 endpoints per category, rather than attempting to process the full ToolBench corpus during the submitted end-to-end run.

## Problem Statement

Build an offline synthetic data generation system that creates multi-turn conversations with multi-step and multi-tool tool-use traces, grounded in a subset of ToolBench-style API/tool schemas.

The system must:

- Ingest ToolBench tool/API definitions.
- Normalize inconsistent raw schemas into a reliable internal tool registry.
- Build a Tool Graph representing tools, endpoints, parameters, domains, and semantic relationships.
- Sample realistic tool chains from that graph with constraints.
- Generate multi-agent conversations where users ask for tasks, assistants call tools, tools return mocked outputs, and assistants complete the task.
- Execute tools offline with stateful, schema-consistent mock responses.
- Evaluate conversations with an LLM-as-judge or deterministic fallback judge.
- Retry or repair low-quality or structurally invalid generations.
- Manage context both inside a conversation and across a generated corpus.
- Run a diversity experiment comparing generation with cross-conversation steering disabled vs enabled.

## Sources And References

### Assignment Source

- Local PDF: `Offline KG+MLE Exercise T2 FTW v1.pdf`
- Extracted locally with `pypdf` from `.venv`.

### ToolBench References

- OpenBMB ToolBench GitHub repository: https://github.com/OpenBMB/ToolBench
- ToolBench paper/repository framing: ToolBench is an open platform for training, serving, and evaluating large language models for tool learning.
- StableToolBench paper for evaluation context: https://arxiv.org/abs/2403.07714

### Project References We Will Produce

- `README.md`: how to install, build, generate, evaluate, and reproduce outputs.
- `DESIGN.md`: architecture, decisions, context management, prompt design, diversity and quality analysis.
- Generated artifacts:
  - normalized registry
  - tool graph
  - generated JSONL dataset
  - evaluation metrics
  - diversity experiment report

## Planned Repository Shape

```text
.
├── README.md
├── DESIGN.md
├── PROJECT_CHECKLIST.md
├── pyproject.toml
├── data/
│   ├── sample_toolbench/
│   └── outputs/
├── artifacts/
├── src/
│   └── kg_mle/
│       ├── cli.py
│       ├── registry/
│       ├── graph/
│       ├── sampler/
│       ├── executor/
│       ├── agents/
│       ├── evaluation/
│       ├── repair/
│       └── utils/
└── tests/
    ├── unit/
    ├── integration/
    └── e2e/
```

## Representative ToolBench-Style Subset

The submitted end-to-end workflow will use a curated subset that is intentionally broad enough to exercise the graph, sampler, executor, generator, judge, repair loop, and diversity metrics while remaining reviewable.

Planned categories:

- Finance
- Sports
- AI / ML
- Entertainment
- Travel
- Gaming
- Events
- Food / Restaurants
- Weather

Planned endpoint count:

- 4-5 endpoints per category.
- Approximately 36-45 endpoints total.

Design intent:

- Include single-domain chains such as `search_hotels -> get_hotel_details -> book_hotel`.
- Include cross-domain chains spanning 2 domains such as `search_team -> get_schedule -> create_calendar_event`.
- Include a smaller number of richer chains spanning 3-4 domains, for example:
  - `travel/search_flights -> weather/get_forecast -> events/search_events -> food/search_restaurants`
  - `sports/search_team -> sports/get_schedule -> weather/get_forecast -> events/create_calendar_event`
  - `entertainment/search_movies -> events/search_events -> food/check_availability -> food/make_reservation`
  - `ai_ml/list_models -> ai_ml/estimate_inference_cost -> finance/get_quote -> finance/create_price_alert`

Implementation note:

- These categories are representative of RapidAPI/ToolBench-style API categories, and this design choice should be documented in `DESIGN.md`.
- The loader should still be written against generic ToolBench-style JSON so the sample subset is a reproducible fixture, not a hardcoded special case.

## Implementation Checklist

### 1. Project Scaffold And CLI

- [x] Create `pyproject.toml` with package metadata, dependencies, test config, and console script.
- [x] Create Python package under `src/kg_mle`.
- [x] Add CLI entrypoint.
- [x] Implement `kgmle build`.
- [x] Implement `kgmle generate`.
- [x] Implement `kgmle evaluate`.
- [x] Add consistent artifact paths and config handling.
- [x] Add logging with useful progress messages.
- [x] Add unit tests for CLI argument parsing or command smoke tests.
- [x] Add README usage commands.

Testing reminder:

- [x] After CLI scaffold, run `pytest`.
- [x] Run CLI smoke commands with `--help`.

### 2. Tool Registry

- [x] Define normalized internal data models for:
  - tool
  - endpoint
  - parameter
  - response schema
  - domain/category
- [x] Implement loader for ToolBench-style raw JSON.
- [x] Handle missing descriptions, missing parameter types, malformed schemas, and incomplete response schemas.
- [x] Add schema normalization defaults.
- [x] Persist normalized registry artifact as JSON.
- [x] Add validation for required normalized fields.
- [ ] Document data model choices in `DESIGN.md`.
- [x] Add unit tests for clean input.
- [x] Add unit tests for messy/incomplete input.

Testing reminder:

- [x] Every registry normalization rule should have at least one test.

### 3. Tool Graph

- [ ] Define graph node types:
  - domain
  - tool
  - endpoint
  - parameter
  - output field
- [ ] Define graph edge types:
  - tool belongs to domain
  - tool exposes endpoint
  - endpoint requires parameter
  - endpoint returns field
  - endpoint output can satisfy later parameter
  - semantically related endpoints
- [ ] Build graph from normalized registry.
- [ ] Persist graph artifact as JSON.
- [ ] Add graph summary metrics.
- [ ] Document graph schema and why these relationships support sampling.
- [ ] Add unit tests for graph construction.

Testing reminder:

- [ ] Test that output-to-input edges exist for ID-like fields.
- [ ] Test graph artifact can round-trip through JSON.

### 4. Tool-Chain Sampler

- [ ] Implement sampler that uses the graph, not hardcoded chains.
- [ ] Support constraints:
  - exact number of steps
  - minimum number of distinct tools
  - required domain
  - required multi-step dependency
  - allow/disallow parallel pattern
- [ ] Implement deterministic sampling with seed.
- [ ] Track sampling metadata.
- [ ] Support cross-conversation steering toggle.
- [ ] Add steering penalties for overused domains, tools, endpoint pairs, and chain patterns.
- [ ] Document constraint interface in `DESIGN.md`.
- [ ] Add unit tests for constraints.
- [ ] Add unit tests for deterministic sampling.

Testing reminder:

- [ ] Test that `--no-cross-conversation-steering` produces reproducible unsteered samples.
- [ ] Test that steering changes distribution across a corpus.

### 5. Offline Tool Execution

- [ ] Implement schema-derived mock output generation.
- [ ] Maintain per-conversation session state.
- [ ] Generate stable IDs and names from seeded random state.
- [ ] Ensure later tool calls can use IDs returned by earlier tool outputs.
- [ ] Validate tool-call arguments against normalized endpoint parameters.
- [ ] Validate mocked outputs against expected lightweight schema.
- [ ] Add deterministic fallback behavior when schemas are incomplete.
- [ ] Document offline execution model in `DESIGN.md`.
- [ ] Add unit tests for output generation.
- [ ] Add unit tests for chained ID grounding.

Testing reminder:

- [ ] Test that booking/detail/update endpoints use IDs produced by search/list/create endpoints.

### 6. Multi-Agent Conversation Generator

- [ ] Define agent roles:
  - scenario planner
  - user simulator
  - assistant/tool caller
  - tool executor
  - final response writer
- [ ] Implement at least one structured-output agent.
- [ ] Generate role-tagged messages.
- [ ] Generate conversations with natural disambiguation when fields are missing.
- [ ] Ensure 50-60% of output has at least 3 tool calls and at least 2 distinct tools.
- [ ] Generate varied conversation lengths.
- [ ] Attach metadata for tools, domains, seed, chain, constraints, and generation mode.
- [ ] Document agent communication protocol in `DESIGN.md`.
- [ ] Add unit tests for conversation schema.
- [ ] Add integration test for a generated multi-step chain.

Testing reminder:

- [ ] After implementing generation, generate a small JSONL sample and inspect manually.
- [ ] Add tests before expanding behavior.

### 7. LLM-As-Judge And Evaluation

- [ ] Define quality dimensions, likely:
  - tool correctness
  - grounding/coherence
  - task completion
  - naturalness
- [ ] Implement evaluator interface.
- [ ] Implement deterministic/local judge fallback for offline testability.
- [ ] Optionally support real LLM judge through environment configuration.
- [ ] Store judge scores in each conversation record.
- [ ] Implement `kgmle evaluate` summary metrics.
- [ ] Document judge dimensions and thresholds in `DESIGN.md`.
- [ ] Add unit tests for judge output parsing.
- [ ] Add unit tests for scoring edge cases.

Testing reminder:

- [ ] Tests should not require network or real LLM credentials.
- [ ] Real LLM support, if added, should be optional.

### 8. Retry And Repair

- [ ] Define validation failures:
  - missing required metadata
  - invalid tool call
  - ungrounded chained argument
  - low judge score
  - malformed message sequence
- [ ] Implement repair strategy:
  - regenerate missing arguments from prior outputs
  - insert clarification turn when required fields are missing
  - revise final assistant response to match tool outputs
  - rescore repaired conversation
- [ ] Limit repair attempts.
- [ ] Preserve repair history in metadata.
- [ ] Document retry/repair strategy in `DESIGN.md`.
- [ ] Add integration test demonstrating failure, repair, and pass.

Testing reminder:

- [ ] The repair test is explicitly required by the assignment. Do not leave it for the end.

### 9. Context Management

- [ ] Within-conversation grounding:
  - maintain state store of returned IDs and salient fields
  - expose state to tool-call argument builder
  - validate dependencies after each step
- [ ] Cross-conversation steering:
  - maintain corpus-level counters
  - steer away from repeated domains, tools, endpoint pairs, and chain lengths
  - allow disabling with CLI flag
- [ ] Document limitations and scale tradeoffs in `DESIGN.md`.
- [ ] Add tests for state grounding.
- [ ] Add tests for corpus steering counters.

Testing reminder:

- [ ] Context tests should be small and deterministic.

### 10. Diversity Experiment

- [ ] Add CLI flag: `--no-cross-conversation-steering`.
- [ ] Generate Run A with steering disabled and fixed seed.
- [ ] Generate Run B with steering enabled and same seed.
- [ ] Compute at least two diversity metrics:
  - domain entropy
  - distinct endpoint-pair ratio
  - tool coverage ratio
  - chain-pattern diversity
- [ ] Measure quality scores for both runs.
- [ ] Save numeric results to artifact files.
- [ ] Add `DESIGN.md` section: "Diversity & Quality Analysis".
- [ ] Analyze whether diversity improved and whether quality changed.

Testing reminder:

- [ ] Add a test that diversity metric functions return stable values for known fixtures.

### 11. Dataset Output Format

- [ ] Define JSONL record schema.
- [ ] Include:
  - `conversation_id`
  - `messages`
  - `tool_calls`
  - `tool_outputs`
  - `judge_scores`
  - `metadata`
- [ ] Include reproduction metadata:
  - seed
  - generated_at
  - sampler constraints
  - graph artifact version/hash
  - steering enabled/disabled
  - repair attempts
- [ ] Add schema validation.
- [ ] Document metadata schema in README or `DESIGN.md`.
- [ ] Add unit tests for serialization.

Testing reminder:

- [ ] Validate generated JSONL before using it in evaluation.

### 12. End-To-End Workflow

- [x] Provide sample ToolBench-style data in repo.
- [x] Add fixture tests for sample ToolBench-style data shape.
- [x] Add intentionally messy ToolBench-style fixture cases for registry normalization.
- [ ] Run:
  - `kgmle build`
  - `kgmle generate --count 100 --seed 42`
  - `kgmle evaluate`
- [ ] Add E2E test that builds artifacts and generates at least 100 samples.
- [ ] Assert mean judge score exceeds a justified threshold.
- [ ] Save sample generated dataset.
- [ ] Save evaluation metrics.
- [ ] Add README quickstart.

Testing reminder:

- [ ] E2E test must be runnable without external network access.

### 13. Documentation

- [ ] Write `README.md`.
- [ ] Write `DESIGN.md`.
- [ ] Include architecture diagram or text flow.
- [ ] Include agent roles and protocol.
- [ ] Include prompt design and at least one failed iteration.
- [ ] Include context management design.
- [ ] Include diversity experiment results.
- [ ] Include limitations and next steps.
- [ ] Include how to run tests.

Testing reminder:

- [ ] Before final submission, follow README commands from a clean shell.

## Definition Of Done

- [ ] `uv venv .venv` works.
- [ ] `uv pip install -e ".[dev]"` works.
- [ ] `kgmle --help` works.
- [ ] `kgmle build` creates registry and graph artifacts.
- [ ] `kgmle generate --count 100 --seed 42` creates JSONL dataset.
- [ ] `kgmle evaluate` creates metrics.
- [ ] `pytest` passes.
- [ ] `README.md` explains the end-to-end workflow.
- [ ] `DESIGN.md` directly answers every required design section from the assignment.
- [ ] Diversity experiment numbers are included.
- [ ] Known limitations are documented honestly.

## Working Rule For This Project

After implementing any meaningful component:

1. Add or update focused tests.
2. Run the relevant tests.
3. Update this checklist.
4. Only then move to the next feature.
