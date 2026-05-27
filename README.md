# KG MLE Offline Tool-Use Generator

This repository contains a production-oriented MVP for the SAP offline KG + MLE exercise.

![Pipeline workflow](docs/workflow_diagram.png)

Architecture and design rationale are in [DESIGN.md](DESIGN.md). A 3-record
slice of a real seed-42 run (generated, scored, and metrics) lives in
[`docs/sample_outputs/`](docs/sample_outputs) so you can see the output shape
without running anything.

## Install

```powershell
uv venv .venv
uv pip install -e ".[dev]"
kgmle --help
```

The implementation uses a curated ToolBench-style fixture in
`data/sample_toolbench/tools.json` (9 domains, 45 endpoints, with intentional
schema messiness). Everything runs **offline by default** — no API keys needed.

If you want hosted LLM features or Mem0-backed semantic graph expansion, start
from `.env.example`:

```powershell
Copy-Item .env.example .env
```

Only `.env.example` is committed. `.env` is local-only and should contain your
provider keys.

## The pipeline at a glance

```powershell
kgmle build                          # ToolBench JSON  -> registry.json + tool_graph.json
kgmle generate --count 100 --seed 42 # graph           -> conversations.jsonl
kgmle evaluate                       # conversations   -> metrics.json + scored.jsonl
kgmle diversity                      # extra: steering OFF vs ON experiment
```

| Command | Purpose | Reads | Writes |
|---|---|---|---|
| `build` | Normalize the tool schemas and build the tool graph | ToolBench JSON | `registry.json`, `tool_graph.json` |
| `generate` | Sample tool chains and generate role-tagged conversations | graph artifacts | `conversations.jsonl` |
| `evaluate` | Score conversations (deterministic + optional LLM judge), optionally repair | `conversations.jsonl` | `evaluation_metrics.json`, `*_scored.jsonl` |
| `diversity` | *(extra)* Run generation with steering OFF vs ON and compare | ToolBench JSON | `diversity_report.json` + per-run datasets/metrics |

Every parameter below has a sensible default, so each command runs with zero
arguments. The tables mark the flags you'll most commonly set.

## Global options

Placed **before** the command, e.g. `kgmle --use-llm evaluate ...`.

| Flag | Default | What it does |
|---|---|---|
| `--use-llm / --no-use-llm` | `--no-use-llm` | Master switch for hosted-LLM features. See the matrix at the bottom. |
| `--log-level TEXT` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, ...). |
| `--version` | — | Print version and exit. |

## `kgmle build`

Ingest ToolBench-style JSON, normalize it into a registry, and build the tool graph.

| Flag | Default | What it does |
|---|---|---|
| `--input, -i PATH` | `data/sample_toolbench/tools.json` | ToolBench-style JSON file or directory. |
| `--artifacts-dir, -a PATH` | `artifacts/` | Where `registry.json` and `tool_graph.json` are written. |
| `--semantic-graph / --no-semantic-graph` | off | Add `semantic_related` edges via embeddings (extra candidate links). |
| `--semantic-backend TEXT` | `local` | `local` (SentenceTransformer/MiniLM) or `mem0` (HF embeddings + Qdrant). |
| `--semantic-threshold FLOAT` | `0.78` | Minimum cosine score for a semantic edge. |
| `--semantic-top-k INT` | `5` | Semantic neighbours considered per endpoint. |
| `--enrich-registry-fields / --no-...` | on | Deterministic alias/type enrichment (`destination → city`, etc.). |
| `--llm-enrich-registry / --no-...` | off | Use the LLM for structured alias/type suggestions (needs credentials). |
| `--registry-enrichment-threshold FLOAT` | `0.80` | Minimum confidence to accept an LLM enrichment suggestion. |
| `--max-llm-registry-endpoints INT` | `5` | Cap on endpoints sent to the live LLM enricher (bounds cost). |

```powershell
kgmle build                                          # deterministic, offline
kgmle build --semantic-graph --semantic-backend local  # add local semantic edges
```

## `kgmle generate`

Sample constrained tool chains from the graph and turn them into role-tagged
conversations with grounded tool calls.

| Flag | Default | What it does |
|---|---|---|
| `--count, -n INT` | `10` | Number of conversations to generate. |
| `--seed INT` | `42` | Random seed (deterministic: same seed → same dataset). |
| `--output, -o PATH` | `data/outputs/conversations.jsonl` | Output JSONL path. |
| `--artifacts-dir, -a PATH` | `artifacts/` | Where to load `registry.json` / `tool_graph.json` (rebuilds if missing). |
| `--cross-conversation-steering / --no-...` | **on** | Steer the corpus toward diverse domains/tools/endpoints. `--no-...` disables it (Run A of the diversity experiment). |
| `--allow-semantic-edges / --no-...` | off | Let the sampler traverse `semantic_related` edges, not just grounded/same-domain. |
| `--semantic-graph / --no-...` | off | If artifacts are missing, build the graph with semantic expansion on the fly. |
| `--semantic-backend TEXT` | `local` | Backend used when building on the fly. |

```powershell
kgmle generate --count 100 --seed 42
kgmle generate --count 100 --seed 42 --no-cross-conversation-steering   # unsteered
kgmle --use-llm generate --count 20 --seed 42                           # LLM-driven agents
```

## `kgmle evaluate`

Score the generated dataset with deterministic structural metrics and an
optional LLM judge, and optionally run a bounded repair pass.

| Flag | Default | What it does |
|---|---|---|
| `--input, -i PATH` | `data/outputs/conversations.jsonl` | Generated dataset to score. |
| `--output, -o PATH` | `data/outputs/evaluation_metrics.json` | Aggregate + per-record metrics JSON. |
| `--scored-output PATH` | `<output>_scored.jsonl` | Conversations with `metadata.evaluation` added. |
| `--llm-judge / --no-llm-judge` | off | Score each record with the hosted LLM judge (5 dimensions). Also enabled by global `--use-llm`. |
| `--max-llm-judge-records INT` | `10` | Cap on records sent to the live judge (bounds cost). |
| `--repair / --no-repair` | off | Attempt one bounded repair pass on failed/low-scoring records. |
| `--repair-threshold FLOAT` | `8.0` | Deterministic-score threshold below which repair is attempted. |
| `--max-repair-attempts INT` | `1` | Repair attempts per record (0 or 1). |

```powershell
kgmle evaluate                                              # deterministic only, offline
kgmle evaluate --repair --repair-threshold 8.0              # + bounded repair
kgmle --use-llm evaluate --max-llm-judge-records 10         # + hosted LLM judge
```

`evaluate` writes two files: the **metrics JSON** (corpus aggregates +
per-record scores) and a **scored JSONL** (the original conversations with a
`metadata.evaluation` block). The scored JSONL is the training/evaluation-ready
dataset.

## `kgmle diversity` *(the extra command)*

Runs the full generation pipeline **twice with the same seed** — once with
cross-conversation steering OFF (Run A) and once ON (Run B) — then computes
diversity metrics and a side-by-side comparison. This is the experiment that
answers "does steering improve diversity, and at what cost to quality?"

| Flag | Default | What it does |
|---|---|---|
| `--count, -n INT` | `100` | Conversations per run (both runs use the same count). |
| `--seed INT` | `42` | Shared seed for both runs (isolates the steering effect). |
| `--output-dir, -o PATH` | `artifacts/diversity/` | Directory for all run artifacts + the report. |
| `--repair / --no-repair` | off | Run the deterministic repair pass during evaluation of both runs. |
| `--allow-semantic-edges / --no-...` | off | Allow semantic-edge traversal in both runs. |
| `--semantic-graph / --no-...` | off | Build the graph with semantic expansion for both runs. |
| `--semantic-backend TEXT` | `local` | Semantic backend. |
| `--max-llm-judge-records INT` | `10` | Cap on judge records per run (only when `--use-llm`). |

```powershell
kgmle diversity --count 100 --seed 42
kgmle --use-llm diversity --count 100 --seed 42 --max-llm-judge-records 10
```

It writes, under `--output-dir`:

```text
run_a_no_steering.jsonl   run_b_steering.jsonl     # generated datasets
run_a_metrics.json        run_b_metrics.json       # per-run evaluation
run_a_scored.jsonl        run_b_scored.jsonl       # scored conversations
diversity_report.json     # config + per-run diversity/quality + comparison deltas
```

## The `--use-llm` switch

`--use-llm` is one master toggle for all hosted-LLM features. Without it the
whole pipeline runs deterministically and offline. With it (and a configured
provider key), each command lights up the LLM features it supports:

| Command | `--use-llm` enables |
|---|---|
| `build` | Structured-output registry enrichment + semantic graph. |
| `generate` | LLM planner/user/assistant agents (deterministic fallback per turn) + semantic-edge traversal. |
| `evaluate` | LLM judge; with `--repair`, also the LLM repair planner. |
| `diversity` | All of the above for both runs. Generation defaults to deterministic without the flag (the reproducible baseline). |

**Provider config** (in `.env`; see `.env.example`). The default is Gemini;
any of these can be selected with `KG_MLE_LLM_PROVIDER`:

```text
KG_MLE_LLM_PROVIDER=gemini          # gemini | anthropic | groq | openai | deepseek
                                    # | qwen | together | xai | huggingface
                                    # | ollama | lmstudio | vllm
KG_MLE_LLM_MODEL=gemini-2.5-flash-lite
GOOGLE_API_KEY=...                  # or the matching key for your provider
```

LLM features degrade gracefully: a missing key, provider quota error, or
malformed output falls back to the deterministic path and is recorded in the
output metadata. See [DESIGN.md](DESIGN.md) #2 for the full provider table.

## Environment configuration

`.env.example` documents all supported knobs. The practical split is:

```text
No .env needed
  deterministic build / generate / evaluate / diversity

.env recommended
  --use-llm
  --llm-enrich-registry
  --semantic-backend mem0
```

Common setups:

```text
Deterministic / offline only
  no provider keys required

Hosted LLM features
  KG_MLE_LLM_PROVIDER=gemini
  KG_MLE_LLM_MODEL=gemini-2.0-flash-lite-001
  GOOGLE_API_KEY=...

Groq alternative
  KG_MLE_LLM_PROVIDER=groq
  KG_MLE_LLM_MODEL=llama-3.1-8b-instant
  GROQ_API_KEY=...

Local provider
  KG_MLE_LLM_PROVIDER=ollama | lmstudio | vllm
  KG_MLE_LLM_BASE_URL=...
```

Mem0-specific notes:

- Mem0 is only used when you explicitly run `--semantic-backend mem0`.
- The default semantic path is `local` MiniLM, which is simpler and more reproducible.
- Mem0 initialization still expects an LLM provider config even though endpoint cards are added with `infer=False`.

Minimal Mem0-related `.env` values:

```text
KG_MLE_SEMANTIC_BACKEND=mem0
KG_MLE_EMBEDDING_PROVIDER=huggingface
KG_MLE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
KG_MLE_MEM0_LLM_PROVIDER=gemini
KG_MLE_MEM0_LLM_MODEL=gemini-2.0-flash-lite-001
GOOGLE_API_KEY=...
HF_TOKEN=...   # recommended for Hugging Face-backed embedding/model access
```

If you plan to use local semantic search rather than Mem0, the simpler setup is:

```text
KG_MLE_SEMANTIC_BACKEND=local
KG_MLE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Install note:

```powershell
uv pip install -e ".[dev]"
uv pip install -e ".[dev,semantic]"
```

The `semantic` extra is the one that pulls in `sentence-transformers`,
`mem0ai`, and `google-genai` for semantic graph / Mem0 workflows.

## 100-sample run results

Real stats from `build → generate --count 100 --seed 42 → evaluate --repair`
(deterministic agents + deterministic judge):

| Metric | Value |
|---|---|
| records generated / failures | 100 / 0 |
| multi-step (≥3 calls) **and** multi-tool (≥2 tools) | 81% |
| tool-call count distribution | 2:19, 3:36, 4:31, 5:14 |
| message-count distribution | 6:11, 8:28, 10:32, 12:21, 14:8 |
| conversations with a clarification turn | 47 |
| distinct domains covered | 9 / 9 |
| mean deterministic score | 10.0 / 10 |
| schema valid rate | 100% |
| tool response coverage | 100% |
| usable-for-training rate | 100% |
| repair attempts needed | 0 |

The 81% multi-step+multi-tool share clears the 50–60% target, lengths are
varied (2–5 tool calls; 6–14 messages), and 47/100 conversations include a
natural clarification turn before tool calling.

**LLM judge note.** The judge integration works end-to-end (`--use-llm
evaluate`), but a full 100-record run on Gemini's free tier hits `HTTP 429`
after ~2 calls. The pipeline contains this: each quota-limited record stores
`metadata.evaluation.llm_judge = {"error": "..."}` and deterministic metrics
stay intact. The E2E test (`tests/e2e/test_pipeline_100.py`) fakes the judge at
the provider boundary so the integration path is verified in CI without
consuming quota.

## Output format

`generate` writes one JSON object per line. Abbreviated generated record
(a real 2-step grounded conversation, `conv_42_00001`):

```json
{
  "conversation_id": "conv_42_00001",
  "messages": [
    {"role": "user", "content": "I'd like help to find tournaments for a game, then follow up with 1 more action(s) across events, gaming."},
    {"role": "assistant", "content": "Before I proceed, could you tell me which game_id to use for get tournament schedule?",
     "clarification_target_step": 0, "clarification_target_parameter": "game_id"},
    {"role": "user", "content": "For game_id, use any_game_id."},
    {"role": "assistant", "content": null, "tool_calls": [
      {"endpoint_id": "gaming/get_tournament_schedule", "arguments": {"game_id": "any_game_id"}, "call_confidence": 1.0}]},
    {"role": "tool", "endpoint": "gaming/get_tournament_schedule",
     "content": {"tournament_id": "trn_s1wjo3", "start_time": "2026-04-11T09:00", "venue": "Old Town"}},
    {"role": "assistant", "content": null, "tool_calls": [
      {"endpoint_id": "events/create_calendar_event",
       "arguments": {"title": "title_value", "start_time": "2026-04-11T09:00", "location": "Old Town", "event_type": "event_type_value"},
       "call_confidence": 1.0}]},
    {"role": "tool", "endpoint": "events/create_calendar_event",
     "content": {"calendar_event_id": "cal_coofdh", "status": "Status 72"}},
    {"role": "assistant", "content": "Done. I completed: get tournament schedule, create calendar event."}
  ],
  "plan": { "...": "the Plan that drove generation (intent, per-step parameter plans, ambiguous_step_indices)" },
  "metadata": {
    "seed": 42000127,
    "original_chain": ["gaming/get_tournament_schedule", "events/create_calendar_event"],
    "final_chain":    ["gaming/get_tournament_schedule", "events/create_calendar_event"],
    "n_tool_calls": 2,
    "domains": ["events", "gaming"],
    "tools_visited": ["gaming_catalog", "events_calendar"],
    "transition_summary": [
      {"source": "gaming/get_tournament_schedule", "target": "events/create_calendar_event",
       "advance_type": "grounded", "parameter": "start_time"}
    ],
    "clarifications_taken": [
      {"step_index": 0, "parameter_name": "game_id", "initiated_by": "planner"}
    ],
    "deviations_accepted": [],
    "deviations_rejected": []
  }
}
```

Note the **grounded chaining**: `gaming/get_tournament_schedule` returns
`start_time: "2026-04-11T09:00"`, and the next call's `start_time` argument
reuses that exact value (the `transition_summary` records it as a `grounded`
edge on `start_time`). The executor rejects hallucinated values.

`evaluate` adds a `metadata.evaluation` block to each record in the scored JSONL:

```json
"evaluation": {
  "schema_valid": true,
  "role_sequence_valid": true,
  "tool_response_coverage": 1.0,
  "chain_completion": 1.0,
  "error_free_trace": 1.0,
  "deterministic_score": 10.0,
  "llm_judge": null,
  "quality_band": "gold",
  "usable_for_training": true
}
```

With `--use-llm`, `llm_judge` carries the 5-dimension score object (or
`{"error": "..."}` if the provider failed for that record):

```json
"llm_judge": {
  "task_completion": 9.0, "tool_trace_validity": 9.0, "argument_grounding": 9.0,
  "response_grounding": 9.0, "naturalness": 8.0, "overall_score": 9.0,
  "confidence": 9.0, "issues": [], "rationale": "..."
}
```

The metrics JSON carries corpus-level aggregates:

```json
{
  "summary": {
    "conversation_count": 100, "mean_deterministic_score": 10.0,
    "mean_chain_completion": 1.0, "mean_tool_response_coverage": 1.0,
    "schema_valid_rate": 1.0, "llm_judged_count": 0,
    "mean_llm_overall_score": null, "usable_for_training_rate": 1.0
  },
  "repair_summary": {"enabled": true, "attempted": 0, "repaired": 0, "failed": 0, "rejected": 0, "regenerated": 0, "status_counts": {}},
  "records": [ { "conversation_id": "conv_42_00000", "deterministic_score": 10.0, "quality_band": "gold", "usable_for_training": true, "...": "..." } ]
}
```

See [`docs/sample_outputs/`](docs/sample_outputs) for real (trimmed) versions of all three.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -m "not live"
```

193 non-live tests run offline without credentials. Five `@pytest.mark.live` tests run
against a real provider when an API key is present and skip cleanly otherwise.
The E2E test (`tests/e2e/test_pipeline_100.py`) builds artifacts, generates 100
conversations, evaluates them through the LLM-judge interface with a
deterministic fake provider, and asserts the mean judge score exceeds the
documented threshold (8.0/10).
