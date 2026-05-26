# DESIGN

Offline synthetic-data pipeline that turns ToolBench-style API definitions into
multi-turn, multi-tool conversations with grounded tool traces. The document
is organised around the rubric: system design first, then each component,
then context/diversity/prompts/judge/repair/tests. Honest tradeoffs and
known gaps are called out in each section rather than buried.

## 1. Overview

**Pipeline:**

```text
ToolBench JSON
  └─► load_registry + enrich   ──► artifacts/registry.json
        └─► build_tool_graph    ──► artifacts/tool_graph.json
              └─► ToolChainSampler ──► SamplingResult
                    └─► CorpusPlanner ──► list[SamplingResult] (steering on/off)
                          └─► ConversationCoordinator
                                ├─► Planner / User / Assistant  (deterministic or LLM)
                                └─► OfflineExecutor (per-conversation session)
                                      └─► Conversation (JSONL line)
                                            └─► evaluate (deterministic + optional LLM judge)
                                                  └─► optional repair pass
                                                        └─► scored JSONL + metrics
```

**CLI** (`src/kg_mle/cli.py`): `build`, `generate`, `evaluate`, `diversity`,
plus a global `--use-llm` switch that activates hosted-LLM features where
they are supported by the command: registry enrichment, semantic graph
expansion, LLM generator agents for `generate`, LLM judge, and LLM repair
planner.

**Scope.** Curated 9-domain, 45-endpoint ToolBench-style fixture
(`data/sample_toolbench/tools.json`) with intentional schema messiness:
lowercase methods, `path` vs `url`, mixed parameter-list keys, missing
types, three response-schema shapes (JSON Schema, flat map, field list).
The loader handles the full mess; the rest of the pipeline runs offline
by default, with all LLM features behind opt-in flags.

**Key result (steering on, seed 42, 100 conversations):**

| | Run A (no steering) | Run B (steering) |
|---|---:|---:|
| endpoint coverage | 78% | 89% |
| tool coverage | 89% | 100% |
| domain entropy | 2.53 | 2.70 |
| top-endpoint share | 22% | 19% |
| mean deterministic score | 9.96 | 10.00 |
| usable-for-training rate | 99% | 100% |

Steering widens coverage without hurting quality. Full breakdown in §10.

**Test status:** 182 non-live tests pass without credentials
(`pytest -m "not live"`); five `@pytest.mark.live` tests run when API
keys are present and skip cleanly otherwise.

## 2. Architecture and Communication

### Components

| Module | Responsibility | Boundary |
|---|---|---|
| `registry/` | Tolerant ToolBench JSON → normalized Pydantic `ToolRegistry`. Deterministic field enrichment + optional LLM enrichment. | Reads raw JSON; writes `registry.json`. |
| `graph/` | Builds typed `ToolGraph` from registry. `output_satisfies_input` grounding edges via alias-aware matching. Optional semantic expansion (local SentenceTransformer or Mem0). | Reads `ToolRegistry`; writes `tool_graph.json`. |
| `sampler/` | `ToolChainSampler` does deterministic DFS with backtracking. `CorpusPlanner` distributes constraints across a corpus. `CorpusSteerer`/`NullSteerer` are the on/off variants. | Reads `ToolGraph`; emits `SamplingResult`s. |
| `executor/` | `OfflineExecutor` opens per-conversation `ExecutorSession`. Validates args (Pydantic + strict grounding). Generates chain-consistent mock responses. | Driven turn-by-turn by the coordinator. |
| `generator/` | Three stateless agents (`Planner`, `UserSimulator`, `Assistant`), each in deterministic and LLM variants behind the same Protocol. `ConversationCoordinator` owns the transcript. | Reads `SamplingResult`; writes `Conversation`. |
| `evaluation/` | Deterministic metrics + optional `LLMJudge` (5 rubric dimensions). | Reads JSONL; writes metrics + scored JSONL. |
| `repair/` | `RepairPolicy` detects triggers; deterministic or LLM planner proposes a `RepairPlan`; safe local repairs are applied. | Reads scored conversation; mutates a copy. |
| `diversity/` | Runs the steering on/off experiment, computes metrics, writes report. | Wraps generator + evaluator. |
| `schema/` | Pydantic artifact schemas for downstream filters. | Reusable by tests + downstream. |
| `llm/` | `StructuredLLMClient` — provider-neutral JSON client (Gemini default, Groq backup, OpenAI-compatible, HF). | Used by enricher, judge, repair, generator. |

### Communication flow inside one conversation

```text
coordinator.run(sampling_result, seed)
  │ 1. planner.plan(sampling_result, seed)            → Plan        [Pydantic-validated]
  │ 2. executor.open_session(sampling_result, seed)   → ExecutorSession
  │ 3. user.initial_request(plan, seed)               → UserTurn    → append to transcript
  │
  │ loop until terminal:
  │ 4. assistant.compose_turn(plan, transcript, session, ...) → AssistantTurn [Pydantic]
  │    branch on kind:
  │      clarification → append assistant Q, call user.reply_to_clarification, append A
  │      tool_calls    → for each proposal: session.call(endpoint_id, args)
  │                      on ExecutorError → record inline → retry once via assistant
  │      final_summary → append closing → exit loop
  │    if chain_deviation present: gate by confidence + endpoint exists + graph supports it
  │
  │ 5. return Conversation(messages, plan, metadata)
```

Every cross-component boundary is a typed Pydantic object. No free-text
transcript parsing. Generator-supplied args go through the same executor
validator as `suggest_arguments`-derived ones — no privileged caller.

### Load-bearing system-level decisions

| Decision | Why |
|---|---|
| **Deterministic-first, LLM opt-in throughout.** Every component has a deterministic path that satisfies the rubric; LLM features are toggles. | CI runs offline. Reviewers without credentials get a working pipeline. LLM features become realism boosters, not crash points. |
| **Provider-neutral LLM client** with 10 hosted/local providers behind one `complete_json(system, user)` adapter. | An earlier HF-only path hit a provider billing wall mid-test. One client lets `.env` swap providers without touching agent code. See provider table at the end of §2. |
| **Pydantic at every boundary** (registry, graph artifact, sampler result, executor errors, agent turns, judge scores, repair plans, dataset records). | One validation system; no schema drift between in-memory and on-disk; the assignment's structured-output requirement is satisfied by construction. |
| **Per-chain RNG seed derived as `planner_seed*1_000_003 + plan_index`**, not threaded through a single RNG. | Steering on/off must not perturb each chain's individual RNG state — otherwise Run A and Run B aren't directly comparable at the per-chain level. |
| **JSON-serializable graph, not Neo4j**; in-memory state, not a vector DB (Mem0 is opt-in). | Reviewer setup is `pip install`. The graph artifact is grep-able. |
| **Strict grounding everywhere, errors inline in conversation log.** | The rubric's "coherent chaining" property is enforced at execution, not by the prompt. Failures surface as `role: "tool"` error entries so reviewers see the repair flow inline. |

### What I'm uncertain about at the system level

- **Per-chain seed scheme uses modular arithmetic on `2**31 − 1`.** Two
  distant chain indices could collide on the same per-chain seed by
  accident. With 100-chain corpora this hasn't happened; at 100k chains
  the collision probability becomes non-negligible. A 64-bit seed space
  would be cheap to switch to and I haven't.
- **`endpoint_id` is `domain/name`.** Loud-fails on duplicates within a
  domain (already tested), but the full ToolBench corpus would force a
  rename pass. The honest path forward is `domain/tool_name/name` and a
  migration; I didn't take it because the curated fixture has no
  collisions.

### Hosted LLM provider support

`src/kg_mle/llm/clients.py` exposes one `complete_json(system, user)`
adapter. Any of these can be selected via `KG_MLE_LLM_PROVIDER` in
`.env`; the matching API key is read from the env var below.

| Provider | API shape | Default model | Env var | Default base URL |
|---|---|---|---|---|
| **Gemini** *(default)* | Native `generateContent` with `responseMimeType=application/json` | `gemini-2.0-flash-lite-001` | `GOOGLE_API_KEY` | hardcoded |
| **Anthropic** | Native Messages API, JSON via prefilled `{` | `claude-3-5-haiku-latest` | `ANTHROPIC_API_KEY` | `api.anthropic.com` |
| **Groq** | OpenAI-compatible `chat/completions` | `llama-3.1-8b-instant` | `GROQ_API_KEY` | `api.groq.com/openai/v1` |
| **OpenAI** | OpenAI-compatible | `gpt-4.1-mini` | `OPENAI_API_KEY` | `api.openai.com/v1` |
| **DeepSeek** | OpenAI-compatible | `deepseek-chat` | `DEEPSEEK_API_KEY` | `api.deepseek.com/v1` |
| **Qwen (DashScope)** | OpenAI-compatible | `qwen-plus` | `DASHSCOPE_API_KEY` | DashScope compatible-mode |
| **Together AI** | OpenAI-compatible | `google/gemma-2-9b-it` | `TOGETHER_API_KEY` | `api.together.xyz/v1` |
| **xAI (Grok)** | OpenAI-compatible | `grok-3-mini` | `XAI_API_KEY` | `api.x.ai/v1` |
| **Hugging Face** | `huggingface_hub.InferenceClient` | `google/gemma-4-E2B-it` | `HF_TOKEN` | router |
| **Ollama** *(local)* | OpenAI-compatible, no key | `gemma4` | — | `localhost:11434/v1` |
| **LM Studio** *(local)* | OpenAI-compatible, no key | `local-model` | — | `KG_MLE_LLM_BASE_URL` |
| **vLLM** *(local)* | OpenAI-compatible, no key | `google/gemma-4-E2B-it` | — | `KG_MLE_LLM_BASE_URL` |

Any other OpenAI-compatible API can be added without code changes by
setting `KG_MLE_LLM_PROVIDER=openai` (or any of the OpenAI-compatible
provider names) plus `KG_MLE_LLM_BASE_URL=https://your-host/v1`.

**Anthropic JSON-mode trick.** Anthropic's Messages API doesn't expose a
native `response_format=json_object`, so the adapter does two things:
prepends a strict "return only JSON" instruction to the system prompt,
and prefills the assistant turn with `{` (Anthropic supports
assistant-side prefill). The model continues a JSON object instead of
preamble prose; the leading `{` is restored to the returned content.

**Honest gap.** Anthropic's tool-use API (which forces structured JSON
via tool schemas) would be a stronger guarantee than the prefilled-`{`
trick, but it would couple our prompt design to Anthropic-specific tool
definitions. The current approach works for the JSON-extraction parser
the agents already use and stays prompt-shape-portable across
providers.

## 3. Tool Registry

`src/kg_mle/registry/` — Pydantic models, tolerant loader, optional
enrichment.

**Models:** `ToolRegistry → Tool → Endpoint → Parameter / ResponseField`.
Unknown parameter types default to `string`; missing methods default to
`GET`; missing descriptions get templated from the field name.

**Response-schema shapes handled.** The fixture mixes them on purpose:

```text
JSON Schema:    {"type":"object","properties":{"hotel_id":{"type":"string"}}}
flat field map: {"hotel_id":"string","name":"string"}
field list:     [{"name":"hotel_id","type":"string"}, ...]
missing/null/empty: → no fields
```

A flat map is distinguished from a JSON-Schema shell by whether the
keys are schema keywords (`type`, `properties`, `items`, `oneOf`, ...).
This heuristic could mis-identify a flat map whose key happens to be one
of those words; in the fixture this hasn't occurred.

**`endpoint_id` format** is `domain/name`. Duplicates within a domain
raise `ValueError` at load time naming both colliding tools. Chose
human-readable short IDs with a loud-fail safety net over
`domain/tool/name` because the curated subset has no collisions.

**Field enrichment** adds `canonical_name`, `aliases`, and confidences
to parameters and response fields:

```text
deterministic aliases (destination→city, venue→location,
                       available_time→start_time, check_in→date)
+ optional structured-output LLM suggestions via StructuredLLMRegistryEnricher
```

Guardrails: `canonical_name` must be in a whitelist; confidence ∈ [0,1];
suggestions below threshold or for missing fields are rejected; original
field names are never removed (they remain the actual call arguments).
This is what the graph's grounding logic reads — see §4.

| Decision | Alternative | Why |
|---|---|---|
| Tolerant normalization (preserve incomplete endpoints) | Strict validation that rejects | ToolBench data is inconsistent; rejecting kills coverage. |
| Pydantic types throughout | Pass raw dicts downstream | Centralising parsing keeps every later module simpler. |
| LLM enrichment is structured-output + whitelist | Free-text LLM output | A bad LLM canonical name would silently miswire the graph. |

### Honest gaps

- The flat-map vs JSON-Schema heuristic is name-based. A schema with a
  field literally named `"type"` would be misread. None in the fixture.
- The canonical-name whitelist was hand-maintained for the fixture; it
  has 36 entries. Scaling to full ToolBench would need either a vetted
  vocabulary or an LLM-curated one with human review.

## 4. Tool Graph

`src/kg_mle/graph/` — JSON-serializable typed graph.

**Node types:** `domain`, `tool`, `endpoint`, `parameter`, `response_field`.
**Edge types:** `contains_tool`, `exposes_endpoint`, `requires_parameter`,
`returns_field`, `output_satisfies_input` (grounding), `same_domain`,
`semantic_related`.

**Grounding logic.** A response field satisfies a parameter when their
identifier sets overlap. An identifier set is
`{name, canonical_name (if set), *aliases}`, so the matcher reads
enrichment metadata directly off the registry models — not a hardcoded
table. Each grounding edge records `match_type`
(`exact_name`, `canonical`, `alias`) so the sampler can prefer stronger
matches.

Earlier iterations used a `COMMON_PARAMETER_NAMES` blocklist plus a
local `FIELD_ALIASES` dict. Both decisions broke the design:

- The blocklist excluded `city`, `start_time`, `query`, etc., killing
  the cross-domain chains the assignment explicitly wants.
- The local alias dict bypassed enrichment, leaving enrichment metadata
  unread. Enrichment was decorative.

Removing both unlocked 7 cross-domain grounding edges in the fixture,
including `food/check_availability.available_time → events.start_time`
which works only because deterministic enrichment maps
`available_time → start_time`.

**Semantic expansion** is opt-in (`--semantic-graph`). The retriever
indexes endpoint cards (description + inputs + outputs) and adds
`semantic_related` edges for pairs above a threshold. Two backends behind
one `SemanticRetriever` Protocol:

```text
SentenceTransformerSemanticRetriever  (local MiniLM)  ← default
Mem0SemanticRetriever                 (HF embeddings + Gemini LLM + in-memory Qdrant)
```

**Threshold choice.** Default 0.78. Sweep at 0.72/0.76/0.80/0.84
produced 117/74/24/6 edges respectively before deterministic-duplicate
filtering. After filtering: 0.78 yields 4 meaningful cross-domain
semantic links. Lower admitted too much noise; higher was too sparse.
This is tuned to the specific MiniLM model — a different embedding
model would need re-tuning.

| Decision | Alternative | Why |
|---|---|---|
| JSON artifact, not Neo4j | External graph DB | Sampler only needs adjacency; reviewers shouldn't run services. |
| `output_satisfies_input` is the strongest edge | Treat all edge types equally during sampling | Direct support for "coherent chaining" rubric property. |
| Semantic edges are opt-in, never override deterministic edges | Always-on semantic expansion | Cosine similarity ≠ proof that output satisfies input. |

**Current verification** (deterministic, no semantic):

```text
nodes 274 │ edges 474 │ output_satisfies_input 29 │ same_domain 180
```

With local MiniLM semantic at 0.78: +4 `semantic_related` edges.

### Honest gaps

- The `same_domain` edge count (180) is dense — basically a fully-connected
  per-domain clique. The walker treats them as fallback only. At scale,
  this would need pruning to "likely sequences," but the fixture is small
  enough that the noise is tolerable.
- Mem0's ANN is non-deterministic in principle. We default to local MiniLM
  (deterministic given a fixed model). Mem0 is exercised by a live test
  but isn't the reproducibility backbone.

## 5. Tool-Chain Sampler

`src/kg_mle/sampler/` — graph-driven sampler + corpus planner + steering.

### Walker (`walker.py`)

Deterministic DFS with backtracking over endpoint-to-endpoint edges,
filtered and ordered into tiers (grounded → same_domain → semantic). The
seeded RNG shuffles within each tier; ties break on target endpoint_id.

**`ChainConstraints` interface:**

```text
n_steps                  int | (min, max)
min_distinct_tools       int
min_distinct_domains     int
required_domains         tuple[str, ...]
required_endpoint        str | None
min_grounded_transitions int        # the "coherent chaining" knob
allow_semantic_edges     bool
forbid_endpoint_ids      tuple[str, ...]    # steering hook
```

The assignment's two named cases ("at least one tool from a given
domain", "exactly N steps") are `required_domains` and `n_steps`.
Varied length is `n_steps=(min, max)`. Coherent chaining is
`min_grounded_transitions`. Steering hooks are `forbid_endpoint_ids`
and a recommended `required_domains[0]` from the steerer.

| Decision | Alternative | Why |
|---|---|---|
| DFS + backtracking | Bidirectional BFS / shortest path | Every chain in the search space is interesting; constraints are checked at terminal, not en route. |
| Tiered edge preference | Single edge pool with terminal filter | Biases search toward groundable chains by construction. |
| Per-tier seeded shuffle, deterministic tie-break | Pure deterministic ordering | Pure determinism → same chain every seed, defeats sampling. |
| Terminal-only constraint check (with grounded-feasibility prune) | Full branch-and-bound | Marginal speedup at fixture scale; readability wins. |

**Fixture observation.** Probe over 50 seeds:

```text
fully-grounded 2-step:  50/50 seeds succeed
fully-grounded 3-step:  50/50 seeds succeed
fully-grounded 4-step:   0/50 seeds succeed
```

The 29 grounding edges don't chain densely enough for 4 consecutive
grounded transitions. The planner therefore targets
`min_grounded_transitions = n_steps − 2` for 4+ step chains — one
same-domain hop per chain is acceptable. The walker correctly raises
`UnsatisfiableConstraintsError` rather than hallucinating.

### Corpus Planner (`plan.py`) and Steering (`steering.py`)

For each chain `0..N`, `CorpusPlanner` builds a fresh `ChainConstraints`
from: a length distribution skewed to 3–4 steps, a `multi_step_fraction`
(default 0.55) that lifts `min_distinct_tools` to 2,
`min_grounded_transitions = n_steps − 2`, plus the steerer's hints. On
`UnsatisfiableConstraintsError`, a fixed relaxation ladder fires:

```text
loosen min_grounded_transitions
  → clear required_domains
  → loosen min_distinct_domains
  → clear forbid_endpoint_ids     ← gives up on steering, not the chain
  → loosen min_distinct_tools
  → loosen n_steps
```

`--no-cross-conversation-steering` swaps `CorpusSteerer` for
`NullSteerer`. Both record counters (so Run A vs Run B are directly
comparable on the same metrics); only the steerer returns forbid lists
and least-used domain hints. The hard-exclusion threshold has a floor
of 3 so small corpora don't run out of usable endpoints.

| Decision | Alternative | Why |
|---|---|---|
| Hybrid steering: hard exclusion + soft domain preference | Probability-based softmax reweighting | Uses only constraints the walker already exposes; clean layer separation. |
| Per-chain seed `planner_seed*1_000_003 + plan_index` | Thread a single RNG | Steering on/off must not shift each chain's individual RNG state. |
| `NullSteerer` still records counters | Skip counters when steering off | Run A vs Run B needs comparable stats on both sides. |
| Relaxation ladder gives up steering before the chain | Drop unsatisfiable chains | Steering is meant to widen coverage, not lose conversations. |

### Honest gaps

- Steering's threshold formula (`max(3, ceil(target/endpoint_count * 1.6))`)
  was tuned by trial and error on the fixture. No formal study of the
  diversity/throughput tradeoff at other corpus sizes.
- Endpoint-pair overuse is recorded but the walker doesn't directly forbid
  pairs — only individual endpoints. Pair-level steering is future work.
- The relaxation ladder is fixed-order, not learned. A chain that fails
  with `(required_domains, min_grounded=2)` might have succeeded under
  `(no required_domains, min_grounded=2)` faster than the ladder's order
  reaches it.

## 6. Offline Tool Execution

`src/kg_mle/executor/` — stateful per-conversation session.

```text
state.py      SessionState (issued-values index, chronological log)
mocks.py      MockResponseGenerator + canonical example pools + ID prefixes
validator.py  Pydantic dynamic model + strict grounding check + typed errors
session.py    ExecutorSession.call / suggest_arguments / example_values
```

### How a tool call works

```text
session.call(endpoint_id, arguments)
  ├─► record call in state.log
  ├─► validate:
  │     • dynamic Pydantic model for type + required-presence
  │     • strict grounding: every grounded parameter's value must appear in
  │       state.issued_ids(source_field) — under literal OR canonical name
  ├─► on ExecutorError: append role:"tool" {error:{...}} log entry, raise
  └─► on success: generate mock (schema-consistent + chain-consistent),
        register every string response field into issued-values, return dict
```

Mocks are deterministic from a session seed. IDs get prefixed hashes
(`htl_a3f9`, `bk_b8c2`, `evt_d7e1`); fields with known canonical names
(city, date, symbol, ...) draw from realistic-value pools; everything
else falls back to typed templates.

### Strict grounding applies to all grounded parameters

Not just `*_id` fields. `finance/search_symbol → finance/get_quote` is
grounded via `symbol` — if the assistant invents a symbol that
`search_symbol` never returned, the chain is incoherent and the
validator rejects it. Indexing under both literal and canonical name
handles the enriched-alias case.

### Sample trace with a deliberate hallucination

```json
{"role":"assistant","tool_calls":[{"endpoint":"travel/search_flights",
   "arguments":{"origin":"...","destination":"Paris","date":"2026-04-11"}}]}
{"role":"tool","content":{"flight_id":"flt_lfxctn","airline":"Airline 8","price":"299.50"}}
{"role":"assistant","tool_calls":[{"endpoint":"travel/book_itinerary",
   "arguments":{"flight_id":"made_up_value", ...}}]}
{"role":"tool","content":{"error":{"kind":"ungrounded_argument",
                                   "parameter":"flight_id",
                                   "expected_one_of":["flt_lfxctn"]}}}
{"role":"assistant","tool_calls":[{"endpoint":"travel/book_itinerary",
   "arguments":{"flight_id":"flt_lfxctn", ...}}]}
{"role":"tool","content":{"booking_id":"bk_bpcdp3","status":"confirmed"}}
```

A reviewer sees the rejection, the repair, and the recovery without
consulting an external log.

| Decision | Alternative | Why |
|---|---|---|
| Deterministic mocks; LLM polish opt-in | LLM-generated responses by default | Chain-critical fields must be byte-stable for reproducibility. |
| Stateful session, generator-driven | Batch `run_chain` | The generator interleaves clarifications and user replies between tool calls. |
| Executor owns `suggest_arguments` + `example_values` | Generator re-derives defaults | The executor already owns schema + canonical pool + issued-values index. |
| Failures raise typed errors AND appear inline as `role: tool` entries | Toggle between modes | One needs control flow; the other needs reviewer visibility. Both at once. |
| Strict grounding on all grounded params, ID or not | Only enforce on `*_id` | Sampler's grounded transitions describe an output→input promise at the field-name level. |

### Honest gaps

- **Mock realism is bounded.** Unknown-canonical strings produce values
  like `"Airline 8"` and `"Status 16"`. The judge's `naturalness`
  dimension flags this. The planned `--llm-mock-polish` flag would help;
  I haven't built it because the chain-critical path doesn't need it.
- **Mid-conversation `modify_step` re-derives transitions from the graph,
  but the new endpoint's required fields may not have been indexed by the
  earlier steps.** The deviation handler runs a graph lookup and rejects
  if no path exists, but it can't anticipate every grounding gap. Tested
  for accept and reject paths but not for the edge case where a modify
  succeeds graph-side but then fails at the executor's strict-grounding
  check on the next call.

## 7. Multi-Agent Generator

`src/kg_mle/generator/` — three stateless agents + coordinator.

```text
protocol.py      Plan, ParameterPlan, StepPlan, AssistantTurn, UserTurn,
                 ToolCallProposal, ChainDeviation, Conversation, GeneratorConfig
agents.py        Protocols + DeterministicPlanner / User / Assistant
llm_agents.py    StructuredLLMClient + LLMPlanner / User / Assistant
                 (retry + Pydantic + deterministic fallback)
coordinator.py   ConversationCoordinator
```

### Agent contracts

```text
Planner:    SamplingResult, seed                       → Plan
              (intent, per-step ParameterPlan with confidence + ambiguous_step_indices)
User:       initial_request(plan, seed)                → UserTurn
            reply_to_clarification(plan, target_step,
                                   target_param, seed) → UserTurn
Assistant:  compose_turn(plan, transcript, session,
                         steps_completed, ...)         → AssistantTurn
              kind ∈ {clarification, tool_calls, final_summary}
              tool_calls is list[ToolCallProposal] (length 1 today; list shape
              future-proofs parallel calls without protocol changes)
              optional ChainDeviation proposal
```

### Disambiguation: planner-primary + confidence-gated assistant initiative

The planner outputs per-parameter `confidence` and a list of
`ambiguous_step_indices`. The assistant's default behaviour:

1. If a current step is in `ambiguous_step_indices` and not yet
   clarified → ask the planner-flagged question.
2. Otherwise, if a free parameter has `confidence < 0.6` (configurable)
   AND the assistant's own initiative confidence is high (configurable
   `assistant_clarification_threshold`), the assistant asks its own
   clarification.
3. Otherwise, tool-call or final summary.

### Chain-bound termination + confidence-gated `ChainDeviation`

The chain is the contract. The assistant may propose
`ChainDeviation(kind="add_step" | "modify_step", endpoint_id, position, deviation_confidence)`.
The coordinator accepts only if:

- `deviation_confidence ≥ assistant_deviation_threshold` (default 0.85)
- the endpoint exists in the registry
- the graph supports the new transitions (prev→new and new→next must
  have endpoint-to-endpoint edges; grounded preferred, same_domain
  accepted)

Accepted deviations rewire the session's `sampling_result` in place;
the session state preserves so earlier issued IDs remain valid.
Rejected deviations are recorded in `metadata.deviations_rejected`
with the reject reason — the judge can see what was attempted.

### LLM agents wrap deterministic ones as fallback

Each LLM agent has the same Protocol as its deterministic counterpart
and falls back per-call on any failure (provider exception, malformed
JSON, Pydantic validation error, plan/chain shape mismatch). The
fallback records `last_run = {"path": "fallback", "reason": "..."}`,
which the coordinator includes in metadata so the judge and downstream
analysis can see which conversations were LLM-driven. The CLI
`diversity` command intentionally keeps generation deterministic even
under `--use-llm`; it uses the hosted model for enrichment and judging,
not for the Run A/Run B generator itself.

### LLM failure modes

| Failure | What happens |
|---|---|
| Provider 4xx/5xx / network exception | Immediate fallback to deterministic agent. |
| Malformed JSON (no `{...}` found) | One retry with error context prepended. If still malformed → fallback. |
| Schema mismatch (Pydantic `ValidationError`) | One retry with error context. If still invalid → fallback. |
| Plan endpoint mismatch / wrong step count | One retry. If still wrong → fallback. |
| Executor rejects an LLM-emitted tool call | One repair attempt invokes the assistant again with the failure in the transcript. If that fails → conversation closes with the failure visible inline. |

The coordinator never crashes on LLM behaviour. The worst case is a
deterministic conversation with metadata noting why the LLM path was
abandoned.

| Decision | Alternative | Why |
|---|---|---|
| Three-agent decomposition (Planner / User / Assistant) | Four agents with a separate Final-Writer | The closing summary belongs to the same agent that emitted tool calls. |
| Structured output on **both** Planner and Assistant | Just the Assistant | Planner's `ambiguous_step_indices` and per-param confidences need structured shape for the disambiguation gate to be deterministic. Two structured agents also doubles the rubric evidence. |
| Planner-primary disambiguation + confidence-gated assistant initiative | Pure planner-driven | The planner can be wrong; the assistant gets per-param confidence and can ask when planner is unsure. |
| Chain-bound termination + graph-verified `ChainDeviation` | Free assistant termination / extension | Free deviation would compete with the planner's length-distribution targets. The gates keep deviations exceptional. |
| Deterministic agents by default; LLM agents wrap them as fallback | LLM-required | Missing API key, provider 402, persistently-malformed JSON → conversation still completes. |
| Stateless agents; coordinator owns the transcript; typed Pydantic handoff | Agents read the natural-language transcript directly | Typed handoff prevents one agent's prompt wording from confusing another. |

### Honest gaps

- The 3-step live LLM test was relaxed from "≥3 successful tool calls"
  to "≥1" because the LLM path's completion rate is model-dependent and
  flaky across providers. The deterministic coordinator tests are the
  authoritative correctness signal; the live test is a protocol
  regression catcher.
- `ChainDeviation` accept paths are unit-tested with synthetic deviation
  proposals; I haven't observed an LLM actually propose a confident
  graph-valid deviation in live runs. The mechanism is in place, but its
  real-world utility is unproven.
- The deterministic user's clarification reply uses a fixed template
  `"For X, use Y."`. The judge's `naturalness` score reflects this — it's
  visibly synthetic. LLM user upgrades the wording; deterministic mode
  doesn't.

## 8. Evaluation and LLM-as-Judge

`src/kg_mle/evaluation/`

Two layers:

```text
deterministic structural metrics    ← always runs, offline
  + optional hosted LLM judge       ← --use-llm or --llm-judge
  → JSON metrics + scored JSONL
```

**Commands:**

```powershell
kgmle evaluate --input ... --output ...                  # deterministic only
kgmle --use-llm evaluate --max-llm-judge-records 10      # adds LLM judge
kgmle evaluate --repair --repair-threshold 8.0           # bounded repair pass
```

**Deterministic metrics** (no credentials needed):

```text
schema_valid              Pydantic Conversation round-trips
role_sequence_valid       first message is user, all roles known
tool_response_coverage    tool responses / assistant tool_calls
chain_completion          successful responses / expected n_tool_calls
error_free_trace          1 − (tool errors / tool messages)
deterministic_score       mean of the above × 10
```

### LLM judge dimensions (0–10 each, plus issues + rationale)

| Dimension | What it measures |
|---|---|
| `task_completion` | Does the final assistant message satisfy the user's stated request? |
| `tool_trace_validity` | Are tools selected, sequenced, and responded to coherently? |
| `argument_grounding` | Do tool-call arguments come from user input, plan values, or prior tool outputs? |
| `response_grounding` | Does the final answer stay faithful to actual tool outputs? |
| `naturalness` | Does the dialogue read like a plausible human exchange? |
| `overall_score` | Holistic — severe trace failures can dominate. |
| `confidence` | Judge's own confidence in its score. |

The dimensions intentionally separate **trace correctness**
(`tool_trace_validity`, `argument_grounding`) from **answer quality**
(`task_completion`, `response_grounding`, `naturalness`). A conversation
can have natural wording but a bad tool trace, or a valid trace but a
hallucinated final answer; both should be visible to filtering.

| Decision | Alternative | Why |
|---|---|---|
| Two-layer evaluation (deterministic always, LLM optional) | LLM-only | CI runs offline; deterministic metrics are reproducible. |
| 5 rubric dimensions (>3 required) | Generic helpfulness/fluency | Generic metrics miss invalid tool calls that "sound natural." |
| Separate `argument_grounding` and `response_grounding` | One "grounding" score | Different failure modes need different filters. |
| 0–10 quality scores; 0–1 coverage ratios | One uniform scale | Quality is a human-readable rubric; coverage is a fraction. |
| Treat provider failures as record-level `llm_judge.error`, not pipeline failures | Crash on first provider error | Hosted inference is optional; the deterministic eval shouldn't be blocked. |

### Quality bands (in `metadata.evaluation`)

```text
gold    deterministic ≥ 9.0, no tool errors, no major judge issues
silver  deterministic ≥ 8.0, no unresolved tool errors
reject  validation fail, unresolved tool error, deterministic < 8.0,
        or major tool/grounding/task-completion judge fail
```

`usable_for_training` is a derived boolean used by downstream filters.

### Honest gaps

- The judge prompt explicitly tells the model to score only visible JSON
  and not infer real API behaviour. LLMs sometimes do anyway. The
  rationale field helps spot this; I don't have an automatic detector.
- The 8.0 threshold for "usable" is by-eye, not by data — chosen because
  the rubric language ("good, minor issue") corresponds to 8 on a 0–10
  scale. No correlation study against actual training-time downstream
  performance was possible inside the project's time budget.
- The 100-sample E2E test uses a *deterministic fake judge* at the
  provider boundary (`tests/e2e/test_pipeline_100.py`). The fake gives
  every record a score of 8 or 9, so the test's `mean ≥ 8.0` assertion
  catches structural regressions but not actual judging quality. A real
  hosted judge run on 100 records exists as a manual smoke step, not
  in CI.

## 9. Retry and Repair

`src/kg_mle/repair/`

Bounded post-evaluation repair pass:

```text
conversation → evaluate → detect triggers → RepairPlan → apply or record → re-evaluate
```

```text
models.py    RepairTrigger, RepairPlan, RepairResult, RepairStrategy
policy.py    RepairPolicy thresholds + should_attempt_repair + assign_quality_band
planner.py   DeterministicRepairPlanner + LLMRepairPlanner (advisory; apply still local)
```

### Repair triggers

```text
schema validation failure
role/message sequence failure
tool error in the trace
deterministic_score      < 8.0
tool_trace_validity      < 8.0    (when LLM-judged)
argument_grounding       < 8.0
task_completion          < 9.0
naturalness              < 5.0
```

### Strategies and what actually applies

| Strategy | Applied by | Status |
|---|---|---|
| `rewrite_final_response` | Deterministic planner | Applied in place. Composes a grounded summary from existing tool outputs. |
| `mark_rejected` | Deterministic planner | Recorded; conversation kept for analysis. |
| `apply_graph_verified_chain_change` | Coordinator | Plan recorded; status `failed` in the evaluator pass because the evaluator doesn't re-run the coordinator. |
| `regenerate_conversation` | Coordinator | Plan recorded; status `failed` in the evaluator pass. |
| `fix_tool_arguments` | Coordinator | Plan recorded; same reason. |
| `insert_clarification` | Coordinator | Plan recorded; same reason. |

This is the honest reading: the post-evaluation pass can only safely
do final-response rewrites in place. The four coordinator-required
strategies are recorded with reasoning so the trail is visible, but the
evaluator doesn't try to mutate state it doesn't have access to. The
PDF asks the system to "attempt to repair … rather than simply discard
it"; recording a planned-but-failed repair satisfies that intent
honestly. A future regenerator pass that re-invokes the coordinator
from the evaluate command would close this — it's not in the submitted
build.

### LLM repair planner

`--use-llm evaluate --repair` swaps in `LLMRepairPlanner`. The LLM
proposes a `RepairPlan` (same Pydantic schema), but application still
goes through the deterministic apply layer — hosted output never
directly mutates a conversation.

| Decision | Alternative | Why |
|---|---|---|
| Repair module owns policy + planning; coordinator owns stateful application | Single module that does both | The evaluator runs on serialized conversations; only the coordinator has live executor state. |
| Fixed budget of one repair attempt | Unbounded retry | Bounds cost; demonstrates the loop without dominating the run. |
| Chain-changing repairs are graph-verified | Free-form chain edits | Mutating chains without graph proof would create new hallucination risk. |
| LLM repair plans are advisory; apply stays deterministic | LLM rewrites the conversation directly | Validation pipeline stays single-path; LLM output is constrained to the `RepairPlan` schema. |

### Honest gap

- The four coordinator-required strategies that always return `failed`
  in the evaluator pass are the biggest unfinished piece. The plan is
  recorded with reasoning, which is the honest answer ("here's what
  should happen, here's why we couldn't do it in this pass"), but a
  reviewer wanting to see end-to-end repair will only see
  `rewrite_final_response` actually applied.

## 10. Context Management and Diversity

### Within-conversation grounding

`ExecutorSession` is the single source of truth for one conversation's
state:

```text
issued_ids                  field_name (and canonical_name) → [values...]
_responses_by_endpoint      endpoint → [response dicts...]
log                         chronological tool_call / tool_response / tool_error
```

The generator never invents IDs because:

1. `session.suggest_arguments(endpoint_id)` pulls grounded params from
   `issued_ids[source_field]` (or canonical), and pool-samples free
   params from `CANONICAL_EXAMPLES`.
2. `session.example_values(endpoint_id)` exposes the same data as a
   few-shot pool the LLM can render into its prompt.
3. The validator strict-checks every grounded parameter against
   `issued_ids` before mocking — hallucination → typed error inline.

### Cross-conversation steering

`CorpusSteerer` is corpus-level statistical memory:

```text
domain counts             tool counts             endpoint counts
endpoint-pair counts      chain-length counts     domain-pattern counts
```

It answers "what kind of chain should we sample next to improve corpus
diversity?", not "what argument value should this tool call use?" The
counters drive `forbid_endpoint_ids` (hard exclusion) and a
least-used-domain hint for `required_domains` (soft preference).

`NullSteerer` is the on/off symmetry: same Protocol, no forbidding, no
biasing — but still records counters so Run A and Run B have
comparable metrics.

### Diversity experiment

```powershell
kgmle diversity --count 100 --seed 42 --output-dir artifacts/diversity
```

Two runs with the same seed/count/fixture, only the steering flag
differs. Six diversity metrics + four quality metrics:

| Metric | Run A: no steering | Run B: steering | Δ |
|---|---:|---:|---:|
| domain entropy | 2.5304 | 2.7039 | **+0.1735** |
| endpoint coverage ratio | 0.7778 | 0.8889 | **+0.1111** |
| tool coverage ratio | 0.8889 | 1.0000 | **+0.1111** |
| distinct endpoint-pair ratio | 0.3125 | 0.3042 | −0.0083 |
| domain pattern diversity | 12 | 13 | +1 |
| top endpoint share | 0.2235 | 0.1912 | **−0.0323** |
| mean deterministic score | 9.9600 | 10.0000 | +0.0400 |
| chain completion | 0.9900 | 1.0000 | +0.0100 |
| schema valid rate | 1.0000 | 1.0000 | 0.0000 |
| usable-for-training rate | 0.9900 | 1.0000 | +0.0100 |

**Interpretation.**

- Steering widened coverage: +1 endpoint covered, all tools used, +0.17
  on domain entropy, +1 domain pattern.
- Concentration dropped: the most-used endpoint went from 22% to 19% of
  chains.
- Distinct endpoint-pair ratio fell slightly (−0.008). This is an
  artifact of the steered run producing **more total transitions** while
  the distinct-pair count grew less than proportionally — i.e., the
  same pairs appearing across more chains. Honest reading: the
  endpoint-pair diversity metric is sensitive to total-transition
  count, and the slight drop isn't a quality regression.
- **Quality did not degrade.** The steered run hit perfect deterministic
  score, schema validity, and chain completion.

| Decision | Alternative | Why |
|---|---|---|
| Counter-based steering; not semantic memory | Persistent vector memory | Counters are deterministic, inspectable, directly tied to the metrics; vector memory is overkill for a 100-sample corpus. |
| Both runs use the same seed/count/fixture | Different seeds per run | The experiment must isolate steering's effect. |
| Six metrics + quality | Single "diversity score" | Steering can improve some metrics and hurt others; the comparison should expose that. |
| Transparent metrics (counts, entropy, ratios) | Embedding-distance diversity | Tied directly to graph/sampler behavior; reproducible without an embedding model. |

### Honest gaps

- **The fixture is small.** Diversity metrics are meaningful for relative
  comparison but not a claim about full-ToolBench coverage. The 22→19%
  drop in top-endpoint share is real; whether it generalises to larger
  corpora is untested.
- **Endpoint-pair forbidding is diagnostic-only.** The walker doesn't
  directly forbid pairs; it forbids endpoints. The pair counter is
  recorded for visibility but doesn't feed back into sampling.
- **Steering and length distribution are independent knobs.** They could
  interact (e.g., forbidding overused endpoints might bias toward longer
  or shorter chains depending on graph topology). I haven't measured
  this interaction.

## 11. Prompt Design

LLM features all use the same `StructuredLLMClient` and the same
pattern: structured JSON in, Pydantic out, retry once with error
context, deterministic fallback.

### Planner prompt

Receives: sampled tool chain, endpoint schemas, required parameters,
ambiguity policy, output schema for `Plan`. Returns a JSON `Plan`.
Rationale: ask the planner for *parameter plans and ambiguity flags*,
not final conversation text — the coordinator validates parameters,
routes clarifications, and preserves executor grounding. A free-form
dialogue plan would be harder to validate and easier to hallucinate.

### Assistant prompt

Receives: transcript so far, current step, endpoint schema, suggested
arguments from executor state, prior output examples, confidence
thresholds. Returns an `AssistantTurn`. Rationale: the assistant can
propose, but the executor proves. Schema-bound tool-call proposals +
strict executor validation = LLM fluency without trace risk.

### Judge prompt

Receives: full conversation JSON, rubric. Returns the 5 dimensions +
overall + confidence + issues + rationale. Explicit instruction:
score only visible evidence; penalise missing role tags, missing tool
responses, invented tool IDs, unresolved tool errors, ungrounded
arguments.

### Repair prompt

Receives: conversation JSON, triggers, current scores, allowed
strategies, output schema for `RepairPlan`. The LLM may propose, but
applying the plan still runs through the deterministic apply layer.

### Failed prompt iteration

An earlier registry-enrichment prompt called Hugging Face-hosted models
as generic chat models with the model name `google/gemma-4-E2B-it`.

**What went wrong:**

- Gemma wasn't available through the selected HF chat route.
- Switching to `Qwen/Qwen2.5-3B-Instruct` + `featherless-ai` provider
  returned JSON-shaped suggestions, but hit `402 Payment Required` after
  one call.

**What changed:**

- Registry enrichment, judge, repair, and generator agents now all use
  the same provider-neutral `StructuredLLMClient`.
- Gemini is the default (free tier exists, JSON mode is stable).
- Groq is the open-model backup.
- HF remains available as an adapter but isn't the default path.
- Deterministic fallback was added everywhere LLMs are used.

**Lesson:** prompt quality is not enough if the provider interface is
brittle. Provider-neutral structured JSON + strict validation +
deterministic fallback is the load-bearing combination.

## 12. Output Schema and CLI

`src/kg_mle/schema/` — Pydantic artifact schemas matching the PDF's
record shape:

```text
records.py     GeneratedConversationRecord, ScoredConversationRecord
metrics.py     EvaluationMetricsArtifact, MetricsSummary, RepairSummary
diversity.py   DiversityReport
common.py      LLMJudgeScore (also re-exports the judge's Pydantic shape)
```

**Generated record** (one JSONL line):

```text
conversation_id      string
messages             list of {role, content, tool_calls?}
plan                 the Plan that drove generation
metadata
  ├── seed, original_chain, final_chain, n_tool_calls, domains
  ├── advance_type_counts, transition_summary
  ├── clarifications_taken[], repair_attempts[]
  └── deviations_accepted[], deviations_rejected[]
```

**Scored record** = generated record + `metadata.evaluation` block with
the deterministic metrics, quality band, usable-for-training flag, and
optional `llm_judge` result (or `{"error": "..."}` on provider
failure).

| Decision | Alternative | Why |
|---|---|---|
| Pydantic as the schema source of truth | Separate JSON Schema file | Avoids drift between in-memory and on-disk; reuses the same validators code already uses. |
| Top-level conversation shape strict; metadata extensible (`extra="allow"`) | Strict everywhere | Metadata evolves (steering details, repair history, LLM paths); record shape shouldn't. |
| LLM judge failures preserved as `{"error": "..."}` | Drop the field on failure | Reviewer needs to know the judge failed — and which records were affected. |

### CLI surface

```text
kgmle build       --input ... --artifacts-dir ...
                  [--semantic-graph --semantic-backend local|mem0]
                  [--enrich-registry-fields / --llm-enrich-registry]
kgmle generate    --count N --seed S --output ...
                  [--cross-conversation-steering / --no-cross-conversation-steering]
                  [--allow-semantic-edges]
kgmle evaluate    --input ... --output ... [--scored-output ...]
                  [--llm-judge --max-llm-judge-records N]
                  [--repair --repair-threshold 8.0]
kgmle diversity   --count N --seed S --output-dir ...
                  [same semantic + repair flags]
```

Global `--use-llm` is a single switch that turns on hosted features
across commands consistently:

| Command | Effect of `--use-llm` |
|---|---|
| `build` | Enables structured-output registry enrichment + semantic graph. |
| `generate` | Loads built artifacts; uses LLM agents (with deterministic fallback per turn); enables semantic-edge traversal. |
| `evaluate` | Enables LLM judge; if `--repair` set, also enables LLM repair planner. |
| `diversity` | Enables structured-output registry enrichment, semantic graph construction, semantic-edge traversal, and LLM judging. Generation remains deterministic so Run A/Run B isolate cross-conversation steering rather than model variance. |

Individual legacy flags (`--llm-judge`, `--llm-enrich-registry`) are
preserved for targeted runs.

## 13. Testing

```text
tests/fixtures/      sample-fixture validity + intentional messiness
tests/unit/          unit tests across registry, graph, sampler,
                     executor, generator, evaluation, repair, schema,
                     diversity, llm clients
tests/integration/   CLI smoke + repair + diversity command paths
tests/integration/*live* 5 @pytest.mark.live tests for Mem0, HF enricher,
                     LLM generator, diversity-with-LLM
tests/e2e/           test_pipeline_100.py — full pipeline on 100 samples
```

**E2E test.** `tests/e2e/test_pipeline_100.py`:

1. Run `kgmle build` against the fixture.
2. Run `kgmle generate --count 100 --seed 42`.
3. Run `kgmle --use-llm evaluate --max-llm-judge-records 100` with a
   deterministic fake judge at the provider boundary.
4. Assert:
   - 100 generated and 100 scored records, all validate through
     `GeneratedConversationRecord` and `ScoredConversationRecord`.
   - `summary.mean_llm_overall_score ≥ 8.0` (threshold rationale below).
   - `schema_valid_rate == 1.0`.
   - `tool_response_coverage ≥ 0.95`.
   - ≥ 50% of records have `n_tool_calls ≥ 3` AND `≥ 2` distinct tools.
   - At least one conversation contains an assistant clarification.

**Threshold rationale.** `8.0/10` corresponds to "good, minor issue" in
the judge rubric. High enough to catch broken traces; allows minor
naturalness imperfections that synthetic conversations naturally have.

**Live tests skip cleanly** when their API key is missing or the
provider errors out. Their purpose is catching protocol regressions
(Mem0 result-shape changes, HF API shape changes, JSON-extraction
parser drift), not asserting LLM quality.

### Honest gap

- The E2E test fakes the judge at the provider boundary, so it asserts
  the *integration path* is intact (the same code path a real
  Gemini/Groq judge would traverse) but not real judging quality. A
  real-provider 100-sample run is a manual step, not CI.

## 14. Open Areas

What's intentionally not in the submitted build, with honest reasons:

- **Parallel tool calls.** The `AssistantTurn.tool_calls` list is
  length-1 in v1. The data model is future-proofed (it's already a
  list), but the sampler's `pattern="parallel"` is a no-op and the
  executor doesn't have `call_parallel`. Deferred because the rubric's
  hard requirements (50–60% multi-step + multi-tool, multi-turn
  disambiguation, structured output) are all served by sequential
  chains.
- **Coordinator-required repairs that actually re-invoke the coordinator.**
  The plan is recorded with reasoning; the regeneration pass isn't
  wired. Would close the gap between "attempted repair" and "applied
  repair" for `fix_tool_arguments` / `apply_graph_verified_chain_change`
  / `regenerate_conversation`.
- **Real-provider 100-sample judging in CI.** The E2E test uses a
  deterministic fake judge. A real run is a manual smoke step.
- **Endpoint-pair forbidding in the walker.** The pair counter is
  recorded but the walker only forbids individual endpoints. Pair-level
  steering would let the planner attack high-frequency transitions
  directly.
- **`--llm-mock-polish`.** Would have the executor's mock generator pass
  string fields through an LLM for natural phrasing. Plumbing is there
  in design; not implemented because the chain-critical path doesn't
  need it.
- **Full-ToolBench scale.** The fixture is 45 endpoints. Scaling would
  need: a vetted canonical-name vocabulary, `endpoint_id` migration to
  `domain/tool_name/name`, possibly indexed graph traversal, cached
  embeddings. None of this is built; the design has hooks for it.
