# DESIGN

## 1. Executive Summary

This project builds an offline synthetic data generation pipeline for multi-turn, multi-tool conversations grounded in ToolBench-style API definitions.

The submitted workflow is intentionally scoped as a production-oriented MVP:

- It uses a curated, reproducible ToolBench-style subset instead of the full ToolBench corpus.
- It implements the required architecture as real package modules and CLI commands.
- It keeps the default pipeline deterministic and runnable without private credentials.
- It supports optional semantic graph expansion using local embeddings and a Mem0 adapter.

Current pipeline:

```text
ToolBench-style JSON
-> normalized registry
-> tool graph
-> sampler
-> offline executor
-> conversation generator
-> evaluator / repair loop
-> JSONL dataset
```

Implemented so far:

```text
ToolBench-style JSON
-> normalized registry
-> deterministic + optional semantic tool graph
```

## 2. ToolBench Interpretation

ToolBench provides raw API/tool definitions: categories, tool names, endpoint names, parameters, descriptions, HTTP methods, paths, and sometimes response schemas.

In this project, ToolBench-style definitions are used as the source of truth for what tools exist. We are not reusing ToolBench conversations as generated output. The generated conversations will be synthetic.

Design decision:

- Use ToolBench-style API definitions as schema grounding.
- Generate new conversations and traces from those definitions.

Reason:

- The assignment asks for synthetic conversation generation grounded in ToolBench schemas, not for replaying existing ToolBench data.

## 3. Scope And Dataset Fixture

The fixture at `data/sample_toolbench/tools.json` contains:

- 9 categories: Finance, Sports, AI / ML, Entertainment, Travel, Gaming, Events, Food, Weather.
- 45 total endpoints.
- Intentional inconsistencies that mimic raw ToolBench schema issues.

Examples of intentional messiness:

- lowercase HTTP methods
- missing methods
- `path` instead of `url`
- `response` instead of `response_schema`
- null or empty response schemas
- `required` instead of `required_parameters`
- `optional_params` and `optionalParameters`
- missing or unknown parameter types

Design decision:

- Use a curated representative subset for the submitted run.
- Keep the loader generic enough for broader ToolBench-style JSON.

Alternative considered:

- Process the full ToolBench corpus immediately.

Reason rejected:

- The exercise is time-boxed. A curated subset gives reproducible review behavior while still exercising registry normalization, graph construction, semantic search, sampling, execution, evaluation, and repair.

## 4. Packaging And CLI

The project uses:

- `pyproject.toml`
- `hatchling` for packaging
- `typer` for CLI
- `rich` for logs and command output
- `pydantic` for typed internal models

CLI surface:

```text
kgmle build
kgmle generate
kgmle evaluate
```

Current implemented command:

```text
kgmle build
```

It writes:

```text
artifacts/registry.json
artifacts/tool_graph.json
```

Design decision:

- Build an installable package with a real console command instead of loose scripts.

Reason:

- The assignment will be reviewed end-to-end. A package and CLI make the workflow easier to run, test, and inspect.

Alternative considered:

- Plain scripts and `argparse`.

Reason rejected:

- That would be simpler initially but weaker as a production-style submission.

## 5. Configuration And Secrets

Default paths and model settings live in `src/kg_mle/config.py`.

The project loads local `.env` values through `python-dotenv`.

Important files:

```text
.env.example   committed template only
.env           local secrets/config, ignored by git
```

Current model defaults:

```text
KG_MLE_LLM_PROVIDER=huggingface
KG_MLE_LLM_MODEL=google/gemma-4-E2B-it
KG_MLE_SEMANTIC_BACKEND=local
KG_MLE_EMBEDDING_PROVIDER=huggingface
KG_MLE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
KG_MLE_SEMANTIC_THRESHOLD=0.78
KG_MLE_SEMANTIC_TOP_K=5
```

Design decision:

- Keep secrets out of git.
- Use `.env.example` only as documentation.
- Use deterministic fallbacks so reviewers do not need private keys.

Alternative considered:

- Require API keys for all generation/evaluation.

Reason rejected:

- Reviewer environments may not have credentials. The required workflow should still run offline.

## 6. Tool Registry

The registry is the normalized source of truth for tools and endpoints.

Raw input:

```text
data/sample_toolbench/tools.json
```

Normalized output:

```text
artifacts/registry.json
```

Core models:

- `ToolRegistry`
- `Tool`
- `Endpoint`
- `Parameter`
- `ResponseField`

The registry normalizes:

- category/domain names
- endpoint names
- HTTP methods
- paths
- required and optional parameters
- parameter types
- response fields

Unknown or missing parameter types default to `string`.

Design decision:

- Normalize tolerantly and preserve endpoints whenever possible.

Alternative considered:

- Strict validation that rejects incomplete endpoints.

Reason rejected:

- ToolBench-style data is inconsistent. Rejecting incomplete endpoints would reduce coverage and make the pipeline brittle.

Alternative considered:

- Use raw JSON dictionaries directly in graph, sampler, and executor.

Reason rejected:

- Every downstream module would need repeated defensive parsing. Centralizing normalization keeps later components simpler and easier to test.

### Response Schema Shapes

ToolBench-style data uses several response-schema shapes inconsistently. The
loader handles four:

```text
JSON Schema:        {"type": "object", "properties": {"hotel_id": {"type": "string"}}}
flat field map:     {"hotel_id": "string", "name": "string"}
list of field dicts:[{"name": "hotel_id", "type": "string"}, ...]
missing/null/empty: -> empty list (no fields)
```

A flat map is distinguished from a JSON-Schema shell by checking whether the
keys look like JSON-Schema keywords (`type`, `properties`, `items`, `oneOf`,
etc.). If they do not, the dict is treated as a flat field map.

### Endpoint ID Uniqueness

`endpoint_id` is formatted as `domain/name`. If two tools in the same domain
expose endpoints with the same name, the loader raises `ValueError` with both
tool names. This prevents silent dict-keyed overwrites in downstream consumers.

If you scale to full ToolBench data and hit a collision, either rename the
colliding endpoints in the input or extend the ID format to include
`tool_name`. The choice was made deliberately to keep `endpoint_id` short and
human-readable in the curated subset, with a loud-fail safety net.

### Removed: `raw_schema` Field

Earlier iterations stored `raw_schema: dict[str, Any]` on every `Endpoint` and
persisted it to `artifacts/registry.json`. It roughly doubled artifact size
(~145 KB → ~97 KB after removal) and risked downstream code accidentally
reading raw, unnormalized fields. It is removed. Debugging tools that need the
raw payload should re-read the source JSON.

### Registry Field Enrichment

The registry also supports a constrained enrichment layer for canonical names, aliases, and type hints.

Default behavior:

```text
deterministic normalization
-> deterministic alias enrichment
-> optional structured LLM-style suggestions
```

Deterministic examples:

```text
destination -> city
venue -> location
available_time -> start_time
check_in -> date
```

The enrichment layer stores metadata on parameters and response fields:

```text
canonical_name
aliases
type_confidence
alias_confidence
enrichment_source
```

Design decision:

- Do not let enrichment rename or delete original fields.
- Preserve original schema names and add canonical metadata.

Reason:

- Original parameter names are needed for actual tool-call arguments.
- Canonical metadata helps the graph and sampler reason across near-equivalent fields without losing traceability.

Optional LLM-style suggestions are represented as structured Pydantic objects:

```text
FieldEnrichmentSuggestion
├── endpoint_id
├── target: parameter | response_field
├── field_name
├── canonical_name
├── aliases
├── type_hint
├── confidence
├── reason
└── source
```

Guardrails:

- suggestions must validate through Pydantic
- `canonical_name` must be in the allowed canonical vocabulary
- `type_hint` must be one of the internal registry types
- confidence must be between 0 and 1
- suggestions below threshold are rejected
- suggestions for missing fields are rejected
- original field names are never removed
- deterministic normalization remains the fallback

Design decision:

- Use LLM-style structured enrichment only for unresolved alias/type cases, not as the primary registry normalizer.

Reason:

- The registry is the base contract for graph construction. Keeping deterministic normalization first preserves reproducibility, while structured enrichment demonstrates LLM-oriented schema repair capability in a controlled way.

Future path:

- Wire `FieldEnrichmentSuggestion` to an actual provider-backed structured-output LLM call.
- Cache accepted enrichment suggestions into artifacts for reproducibility.
- Compare deterministic-only vs enriched graph quality in the diversity experiment.

Implementation status:

- The structured suggestion schema, validation, confidence gating, and application logic are implemented.
- A Hugging Face registry enricher is implemented behind `--llm-enrich-registry`.
- Local live test with `KG_MLE_LLM_MODEL=google/gemma-4-E2B-it` reached Hugging Face, but the HF router reported that the configured model is not available as a chat model and has no text-generation provider mapping.
- Live test with `KG_MLE_LLM_MODEL=Qwen/Qwen2.5-3B-Instruct` and `KG_MLE_HF_PROVIDER=featherless-ai` successfully reached a provider-backed conversational endpoint and returned JSON-shaped suggestions, proving the integration path works. A full run was stopped by provider billing/credit limits (`402 Payment Required`), so live LLM enrichment remains explicit opt-in.
- A `@pytest.mark.live` integration test (`tests/integration/test_hf_enricher_live.py`) now exercises the enricher against 2 endpoints when `HF_TOKEN` is present and the provider is reachable. It asserts the report is well-formed (whitelisted `canonical_name`, valid confidence, field application reflects the suggestion) but does not assert specific suggestions — LLM output is non-deterministic by design.
- Therefore live Gemma enrichment is not enabled by default. The deterministic and fake structured-enrichment paths remain fully tested.
- To use live registry enrichment, configure a Hugging Face-hosted instruction model that supports the Inference API/router, or configure a future local provider such as Ollama, LM Studio, vLLM, or LiteLLM for Gemma.

## 7. Tool Graph

The graph is a typed, JSON-serializable artifact.

Output:

```text
artifacts/tool_graph.json
```

Core models:

- `GraphNode`
- `GraphEdge`
- `ToolGraph`

Node types:

- `domain`
- `tool`
- `endpoint`
- `parameter`
- `response_field`

Edge types:

- `contains_tool`
- `exposes_endpoint`
- `requires_parameter`
- `returns_field`
- `output_satisfies_input`
- `same_domain`
- `bridge`
- `semantic_related`

Design decision:

- Use a JSON-serializable graph instead of Neo4j.

Reason:

- The current graph is small enough for memory.
- Reviewers do not need to run an external graph database.
- JSON artifacts are easy to inspect.
- The sampler only needs adjacency-style traversal.

Alternative considered:

- Neo4j or another graph database.

Reason rejected:

- Adds service setup, deployment overhead, slower tests, and operational complexity that does not improve the submitted MVP.

Future path:

- The typed graph artifact can be exported to Neo4j later if full ToolBench scale needs indexed graph traversal or interactive graph analysis.

## 8. Deterministic Graph Edges

The graph builder always creates deterministic edges.

Structural edges:

```text
domain -> tool
tool -> endpoint
endpoint -> parameter
endpoint -> response_field
```

Grounding edges:

```text
source endpoint returns X
target endpoint requires X
=> output_satisfies_input
```

Example:

```text
travel/search_hotels returns hotel_id
travel/get_hotel_details requires hotel_id
travel/book_itinerary requires hotel_id
```

Derived edges:

```text
travel/search_hotels -> travel/get_hotel_details
travel/search_hotels -> travel/book_itinerary
```

Design decision:

- Treat `output_satisfies_input` as the strongest edge type.

Reason:

- It directly supports grounded tool chaining and reduces hallucinated arguments.

### Alias-Aware Field Matching

A response field satisfies a parameter when their identifier sets overlap. An
identifier set is `{name, canonical_name (if set), *aliases}`. This means the
matcher reads enrichment metadata directly off `Parameter` and `ResponseField`
rather than from a hardcoded local alias table.

Each grounding edge records the match strength in metadata:

```text
exact_name: parameter.name == field.name
canonical:  match via canonical_name on either side
alias:      match via an alias entry only
```

The sampler can later prefer `exact_name` edges over `alias` edges when
weighting chain candidates.

### What Changed And Why

An earlier iteration of `_field_satisfies_parameter` used a module-level
`COMMON_PARAMETER_NAMES` blocklist and a module-level `FIELD_ALIASES` dict.
Both decisions hurt the graph:

- The blocklist excluded `city`, `date`, `location`, `start_time`, `query`,
  and other common fields. This made cross-domain chains like
  `weather/get_forecast(city)` ungroundable from any other endpoint that
  returned a `city` field, even though that is exactly the cross-domain
  chaining the assignment requires.
- The hardcoded alias dict bypassed the structured enrichment layer. The
  enrichment work set `canonical_name = "city"` on `destination` parameters
  but the matcher never read it; only the local table did. Enrichment was
  effectively decorative.

Removing the blocklist and routing matching through enrichment metadata
unlocked 7 new cross-domain grounding edges in the fixture, including one
(`food/check_availability.available_time -> events.start_time`) that is only
possible because deterministic enrichment recognises `available_time` as a
canonical alias of `start_time`. The same alias has been in the codebase
since enrichment landed but was unused by the graph until now.

The risk of false-positive matches (e.g., every endpoint returning `city`
chaining into every endpoint that consumes `city`) is real but is a
*sampler* concern, not a graph-construction concern. The graph should expose
the truth; the sampler can downweight low-information matches using the
`match_type` metadata recorded on each edge.

## 9. Semantic Graph Expansion

The graph builder supports optional semantic expansion.

Current command:

```powershell
kgmle build --semantic-graph --semantic-backend local
```

The semantic retriever indexes endpoint cards:

```text
Endpoint: travel/search_hotels
Domain: travel
Description: Find hotels in a city.
Inputs: city, check_in, max_price
Outputs: hotel_id, hotel_name, nightly_price
```

The builder then adds `semantic_related` edges for high-scoring endpoint pairs.

Design decision:

- Use semantic expansion as optional candidate discovery, not as the grounding backbone.

Reason:

- Semantic similarity can suggest realistic adjacent tools, but it does not prove that one tool's output satisfies another tool's input.

## 10. Mem0, Local Embeddings, And Model Choices

The assignment mentions mem0 as an option for vector-backed retrieval. The code includes a `Mem0SemanticRetriever` behind the same interface as the local retriever.

Local test finding:

- `mem0ai==2.0.2` successfully installed.
- `Memory.from_config(...)` exists.
- Mem0 initialized the Hugging Face embedder successfully.
- Mem0 then initialized an LLM even though the graph builder only needed semantic search.
- Without explicit LLM config, Mem0 defaulted to OpenAI and failed without `OPENAI_API_KEY`.
- The Mem0 adapter now passes explicit Gemini config when `GOOGLE_API_KEY` is available.
- The adapter uses `infer=False` when adding endpoint cards to Mem0, because endpoint cards are already structured and do not need LLM-based memory extraction.
- The adapter uses Mem0 search filters instead of deprecated top-level entity parameters.
- The adapter configures Qdrant for MiniLM's 384-dimensional embeddings.

Final Mem0 verification:

```text
kgmle build --semantic-graph --semantic-backend mem0
```

With `HF_TOKEN`, `GOOGLE_API_KEY`, and Gemini config present in `.env`, Mem0 successfully built the graph:

```text
nodes: 274
edges: 472
semantic_related: 4
```

The accepted Mem0 semantic edges matched the local MiniLM backend:

```text
sports/get_schedule -> gaming/get_tournament_schedule
gaming/get_tournament_schedule -> sports/get_schedule
travel/book_itinerary -> events/book_tickets
events/book_tickets -> travel/book_itinerary
```

Conclusion:

- Mem0 is supported as an optional backend, but it is not the safest default for an offline review environment unless LLM credentials are configured.
- A Hugging Face token alone is not sufficient for Mem0 2.0.2's memory pipeline because Mem0 also expects an LLM provider.
- Gemini is the preferred Mem0 LLM provider for this project because Google AI Studio has accessible free-tier usage.

Default semantic backend:

```text
SentenceTransformerSemanticRetriever
model: sentence-transformers/all-MiniLM-L6-v2
```

Optional Mem0 LLM config:

```text
KG_MLE_MEM0_LLM_PROVIDER=gemini
KG_MLE_MEM0_LLM_MODEL=gemini-2.0-flash-lite-001
GOOGLE_API_KEY=<local secret>
```

### Why `infer=False` For Mem0 Endpoint Cards

Mem0 is designed for user/conversation memory. In that default mode, `infer=True` lets an LLM extract cleaner memories from free-form text.

Example where LLM inference is useful:

```text
Raw: "I usually book vegetarian-friendly restaurants near my hotel."
Inferred memory: "User prefers vegetarian-friendly restaurants near their hotel."
```

Our graph-building input is different. Endpoint cards are already structured:

```text
Endpoint: travel/search_hotels
Domain: travel
Description: Find hotels in a city.
Inputs: city, check_in, max_price
Outputs: hotel_id, hotel_name, nightly_price
```

Letting an LLM rewrite these cards would add cost, latency, and nondeterminism without improving the schema facts. It could also accidentally omit parameter names or response fields that are important for graph construction.

Therefore the Mem0 adapter uses:

```text
infer=False
```

This means:

- store the endpoint card as-is
- create embeddings
- make it searchable
- do not ask an LLM to extract or rewrite the memory

This is the correct behavior for deterministic schema graph construction.

### Mem0 Test Coverage

The `Mem0SemanticRetriever` constructor accepts an injected `memory` object
so its behavior can be tested without a live Mem0 install or provider
credentials. Unit tests cover:

- indexing always calls `Memory.add(..., infer=False)` with the correct
  `user_id` and `endpoint_id` metadata
- search-result parsing prefers `metadata.endpoint_id`, falls back to looking
  up the previously-indexed text, and skips results that cannot be resolved
- `search.filters` uses `{"user_id": "kg_mle_tool_graph"}`, not the
  deprecated top-level entity parameters
- results are capped at the requested `top_k`

The real Mem0 path (with Hugging Face embeddings, Gemini LLM, and an
in-memory Qdrant store) is now covered by a live integration test
(`tests/integration/test_mem0_live.py`) that runs when `HF_TOKEN` and
`GOOGLE_API_KEY` are present and skips otherwise. It asserts the
index/search contract — not specific semantic matches, because ANN is
approximate. The injected-memory unit tests remain the credential-free
safety net for the result-parsing contract.

## 10.1 LLM Provider Orchestration

The project separates LLM provider configuration from business logic.

Provider config is loaded through:

```text
src/kg_mle/llm/providers.py
```

Supported provider patterns:

```text
gemini      -> GOOGLE_API_KEY
openai      -> OPENAI_API_KEY
anthropic   -> ANTHROPIC_API_KEY
deepseek    -> DEEPSEEK_API_KEY
qwen        -> DASHSCOPE_API_KEY
together    -> TOGETHER_API_KEY
huggingface -> HF_TOKEN
ollama      -> KG_MLE_LLM_BASE_URL
lmstudio    -> KG_MLE_LLM_BASE_URL
vllm        -> KG_MLE_LLM_BASE_URL
```

Design decision:

- Use a lightweight provider configuration layer instead of hardcoding one vendor.

Reason:

- The assignment should be runnable by reviewers with different available credentials.
- It lets the future generator, judge, and repair loop swap between Gemini, OpenAI, Claude, DeepSeek, Qwen, Gemma via local runtimes, or deterministic fallbacks.
- It avoids coupling the core pipeline to a single API vendor.

Gemma 4 note:

- Gemma 4 remains the preferred open model family for future generation/judging.
- For Mem0 specifically, Gemma 4 can be used through local/provider-compatible routes such as Ollama, LM Studio, vLLM, Together, or LiteLLM if configured.
- MiniLM remains the embedding model because embeddings and instruction generation are different tasks.

Why MiniLM:

- free
- CPU-friendly
- small
- widely used for sentence similarity
- works well for endpoint-card retrieval

LLM model policy:

```text
Preferred future generation/judge model: google/gemma-4-E2B-it
Embedding model: sentence-transformers/all-MiniLM-L6-v2
```

Design decision:

- Use Gemma 4 as the preferred future instruction model.
- Use MiniLM for embeddings.

Reason:

- Generation/judging and embedding are different tasks. Gemma is appropriate for instruction generation; MiniLM is appropriate for lightweight vector similarity.

## 11. Reducing Hallucination And Nondeterminism

Hallucination controls already designed:

- normalize schemas before graph construction
- use exact output-to-input edges for grounded chains
- preserve response fields and required parameters
- later executor will maintain per-conversation state for returned IDs
- later validator will reject ungrounded arguments

Semantic nondeterminism controls:

- semantic graph expansion is optional
- deterministic graph edges always exist
- tests use `FakeSemanticRetriever`
- semantic matches are sorted by score descending, then endpoint ID
- self-edges are removed
- semantic edges are not added when a deterministic source-target pair already exists
- thresholds and `top_k` are fixed and stored in metadata
- accepted semantic edges are persisted into `tool_graph.json`

Threshold experiment with local MiniLM before deterministic duplicate filtering:

```text
threshold 0.72 -> 117 semantic edges
threshold 0.76 -> 74 semantic edges
threshold 0.80 -> 24 semantic edges
threshold 0.84 -> 6 semantic edges
```

After excluding source-target pairs already covered by deterministic graph construction:

```text
threshold 0.76 -> 9 semantic edges
threshold 0.78 -> 4 semantic edges
threshold 0.80 -> 2 semantic edges
```

Chosen default:

```text
KG_MLE_SEMANTIC_THRESHOLD=0.78
```

Reason:

- `0.76` admitted more borderline matches.
- `0.80` was too sparse.
- `0.78` preserved a small number of meaningful cross-domain semantic links.

## 12. Current Graph Verification

Deterministic graph:

```text
nodes: 274
edges: 474

node types:
domain: 9
tool: 9
endpoint: 45
parameter: 101
response_field: 110

edge types:
contains_tool: 9
exposes_endpoint: 45
requires_parameter: 101
returns_field: 110
output_satisfies_input: 29
same_domain: 180
```

`output_satisfies_input` match-type breakdown:

```text
exact_name: 28
canonical:  1   (food/check_availability.available_time -> events.start_time)
```

Cross-domain grounding edges (the cases that hardcoded blocklists previously suppressed):

```text
sports/get_player_stats          -> gaming/recommend_games        (player_id)
entertainment/search_live_shows  -> events/create_calendar_event  (start_time)
gaming/search_games              -> sports/get_game_odds          (game_id)
gaming/search_games              -> events/create_calendar_event  (title)
gaming/get_tournament_schedule   -> events/create_calendar_event  (start_time)
food/check_availability          -> events/create_calendar_event  (available_time -> start_time, canonical)
weather/recommend_outdoor_window -> events/create_calendar_event  (start_time)
```

Semantic graph with local MiniLM at threshold `0.78`:

```text
nodes: 274
edges: 478
semantic_related: 4
```

Accepted semantic edges:

```text
sports/get_schedule -> gaming/get_tournament_schedule
gaming/get_tournament_schedule -> sports/get_schedule
travel/book_itinerary -> events/book_tickets
events/book_tickets -> travel/book_itinerary
```

## 13. Testing Strategy

Tests are organized by purpose:

```text
tests/fixtures/
tests/unit/
tests/integration/
tests/e2e/
```

Current coverage:

- fixture validity and intentional messiness
- path/config helpers
- `.env` config override behavior
- CLI smoke behavior
- registry normalization (clean, messy, flat-dict responses, list-shaped responses, JSON-Schema shells)
- registry persistence
- registry enrichment (deterministic, structured-output suggestions, Hugging Face enricher with mocked client)
- duplicate `endpoint_id` rejection
- graph model validation
- graph builder construction
- grounding edge creation, including alias-driven grounding produced by enrichment
- cross-domain grounding via common fields (`city`, `start_time`, etc.)
- semantic edge filtering
- local semantic retriever behavior through a mocked model
- Mem0 retriever indexing and search-result parsing via an injected fake `Memory`

Current result:

```text
51 passed
```

### Live Integration Tests

Two `@pytest.mark.live` tests in `tests/integration/` exercise the real
external paths when credentials are present and skip cleanly otherwise:

- `test_mem0_live.py` — indexes 4 endpoint cards through real Mem0 (HF
  embeddings, Gemini LLM, in-memory Qdrant) and asserts the search results
  are drawn from the indexed set, scored, and sorted descending. It does
  not assert specific semantic matches because ANN + model drift would
  make that flaky.
- `test_hf_enricher_live.py` — runs the HF registry enricher against 2
  endpoints (capped via `max_llm_endpoints`) and asserts the report is
  well-formed: every accepted suggestion has a whitelisted
  `canonical_name`, a confidence in `[0, 1]`, and is reflected on the
  actual field. It does not assert that any specific field gets enriched.

Both tests skip — not fail — on missing credentials, provider rejection
(402 billing, model unavailable), or transient outages. The intent is
to catch *protocol regressions* (Mem0 result-shape change, HF API
shape change, JSON-extraction parser drift) without flaking CI on
provider availability.

Design decision:

- Add tests immediately after each component.

Reason:

- The assignment explicitly requires unit, integration, and end-to-end tests. Keeping tests close to each implementation step prevents a late testing scramble.

## 14. Open Design Areas

Still to implement:

- Tool-chain sampler
- Cross-conversation steering
- Offline executor with stateful mocked outputs
- Multi-agent conversation generator
- Deterministic evaluator plus optional Gemma-backed judge
- Retry/repair loop
- Diversity experiment
- End-to-end test that generates at least 100 samples
