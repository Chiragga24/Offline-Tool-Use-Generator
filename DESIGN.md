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
KG_MLE_LLM_PROVIDER=gemini
KG_MLE_LLM_MODEL=gemini-2.0-flash-lite-001
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

### Hosted LLM Provider Choice

The default hosted LLM provider is Gemini, with Groq as the preferred
backup for open-model inference.

Design decision:

- Use Gemini for optional generation and judge calls.
- Keep Groq behind the same structured JSON client as an OpenAI-compatible
  backup.
- Keep Hugging Face as an optional adapter, not the default path.

Reason:

- Gemini has a practical free-tier path for API testing and supports
  JSON-shaped generation well enough for structured-output prompts.
- Groq gives a second hosted path for open models without downloading
  weights locally.
- Hugging Face Inference Providers are useful, but the Qwen live test hit
  provider billing limits, so HF is too volatile to be the default reviewer
  path.

Implementation:

```text
src/kg_mle/llm/clients.py
```

`StructuredLLMClient` supports:

- Gemini REST `generateContent` with `responseMimeType=application/json`
- Groq `/openai/v1/chat/completions` with `response_format=json_object`
- other OpenAI-compatible providers through `KG_MLE_LLM_BASE_URL`
- Hugging Face as a retained fallback adapter

The generator agents depend only on `complete_json(...)`, so switching
providers does not change planner/user/assistant logic.

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

## 13.5 Tool-Chain Sampler (Walker)

The sampler walks the tool graph to produce candidate chains. It is
graph-driven (the assignment's hard requirement) and constraint-driven
(the assignment's constrained-sampling requirement).

```text
src/kg_mle/sampler/
├── constraints.py    # ChainConstraints, SamplingResult, Transition
├── walker.py         # ToolChainSampler — DFS with backtracking
├── steering.py       # CorpusSteerer, NullSteerer, CorpusCounters
└── plan.py           # CorpusPlanner — orchestrates a corpus
```

### How the Sampler Works (Short Version)

1. The walker is built once from a `ToolGraph`. It indexes every
   `output_satisfies_input`, `same_domain`, and `semantic_related` edge
   into a per-endpoint adjacency list; structural edges
   (`contains_tool`, `exposes_endpoint`, etc.) are ignored because they
   describe graph anatomy, not chain advancement.
2. A `ChainConstraints` object names what the caller wants: chain length,
   required domains, minimum grounded transitions, etc.
3. The walker chooses a start endpoint (deterministic from the seed,
   biased toward `required_domains` / `required_endpoint`).
4. From the start, it does depth-first search with backtracking. At each
   step it lists outgoing edges, filters by `allow_semantic_edges`,
   sorts them into tiers (grounded → same_domain → semantic), shuffles
   within each tier using the seeded RNG, and tries each in order.
5. When the chain reaches the target length, a terminal check enforces
   `min_distinct_tools`, `min_distinct_domains`, `required_domains`,
   `min_grounded_transitions`, and `required_endpoint`. If any fail, the
   walker backtracks and tries another candidate.
6. If no chain in the search space satisfies the constraints, it raises
   `UnsatisfiableConstraintsError` with diagnostics — never a
   hallucinated chain.

### Design Decisions

**Design decision:** Use a graph-walking sampler with DFS and
backtracking, not goal-directed path search.

**Alternative considered:** Bidirectional BFS from a start node toward a
constraint-satisfying terminal endpoint.

**Reason rejected:** BFS would optimise for shortest path; here every
chain in the search space is interesting, and constraint satisfaction
is checked at the terminal, not en route. DFS with backtracking
explores the same search space with simpler code and matches the
"propose a chain that satisfies constraints" framing — not "find the
shortest chain that does."

---

**Design decision:** Tiered edge preference — grounded > same_domain >
semantic — rather than a single edge pool.

**Alternative considered:** Treat all edge types as equivalent and
filter by constraint at the terminal check only.

**Reason rejected:** The assignment's "coherent chaining" property
rewards chains where step N is grounded in step N−1 outputs. Treating
grounded and same_domain edges as equivalent during the walk made it
significantly more likely that backtracking would explore unproductive
same_domain or semantic paths before stumbling into a grounded one.
Tiered ordering biases the search toward groundable chains by
construction.

---

**Design decision:** `semantic_related` edges are opt-in via
`allow_semantic_edges=True`, not on by default.

**Alternative considered:** Include `semantic_related` edges in the
default search space, weighted lower than `same_domain`.

**Reason rejected:** Semantic edges are loosely justified (cosine
similarity, not a structural relationship). They are useful for
expanding the *candidate* set of related endpoints but are not strong
evidence that one tool's output satisfies another's input. The walker
should expose them only when the caller explicitly asks; otherwise
the judge's "tool correctness" score would suffer.

---

**Design decision:** Per-tier seeded shuffle with `(target endpoint_id)`
tie-breaks.

**Alternative considered:** Pure deterministic ordering (sort by edge
source/target ids only) with no shuffle.

**Reason rejected:** Pure determinism would produce the same chain for
every seed, defeating the purpose of seeded sampling and making
multi-chain corpora repetitive. Tie-breaking on endpoint_id within the
shuffled tier keeps single-seed runs reproducible even if the graph's
edge ordering changes between builds (Pydantic dump order is stable,
but we don't want to depend on that).

---

**Design decision:** Terminal-only constraint check (length match,
distinct counts, required domains, grounded transitions).

**Alternative considered:** Branch-and-bound — prune partial chains
that can't satisfy constraints from here.

**Reason rejected:** We *do* prune the obvious infeasibility (e.g., if
`min_grounded_transitions` needs more grounded edges than the remaining
steps can supply, the walker rejects non-grounded candidates
immediately). But full branch-and-bound on distinct-tools /
distinct-domains would complicate the code for marginal speedup at the
fixture size we're working with (45 endpoints, chain length ≤ 5).
Terminal-only checks keep the walker readable.

### Determinism Guarantee

Same seed + same constraints + same graph = same chain. The seeded
`random.Random` instance threads through every shuffle (start ordering,
candidate ordering within each tier). Tests cover this explicitly.

### Fixture Grounding-Density Limit

A 50-seed probe revealed:

```text
fully-grounded 2-step chains: 50/50 seeds succeed
fully-grounded 3-step chains: 50/50 seeds succeed
fully-grounded 4-step chains:  0/50 seeds succeed
```

The 29 grounding edges in the curated fixture don't chain densely
enough to produce a fully-grounded 4-step path. This is a fixture
observation, not a walker limitation. The planner targets
`min_grounded_transitions = n_steps − 2` for 4+ step chains — one
same_domain fallback per chain is acceptable. The grounded transitions
remain the ones the executor uses for chain-consistent ID mocking;
same_domain transitions get argument-from-user-intent synthesis.

## 13.6 Corpus Planner and Cross-Conversation Steering

The planner is the layer between the dataset properties the assignment
calls out (50–60% multi-step, varied lengths, balanced domain coverage,
coherent chaining) and the walker's per-chain constraint API. It owns
the corpus-level loop, the steering counters, and the relaxation
strategy that protects against unsatisfiable constraints.

### How the Planner Works (Short Version)

1. For each chain in `range(target_count)`, the planner builds a fresh
   `ChainConstraints` instance derived from:
   - a length distribution skewed toward 3–4 steps,
   - a `multi_step_fraction` (default 0.55) that bumps
     `min_distinct_tools` to 2 when applicable,
   - `min_grounded_transitions = n_steps − 2` (or stricter for short
     chains),
   - the steerer's recommendations: least-used domain as
     `required_domains[0]`, currently-overused endpoints as
     `forbid_endpoint_ids`.
2. It calls the walker with a per-chain seed derived deterministically
   from the planner seed plus the chain index.
3. On `UnsatisfiableConstraintsError`, the planner runs through a fixed
   relaxation ladder (loosen grounded → clear required_domains → loosen
   distinct_domains → clear forbid list → loosen distinct_tools →
   loosen n_steps), retrying up to `max_relaxation_attempts` times.
4. The steerer records every successful chain — domains, tools,
   endpoints, transitions, length, domain pattern — and exposes the
   summary via `report.counters_summary`.
5. `--no-cross-conversation-steering` swaps `CorpusSteerer` for
   `NullSteerer`. `NullSteerer` still records counters (so Run A and
   Run B have comparable stats) but returns empty forbid lists and
   alphabetical-only domain hints. The planner's main loop is
   unchanged.

### Design Decisions

**Design decision:** Hybrid steering — hard exclusion (`forbid_endpoint_ids`)
plus soft preference (`least_used_domains` biasing
`required_domains`).

**Alternative considered:** Probability-based reweighting — full
softmax over endpoint usage counts.

**Reason rejected:** Probability reweighting would require the walker
to support weighted sampling, which it doesn't, or for the planner to
intercept and rewrite the walker's edge ordering. Both bleed steering
concerns into the walker. Hard exclusion plus required-domain bias
uses only the constraints the walker already exposes — clean
separation of layers.

---

**Design decision:** Threshold for hard exclusion has a hard floor of 3,
plus a per-corpus baseline of `target_count / endpoint_count * 1.6`.

**Alternative considered:** Pure baseline-based threshold,
no floor.

**Reason rejected:** Without the floor, small corpora
(`target_count < endpoint_count`) hit a threshold of 1 or 2 — any
endpoint used at all gets forbidden after a few chains, and the
planner runs out of usable endpoints within a dozen samples. The floor
of 3 lets every endpoint get a few uses before steering kicks in, so
the diversity gain doesn't come at the cost of generation throughput.

---

**Design decision:** `NullSteerer` still records counters even though it
returns no penalties.

**Alternative considered:** When steering is off, don't track counters
at all.

**Reason rejected:** The diversity experiment requires Run A and Run B
to be directly comparable on the same metrics. Without counters in
Run A, "did steering increase endpoint coverage?" can't be answered
quantitatively. The cost of always recording is one Counter increment
per result — negligible.

---

**Design decision:** Per-chain seeds derived as
`(planner_seed * 1_000_003 + plan_index) % (2**31 − 1)`.

**Alternative considered:** Thread a single RNG through the entire
corpus loop.

**Reason rejected:** A single RNG means turning steering on/off changes
the RNG state for every subsequent chain (because steering-driven
constraint shaping consumes a different number of RNG draws). That
makes Run A and Run B harder to compare — the *same* chain index would
draw a different walker seed. Per-chain seeds keep each chain's RNG
state stable across the steering flag.

---

**Design decision:** Relaxation ladder rather than abandoning chains
that fail.

**Alternative considered:** On `UnsatisfiableConstraintsError`, just
record the failure and move on.

**Reason rejected:** Steering's whole point is to widen coverage by
forbidding overused endpoints. As the forbid list grows, constraint
sets get harder to satisfy. Without relaxation, the planner would
silently drop chains in the back half of long runs — exactly when
diversity matters most. The relaxation ladder gives up steering for
one chain rather than losing the chain entirely; the steerer still
records what was sampled.

### Empirical Diversity Contrast

Smoke run with `target_count=100`, `seed=42`, same fixture:

```text
                              Run A (no steering)   Run B (steering)
distinct endpoints used       35 / 45               40 / 45
distinct domains              8                     9
top-endpoint share            76%                   65%
multi-step + multi-tool       81%                   81%
chains generated              100                   100
relaxation fallbacks          0                     0
```

Steering widens endpoint coverage by 5 endpoints and one domain, and
reduces top-endpoint concentration by 11 percentage points, with no
loss in multi-step coverage. These are the seeds for the formal
diversity metrics computed in §15 (TBD).

## 13.7 Offline Tool Execution Model

The executor is the layer that turns a sampled chain into a realised
sequence of (tool_call, tool_response) pairs without invoking real APIs.
It is the assignment §3 requirement, and it is also the within-conversation
grounding implementation called out in §5.1.

```text
src/kg_mle/executor/
├── state.py       # SessionState, LogEntry — per-conversation memory
├── mocks.py       # MockResponseGenerator + canonical example pools
├── validator.py   # Pydantic-shaped + grounded-arg validation, typed errors
└── session.py     # ExecutorSession, OfflineExecutor — what the generator drives
```

### How It Works (Short Version)

1. `OfflineExecutor` is constructed once per pipeline run; it holds the
   registry handle.
2. For each conversation, the generator calls `open_session(sampling_result, seed=...)`.
   That returns a fresh `ExecutorSession` whose `SessionState` starts
   empty.
3. On every tool call:
   - the session records the call in the conversation log,
   - validates arguments via a dynamic Pydantic model + a strict grounding
     check that compares each grounded parameter to the session's
     issued-values index,
   - on failure, records a `tool_error` entry in the log and raises a
     typed exception (the repair loop reads both),
   - on success, generates a schema-consistent mock response, registers
     every string-valued response field into the issued-values index
     (under literal *and* canonical name), and records the response in
     the log.
4. The generator can preview-and-shape calls via:
   - `suggest_arguments(endpoint_id)` — plausible defaults derived from
     session state for grounded params and a canonical example pool for
     free params;
   - `example_values(endpoint_id)` — few-shot pool the generator can
     show the LLM: real issued IDs for grounded params, canonical
     example pool for free params.

### Design Decisions

**Design decision:** Mocks are deterministic-by-default; an LLM polish
pass is opt-in only.

**Alternative considered:** LLM-generated responses by default,
deterministic only as a fallback.

**Reason rejected:** Every response field that the next call grounds
into must be byte-stable for the conversation to be reproducible from
a seed. LLM responses are non-deterministic in temperature ≥ 0 and
costly across hundreds of calls. Deterministic mocks for the chain
critical path plus optional LLM polish for descriptive fields (planned
flag `--llm-mock-polish`) keeps the offline pipeline functional without
credentials while preserving the upgrade path.

---

**Design decision:** The executor is a *stateful session* the generator
drives one tool call at a time.

**Alternative considered:** A batch `run_chain(sampling_result)` that
executes the entire chain at once and returns a list of
(call, response) pairs.

**Reason rejected:** The multi-agent generator needs to interleave
clarifying questions, user replies, and tool calls. With a batch
runner, the executor would have to know about non-tool turns or be
called repeatedly with growing chain prefixes — both ugly. A stateful
session matches the conversation's actual control flow.

---

**Design decision:** Argument *synthesis* lives in the executor
(`suggest_arguments`, `example_values`); argument *override* is the
generator's prerogative.

**Alternative considered:** Argument synthesis is the generator's job;
the executor only validates and mocks.

**Reason rejected:** The executor already owns the schema, the
canonical example pool, and the issued-values index — three of the
four inputs needed to pick a plausible default. Making the generator
re-derive defaults would duplicate this knowledge across modules.
Instead, the executor exposes `suggest_arguments` as the easy-path API
and `example_values` as a few-shot pool the generator can render into
the LLM prompt. Generator overrides go through the same validator, so
there is no privileged caller and no trust gap.

---

**Design decision:** Failures raise typed exceptions *and* appear as
`role: "tool"` entries in the conversation log. No "permissive mode"
flag.

**Alternative considered:** Toggle the executor between "raise on
violation" and "return mock error payload" via a flag.

**Reason rejected:** The repair loop needs control flow (catch on
failure, retry with corrected args). A typed exception is the cleanest
control-flow signal. The log entry is what gives the reviewer
visibility — a flag would have made one of those two needs hard to
satisfy. Surfacing both at once means the conversation trace tells the
full story of what went wrong, in the same stream the assistant's good
calls live in.

---

**Design decision:** Strict grounding applies to all grounded
parameters, not just ID-shaped ones.

**Alternative considered:** Only enforce grounding on `*_id`-shaped
parameters; let other grounded params (city, symbol, date) pass through
without a session-state check.

**Reason rejected:** The sampler's grounded transitions describe an
output-to-input promise at the field-name level, not the ID level.
`finance/search_symbol -> finance/get_quote` is grounded via `symbol`;
if the assistant invented a symbol that `search_symbol` never returned,
the chain is incoherent and the judge's "tool correctness" score
should penalize it. Strict-everywhere keeps the executor honest about
what "groundedness" means.

---

**Design decision:** Session state registers every string-valued
response field, not just IDs, keyed by both literal and canonical
name.

**Alternative considered:** Register only ID-shaped fields.

**Reason rejected:** A direct consequence of the strict-grounding
decision above. Non-ID grounded transitions (the `symbol` case) need
the same lookup path as ID grounded transitions. Indexing under both
literal and canonical name handles the case where the enrichment layer
mapped the field to a canonical (`destination -> city`) and the next
endpoint's required parameter is the canonical form.

### Sample Trace (Conversation Log)

A 3-step chain with a deliberate hallucinated grounded argument in step 2:

```json
{"role": "assistant", "tool_calls": [{"endpoint": "travel/search_flights",
   "arguments": {"origin": "...", "destination": "Paris", "date": "2026-04-11"}}]}
{"role": "tool", "endpoint": "travel/search_flights",
   "content": {"flight_id": "flt_lfxctn", "airline": "Airline 8", "price": "299.50"}}
{"role": "assistant", "tool_calls": [{"endpoint": "travel/book_itinerary",
   "arguments": {"flight_id": "made_up_value", "hotel_id": "...", "traveler_name": "..."}}]}
{"role": "tool", "endpoint": "travel/book_itinerary",
   "content": {"error": {"kind": "ungrounded_argument",
                          "parameter": "flight_id",
                          "expected_one_of": ["flt_lfxctn"]}}}
{"role": "assistant", "tool_calls": [{"endpoint": "travel/book_itinerary",
   "arguments": {"flight_id": "flt_lfxctn", "hotel_id": "...", "traveler_name": "..."}}]}
{"role": "tool", "endpoint": "travel/book_itinerary",
   "content": {"booking_id": "bk_bpcdp3", "status": "Status 16"}}
```

A reviewer can trace the rejection, the repair, and the recovery
without consulting any external log.

## 13.8 Multi-Agent Conversation Generator

The generator turns one sampled chain into a role-tagged conversation
matching the assignment's dataset record shape. Three stateless agents
communicate through a typed Pydantic protocol; the coordinator owns
the transcript and drives them in sequence.

```text
src/kg_mle/generator/
├── protocol.py     # Pydantic models: Plan, AssistantTurn, UserTurn,
│                   #   ChainDeviation, GeneratorConfig, Conversation
├── agents.py       # Protocols + DeterministicPlanner / User / Assistant
├── llm_agents.py   # StructuredLLMClient + LLMPlanner / User / Assistant
│                   #   with retry + Pydantic + deterministic fallback
└── coordinator.py  # ConversationCoordinator
```

### How It Works (Short Version)

1. The coordinator receives a `SamplingResult` and a seed.
2. It calls the planner once: `Planner.plan(...) -> Plan`. Plan is
   Pydantic-validated, including per-parameter confidences and
   ambiguous-step indices.
3. It opens an `ExecutorSession` (from §13.7) for the conversation.
4. It calls the user simulator to produce the initial request, appended
   to the transcript.
5. Main loop: at each turn it calls the assistant
   (`Assistant.compose_turn(...) -> AssistantTurn`). Based on
   `assistant_turn.kind`:
   - **clarification** → append assistant question, call user to reply,
     append user turn. Update the relevant `ParameterPlan` with the
     supplied value.
   - **tool_calls** → for each `ToolCallProposal`, append the
     assistant's tool_call message, invoke `session.call(...)`, append
     the resulting tool response (or, on `ExecutorError`, the error and
     a single repair retry).
   - **final_summary** → append the closing message and exit.
6. Every turn additionally inspects `assistant_turn.chain_deviation`.
   Accepted deviations rewire the `SamplingResult` in place; rejected
   ones (low confidence / unknown endpoint / no graph path) are
   recorded in metadata so the judge can see the proposal.
7. The coordinator returns a `Conversation` Pydantic model carrying
   `messages`, the validated `Plan`, and a metadata dict ready for the
   dataset's JSONL line.

### Design Decisions

**Design decision:** Three-agent decomposition — Planner, User, Assistant.

**Alternative considered:** A four-agent decomposition with a separate
Final-Writer agent producing the closing assistant message.

**Reason rejected:** The closing summary is one short turn that
naturally belongs to the same agent that emitted the tool calls; the
plan and transcript carry enough context. Splitting it into a fourth
agent adds an LLM hop with no quality gain.

---

**Design decision:** Structured output on *both* Planner and Assistant
(the assignment requires ≥1; we use 2).

**Alternative considered:** Structured output only on the Assistant's
tool calls; Planner emits free-text intent + a regex-parsed parameter
list.

**Reason rejected:** Free-text planning is brittle: a slightly
different phrasing breaks the parser, and the ambiguity-injection
behaviour (which drives planner-driven disambiguation) requires a
structured `ambiguous_step_indices` list. Pydantic on both agents gives
us validation, retry-with-error-context, and clean fallback semantics
in one mechanism. Two structured-output agents is also stronger
rubric evidence than one.

---

**Design decision:** Planner-primary disambiguation with confidence-gated
assistant initiative.

**Alternative considered:** Pure planner-driven (assistant never asks
beyond the plan).

**Reason rejected:** The planner can be wrong — especially in LLM mode
where a parameter the planner thought was a canonical example might
actually be ambiguous in context. The assistant gets the per-parameter
`confidence` from the plan; when `confidence <
planner_param_low_confidence` AND its own
`assistant_clarification_confidence ≥
assistant_clarification_threshold`, it asks an additional clarification.
Both thresholds are configurable. The default behaviour is
planner-driven; the gate makes assistant initiative possible but rare.

Tested: `test_clarification_does_not_fire_when_ambiguity_zero` and the
ambiguity-injection planner tests prove the planner-driven path; the
gate logic is exercised through the deterministic assistant's branch
2 in `compose_turn`.

---

**Design decision:** Chain-bound termination with confidence-gated,
graph-verified `ChainDeviation` proposals.

**Alternative considered:** Assistant-decided termination — assistant
can quit early or extend freely.

**Reason rejected:** Free assistant termination would compete with the
planner's length distribution targets (which hit the
varied-conversation-length rubric requirement) and make Run A vs Run B
diversity comparison muddier.

The gate has three checks: (i) `deviation_confidence ≥
assistant_deviation_threshold` (default 0.85); (ii) endpoint exists in
the registry; (iii) the graph supports the new transitions — for
`add_step`, `prev → new` AND `new → next` edges must exist (grounded
preferred, same_domain accepted); for `modify_step`, both inbound and
outbound transitions must be re-derivable. The session's
`sampling_result` is updated in place when accepted; session state
(issued IDs, log) is preserved. Tested across five coordinator tests
covering both accepted and rejected variants.

`modify_step` is moderate complexity — the implementation reuses the
graph's existing edge-metadata to re-derive `Transition` objects, so
the same data structure that drove sampling drives mid-conversation
chain repair.

---

**Design decision:** Deterministic agents by default; LLM agents are
opt-in and *wrap* the deterministic agent as fallback.

**Alternative considered:** LLM-required, fail loudly without credentials.

**Reason rejected:** Same pattern as the executor's mocks and the
semantic graph: CI must run offline. The deterministic agents produce
structurally valid conversations that satisfy every hard rubric
requirement (multi-turn disambiguation, valid tool calls, role
tagging, coherent chaining). LLM agents add natural-language realism.
A missing API key, a provider 402, or persistently-malformed LLM JSON
all fall back transparently; the metadata records the path
(`{"path": "llm", "retries": 0}` vs `{"path": "fallback",
"reason": "..."}`) so the judge and the diversity experiment can read
which conversations were LLM-driven.

---

**Design decision:** Stateless agents, transcript owned by the
coordinator, typed Pydantic handoff at every boundary.

**Alternative considered:** Agents read the natural-language
transcript directly and parse it for context.

**Reason rejected:** Free-text transcript parsing makes one agent's
prompt wording confuse another. Typed handoff means the planner emits
a `Plan` object the assistant reads as a structured input — not as
text. The transcript is only consulted by the assistant for tool-call
history; clarification routing happens via `ClarificationTarget` on
the assistant turn, not by inferring from natural language.

### LLM Failure Modes and Where They Go

| Failure | What happens |
|---|---|
| Provider 4xx/5xx / network exception | Immediate fallback to deterministic agent. `last_run.path="fallback"`, `reason` carries exception text. |
| Malformed JSON (no `{...}` found) | One retry with error context prepended. If still malformed → fallback. |
| Schema mismatch (Pydantic `ValidationError`) | One retry with error context. If still invalid → fallback. |
| Plan endpoint mismatch / wrong step count | One retry. If still wrong → fallback. |
| Executor rejects an LLM-emitted tool call | One repair attempt invokes the assistant again with the failure in the transcript. If that fails → conversation closes; the failure is visible inline (per §13.7). |

The coordinator never crashes on LLM behaviour. The worst case is a
deterministic conversation with metadata noting why the LLM path
was abandoned.

---

**Design decision:** Provider-neutral structured JSON client for LLM agents.

**Alternative considered:** Keep `llm_agents.py` directly tied to the
Hugging Face `InferenceClient`.

**Reason rejected:** The Qwen path through Hugging Face reached a live
provider but hit billing/credit limits. That made the assignment's
LLM path depend on a provider-specific payment wall rather than on the
project's own design. The generator now uses `kg_mle.llm.StructuredLLMClient`,
which supports Gemini as the default hosted path and Groq as a backup,
while preserving HF as an optional adapter.

The provider adapter returns raw JSON text only. Pydantic validation,
retry-with-error-context, confidence gates, and deterministic fallback
remain in the agent layer. This keeps provider plumbing separate from
generation policy.

### Live Integration Tests

`tests/integration/test_llm_generator_live.py` runs the full
LLM-driven pipeline against the configured Gemini, Groq, or Hugging Face
provider when the matching API key is present:

- `test_llm_generator_live_produces_structurally_valid_conversation`
  — 2-step chain end-to-end, asserts role tags, ≥1 successful tool
  call, all agents emit coherent `last_run` paths.
- `test_llm_generator_live_runs_multi_step_chain_to_completion` —
  3-step chain, asserts the LLM pipeline reaches the executor and
  produces ≥1 successful tool call. Specific completion rates are
  LLM-quality matters, not protocol regressions; the deterministic
  coordinator tests already prove the coordinator drives a 3-step
  chain to completion.

Both skip without credentials. Both treat provider exceptions as
"skip, not fail" — provider availability is volatile and CI's job is
catching protocol regressions.

## 14. Evaluation And LLM-As-Judge

The evaluator has two layers:

```text
deterministic structural metrics
-> optional hosted LLM judge
-> JSON metrics artifact
```

Command:

```powershell
kgmle evaluate --input data/outputs/conversations.jsonl --output data/outputs/evaluation_metrics.json
kgmle evaluate --llm-judge --max-llm-judge-records 10
```

The command writes two outputs:

```text
evaluation_metrics.json       aggregate + per-record metrics
evaluation_metrics_scored.jsonl
                              original conversations with metadata.evaluation populated
```

Deterministic metrics are always available and do not require credentials:

- schema validity through the `Conversation` Pydantic model
- role sequence validity
- assistant tool-call to tool-response coverage
- chain completion against expected `metadata.n_tool_calls`
- tool error rate
- aggregate deterministic score

Score ranges:

- rubric scores use a 0-10 scale
- `deterministic_score` and `mean_deterministic_score` use a 0-10 scale
- LLM judge dimensions and `mean_llm_overall_score` use a 0-10 scale
- coverage/rate fields such as `chain_completion`, `tool_response_coverage`,
  and `schema_valid_rate` remain 0-1 ratios because they are fractions, not
  rubric scores

The optional LLM judge uses the same provider-neutral
`StructuredLLMClient` as the generator. The default hosted path is Gemini;
Groq can be used by switching `KG_MLE_LLM_PROVIDER=groq`.

### Evaluation Design Principles

The evaluator follows these design principles:

- **Offline-first:** deterministic metrics run without credentials, network,
  or a hosted model.
- **Human-readable scale:** rubric scores are 0-10 because reviewers and
  hiring panels can interpret them faster than 0-1 decimals.
- **Separation of scores and rates:** quality scores are 0-10, while
  mathematical coverage ratios remain 0-1.
- **Structured judge output:** the LLM must return a Pydantic-validated JSON
  object, not free-form prose.
- **Visible-evidence judging:** the LLM judge is instructed to score only the
  supplied conversation JSON and not infer real API behavior.
- **Failure containment:** provider failures are stored in the record's
  `llm_judge.error` field instead of crashing evaluation.
- **Bounded cost:** `--max-llm-judge-records` limits how many records are sent
  to the hosted judge.
- **Single LLM switch:** `kgmle --use-llm ...` is the global opt-in for
  optional hosted-model behavior. Individual legacy flags such as
  `evaluate --llm-judge` remain supported for targeted runs, but new optional
  LLM behavior should hang off the global switch.
- **Provider portability:** the judge uses the same `StructuredLLMClient` as
  generation, so Gemini, Groq, and OpenAI-compatible providers can be swapped
  through `.env`.

### Evaluation Design Decisions

**Design decision:** Two-layer evaluation: deterministic metrics first,
optional LLM-as-judge second.

**Reason:** Deterministic metrics guarantee that every reviewer can run
evaluation offline. The LLM judge adds qualitative signal when credentials
are available, but it is not required for the pipeline to function.

---

**Design decision:** Use dimensions tied to tool-use training quality rather
than generic chatbot quality.

**Reason:** Training data for tool-use agents must reward correct action
traces, grounded arguments, and faithful final responses. Generic dimensions
such as "helpfulness" or "fluency" would miss invalid tool calls that still
sound natural.

---

**Design decision:** Separate `argument_grounding` from
`response_grounding`.

**Reason:** These are different failure modes. A model can pass an invented
ID into a tool even if the final answer sounds grounded, or it can call tools
correctly and then hallucinate the final response. Keeping them separate
makes filtering and repair more targeted.

---

**Design decision:** Use a 0-10 scale for quality scores while leaving
coverage metrics as 0-1 ratios.

**Reason:** 0-10 is easier for humans to read in review artifacts. Coverage
metrics such as `chain_completion` are mathematical fractions, so converting
them to 0-10 would make them less precise and less conventional.

---

**Design decision:** Store scores in both evaluation artifacts and scored
conversation metadata.

**Reason:** Aggregate metrics are useful for comparing runs, but downstream
training/filtering needs per-conversation scores colocated with the
conversation record. The raw generated JSONL is preserved; `kgmle evaluate`
writes a separate scored JSONL.

---

**Design decision:** Treat hosted-provider failures as record-level judge
errors, not pipeline failures.

**Reason:** Gemini/Groq/HF quotas and availability are external to the
project. A provider error should not invalidate deterministic evaluation or
block the review workflow.

---

**Design decision:** Use one global `--use-llm/--no-use-llm` CLI switch for
optional hosted-model behavior.

**Reason:** The project will add optional LLM behavior in multiple places
(judge, repair planner, and possibly generation polish). A single top-level
switch avoids a sprawl of unrelated flags and makes runs easier to reproduce.
The existing `evaluate --llm-judge` flag remains supported for compatibility,
but `kgmle --use-llm evaluate` is the preferred workflow.

### Judge Dimensions

The judge dimensions are chosen specifically for training tool-use agents:

- `task_completion` (0-10): measures whether the final assistant response
  actually satisfies the user's request. This is the main supervised signal
  for whether the trajectory is worth training on.
- `tool_trace_validity` (0-10): measures whether the assistant selected valid
  tools, called them in a coherent order, and received compatible tool
  responses. Tool-use agents need valid action traces, not just good final
  prose.
- `argument_grounding` (0-10): measures whether tool-call arguments come from
  user input, planned values, or prior tool outputs. This catches fabricated
  IDs and unsupported chained arguments, which are among the most harmful
  errors for tool-learning data.
- `response_grounding` (0-10): measures whether the assistant's final answer
  stays faithful to the tool outputs. This prevents training examples where
  the model calls tools correctly but then hallucinates the answer.
- `naturalness` (0-10): measures whether the dialogue reads like a plausible
  user-assistant interaction. This matters because the dataset is intended
  for conversational agents, not only API-call planners.
- `overall_score` (0-10): holistic score used for filtering and comparing
  runs. It is not required to be a simple average; severe tool or grounding
  failures can dominate.
- `confidence` (0-10): judge confidence in its own score, used to identify
  records that may need deterministic review or re-judging.

The dimensions intentionally separate *trace correctness* from *answer
quality*. A conversation can have natural wording but a bad tool trace, or
a valid tool trace but an ungrounded final answer; both cases should be
visible to the filtering logic.

Design decision:

- Make deterministic evaluation the default.
- Make LLM-as-judge opt-in through global `--use-llm`; keep
  `evaluate --llm-judge` as a compatibility alias.
- Validate judge output with Pydantic before it is accepted.

Reason:

- Reviewers can run evaluation offline.
- LLM judging demonstrates rubric-style qualitative assessment without
  making the project fail when hosted inference is unavailable.
- Structured judge output keeps scores machine-readable and prevents
  free-text-only evaluation from becoming hard to aggregate.

Judge schema:

```text
task_completion     0-10
tool_trace_validity 0-10
argument_grounding  0-10
response_grounding  0-10
naturalness         0-10
overall_score       0-10
confidence          0-10
issues
rationale
```

The judge prompt explicitly tells the model to score only visible JSON,
penalize missing role tags, missing tool responses, invented tool IDs,
unresolved tool errors, and ungrounded arguments.

Each scored conversation stores the evaluation under:

```text
metadata.evaluation
```

This includes deterministic metrics, the 0-10 deterministic score, and the
optional LLM judge result or provider error. The original generated input
JSONL is not overwritten; `kgmle evaluate` writes a separate scored JSONL so
reviewers can compare raw vs evaluated artifacts.

## 15. Retry And Repair

The assignment requires the system to attempt repair when a generated
conversation fails validation or scores below a quality threshold. Repair is
implemented as an evaluation-time bounded pass:

```text
conversation
-> evaluate
-> detect validation/quality triggers
-> create RepairPlan
-> apply safe deterministic repair or mark coordinator-required repair
-> re-evaluate once
-> write repair history into metadata and metrics
```

Command:

```powershell
kgmle evaluate --repair --repair-threshold 8.0 --max-repair-attempts 1
```

Repair module:

```text
src/kg_mle/repair/
├── models.py   # RepairTrigger, RepairPlan, RepairResult
├── policy.py   # thresholds, trigger detection, quality bands
└── planner.py  # deterministic + LLM repair planners, safe local repairs
```

### Repair Triggers

Repair is attempted when any of the following are observed:

- schema validation failure
- role/message validation failure
- tool error in the trace
- `deterministic_score < 8.0`
- `tool_trace_validity < 8.0`
- `argument_grounding < 8.0`
- `task_completion < 9.0`
- `naturalness < 5.0`

The default repair budget is one attempt per conversation.

### Repair Plans

Repair returns a structured plan rather than free-form instructions:

```text
RepairPlan
├── strategy
├── triggers
├── reason
├── target_step / target_endpoint
├── proposed_arguments
├── proposed_final_response
├── proposed_chain_change
├── requires_coordinator
└── confidence
```

Current strategies:

- `rewrite_final_response`
- `apply_graph_verified_chain_change`
- `regenerate_conversation`
- `mark_rejected`
- `fix_tool_arguments`
- `insert_clarification`

The deterministic planner applies only local safe repairs in the evaluator:
currently this means final-response rewrites grounded in existing tool
outputs. Repairs that need the coordinator, executor state, or graph mutation
are planned and recorded as coordinator-required. This keeps the submitted
pipeline honest: it attempts repair, records the required action, and does
not pretend to re-run a stateful conversation when the required state is not
available in the evaluation artifact.

When the global LLM switch is enabled, the evaluator uses an LLM repair
planner:

```powershell
kgmle --use-llm evaluate --repair
```

The LLM planner still returns the same Pydantic `RepairPlan` schema. If the
model returns malformed JSON, an invalid strategy, or out-of-range confidence,
the planner falls back to the deterministic planner. Applying the plan still
goes through the deterministic repair application layer so hosted-model output
does not directly mutate conversations.

### Repair Metadata

Every scored conversation includes repair information when a repair was
attempted:

```text
metadata.repair_history
metadata.evaluation.repair
metadata.evaluation.quality_band
metadata.evaluation.usable_for_training
```

The metrics JSON also includes:

```text
repair_summary
├── enabled
├── attempted
├── repaired
├── failed
├── rejected
├── regenerated
└── status_counts
```

Quality bands:

- `gold`: deterministic score >= 9, no tool errors, and no major LLM judge
  issues when judge scores are present.
- `silver`: deterministic score >= 8 and no unresolved tool errors.
- `reject`: validation failure, unresolved tool error, deterministic score <
  8, or major tool/grounding/task-completion judge failure.

### Repair Design Decisions

**Design decision:** Use a hybrid repair model.

**Reason:** Some issues are best repaired inline during generation, where the
coordinator has executor state. Others are discovered only after evaluation
or LLM judging. The current implementation provides the post-generation pass
and records coordinator-required repairs; future work can hand those plans
back into the coordinator for full regeneration.

---

**Design decision:** Repair module owns policy and planning; coordinator owns
stateful application.

**Reason:** The repair module should decide what needs to change. The
coordinator/executor should apply changes that require conversation state,
tool outputs, or graph validation. This separation prevents the evaluator
from duplicating generation logic.

---

**Design decision:** Fixed repair budget of one attempt.

**Reason:** One repair attempt demonstrates the required retry/repair loop
while bounding hosted-model cost, latency, and nondeterminism.

---

**Design decision:** Deterministic repair first, optional LLM repair planner
behind global `--use-llm`.

**Reason:** The project is offline-first. Deterministic repair is testable
without credentials. LLM repair uses the same provider-neutral client and the
same `RepairPlan` schema, so it can improve planning quality without bypassing
validation or deterministic application guardrails.

---

**Design decision:** Chain-changing repairs are allowed only when
graph/coordinator validation can prove them.

**Reason:** Low scores can mean the sampled chain itself is bad, but changing
chains without graph validation would create new hallucination risk. The
planner records graph-verified-chain-change as the appropriate strategy,
but the evaluator does not mutate chains directly.

## 16. Context Management

Context is managed at two levels:

```text
within-conversation context
cross-conversation corpus context
```

The two serve different purposes. Within-conversation context keeps one tool
trace grounded. Cross-conversation context makes the generated corpus diverse.

### Within-Conversation Context

Within one conversation, `ExecutorSession` and `SessionState` are the source
of truth.

They track:

- successful tool calls
- failed tool calls
- tool responses
- issued IDs
- issued string fields
- canonical field aliases
- chronological tool log

The generator accesses this state through narrow APIs:

```python
session.suggest_arguments(endpoint_id)
session.example_values(endpoint_id)
session.call(endpoint_id, arguments)
```

Design decision:

- Keep runtime context in executor state, not only in prompt text.

Reason:

- Grounding must be machine-checkable. A prompt can suggest prior values, but
  the executor state can prove whether a value was actually returned by an
  earlier tool.

### Grounding And Canonical Aliases

When a tool returns a string field, the session registers it under both the
literal field name and the canonical name when present.

Example:

```text
food/check_availability returns available_time
available_time canonical_name = start_time
events/create_calendar_event requires start_time
```

The session stores the returned value under:

```text
available_time
start_time
```

This lets a later `start_time` parameter use the previous
`available_time` output without inventing a new value.

Design decision:

- Strict grounding applies to all grounded parameters, not only ID-shaped
  parameters.

Reason:

- ToolBench-style chains can ground non-ID values such as `symbol`, `city`,
  `date`, and `start_time`. If the assistant invents one of those values
  instead of using the previous output, the trace should fail.

### LLM Prompt Context

LLM agents receive compact structured context:

- plan
- transcript so far
- current step
- suggested arguments
- example values
- relevant thresholds

Design decision:

- Pass structured context instead of large unbounded memory dumps.

Reason:

- It reduces hallucination surface area, lowers token cost, and makes LLM
  outputs easier to validate through Pydantic.

### Cross-Conversation Context

Cross-conversation context is corpus-level statistical memory, implemented by
`CorpusSteerer` and `CorpusCounters`.

It tracks:

- domain counts
- tool counts
- endpoint counts
- endpoint-pair counts
- chain length counts
- domain-pattern counts

Flow:

```text
sample conversation N
-> record domains/tools/endpoints/pairs
-> update corpus counters
-> shape constraints for conversation N+1
```

This context answers:

```text
What kind of chain should we sample next to improve corpus diversity?
```

It does not answer:

```text
What argument value should this tool call use?
```

That second question belongs to within-conversation executor state.

Design decision:

- Use statistical counters, not long-term semantic memory, for
  cross-conversation steering.

Reason:

- The assignment needs diversity control and a steering-on/off experiment.
  Counters are deterministic, inspectable, cheap, and directly tied to the
  diversity metrics. Persistent user memory would add privacy and complexity
  without improving the required tool-use dataset.

### Steering And Relaxation

The corpus planner can use cross-conversation counters to:

- prefer underused domains
- forbid overused endpoints
- diagnose overused endpoint pairs
- vary chain length and domain patterns

If steering over-constrains sampling, the planner relaxes constraints rather
than dropping the record.

Relaxation order:

1. reduce grounded-transition requirement
2. remove required domains
3. reduce distinct-domain requirement
4. stop forbidding overused endpoints
5. reduce distinct-tool requirement
6. reduce chain length only as a last resort

Design decision:

- Relax steering before failing generation.

Reason:

- A valid slightly less-diverse conversation is better than silently losing a
  record, especially in a fixed-size corpus.

### Mem0 And Semantic Context

Mem0 is used only for optional semantic graph expansion, not as runtime user
memory.

Endpoint cards are added with `infer=False`.

Design decision:

- Do not let Mem0's LLM memory extraction rewrite endpoint cards.

Reason:

- Endpoint cards are already structured schema facts. LLM rewriting would add
  cost, latency, nondeterminism, and the risk of dropping parameter names.

### Repair Context

Inline repair during generation can use live executor context. Post-generation
repair has only the serialized conversation, metrics, and tool outputs.

Design decision:

- Post-generation repair applies only safe local edits directly. Repairs that
  need executor state, graph mutation, or chain regeneration are planned and
  recorded as coordinator-required.

Reason:

- The evaluator should not pretend it can safely mutate a stateful trace after
  the session has ended.

### Context Management Limitations And Scale Tradeoffs

Limitations:

- Cross-conversation steering is counter-based, so it cannot understand deep
  semantic novelty by itself.
- Endpoint-pair overuse is recorded and surfaced for diagnostics, but the
  current walker does not yet support direct pair-level forbidding.
- Within-conversation context is per-session only; no user preferences persist
  across conversations.
- Post-generation repair cannot reconstruct full executor state unless the
  coordinator is rerun.

Scale tradeoffs:

- In-memory session state is simple and fast for the curated subset and 100+
  sample runs.
- Corpus counters are O(number of generated chains) and cheap to serialize.
- Full ToolBench scale may require indexed graph traversal, cached embeddings,
  and possibly a graph/vector store for artifact exploration.
- We avoid external databases in the submitted MVP so reviewers can run the
  pipeline without service setup.

### Context Tests

Implemented tests cover:

- strict grounding rejects hallucinated grounded values
- previous tool outputs validate when reused
- canonical response aliases ground later parameters
- `example_values()` exposes issued prior outputs to LLM agents
- steering counters track domains, tools, endpoints, endpoint pairs, chain
  lengths, and domain patterns
- steering on/off produces different corpora while both runs retain
  comparable counters

## 17. Open Design Areas

Still to implement:
- Diversity experiment
- End-to-end test that generates at least 100 samples
