"""ExecutorSession — the stateful, per-conversation executor surface.

This is what the multi-agent generator drives. It looks like a session:
open it from a `SamplingResult` plus a seed, then call `session.call(...)`
once per tool invocation. Between calls, the generator can inspect
`session.state` (issued IDs, last response) and `session.example_values`
(few-shot pool for free parameters) to compose the next call.

The session also exposes `suggest_arguments(endpoint_id)` for the
"default plausible arguments" path. The generator can use that directly,
or override the values it cares about — both routes go through the
same validator, so there is no privileged caller.

Lifecycle:

```
session = OfflineExecutor(registry).open_session(sampling_result, seed=...)
for endpoint_id in sampling_result.endpoints:
    args = generator.compose_args(session, endpoint_id)  # may use suggest/examples
    response = session.call(endpoint_id, args)            # raises on invalid
session.state.as_log()  # serializable conversation trace
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kg_mle.executor.mocks import CANONICAL_EXAMPLES, MockResponseGenerator, is_id_parameter
from kg_mle.executor.state import SessionState
from kg_mle.executor.validator import (
    ExecutorError,
    UngroundedArgumentError,
    ValidationContext,
    grounded_parameters_for_endpoint,
    validate_arguments,
)
from kg_mle.registry.models import Endpoint, ToolRegistry
from kg_mle.sampler.constraints import SamplingResult


@dataclass
class ExecutorSession:
    """A live session for one conversation."""

    registry: ToolRegistry
    sampling_result: SamplingResult
    state: SessionState
    mocks: MockResponseGenerator
    strict_extras: bool = False

    _endpoint_index: dict[str, Endpoint] | None = None

    def __post_init__(self) -> None:
        self._endpoint_index = {endpoint.endpoint_id: endpoint for endpoint in self.registry.endpoints}

    @property
    def endpoint_index(self) -> dict[str, Endpoint]:
        assert self._endpoint_index is not None
        return self._endpoint_index

    def call(self, endpoint_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate then mock. Raises on any violation, logging it first."""
        endpoint = self._require_endpoint(endpoint_id)
        context = ValidationContext(
            grounded_parameters=grounded_parameters_for_endpoint(
                endpoint_id, self.sampling_result.transitions
            ),
            strict_extras=self.strict_extras,
        )

        self.state.record_call(endpoint_id, arguments)
        try:
            validate_arguments(endpoint, arguments, self.state, context)
        except ExecutorError as exc:
            self.state.record_error(endpoint_id, arguments, exc.to_log_entry())
            raise

        response = self.mocks.generate(endpoint, self.state)
        issued_field_values = self._collect_issued(endpoint, response)
        self.state.record_response(
            endpoint_id,
            response,
            issued_field_values=issued_field_values,
        )
        return response

    def suggest_arguments(self, endpoint_id: str) -> dict[str, Any]:
        """Return a plausible default-argument dict for one endpoint.

        For grounded parameters, the suggestion uses the most-recently
        issued ID from session state — guaranteed to validate. For free
        parameters (no incoming grounded transition), the suggestion
        draws from `CANONICAL_EXAMPLES` when the canonical_name is known,
        and uses a deterministic-typed placeholder otherwise.

        The generator may override any value; the override goes through
        the same validator on the next `call(...)`.
        """
        endpoint = self._require_endpoint(endpoint_id)
        grounded = grounded_parameters_for_endpoint(endpoint_id, self.sampling_result.transitions)
        out: dict[str, Any] = {}
        for param in endpoint.parameters:
            if not param.required:
                continue
            if param.name in grounded:
                issued = self.state.issued_ids(grounded[param.name]) or self.state.issued_ids(param.name)
                if issued:
                    out[param.name] = issued[-1]
                    continue
            out[param.name] = self._default_value_for(param.name, param)
        return out

    def example_values(self, endpoint_id: str) -> dict[str, list[str]]:
        """Few-shot pool the generator can show the LLM.

        For grounded parameters, returns the IDs the previous step
        actually issued (so the LLM picks from real options, not made-up
        ones). For free parameters with a known canonical_name, returns
        the canonical example pool. Empty list when there is nothing
        useful to show.
        """
        endpoint = self._require_endpoint(endpoint_id)
        grounded = grounded_parameters_for_endpoint(endpoint_id, self.sampling_result.transitions)
        out: dict[str, list[str]] = {}
        for param in endpoint.parameters:
            if param.name in grounded:
                source = grounded[param.name]
                issued = self.state.issued_ids(source) or self.state.issued_ids(param.name)
                out[param.name] = list(issued)
                continue
            canonical = param.canonical_name or param.name
            pool = CANONICAL_EXAMPLES.get(canonical)
            out[param.name] = list(pool) if pool else []
        return out

    # --- internals ------------------------------------------------------

    def _require_endpoint(self, endpoint_id: str) -> Endpoint:
        endpoint = self.endpoint_index.get(endpoint_id)
        if endpoint is None:
            raise UngroundedArgumentError(
                f"Endpoint {endpoint_id!r} is not in the registry.",
                endpoint=endpoint_id,
            )
        return endpoint

    @staticmethod
    def _collect_issued(endpoint: Endpoint, response: dict[str, Any]) -> dict[str, list[str]]:
        """Pull string response values out so the session can index them under
        both literal and canonical names.

        We register every string field (not just IDs), because the sampler's
        grounded transitions are tracked at the field-name / canonical-name
        level — `finance/search_symbol -> finance/get_quote` is grounded via
        `symbol`, not via an ID. Strict grounding (design decision #4)
        therefore requires that any grounded parameter value, ID or not,
        is traceable back to a previous response.
        """
        issued: dict[str, list[str]] = {}
        for field in endpoint.response_fields:
            value = response.get(field.name)
            if not isinstance(value, str):
                continue
            issued.setdefault(field.name, []).append(value)
            if field.canonical_name and field.canonical_name != field.name:
                issued.setdefault(field.canonical_name, []).append(value)
        return issued

    def _default_value_for(self, param_name: str, param: Any) -> Any:
        canonical = param.canonical_name or param_name
        pool = CANONICAL_EXAMPLES.get(canonical)
        if pool:
            return pool[0]
        if param.type == "string":
            return f"{param_name}_value"
        if param.type == "integer":
            return 1
        if param.type == "number":
            return 1.0
        if param.type == "boolean":
            return True
        if param.type == "array":
            return []
        if param.type == "object":
            return {}
        return ""


class OfflineExecutor:
    """Top-level executor; one per pipeline run, owns the registry handle.

    `open_session` produces a fresh `ExecutorSession` for each
    conversation, so per-conversation state never leaks.
    """

    def __init__(self, registry: ToolRegistry, *, strict_extras: bool = False) -> None:
        self._registry = registry
        self._strict_extras = strict_extras

    def open_session(self, sampling_result: SamplingResult, *, seed: int) -> ExecutorSession:
        return ExecutorSession(
            registry=self._registry,
            sampling_result=sampling_result,
            state=SessionState(),
            mocks=MockResponseGenerator(seed=seed),
            strict_extras=self._strict_extras,
        )
