"""Per-conversation session state for the offline executor.

The state is the *single source of truth* for what has happened in one
conversation: every successful tool call, every failed tool call, and
every value the mock layer has issued for an ID-shaped field.

This is also the within-conversation grounding API the assignment's
§5.1 requires. The generator asks `state.issued_ids("hotel_id")` to
find what IDs are available before composing a `hotel_id=...` argument,
and the validator uses the same lookup to verify grounded arguments.

The conversation log produced by `as_log()` mixes tool calls and tool
errors in chronological order. A reviewer reading the trace sees, in
the same stream, "here is the call the assistant made" → "here is the
error the API returned" → "here is the repaired call." This is the
visibility design choice we made for the executor: errors are never
silent, they always appear in the dataset's `messages` stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


LogEntryKind = Literal["tool_call", "tool_response", "tool_error"]


@dataclass(frozen=True)
class LogEntry:
    """One entry in the per-conversation tool log.

    Frozen so the generator can hand entries to the dataset serializer
    without worrying about post-hoc mutation.
    """

    kind: LogEntryKind
    endpoint_id: str
    arguments: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


@dataclass
class SessionState:
    """Mutable state for one conversation."""

    _issued: dict[str, list[str]] = field(default_factory=dict)
    """field_name (and canonical_name) -> list of IDs issued, in order."""

    _responses_by_endpoint: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    """endpoint_id -> list of full response dicts, in call order."""

    log: list[LogEntry] = field(default_factory=list)
    """Chronological mixed log of calls, responses, and errors."""

    def record_call(self, endpoint_id: str, arguments: dict[str, Any]) -> None:
        self.log.append(LogEntry(kind="tool_call", endpoint_id=endpoint_id, arguments=dict(arguments)))

    def record_response(
        self,
        endpoint_id: str,
        response: dict[str, Any],
        *,
        issued_field_values: dict[str, list[str]] | None = None,
    ) -> None:
        """Record a successful response and any ID-shaped values it issued.

        `issued_field_values` is keyed by both the literal field name and
        the canonical_name (when the mock layer knows the canonical), so
        validators can look up under either name.
        """
        self._responses_by_endpoint.setdefault(endpoint_id, []).append(dict(response))
        if issued_field_values:
            for field_name, values in issued_field_values.items():
                self._issued.setdefault(field_name, []).extend(values)
        self.log.append(
            LogEntry(kind="tool_response", endpoint_id=endpoint_id, response=dict(response))
        )

    def record_error(
        self,
        endpoint_id: str,
        arguments: dict[str, Any],
        error: dict[str, Any],
    ) -> None:
        """Record a failed call. The exception is raised by the caller; this
        only logs it so the conversation trace shows the failure inline."""
        self.log.append(
            LogEntry(
                kind="tool_error",
                endpoint_id=endpoint_id,
                arguments=dict(arguments),
                error=dict(error),
            )
        )

    def issued_ids(self, field_name: str) -> tuple[str, ...]:
        """All IDs issued for a given field or canonical name, oldest first."""
        return tuple(self._issued.get(field_name, ()))

    def has_issued(self, field_name: str, value: str) -> bool:
        return value in self._issued.get(field_name, ())

    def last_response(self, endpoint_id: str | None = None) -> dict[str, Any] | None:
        if endpoint_id is not None:
            responses = self._responses_by_endpoint.get(endpoint_id, ())
            return dict(responses[-1]) if responses else None
        for entry in reversed(self.log):
            if entry.kind == "tool_response":
                return dict(entry.response or {})
        return None

    def as_log(self) -> list[dict[str, Any]]:
        """Conversation-log shape for dataset serialization.

        Each tool_call collapses with its tool_response or tool_error so the
        dataset format matches the assignment's example record (a single
        `role: assistant` tool_call followed by a single `role: tool` payload).
        """
        out: list[dict[str, Any]] = []
        pending_call: LogEntry | None = None
        for entry in self.log:
            if entry.kind == "tool_call":
                if pending_call is not None:
                    # call without response — surface as orphan
                    out.append(self._call_to_dict(pending_call))
                pending_call = entry
                continue
            if entry.kind == "tool_response":
                if pending_call is not None:
                    out.append(self._call_to_dict(pending_call))
                    pending_call = None
                out.append(
                    {
                        "role": "tool",
                        "endpoint": entry.endpoint_id,
                        "content": entry.response,
                    }
                )
                continue
            if entry.kind == "tool_error":
                if pending_call is not None:
                    out.append(self._call_to_dict(pending_call))
                    pending_call = None
                out.append(
                    {
                        "role": "tool",
                        "endpoint": entry.endpoint_id,
                        "content": {"error": entry.error},
                    }
                )
        if pending_call is not None:
            out.append(self._call_to_dict(pending_call))
        return out

    @staticmethod
    def _call_to_dict(entry: LogEntry) -> dict[str, Any]:
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "endpoint": entry.endpoint_id,
                    "arguments": entry.arguments or {},
                }
            ],
            "content": None,
        }
