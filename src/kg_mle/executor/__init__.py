"""Offline tool execution model.

The executor produces mock responses consistent with endpoint schemas and
maintains per-conversation session state so multi-step tool chains use
real previous outputs rather than hallucinated placeholders.
"""

from kg_mle.executor.mocks import (
    CANONICAL_EXAMPLES,
    ID_PREFIXES,
    MockResponseGenerator,
)
from kg_mle.executor.session import ExecutorSession, OfflineExecutor
from kg_mle.executor.state import LogEntry, SessionState
from kg_mle.executor.validator import (
    ArgumentTypeError,
    ExecutorError,
    MissingRequiredArgumentError,
    UngroundedArgumentError,
    UnknownArgumentError,
    ValidationContext,
    grounded_parameters_for_endpoint,
    validate_arguments,
)


__all__ = [
    "ArgumentTypeError",
    "CANONICAL_EXAMPLES",
    "ExecutorError",
    "ExecutorSession",
    "ID_PREFIXES",
    "LogEntry",
    "MissingRequiredArgumentError",
    "MockResponseGenerator",
    "OfflineExecutor",
    "SessionState",
    "UngroundedArgumentError",
    "UnknownArgumentError",
    "ValidationContext",
    "grounded_parameters_for_endpoint",
    "validate_arguments",
]
