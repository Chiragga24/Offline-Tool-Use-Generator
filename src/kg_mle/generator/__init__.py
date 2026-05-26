"""Multi-agent conversation generator.

The generator turns a sampled tool chain into a role-tagged conversation
with user requests, assistant clarifying questions, tool calls, tool
responses, and a final summary. Both deterministic and LLM-backed agents
are available behind the same protocols; the coordinator drives them
identically.
"""

from kg_mle.generator.agents import (
    Assistant,
    DeterministicAssistant,
    DeterministicPlanner,
    DeterministicUser,
    Planner,
    UserSimulator,
)
from kg_mle.generator.coordinator import ConversationCoordinator
from kg_mle.generator.llm_agents import (
    LLMAssistant,
    LLMPlanner,
    LLMUser,
    StructuredLLMClient,
    make_llm_agents,
)
from kg_mle.generator.protocol import (
    AssistantTurn,
    ChainDeviation,
    ClarificationTarget,
    Conversation,
    GeneratorConfig,
    ParameterPlan,
    Plan,
    StepPlan,
    ToolCallProposal,
    UserTurn,
)


__all__ = [
    "Assistant",
    "AssistantTurn",
    "ChainDeviation",
    "ClarificationTarget",
    "Conversation",
    "ConversationCoordinator",
    "DeterministicAssistant",
    "DeterministicPlanner",
    "DeterministicUser",
    "GeneratorConfig",
    "LLMAssistant",
    "LLMPlanner",
    "LLMUser",
    "ParameterPlan",
    "Plan",
    "Planner",
    "StepPlan",
    "StructuredLLMClient",
    "ToolCallProposal",
    "UserSimulator",
    "UserTurn",
    "make_llm_agents",
]
