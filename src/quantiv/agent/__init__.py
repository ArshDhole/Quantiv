"""Agent package: typed tools + providers + bounded orchestrator."""

from quantiv.agent.orchestrator import AgentResult, agent_result_to_dict, run_agent
from quantiv.agent.providers import EchoProvider, OpenAICompatProvider, Provider, ProviderDecision
from quantiv.agent.tools import TOOLS, ToolResult, call_tool

__all__ = [
    "TOOLS",
    "ToolResult",
    "call_tool",
    "Provider",
    "ProviderDecision",
    "EchoProvider",
    "OpenAICompatProvider",
    "AgentResult",
    "run_agent",
    "agent_result_to_dict",
]
