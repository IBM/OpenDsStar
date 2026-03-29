"""
Backwards compatibility module - FlexibleAgentBuilder has been renamed to AgentFactory.

This module re-exports AgentFactory as FlexibleAgentBuilder for backwards compatibility.
New code should import AgentFactory directly from agent_factory.
"""

from .agent_factory import AgentFactory, AgentType

# Backwards compatibility - FlexibleAgentBuilder is now AgentFactory
FlexibleAgentBuilder = AgentFactory

__all__ = ["FlexibleAgentBuilder", "AgentFactory", "AgentType"]
