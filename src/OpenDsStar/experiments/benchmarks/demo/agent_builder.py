"""
Demo agent builder - simple placeholder implementation.

This module provides SimpleAgent and SimpleAgentBuilder for demo experiments.
"""

from __future__ import annotations

# Keep SimpleAgent and SimpleAgentBuilder for testing/examples
import logging
from typing import Any, Sequence

from langchain_core.tools import BaseTool

from ...core.context import PipelineContext
from ...interfaces.agent_builder import AgentBuilder

logger = logging.getLogger(__name__)


class SimpleAgent:
    """
    Simple agent that just returns the prompt as answer.

    This is a placeholder. Replace with your actual agent
    (e.g., LangChain agent, LangGraph, custom LLM wrapper, etc.).
    """

    def __init__(self, tools: Sequence[BaseTool]) -> None:
        """
        Initialize agent with tools.

        Args:
            tools: LangChain BaseTool instances available to the agent
        """
        self.tools = list(tools)
        self.tool_names = [t.name for t in tools]


class SimpleAgentBuilder(AgentBuilder):
    """
    Simple agent builder.

    This is a placeholder implementation. Replace with your actual
    agent building logic (e.g., LangChain, LangGraph, etc.).
    """

    def build_agent(
        self,
        ctx: PipelineContext,
        tools: Sequence[BaseTool],
    ) -> Any:
        """
        Build an agent with the given tools.

        Args:
            ctx: Pipeline context
            tools: LangChain BaseTool instances to provide to the agent

        Returns:
            Configured agent
        """
        logger.info(f"building_agent tool_count={len(tools)}")
        return SimpleAgent(tools)
