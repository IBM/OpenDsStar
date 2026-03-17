"""Simple tool builder implementation."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ...core.context import PipelineContext
from ...core.types import BenchmarkEntry
from ...interfaces.tool_builder import ToolBuilder

logger = logging.getLogger(__name__)


class EchoToolInput(BaseModel):
    """Input schema for echo tool."""

    text: str = Field(..., description="The text to echo back")


class EchoTool(BaseTool):
    """Simple echo tool for testing."""

    name: str = "echo"
    description: str = (
        "Echo back the input text. Use this tool to test basic functionality."
    )
    args_schema: type[EchoToolInput] = EchoToolInput

    def _run(self, text: str) -> str:
        """Echo the input text."""
        return text


class EchoToolBuilder(ToolBuilder):
    """
    Simple tool builder that creates an echo tool.

    This is a placeholder implementation. Replace with your actual
    tool building logic (e.g., retrievers, calculators, etc.).
    """

    def build_tools(
        self,
        ctx: PipelineContext,
        benchmarks: Sequence[BenchmarkEntry],
        corpus: Any = None,
    ) -> Sequence[BaseTool]:
        """
        Build tools from benchmark data.

        Args:
            ctx: Pipeline context
            benchmarks: Raw benchmark entries
            corpus: Optional corpus data (not used by this simple tool)

        Returns:
            List of LangChain BaseTool instances
        """
        logger.info("building_echo_tool")
        return [EchoTool()]
