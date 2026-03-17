"""Interface for building tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from langchain_core.tools import BaseTool

from ..core.context import PipelineContext
from ..core.types import BenchmarkEntry


class ToolBuilder(ABC):
    """Interface for building tools from benchmark data and corpus."""

    @property
    def name(self) -> str:
        """Name of this tool builder (used for namespacing)."""
        return self.__class__.__name__

    @abstractmethod
    def build_tools(
        self,
        ctx: PipelineContext,
        benchmarks: Sequence[BenchmarkEntry],
        corpus: Any = None,
    ) -> Sequence[BaseTool]:
        """
        Build tools from benchmark data and optional corpus.

        Args:
            ctx: Pipeline context with config and logger
            benchmarks: Raw benchmark entries
            corpus: Optional corpus data from DataReader.get_data()
                   (can be ignored if not needed by the tool builder)

        Returns:
            Sequence of LangChain BaseTool instances ready for agent use
        """
        raise NotImplementedError
