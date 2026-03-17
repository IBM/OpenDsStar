"""HotpotQA tools builder implementation."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Sequence

from langchain_core.tools import BaseTool

from core.model_registry import ModelRegistry
from tools import VectorStoreTool

from ...core.context import PipelineContext
from ...core.types import BenchmarkEntry
from ...interfaces.tool_builder import ToolBuilder

logger = logging.getLogger(__name__)


class HotpotQAToolsBuilder(ToolBuilder):
    """
    Builds retrieval tools from HotpotQA corpus.

    This builder creates a retriever tool that can search the HotpotQA corpus
    to find relevant documents for answering questions.
    """

    def __init__(
        self,
        tool_model: str = ModelRegistry.WX_MISTRAL_MEDIUM,
        embedding_model: str = ModelRegistry.GRANITE_EMBEDDING,
        temperature: float = 0.0,
        batch_size: int = 8,
    ):
        """
        Initialize the tools builder.

        Args:
            tool_model: Model for tool operations (default: WX_MISTRAL_MEDIUM)
            embedding_model: Embedding model (default: GRANITE_EMBEDDING)
            temperature: Temperature for generation (default: 0.0)
            batch_size: Batch size for processing (default: 8)
        """
        self.tool_model = tool_model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        """Name of this tool builder."""
        return "hotpotqa_tools"

    def build_tools(
        self,
        ctx: PipelineContext,
        benchmarks: Sequence[BenchmarkEntry],
        corpus: Any = None,
    ) -> Sequence[BaseTool]:
        """
        Build retriever tools from HotpotQA corpus.

        Args:
            ctx: Pipeline context with config and logger
            benchmarks: Raw benchmark entries (not used for corpus-based tools)
            corpus: Corpus data from DataReader (Sequence[Document])

        Returns:
            Sequence of LangChain BaseTool instances ready for agent use
        """
        logger.info(f"Building HotpotQA tools, corpus_available={corpus is not None}")

        if corpus is None:
            logger.warning("No corpus available")
            return []

        # Get cache directory from context (will be hotpotqa/cache)
        # Fall back to temp directory if not configured
        if ctx.config.cache_dir:
            cache_dir = ctx.config.cache_dir
        else:
            cache_dir = Path(tempfile.gettempdir()) / "hotpotqa_cache"
            logger.warning(f"No cache_dir configured, using temp: {cache_dir}")

        # Create VectorStoreTool with individual parameters
        retriever_tool = VectorStoreTool(
            corpus=corpus,
            cache_dir=cache_dir,
            name="search_hotpotqa",
            model=self.tool_model,
            temperature=self.temperature,
            embedding_model=self.embedding_model,
            batch_size=self.batch_size,
            chunk_size=1000,
            chunk_overlap=200,
            experiment_name="hotpotqa",
        )

        logger.info(
            f"HotpotQA tools built: tool_count=1, "
            f"tool_names=[{retriever_tool.name}], "
            f"cache_dir={cache_dir}, "
            f"tool_model={self.tool_model}, "
            f"embedding_model={self.embedding_model}"
        )

        return [retriever_tool]
