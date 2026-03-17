"""KramaBench tools builder implementation."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from langchain.tools import BaseTool

from ingestion.docling_based_ingestion.docling_description_builder import (
    DoclingDescriptionBuilder,
)

from ...core.context import PipelineContext
from ...core.types import BenchmarkEntry
from ...interfaces.tool_builder import ToolBuilder
from ..shared_tools import build_file_tools

logger = logging.getLogger(__name__)


class KramaBenchToolsBuilder(ToolBuilder):
    """
    Build retrieval and file-loading tools for the KramaBench corpus.

    Tools produced:
    1. search_files
    2. get_file_info
    3. get_file_content
    4. load_dataframe
    """

    def __init__(
        self,
        cache_dir: str,
        llm: str,
        embedding_model: str,
        temperature: float = 0.0,
        batch_size: int = 8,
    ):
        """
        Initialize the tools builder.

        Args:
            cache_dir: Directory for caching analysis results
            llm: Model for file description generation
            embedding_model: Embedding model
            temperature: Temperature for generation (default: 0.0)
            batch_size: Batch size for processing (default: 8)
        """
        self.cache_dir = cache_dir
        self.llm = llm
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        """Name of this tool builder."""
        return "kramabench_tools"

    def _log_analysis_stats(self, analysis_results: dict[str, dict[str, Any]]) -> None:
        """
        Log summary statistics for corpus analysis results.

        Args:
            analysis_results: Mapping from doc_id to analysis result metadata
        """
        total = len(analysis_results)
        successes = sum(
            1 for result in analysis_results.values() if result.get("success", False)
        )
        failures = total - successes

        unique_errors: set[str] = set()
        for result in analysis_results.values():
            if result.get("success", False):
                continue
            error_msg = str(result.get("fatal_error", "Unknown error"))
            unique_errors.add(error_msg[:100])

        error_samples = list(unique_errors)[:3]
        error_sample_str = (
            "; ".join(f"'{sample}...'" for sample in error_samples)
            if error_samples
            else "N/A"
        )

        logger.info(
            "Analysis results: total=%d, success=%d, failed=%d, unique_errors=%d (samples: %s)",
            total,
            successes,
            failures,
            len(unique_errors),
            error_sample_str,
        )

    def build_tools(
        self,
        ctx: PipelineContext,
        benchmarks: Sequence[BenchmarkEntry],
        corpus: Any = None,
    ) -> Sequence[BaseTool]:
        """
        Build KramaBench tools from the corpus.

        Args:
            ctx: Pipeline context
            benchmarks: Benchmark entries (unused here; tools are corpus-based)
            corpus: Corpus from the data reader

        Returns:
            Sequence of LangChain BaseTool instances ready for agent use
        """
        del ctx
        del benchmarks

        logger.info(
            "Building KramaBench tools; corpus_available=%s", corpus is not None
        )

        if corpus is None:
            logger.warning("No corpus available; returning no tools")
            return []

        logger.info(
            "Initializing DoclingDescriptionBuilder: llm=%s, embedding_model=%s",
            self.llm,
            self.embedding_model,
        )
        builder = DoclingDescriptionBuilder(
            cache_dir=self.cache_dir,
            model=self.llm,
            temperature=self.temperature,
            embedding_model=self.embedding_model,
            batch_size=self.batch_size,
            enable_caching=True,
        )

        logger.info("Processing %d documents...", len(corpus))
        vector_db, analysis_results, path_to_bytes_factory = builder.process_corpus(
            corpus
        )

        self._log_analysis_stats(analysis_results)

        tools = build_file_tools(
            vector_db=vector_db,
            path_to_bytes_factory=path_to_bytes_factory,
        )

        logger.info(
            "KramaBench tools built: tool_count=%d, tool_names=%s",
            len(tools),
            [tool.name for tool in tools],
        )

        return tools
