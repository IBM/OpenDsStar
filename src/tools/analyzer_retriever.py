"""
Analyzer summary retriever tool.

This tool retrieves summaries created by the analyzer agent and provides
document file paths by document name.
"""

import logging
from typing import Any, Dict

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AnalyzerRetrievalInput(BaseModel):
    """Input schema for analyzer retrieval tool."""

    document_name: str = Field(
        ..., description="The name of the document to retrieve the file path for"
    )


class AnalyzerSummaryRetrievalTool(BaseTool):
    """
    Tool for retrieving document file paths by document name.

    The tool is initialized with pre-computed analyzer results (vector DB and summaries).
    The description can be dynamically updated based on different queries without
    re-running the analysis. When invoked, it returns the file path of the requested document.
    """

    name: str = "get_analyzed_document"
    description: str = "Retrieve the file path for a document by name"
    args_schema: type[AnalyzerRetrievalInput] = AnalyzerRetrievalInput

    # Internal attributes - stores the full analysis results and vector DB
    analysis_results: Dict[str, Dict[str, Any]] = {}
    vector_db: Any = None
    top_k: int = 100

    def __init__(
        self,
        vector_db: Any,
        analysis_results: Dict[str, Dict[str, Any]],
        top_k: int = 100,
    ):
        """
        Initialize the analyzer retrieval tool with pre-computed results.

        Args:
            vector_db: Pre-computed vector database with document summaries
            analysis_results: Dictionary mapping document names to full analysis results
            query: Initial query to find relevant summaries for tool description
            top_k: Number of top summaries to include in description
        """
        super().__init__()

        self.vector_db = vector_db
        self.analysis_results = analysis_results
        self.top_k = top_k

    def update_description(self, query: str) -> None:
        """
        Update the tool description based on a new query.

        This allows dynamically changing what summaries are shown in the tool
        description without re-analyzing the documents.

        Args:
            query: Search query to find relevant summaries
            k: Number of summaries to include in description
        """

        # Retrieve top_k summaries based on query
        results = self.vector_db.similarity_search(query, self.top_k)

        # Build description with summaries embedded (deduplicated)
        summary_sections = []
        seen_docs = set()

        for res in results:
            doc_name = res.metadata["doc_id"]

            # Skip if we've already seen this document
            if doc_name in seen_docs:
                continue

            seen_docs.add(doc_name)
            summary = res.page_content

            summary_sections.append(f"**Document: {doc_name}**\n{summary}")

        summaries_text = "\n\n---\n\n".join(summary_sections)

        self.description = (
            f"Retrieve the file path for a document by providing its name.\n\n"
            f"Document Summaries:\n\n{summaries_text}\n\n"
            f"Use this tool to get the file path of a document so you can process or analyze it."
        )

        logger.info(
            f"Updated tool description with {len(summary_sections)} summaries for query: '{query}'"
        )

    def _run(self, document_name: str) -> str:
        """
        Retrieve the file path for a given document.

        Args:
            document_name: Name of the document to retrieve

        Returns:
            String containing the file path of the document
        """
        if document_name not in self.analysis_results:
            available = ", ".join(self.analysis_results.keys())
            return (
                f"Error: Document '{document_name}' not found.\n\n"
                f"Available documents: {available}"
            )

        result = self.analysis_results[document_name]
        file_path = result.get("file_path")

        if not file_path:
            return f"Error: File path not found for document '{document_name}'"

        logger.info(f"Retrieved file path for document: {document_name} -> {file_path}")
        return file_path

    async def _arun(self, document_name: str) -> str:
        """Async version of _run."""
        return self._run(document_name)
