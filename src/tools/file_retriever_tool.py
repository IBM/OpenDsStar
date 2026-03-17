"""File retriever tool for accessing files from a corpus by filename."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import BinaryIO, Callable, Dict, Sequence

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from experiments.core.types import Document

logger = logging.getLogger(__name__)


class FileRetrieverInput(BaseModel):
    """Input schema for file retriever tool."""

    filename: str = Field(
        ..., description="The name of the file to retrieve from the corpus"
    )


class FileRetrieverTool(BaseTool):
    """
    Tool for retrieving files from a document corpus by filename.

    This tool allows access to files in the corpus by their filename.
    It returns a stream factory that can be called to get a fresh stream
    for reading the file content.
    """

    name: str = "get_file"
    description: str = (
        "Retrieve a file from the corpus by its filename. "
        "Returns a stream factory that can be called to read the file content. "
        "Use this when you need to access and read specific files from the corpus."
    )
    args_schema: type[FileRetrieverInput] = FileRetrieverInput

    # Internal attributes - stores the corpus and lookup maps
    corpus: Sequence[Document] = []
    _filename_map: Dict[str, Document] = {}
    _document_id_map: Dict[str, Document] = {}

    def __init__(self, corpus: Sequence[Document]):
        """
        Initialize the file retriever tool with a corpus.

        Builds lookup maps for efficient O(1) file retrieval by filename
        and document_id.

        Args:
            corpus: Sequence of Document objects to search through
        """
        super().__init__()
        self.corpus = corpus

        # Build lookup maps for O(1) access
        self._filename_map = {}
        self._document_id_map = {}

        for doc in corpus:
            # Map by document_id
            self._document_id_map[doc.document_id] = doc

            # Map by filename (extracted from path)
            filename = Path(doc.path).name
            self._filename_map[filename] = doc

            # Also map by full path for exact matches
            self._filename_map[doc.path] = doc

    def _run(self, filename: str) -> Callable[[], BinaryIO] | None:
        """
        Retrieve a file's stream factory by filename.

        Uses O(1) lookup maps for efficient retrieval.

        Args:
            filename: The name of the file to retrieve

        Returns:
            A stream factory (callable that returns a BinaryIO stream) if found,
            None if the file is not found in the corpus
        """
        logger.info(f"Searching for file: {filename}")

        # Try exact filename match first
        doc = self._filename_map.get(filename)

        # If not found, try document_id match
        if doc is None:
            doc = self._document_id_map.get(filename)

        if doc is not None:
            logger.info(f"Found file: {filename} (document_id: {doc.document_id})")
            return doc.stream_factory

        logger.warning(f"File not found in corpus: {filename}")
        return None

    def update_description(self, query: str) -> None:
        """
        Update the tool description based on a query.

        This method is a placeholder for future functionality where the
        description might be dynamically updated based on the query context.

        Args:
            query: The query string to potentially update the description with
        """
        # Placeholder - does nothing for now
        pass
