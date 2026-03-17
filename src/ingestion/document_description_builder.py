"""Interface for building document descriptions and storing them in vector stores."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

from docling_core.types.io import DocumentStream
from langchain_milvus.vectorstores.milvus import Milvus


class DocumentDescriptionBuilder(ABC):
    """
    Interface for building document descriptions and storing them in vector stores.

    Implementations convert documents (from directory or corpus) into:
    1. A Milvus vector store with document descriptions/summaries
    2. A map from filename to analysis results
    3. A map from filename to DocumentStream objects
    """

    @abstractmethod
    def process_directory(
        self, dir_path: Path
    ) -> tuple[Milvus, dict[str, Dict[str, Any]], dict[str, DocumentStream]]:
        """
        Process all files in a directory.

        Args:
            dir_path: Directory containing files to process

        Returns:
            Tuple of:
            - vector_db: Milvus vector store with document summaries
            - analysis_results: Dict mapping filename to analysis results
            - document_streams: Dict mapping filename to DocumentStream objects
        """
        raise NotImplementedError

    @abstractmethod
    def process_corpus(
        self, corpus: Any
    ) -> tuple[Milvus, dict[str, Dict[str, Any]], dict[str, DocumentStream]]:
        """
        Process a corpus of documents.

        Args:
            corpus: Corpus data (format depends on implementation)

        Returns:
            Tuple of:
            - vector_db: Milvus vector store with document summaries
            - analysis_results: Dict mapping filename to analysis results
            - document_streams: Dict mapping filename to DocumentStream objects
        """
        raise NotImplementedError
