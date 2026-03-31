"""Vector store tool for semantic search over document corpus."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Type

from langchain_core.documents import Document as LangChainDocument
from langchain_core.tools import BaseTool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus.vectorstores.milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from tqdm import tqdm

from OpenDsStar.core.milvus_uri import resolve_milvus_uri
from OpenDsStar.core.model_registry import ModelRegistry
from OpenDsStar.experiments.core.types import Document

logger = logging.getLogger(__name__)


class VectorStoreInput(BaseModel):
    """Input schema for vector store search."""

    query: str = Field(..., description="The search query to find relevant documents")
    top_k: int = Field(
        default=5, description="Number of top documents to return (default: 5)"
    )


class VectorStoreTool(BaseTool):
    """
    Vector store tool for semantic search over document corpus.

    Uses Milvus vector database with HuggingFace embeddings for retrieval.
    Can be used across different experiments for document retrieval.
    """

    name: str = "search_corpus"
    description: str = (
        "Performs semantic search over the document corpus to retrieve relevant information. "
        "This tool uses vector embeddings to find documents that are semantically similar to your query, "
        "even if they don't contain exact keyword matches. "
        "Use this when you need to find background information, facts, or context related to a question. "
        "The tool returns a list of document excerpts ranked by relevance, with each excerpt containing "
        "a portion of text from a source document. You can specify how many results to retrieve (default: 5). "
    )
    args_schema: Type[BaseModel] = VectorStoreInput
    return_direct: bool = False

    # Internal attributes
    corpus: Sequence[Document] = []
    cache_dir: Path = Field(default_factory=lambda: Path("/tmp"))
    model: Optional[str] = None
    temperature: float = 0.0
    embedding_model: str = ""
    batch_size: int = 8
    chunk_size: int = 1000
    chunk_overlap: int = 200
    experiment_name: Optional[str] = None
    vector_db: Any = None

    def __init__(
        self,
        corpus: Sequence[Document],
        cache_dir: Path,
        name: str = "search_corpus",
        description: str | None = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        embedding_model: Optional[str] = None,
        batch_size: Optional[int] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        experiment_name: Optional[str] = None,
    ):
        """
        Initialize the vector store tool.

        Args:
            corpus: Sequence of Document objects to index
            cache_dir: Directory to store Milvus database
            name: Name of the tool
            description: Optional custom description for the tool
            model: Model for tool operations (optional, uses default if not provided)
            temperature: Temperature for generation (optional, uses default if not provided)
            embedding_model: Embedding model (optional, uses default if not provided)
            batch_size: Batch size for processing (optional, uses default if not provided)
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            experiment_name: Name of the experiment (for cache directory naming)
        """
        # Use default description if not provided
        if description is None:
            description = (
                "Performs semantic search over the document corpus to retrieve relevant information. "
                "This tool uses vector embeddings to find documents that are semantically similar to your query, "
                "even if they don't contain exact keyword matches. "
                "Use this when you need to find background information, facts, or context related to a question. "
                "The tool returns a list of document excerpts ranked by relevance, with each excerpt containing "
                "a portion of text from a source document. You can specify how many results to retrieve (default: 5). "
            )

        # Call super with only BaseTool fields
        super().__init__(name=name, description=description)

        # Set custom Pydantic fields directly after initialization
        self.corpus = corpus
        self.cache_dir = cache_dir
        self.model = model if model is not None else ModelRegistry.WX_MISTRAL_MEDIUM
        self.temperature = temperature if temperature is not None else 0.0
        self.embedding_model = (
            embedding_model
            if embedding_model is not None
            else ModelRegistry.GRANITE_EMBEDDING
        )
        self.batch_size = batch_size if batch_size is not None else 8
        self.experiment_name = experiment_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_db = None

        # Setup vector store
        self._setup_vector_store()

    def _compute_corpus_hash(self) -> str:
        """
        Compute a hash of the corpus content and configuration.

        Returns:
            SHA256 hash string representing the corpus and config
        """
        hasher = hashlib.sha256()

        # Hash tool configuration (for tool's own cache invalidation)
        config = {
            "model": self.model,
            "temperature": self.temperature,
            "embedding_model": self.embedding_model,
            "batch_size": self.batch_size,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        hasher.update(json.dumps(config, sort_keys=True).encode())

        # Hash corpus documents (sorted by document_id for consistency)
        sorted_corpus = sorted(self.corpus, key=lambda d: d.document_id)
        for doc in sorted_corpus:
            # Hash document metadata
            hasher.update(doc.document_id.encode())
            hasher.update(doc.path.encode())

            # Hash document content
            stream = doc.stream_factory()
            content = stream.read()
            if isinstance(content, bytes):
                hasher.update(content)
            else:
                hasher.update(str(content).encode())

        return hasher.hexdigest()

    def _get_cache_paths(self, corpus_hash: str) -> tuple[Path, Path]:
        """
        Get cache file paths for a given corpus hash.

        Args:
            corpus_hash: Hash of the corpus

        Returns:
            Tuple of (milvus_db_path, hash_marker_path)
        """
        # Include experiment name and number of documents in cache directory name
        num_docs = len(self.corpus)
        cache_subdir = (
            self.cache_dir
            / f"vector_store{''if not self.experiment_name else '_' + self.experiment_name}_{num_docs}_{corpus_hash[:16]}"
        )
        milvus_db_path = cache_subdir / "milvus_corpus.db"
        hash_marker_path = cache_subdir / "corpus_hash.txt"
        return milvus_db_path, hash_marker_path

    def _is_cache_valid(self, corpus_hash: str) -> bool:
        """
        Check if a valid cache exists for the given corpus hash.

        Args:
            corpus_hash: Hash of the corpus

        Returns:
            True if cache exists and is valid
        """
        milvus_db_path, hash_marker_path = self._get_cache_paths(corpus_hash)

        # Check if both files exist
        if not milvus_db_path.exists() or not hash_marker_path.exists():
            return False

        # Verify hash marker content
        try:
            stored_hash = hash_marker_path.read_text().strip()
            return stored_hash == corpus_hash
        except Exception as e:
            logger.warning(f"Error reading cache marker: {e}")
            return False

    def _save_cache_marker(self, corpus_hash: str) -> None:
        """
        Save the corpus hash marker to indicate cache is valid.

        Args:
            corpus_hash: Hash of the corpus
        """
        _, hash_marker_path = self._get_cache_paths(corpus_hash)
        hash_marker_path.parent.mkdir(parents=True, exist_ok=True)
        hash_marker_path.write_text(corpus_hash)

    def _setup_vector_store(self) -> None:
        """Setup the Milvus vector store from the corpus."""
        try:
            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Compute corpus hash
            logger.info("Computing corpus hash for caching...")
            corpus_hash = self._compute_corpus_hash()
            logger.info(f"Corpus hash: {corpus_hash[:16]}...")

            # Check if cache exists
            milvus_db_path, _ = self._get_cache_paths(corpus_hash)

            if self._is_cache_valid(corpus_hash):
                logger.info(
                    f"✓ Found valid cache for corpus, loading from {milvus_db_path}"
                )

                # Create HuggingFace embeddings (runs locally)
                embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)

                # Load existing Milvus vector store with AUTOINDEX for local mode
                self.vector_db = Milvus(
                    embedding_function=embeddings,
                    connection_args={"uri": resolve_milvus_uri(str(milvus_db_path))},
                    collection_name="corpus_documents",
                    auto_id=True,
                    index_params={
                        "metric_type": "COSINE",
                        "index_type": "AUTOINDEX",
                        "params": {},
                    },
                )

                logger.info("✓ Vector store loaded from cache")
                return

            # No valid cache, create new vector store
            logger.info("No valid cache found, creating new vector store...")

            # Ensure cache subdirectory exists
            milvus_db_path.parent.mkdir(parents=True, exist_ok=True)

            # Create HuggingFace embeddings (runs locally)
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)

            # Create Milvus vector store with AUTOINDEX for local mode
            # (HNSW not supported in local Milvus)
            self.vector_db = Milvus(
                embedding_function=embeddings,
                connection_args={"uri": resolve_milvus_uri(str(milvus_db_path))},
                collection_name="corpus_documents",
                auto_id=True,
                index_params={
                    "metric_type": "COSINE",
                    "index_type": "AUTOINDEX",
                    "params": {},
                },
            )

            # Process and add documents with chunking
            chunker = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )

            total_docs = len(self.corpus)
            logger.info(
                f"Starting document processing: {total_docs} documents to process"
            )
            langchain_documents = []

            # Process documents with progress bar
            for doc in tqdm(
                self.corpus, desc="Reading & chunking documents", unit="doc"
            ):
                # Read content from stream
                stream = doc.stream_factory()
                content = stream.read()

                # Decode if bytes
                if isinstance(content, bytes):
                    try:
                        text_content = content.decode("utf-8")
                    except UnicodeDecodeError:
                        logger.warning(f"Skipping binary document: {doc.document_id}")
                        continue
                else:
                    text_content = str(content)

                # Skip empty documents
                if not text_content.strip():
                    continue

                # Create LangChain document
                lc_doc = LangChainDocument(
                    page_content=text_content,
                    metadata={
                        "doc_id": doc.document_id,
                        "path": doc.path,
                        "mime_type": doc.mime_type or "text/plain",
                        **doc.extra_metadata,
                    },
                )

                # Split into chunks
                chunks = chunker.split_documents([lc_doc])
                langchain_documents.extend(chunks)

            # Add documents to vector store with progress
            if langchain_documents:
                total_chunks = len(langchain_documents)
                logger.info(
                    f"Creating embeddings and adding {total_chunks} document chunks to vector store..."
                )
                logger.info(
                    f"Using embedding model: {self.embedding_model} (runs locally)"
                )

                # Add in batches to show progress
                batch_size = 100
                num_batches = (total_chunks + batch_size - 1) // batch_size

                for i in tqdm(
                    range(0, total_chunks, batch_size),
                    desc=f"Embedding & indexing ({num_batches} batches)",
                    unit="batch",
                    total=num_batches,
                ):
                    batch = langchain_documents[i : i + batch_size]
                    self.vector_db.add_documents(batch)

                logger.info(
                    f"✓ Successfully indexed {total_chunks} chunks in vector store"
                )

                # Save cache marker
                self._save_cache_marker(corpus_hash)
                logger.info("✓ Cache saved for future use")
            else:
                logger.warning("No documents to add to vector store")

        except ImportError as e:
            raise ImportError(
                f"Required packages not found: {e}. "
                "Please install langchain-milvus and langchain-huggingface."
            )

    def _run(self, query: str, top_k: int = 20) -> list[str]:
        """
        Search the corpus for relevant documents using semantic search.

        Args:
            query: Search query
            top_k: Number of top documents to return

        Returns:
            List of document content strings, ordered by relevance (most relevant first).
            Each string contains the text content from a document chunk.
            Returns empty list if no documents found or on error.
        """
        if self.vector_db is None:
            logger.error("Vector store not initialized")
            return []

        try:
            # Ensure top_k is valid
            if top_k <= 0:
                top_k = 5

            # Perform similarity search
            results = self.vector_db.similarity_search(query, k=top_k)

            if not results or len(results) == 0:
                logger.info(f"No relevant documents found for query: {query}")
                return []

            # Extract document contents
            document_contents = []
            for i, doc in enumerate(results, 1):
                try:
                    # Get the actual text content
                    content = (
                        doc.page_content if hasattr(doc, "page_content") else str(doc)
                    )

                    # Skip empty content
                    if content and content.strip():
                        document_contents.append(content)
                    else:
                        logger.warning(f"Skipping empty document at position {i}")

                except Exception as doc_error:
                    logger.warning(
                        f"Error extracting content from document {i}: {doc_error}"
                    )
                    continue

            if not document_contents:
                logger.warning(
                    f"Found {len(results)} documents but could not extract content for query: {query}"
                )
                return []

            logger.info(
                f"Retrieved {len(document_contents)} document(s) for query: {query[:50]}..."
            )
            return document_contents

        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []
