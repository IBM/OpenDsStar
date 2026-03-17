"""
Build document descriptions using the analyzer agent for code-based analysis.
"""

import logging
from pathlib import Path
from typing import Any, Dict

from docling_core.types.io import DocumentStream
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_litellm import ChatLiteLLM
from langchain_milvus.vectorstores.milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agents.analyzer.analyzer_graph import (
    AnalyzerGraph,
    prepare_result_from_graph_state_analyzer_agent,
)

from .document_description_builder import DocumentDescriptionBuilder

logger = logging.getLogger(__name__)


class AnalyzerDescriptionBuilder(DocumentDescriptionBuilder):
    """
    Build document descriptions using the analyzer agent for code-based analysis.

    Uses the analyzer agent graph to execute code-based analysis on files,
    then stores the analysis results in a Milvus vector store.
    """

    def __init__(
        self,
        llm: BaseChatModel | str = "watsonx/mistralai/mistral-medium-2505",
        code_timeout: int = 30,
        max_debug_tries: int = 3,
        embedding_model: str = "ibm-granite/granite-embedding-english-r2",
        db_uri: str = "./milvus_analyzer.db",
    ):
        """
        Initialize the analyzer document processor.

        Args:
            llm: Language model for code generation and debugging
            code_timeout: Maximum time in seconds for code execution
            max_debug_tries: Maximum number of debug attempts
            embedding_model: HuggingFace model name for embeddings
            db_uri: Path to Milvus database file
        """
        if isinstance(llm, str):
            llm = ChatLiteLLM(model=llm)
        self.analyzer_graph = AnalyzerGraph(
            llm=llm,
            code_timeout=code_timeout,
            max_debug_tries=max_debug_tries,
        )
        self.embedding_model = embedding_model
        self.db_uri = db_uri

    def _process_files(
        self, file_paths: list[Path], analysis_results: dict[str, Dict[str, Any]]
    ) -> tuple[Milvus, dict[str, Dict[str, Any]], dict[str, DocumentStream]]:
        """
        Common processing logic for files.

        Args:
            file_paths: List of file paths to process
            analysis_results: Dict to store analysis results

        Returns:
            Tuple of (vector_db, analysis_results, document_streams)
        """
        document_streams = {}

        for file_path in file_paths:
            file_path = Path(file_path)
            logger.info(f"Analyzing file: {file_path.name}")

            try:
                # Run the analyzer graph
                state = self.analyzer_graph.invoke(
                    {"filename": str(file_path.absolute())}
                )
                result = prepare_result_from_graph_state_analyzer_agent(dict(state))
                # Store the full file path in the result
                result["file_path"] = str(file_path.absolute())
                analysis_results[file_path.name] = result

                # Create a simple DocumentStream placeholder
                # (analyzer doesn't produce actual DocumentStream objects)
                document_streams[file_path.name] = None

            except Exception as e:
                logger.warning(f"Failed to analyze file {file_path.name}: {e}")
                analysis_results[file_path.name] = {
                    "success": False,
                    "fatal_error": str(e),
                    "answer": "",
                    "logs": "",
                    "outputs": {},
                    "file_path": str(file_path.absolute()),
                }

        # Create vector database
        vector_db = Milvus(
            embedding_function=HuggingFaceEmbeddings(model_name=self.embedding_model),
            connection_args={"uri": self.db_uri},
            auto_id=True,
        )

        # Add summaries to vector DB
        chunker = RecursiveCharacterTextSplitter()
        documents = []
        for filename, result in analysis_results.items():
            if result["success"]:
                # Create a summary document from the logs (printed output)
                summary = result["logs"]
                logger.info(summary)
                if summary:
                    doc = Document(
                        page_content=summary,
                        metadata={
                            "doc_id": filename,
                        },
                    )
                    chunks = chunker.split_documents([doc])
                    documents.extend(chunks)

        if documents:
            vector_db.add_documents(documents)
            logger.info(f"Added {len(documents)} analysis summaries to vector DB")
        else:
            logger.warning("No successful analyses to add to vector DB")

        return vector_db, analysis_results, document_streams

    def process_directory(
        self, dir_path: Path
    ) -> tuple[Milvus, dict[str, Dict[str, Any]], dict[str, DocumentStream]]:
        """
        Process all files in a directory and store summaries in vector DB.

        Args:
            dir_path: Directory containing files to analyze

        Returns:
            Tuple of (vector_db, analysis_results, document_streams)
        """
        dir_path = Path(dir_path)
        logger.info(f"Processing directory: {dir_path}")

        analysis_results = {}
        file_paths = [f for f in dir_path.iterdir() if f.is_file()]

        return self._process_files(file_paths, analysis_results)

    def process_corpus(
        self, corpus: list[Path | str]
    ) -> tuple[Milvus, dict[str, Dict[str, Any]], dict[str, DocumentStream]]:
        """
        Process a corpus of file paths.

        Args:
            corpus: List of file paths to analyze

        Returns:
            Tuple of (vector_db, analysis_results, document_streams)
        """
        logger.info(f"Processing corpus with {len(corpus)} files")

        analysis_results = {}
        file_paths = [Path(f) for f in corpus]

        return self._process_files(file_paths, analysis_results)
