"""Unit tests for KramaBench tools builder components."""

from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain.tools import BaseTool

from experiments.benchmarks.kramabench.tools_builder import KramaBenchToolsBuilder
from experiments.benchmarks.shared_tools import (
    FileContentInput,
    FileContentTool,
    MilvusSearchInput,
    MilvusSearchTool,
)
from experiments.core.context import PipelineConfig, PipelineContext
from experiments.core.types import Document


def create_test_document(doc_id: str, path: str) -> Document:
    """Helper to create test Document instances."""
    return Document(
        document_id=doc_id,
        path=path,
        mime_type="text/plain",
        extra_metadata={},
        stream_factory=lambda: BytesIO(b"test content"),
    )


# ============================================================================
# MilvusSearchTool Tests
# ============================================================================


def test_milvus_search_tool_initialization():
    """Test MilvusSearchTool initialization."""
    mock_db = MagicMock()
    tool = MilvusSearchTool(vector_db=mock_db)

    assert tool.name == "search_files"
    assert "search" in tool.description.lower()
    assert tool.vector_db is mock_db
    assert tool.args_schema == MilvusSearchInput


def test_milvus_search_tool_search():
    """Test MilvusSearchTool search functionality."""
    mock_db = MagicMock()
    mock_result = MagicMock()
    mock_result.metadata = {
        "file_path": "/path/to/test_file.txt",
        "filename": "test_file.txt",
    }
    mock_result.page_content = "Description of test file"
    mock_db.similarity_search.return_value = [mock_result]

    tool = MilvusSearchTool(vector_db=mock_db)
    results = tool._run("test query", top_k=5)

    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], dict)
    assert results[0]["path"] == "/path/to/test_file.txt"
    assert results[0]["filename"] == "test_file.txt"
    assert results[0]["description"] == "Description of test file"
    mock_db.similarity_search.assert_called_once_with("test query", k=5)


def test_milvus_search_tool_multiple_results():
    """Test MilvusSearchTool with multiple results."""
    mock_db = MagicMock()
    mock_results = [
        MagicMock(
            metadata={"file_path": "/path/file1.txt", "filename": "file1.txt"},
            page_content="Description 1",
        ),
        MagicMock(
            metadata={"file_path": "/path/file2.pdf", "filename": "file2.pdf"},
            page_content="Description 2",
        ),
        MagicMock(
            metadata={"file_path": "/path/file3.docx", "filename": "file3.docx"},
            page_content="Description 3",
        ),
    ]
    mock_db.similarity_search.return_value = mock_results

    tool = MilvusSearchTool(vector_db=mock_db)
    results = tool._run("test query", top_k=3)

    assert len(results) == 3
    assert results[0] == {
        "path": "/path/file1.txt",
        "filename": "file1.txt",
        "description": "Description 1",
    }
    assert results[1] == {
        "path": "/path/file2.pdf",
        "filename": "file2.pdf",
        "description": "Description 2",
    }
    assert results[2] == {
        "path": "/path/file3.docx",
        "filename": "file3.docx",
        "description": "Description 3",
    }


def test_milvus_search_tool_no_results():
    """Test MilvusSearchTool when no results found."""
    mock_db = MagicMock()
    mock_db.similarity_search.return_value = []

    tool = MilvusSearchTool(vector_db=mock_db)
    results = tool._run("test query")

    assert results == []


def test_milvus_search_tool_no_db():
    """Test MilvusSearchTool when vector_db is None."""
    tool = MilvusSearchTool(vector_db=None)
    results = tool._run("test query")

    assert results == []


def test_milvus_search_tool_exception_handling():
    """Test MilvusSearchTool handles exceptions gracefully."""
    mock_db = MagicMock()
    mock_db.similarity_search.side_effect = Exception("Database error")

    tool = MilvusSearchTool(vector_db=mock_db)
    results = tool._run("test query")

    assert results == []


def test_milvus_search_tool_missing_metadata():
    """Test MilvusSearchTool with results missing metadata."""
    mock_db = MagicMock()
    mock_result_no_metadata = MagicMock(spec=[])  # No metadata attribute
    mock_result_no_path = MagicMock(
        metadata={}, page_content="No file_path"
    )  # No file_path in metadata
    mock_result_valid = MagicMock(
        metadata={"file_path": "/path/valid.txt", "filename": "valid.txt"},
        page_content="Valid description",
    )
    mock_db.similarity_search.return_value = [
        mock_result_no_metadata,
        mock_result_no_path,
        mock_result_valid,
    ]

    tool = MilvusSearchTool(vector_db=mock_db)
    results = tool._run("test query")

    # Only the valid result should be returned
    assert len(results) == 1
    assert results[0] == {
        "path": "/path/valid.txt",
        "filename": "valid.txt",
        "description": "Valid description",
    }


# ============================================================================
# FileContentTool Tests
# ============================================================================


def test_file_content_tool_initialization():
    """Test FileContentTool initialization."""
    path = "/path/to/file1.txt"
    path_to_bytes_factory: dict[str, Any] = {path: lambda: b"content"}

    tool = FileContentTool(path_to_bytes_factory=path_to_bytes_factory)

    assert tool.name == "get_file_content"
    assert "retrieve" in tool.description.lower()
    assert tool.path_to_bytes_factory == path_to_bytes_factory
    assert tool.args_schema == FileContentInput


def test_file_content_tool_retrieval():
    """Test FileContentTool retrieval functionality."""

    def content_factory() -> bytes:
        return b"Test content"

    path = "/path/to/file1.txt"
    path_to_bytes_factory: dict[str, Any] = {path: content_factory}

    tool = FileContentTool(path_to_bytes_factory=path_to_bytes_factory)
    stream = tool._run(path)

    assert stream is not None
    content = stream.read()
    assert content == b"Test content"


def test_file_content_tool_path_based_access():
    """Test FileContentTool uses path-based access."""

    def content_factory() -> bytes:
        return b"Test content"

    path = "/path/to/file1.txt"
    path_to_bytes_factory: dict[str, Any] = {path: content_factory}

    tool = FileContentTool(path_to_bytes_factory=path_to_bytes_factory)

    # Should be accessible by full path
    assert path in tool.path_to_bytes_factory
    assert tool.path_to_bytes_factory[path] == content_factory


def test_file_content_tool_file_not_found():
    """Test FileContentTool when file not found."""
    path = "/path/to/file1.txt"
    path_to_bytes_factory: dict[str, Any] = {path: lambda: b"content"}

    tool = FileContentTool(path_to_bytes_factory=path_to_bytes_factory)

    # FileContentTool raises ValueError for missing files
    with pytest.raises(ValueError, match="File not found in corpus"):
        tool._run("/nonexistent/path.txt")


def test_file_content_tool_no_content():
    """Test FileContentTool when content not available."""
    path = "/path/to/file1.txt"
    path_to_bytes_factory: dict[str, Any] = {}

    tool = FileContentTool(path_to_bytes_factory=path_to_bytes_factory)

    # FileContentTool raises ValueError for missing files
    with pytest.raises(ValueError, match="File not found in corpus"):
        tool._run(path)


def test_file_content_tool_stream_factory_reusable():
    """Test that stream can be read multiple times by calling tool._run again."""

    def content_factory() -> bytes:
        return b"Test content"

    path = "/path/to/file1.txt"
    path_to_bytes_factory: dict[str, Any] = {path: content_factory}

    tool = FileContentTool(path_to_bytes_factory=path_to_bytes_factory)

    # Get stream multiple times
    stream1 = tool._run(path)
    assert stream1 is not None
    content1 = stream1.read()

    stream2 = tool._run(path)
    content2 = stream2.read()

    assert content1 == b"Test content"
    assert content2 == b"Test content"


# ============================================================================
# KramaBenchToolsBuilder Tests
# ============================================================================


def test_kramabench_tools_builder_initialization():
    """Test KramaBenchToolsBuilder initialization."""
    builder = KramaBenchToolsBuilder(
        llm="gpt-4", embedding_model="text-embedding-ada-002", cache_dir="/tmp/cache"
    )

    assert builder.llm == "gpt-4"
    assert builder.embedding_model == "text-embedding-ada-002"
    assert builder.cache_dir == "/tmp/cache"
    assert builder.name == "kramabench_tools"


def test_kramabench_tools_builder_name_property():
    """Test KramaBenchToolsBuilder name property."""
    builder = KramaBenchToolsBuilder(
        llm="gpt-4", embedding_model="text-embedding-ada-002", cache_dir="/tmp/cache"
    )

    assert builder.name == "kramabench_tools"


@patch("experiments.benchmarks.kramabench.tools_builder.DoclingDescriptionBuilder")
def test_kramabench_tools_builder_build_tools(mock_builder_class):
    """Test KramaBenchToolsBuilder.build_tools method."""
    # Setup mocks
    mock_vector_db = MagicMock()
    mock_analysis_results = {}
    mock_path_to_bytes_factory: dict[str, Any] = {"doc1": lambda: b"content"}

    mock_builder_instance = MagicMock()
    mock_builder_instance.process_corpus.return_value = (
        mock_vector_db,
        mock_analysis_results,
        mock_path_to_bytes_factory,
    )
    mock_builder_class.return_value = mock_builder_instance

    # Create builder and corpus
    builder = KramaBenchToolsBuilder(
        llm="gpt-4", embedding_model="text-embedding-ada-002", cache_dir="/tmp/cache"
    )

    corpus = [create_test_document("doc1", "/path/to/file1.txt")]

    ctx = PipelineContext(config=PipelineConfig())

    # Build tools
    tools = builder.build_tools(ctx=ctx, benchmarks=[], corpus=corpus)

    # Verify DoclingDescriptionBuilder was initialized correctly
    mock_builder_class.assert_called_once_with(
        cache_dir="/tmp/cache",
        model="gpt-4",
        temperature=0.0,
        embedding_model="text-embedding-ada-002",
        batch_size=8,
        enable_caching=True,
    )

    # Verify process_corpus was called
    mock_builder_instance.process_corpus.assert_called_once_with(corpus)

    # Verify tools were created
    assert len(tools) == 4
    assert all(isinstance(tool, BaseTool) for tool in tools)

    # Verify tool names
    tool_names = [tool.name for tool in tools]
    assert "search_files" in tool_names
    assert "get_file_info" in tool_names
    assert "get_file_content" in tool_names
    assert "load_dataframe" in tool_names


@patch("experiments.benchmarks.kramabench.tools_builder.DoclingDescriptionBuilder")
def test_kramabench_tools_builder_no_corpus(mock_builder_class):
    """Test KramaBenchToolsBuilder.build_tools with no corpus."""
    builder = KramaBenchToolsBuilder(
        llm="gpt-4", embedding_model="text-embedding-ada-002", cache_dir="/tmp/cache"
    )

    ctx = PipelineContext(config=PipelineConfig())

    # Build tools with no corpus
    tools = builder.build_tools(ctx=ctx, benchmarks=[], corpus=None)

    # Should return empty list
    assert tools == []

    # DoclingDescriptionBuilder should not be initialized
    mock_builder_class.assert_not_called()


@patch("experiments.benchmarks.kramabench.tools_builder.DoclingDescriptionBuilder")
def test_kramabench_tools_builder_empty_corpus(mock_builder_class):
    """Test KramaBenchToolsBuilder.build_tools with empty corpus."""
    # Setup mocks
    mock_vector_db = MagicMock()
    mock_analysis_results = {}
    mock_path_to_bytes_factory: dict[str, Any] = {}

    mock_builder_instance = MagicMock()
    mock_builder_instance.process_corpus.return_value = (
        mock_vector_db,
        mock_analysis_results,
        mock_path_to_bytes_factory,
    )
    mock_builder_class.return_value = mock_builder_instance

    builder = KramaBenchToolsBuilder(
        llm="gpt-4", embedding_model="text-embedding-ada-002", cache_dir="/tmp/cache"
    )

    ctx = PipelineContext(config=PipelineConfig())
    corpus = []

    # Build tools with empty corpus
    tools = builder.build_tools(ctx=ctx, benchmarks=[], corpus=corpus)

    # Should still create tools (even if empty)
    assert len(tools) == 4
    mock_builder_instance.process_corpus.assert_called_once_with(corpus)


@patch("experiments.benchmarks.kramabench.tools_builder.DoclingDescriptionBuilder")
def test_kramabench_tools_builder_multiple_documents(mock_builder_class):
    """Test KramaBenchToolsBuilder.build_tools with multiple documents."""
    # Setup mocks - use paths as keys (matching new design)
    mock_vector_db = MagicMock()
    mock_analysis_results = {}
    mock_path_to_bytes_factory: dict[str, Any] = {
        "/path/to/file1.txt": lambda: b"content1",
        "/path/to/file2.pdf": lambda: b"content2",
        "/path/to/file3.docx": lambda: b"content3",
    }

    mock_builder_instance = MagicMock()
    mock_builder_instance.process_corpus.return_value = (
        mock_vector_db,
        mock_analysis_results,
        mock_path_to_bytes_factory,
    )
    mock_builder_class.return_value = mock_builder_instance

    builder = KramaBenchToolsBuilder(
        llm="gpt-4", embedding_model="text-embedding-ada-002", cache_dir="/tmp/cache"
    )

    corpus = [
        create_test_document("doc1", "/path/to/file1.txt"),
        create_test_document("doc2", "/path/to/file2.pdf"),
        create_test_document("doc3", "/path/to/file3.docx"),
    ]

    ctx = PipelineContext(config=PipelineConfig())

    # Build tools
    tools = builder.build_tools(ctx=ctx, benchmarks=[], corpus=corpus)

    # Verify tools were created
    assert len(tools) == 4

    # Get the file content tool - cast to FileContentTool to access attributes
    content_tool = next(tool for tool in tools if tool.name == "get_file_content")
    assert isinstance(content_tool, FileContentTool)

    # Verify all file paths are mapped
    assert "/path/to/file1.txt" in content_tool.path_to_bytes_factory
    assert "/path/to/file2.pdf" in content_tool.path_to_bytes_factory
    assert "/path/to/file3.docx" in content_tool.path_to_bytes_factory


# ============================================================================
# Integration Tests - Tool Interoperability
# ============================================================================


def test_search_to_content_tool_integration():
    """Test that paths from search tool can be used with content tool."""
    # Create mock vector DB that returns paths and filenames
    mock_db = MagicMock()
    mock_results = [
        MagicMock(
            metadata={"file_path": "/path/to/file1.txt", "filename": "file1.txt"},
            page_content="Description of file1",
        ),
        MagicMock(
            metadata={"file_path": "/path/to/file2.pdf", "filename": "file2.pdf"},
            page_content="Description of file2",
        ),
    ]
    mock_db.similarity_search.return_value = mock_results

    # Create content factories
    def content_factory_1() -> bytes:
        return b"Content of file1"

    def content_factory_2() -> bytes:
        return b"Content of file2"

    # Use full paths as keys in path_to_bytes_factory
    path_to_bytes_factory: dict[str, Any] = {
        "/path/to/file1.txt": content_factory_1,
        "/path/to/file2.pdf": content_factory_2,
    }

    # Create tools
    search_tool = MilvusSearchTool(vector_db=mock_db)
    content_tool = FileContentTool(path_to_bytes_factory=path_to_bytes_factory)

    # Step 1: Search for files
    results = search_tool._run("test query", top_k=2)
    assert len(results) == 2
    paths = [r["path"] for r in results]
    assert "/path/to/file1.txt" in paths
    assert "/path/to/file2.pdf" in paths

    # Step 2: Use paths to retrieve content
    for result in results:
        path = result["path"]
        stream = content_tool._run(path)
        assert stream is not None, f"Failed to retrieve content for {path}"

        # Verify we can read the content
        content = stream.read()
        assert len(content) > 0, f"Empty content for {path}"
        assert isinstance(content, bytes), f"Content should be bytes for {path}"


def test_search_to_content_tool_with_digest_in_doc_id():
    """Test that content tool works with full paths regardless of doc_id format."""
    # Create mock vector DB that returns paths
    mock_db = MagicMock()
    mock_results = [
        MagicMock(
            metadata={"file_path": "/path/to/report.pdf", "filename": "report.pdf"},
            page_content="Report description",
        ),
    ]
    mock_db.similarity_search.return_value = mock_results

    # Content factory
    def content_factory() -> bytes:
        return b"Report content"

    # path_to_bytes_factory uses full path
    path = "/path/to/report.pdf"
    path_to_bytes_factory: dict[str, Any] = {
        path: content_factory,
    }

    # Create tools
    search_tool = MilvusSearchTool(vector_db=mock_db)
    content_tool = FileContentTool(path_to_bytes_factory=path_to_bytes_factory)

    # Search returns path, filename and description
    results = search_tool._run("test query")
    assert len(results) == 1
    assert results[0]["path"] == path
    assert results[0]["filename"] == "report.pdf"
    assert results[0]["description"] == "Report description"

    # Content tool should find it using the path
    stream = content_tool._run(results[0]["path"])
    assert stream is not None

    # Verify content is accessible
    content = stream.read()
    assert content == b"Report content"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
