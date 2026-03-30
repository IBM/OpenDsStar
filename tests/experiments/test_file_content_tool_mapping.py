"""Test that FileContentTool correctly maps paths from search results to bytes factories.

This test verifies that all paths are consistently relative throughout the system:
1. DoclingDescriptionBuilder stores relative paths as keys in doc_id_to_bytes_factory
2. MilvusSearchTool returns relative paths from metadata
3. FileContentTool._run() receives relative paths and successfully looks them up
"""

from io import BytesIO
from unittest.mock import Mock

import pytest

from OpenDsStar.experiments.benchmarks.shared_tools import (
    FileContentTool,
    MilvusSearchTool,
)
from OpenDsStar.experiments.core.types import Document


def test_file_content_tool_with_relative_paths():
    """
    Test that FileContentTool works correctly with relative paths.

    After the fix:
    - DoclingDescriptionBuilder stores relative paths (e.g., "data/file.csv")
    - MilvusSearchTool returns relative paths from metadata
    - FileContentTool._run() receives relative paths and finds the bytes factory
    """
    # Setup: Create a bytes factory with RELATIVE path as key
    relative_path = "data/test_file.csv"
    test_data = b"test,data\n1,2\n3,4"

    def bytes_factory() -> bytes:
        return test_data

    path_to_bytes_factory = {relative_path: bytes_factory}  # Key is RELATIVE path

    # Create corpus with relative path
    _ = [
        Document(
            document_id="test_doc",
            path=relative_path,
            mime_type="text/csv",
            extra_metadata={},
            stream_factory=lambda: BytesIO(test_data),
        )
    ]

    # Create the tool
    tool = FileContentTool(path_to_bytes_factory=path_to_bytes_factory)

    # After fix: relative paths work correctly
    stream = tool._run(path=relative_path)
    assert stream is not None, "Tool should find file with relative path"

    # Verify the content is correct
    content = stream.read()
    assert content == test_data

    # Absolute paths should NOT work (system uses relative paths only)
    absolute_path = f"/full/path/{relative_path}"
    with pytest.raises(ValueError, match="File not found in corpus"):
        tool._run(path=absolute_path)


def test_milvus_search_returns_relative_paths():
    """
    Test that MilvusSearchTool returns relative paths from metadata.

    After the fix: the search tool returns relative paths that match
    the keys in path_to_bytes_factory.
    """
    # Mock vector_db that returns results with relative paths
    mock_vector_db = Mock()

    # Create a mock document with RELATIVE path in metadata
    mock_doc = Mock()
    mock_doc.metadata = {
        "file_path": "data/file.csv",  # RELATIVE path
        "filename": "file.csv",
        "doc_id": "test_doc",
    }
    mock_doc.page_content = "Test description"

    mock_vector_db.similarity_search.return_value = [mock_doc]

    # Create search tool
    search_tool = MilvusSearchTool(vector_db=mock_vector_db)

    # Search returns relative paths
    results = search_tool._run(query="test query", top_k=1)

    assert len(results) == 1
    assert results[0]["path"] == "data/file.csv"
    assert not results[0]["path"].startswith("/"), "Search returns RELATIVE path"


def test_end_to_end_search_and_retrieve_success():
    """
    End-to-end test showing the complete flow works correctly after the fix:
    1. Search returns relative path
    2. User passes that path to get_file_content
    3. Tool successfully finds and retrieves the file
    """
    # Setup: bytes factory with relative path
    relative_path = "data/sales.csv"
    test_data = b"product,sales\nA,100\nB,200"

    def bytes_factory() -> bytes:
        return test_data

    path_to_bytes_factory = {relative_path: bytes_factory}  # Key is RELATIVE

    # Create content tool
    content_tool = FileContentTool(path_to_bytes_factory=path_to_bytes_factory)

    # Mock search tool that returns relative path (after fix)
    mock_vector_db = Mock()
    mock_doc = Mock()
    mock_doc.metadata = {
        "file_path": relative_path,  # Search returns RELATIVE path
        "filename": "sales.csv",
        "doc_id": "sales",
    }
    mock_doc.page_content = "Sales data with product and sales columns"
    mock_vector_db.similarity_search.return_value = [mock_doc]

    search_tool = MilvusSearchTool(vector_db=mock_vector_db)

    # Step 1: Agent searches for files
    search_results = search_tool._run(query="sales data", top_k=1)
    assert len(search_results) == 1

    # Step 2: Agent retrieves file content using path from search
    file_path_from_search = search_results[0]["path"]
    assert file_path_from_search == relative_path, "Search returns relative path"

    # Step 3: SUCCESS - Content tool finds the file
    stream = content_tool._run(path=file_path_from_search)
    assert stream is not None, "Content tool successfully finds file with relative path"

    # Step 4: Verify content is correct
    content = stream.read()
    assert content == test_data, "Retrieved content matches original data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
