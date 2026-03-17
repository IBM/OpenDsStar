"""Standalone test to verify MilvusSearchTool changes."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Get the project root
project_root = Path(__file__).parent.parent

sys.path.insert(0, str(project_root / "src"))

spec = importlib.util.spec_from_file_location(
    "shared_tools",
    project_root / "src" / "experiments" / "benchmarks" / "shared_tools.py",
)
shared_tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared_tools)

MilvusSearchTool = shared_tools.MilvusSearchTool

# Create mock vector DB
mock_db = MagicMock()
mock_result = MagicMock()
mock_result.metadata = {"filename": "test_file.txt"}
mock_result.page_content = "This is a test file description"
mock_db.similarity_search.return_value = [mock_result]

# Create and test the tool
tool = MilvusSearchTool(vector_db=mock_db)
results = tool._run("test query", top_k=5)

print("Test Results:")
print(f"Number of results: {len(results)}")
print(f"Result type: {type(results[0])}")
print(f"Result content: {results[0]}")
print(f'Filename: {results[0]["filename"]}')
print(f'Description: {results[0]["description"]}')

# Verify the results
assert len(results) == 1, "Expected 1 result"
assert isinstance(results[0], dict), "Expected dict result"
assert results[0]["filename"] == "test_file.txt", "Filename mismatch"
assert (
    results[0]["description"] == "This is a test file description"
), "Description mismatch"

print("\n✓ All tests passed! MilvusSearchTool now returns filename and description.")
