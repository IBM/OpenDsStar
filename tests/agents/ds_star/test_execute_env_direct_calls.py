"""Integration tests for direct tool call handling in execution environment."""

from agents.ds_star.ds_star_execute_env import execute_user_code
from agents.ds_star.ds_star_state import DSState


class TestExecuteEnvDirectCalls:
    """Test direct tool call handling in execute_user_code."""

    def test_direct_tool_call_works(self):
        """Test that direct tool calls work when tools are in environment."""
        # Create a mock tool
        call_log = []

        def mock_search_tool(query: str, top_k: int = 5):
            call_log.append({"query": query, "top_k": top_k})
            return [{"path": "/data/test.csv", "filename": "test.csv"}]

        tools = {"search_files": mock_search_tool}

        # Code with direct tool call
        code = """
results = search_files(query="test data", top_k=3)
outputs["results"] = results
"""

        state = DSState(user_query="test", tools=tools, steps=[])

        # Execute with auto-transform enabled (default)
        logs, outputs = execute_user_code(code, state, tools, timeout=5)

        # Tool should have been called
        assert len(call_log) == 1
        assert call_log[0]["query"] == "test data"
        assert call_log[0]["top_k"] == 3

        # Results should be in outputs
        assert "results" in outputs
        assert len(outputs["results"]) == 1

    def test_multiple_direct_calls(self):
        """Test multiple direct tool calls."""
        call_log = []

        def mock_search_tool(query: str):
            call_log.append(("search", query))
            return [{"path": "/data/file.csv"}]

        def mock_content_tool(path: str):
            call_log.append(("content", path))
            # Return picklable data instead of a lambda
            return b"test,data\n1,2"

        tools = {
            "search_files": mock_search_tool,
            "get_file_content": mock_content_tool,
        }

        code = """
results = search_files(query="test")
content = get_file_content(path=results[0]["path"])
outputs["found"] = len(results)
outputs["has_content"] = isinstance(content, bytes)
"""

        state = DSState(user_query="test", tools=tools, steps=[])
        logs, outputs = execute_user_code(code, state, tools, timeout=5)

        # Both tools should have been called
        assert len(call_log) == 2
        assert call_log[0] == ("search", "test")
        assert call_log[1] == ("content", "/data/file.csv")

        assert outputs["found"] == 1
        assert outputs["has_content"] is True

    def test_sequential_direct_calls(self):
        """Test sequential direct tool calls."""
        call_log = []

        def mock_search(query: str):
            call_log.append(("search", query))
            return [{"path": "/data/file.csv"}]

        def mock_content(path: str):
            call_log.append(("content", path))
            # Return picklable data instead of a lambda
            return b"data"

        tools = {
            "search_files": mock_search,
            "get_file_content": mock_content,
        }

        code = """
results = search_files(query="test")
file_path = results[0]["path"]
content = get_file_content(path=file_path)
outputs["has_content"] = isinstance(content, bytes)
"""

        state = DSState(user_query="test", tools=tools, steps=[])
        logs, outputs = execute_user_code(code, state, tools, timeout=5)

        # Both tools should be called
        assert len(call_log) == 2
        assert call_log[0] == ("search", "test")
        assert call_log[1] == ("content", "/data/file.csv")

    def test_direct_call_with_kwargs(self):
        """Test direct calls preserve keyword arguments."""
        call_log = []

        def mock_tool(query: str, top_k: int = 5, filter_type: str = "all"):
            call_log.append(
                {
                    "query": query,
                    "top_k": top_k,
                    "filter_type": filter_type,
                }
            )
            return []

        tools = {"search_files": mock_tool}

        code = """
results = search_files(query="test", top_k=10, filter_type="csv")
outputs["done"] = True
"""

        state = DSState(user_query="test", tools=tools, steps=[])
        logs, outputs = execute_user_code(code, state, tools, timeout=5)

        assert len(call_log) == 1
        assert call_log[0]["query"] == "test"
        assert call_log[0]["top_k"] == 10
        assert call_log[0]["filter_type"] == "csv"

    def test_no_transform_for_non_tools(self):
        """Test that non-tool functions are not transformed."""
        tools = {"search_files": lambda query: []}

        code = """
def my_helper(x):
    return x * 2

result = my_helper(5)
outputs["result"] = result
"""

        state = DSState(user_query="test", tools=tools, steps=[])
        logs, outputs = execute_user_code(code, state, tools, timeout=5)

        # Should work fine - my_helper is not a tool
        assert outputs["result"] == 10

    def test_direct_call_with_positional_args(self):
        """Test direct calls with positional arguments."""
        call_log = []

        def mock_tool(query, top_k=5):
            call_log.append({"query": query, "top_k": top_k})
            return [{"result": query}]

        tools = {"search_files": mock_tool}

        code = """
results = search_files("test query", 10)
outputs["count"] = len(results)
"""

        state = DSState(user_query="test", tools=tools, steps=[])
        logs, outputs = execute_user_code(code, state, tools, timeout=5)

        assert len(call_log) == 1
        assert call_log[0]["query"] == "test query"
        assert call_log[0]["top_k"] == 10
        assert outputs["count"] == 1

    def test_direct_call_mixed_args(self):
        """Test direct calls with both positional and keyword arguments."""
        call_log = []

        def mock_tool(query, top_k=5, filter_type="all"):
            call_log.append(
                {
                    "query": query,
                    "top_k": top_k,
                    "filter_type": filter_type,
                }
            )
            return []

        tools = {"search_files": mock_tool}

        code = """
results = search_files("test", 10, filter_type="csv")
outputs["done"] = True
"""

        state = DSState(user_query="test", tools=tools, steps=[])
        logs, outputs = execute_user_code(code, state, tools, timeout=5)

        assert len(call_log) == 1
        assert call_log[0]["query"] == "test"
        assert call_log[0]["top_k"] == 10
        assert call_log[0]["filter_type"] == "csv"

    def test_error_in_direct_call(self):
        """Test error handling when direct tool call raises error."""

        def mock_tool(query: str):
            raise ValueError("Tool error")

        tools = {"search_files": mock_tool}

        code = """
results = search_files(query="test")
outputs["results"] = results
"""

        state = DSState(user_query="test", tools=tools, steps=[])
        logs, outputs = execute_user_code(code, state, tools, timeout=5)

        # Should capture the error
        assert "_error" in outputs
        # Just verify an error was captured - the exact message may vary
        assert isinstance(outputs["_error"], str) and len(outputs["_error"]) > 0
