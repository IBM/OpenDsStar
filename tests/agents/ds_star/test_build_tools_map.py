"""
Test build_tools_map function with various tool calling patterns.
"""

from OpenDsStar.agents.ds_star.ds_star_utils import build_tools_map


class MockTool:
    """Mock tool for testing."""

    def __init__(self, name: str, func=None):
        self.name = name
        self.func = func


def test_build_tools_map_with_positional_args():
    """Test that tools can be called with positional arguments."""

    def sample_func(arg1, arg2):
        return f"{arg1}-{arg2}"

    tool = MockTool("test_tool", func=sample_func)
    tools_map = build_tools_map([tool])

    # Test with positional args
    result = tools_map["test_tool"]("hello", "world")
    assert result == "hello-world"


def test_build_tools_map_with_keyword_args():
    """Test that tools can be called with keyword arguments."""

    def sample_func(arg1, arg2):
        return f"{arg1}-{arg2}"

    tool = MockTool("test_tool", func=sample_func)
    tools_map = build_tools_map([tool])

    # Test with keyword args
    result = tools_map["test_tool"](arg1="hello", arg2="world")
    assert result == "hello-world"


def test_build_tools_map_with_mixed_args():
    """Test that tools can be called with mixed positional and keyword arguments."""

    def sample_func(arg1, arg2, arg3="default"):
        return f"{arg1}-{arg2}-{arg3}"

    tool = MockTool("test_tool", func=sample_func)
    tools_map = build_tools_map([tool])

    # Test with mixed args
    result = tools_map["test_tool"]("hello", "world", arg3="custom")
    assert result == "hello-world-custom"


def test_build_tools_map_single_positional_arg():
    """Test the common case of a single positional argument (like file_path)."""

    def get_content(file_path):
        return f"content of {file_path}"

    tool = MockTool("get_file_content", func=get_content)
    tools_map = build_tools_map([tool])

    # This is the pattern that was failing before
    result = tools_map["get_file_content"]("path/to/file.csv")
    assert result == "content of path/to/file.csv"


def test_build_tools_map_with_invoke_method():
    """Test tools that use .invoke() method (LangChain style)."""

    class InvokeTool:
        name = "invoke_tool"

        def invoke(self, input_dict):
            if isinstance(input_dict, dict):
                return f"invoked with {input_dict}"
            else:
                return f"invoked with {input_dict}"

    tool = InvokeTool()
    tools_map = build_tools_map([tool])

    # Test with kwargs (standard LangChain pattern)
    result = tools_map["invoke_tool"](query="test")
    assert "invoked with" in result

    # Test with single positional arg
    result = tools_map["invoke_tool"]("single_arg")
    assert "invoked with single_arg" in result


def test_build_tools_map_callable_tool():
    """Test tools that are directly callable."""

    class CallableTool:
        name = "callable_tool"

        def __call__(self, *args, **kwargs):
            return f"called with args={args}, kwargs={kwargs}"

    tool = CallableTool()
    tools_map = build_tools_map([tool])

    # Test with positional args
    result = tools_map["callable_tool"]("arg1", "arg2")
    assert "args=('arg1', 'arg2')" in result

    # Test with keyword args
    result = tools_map["callable_tool"](key1="val1", key2="val2")
    assert "kwargs=" in result and "key1" in result


def test_build_tools_map_normalization():
    """Test that tool results are normalized (e.g., stream factories are called)."""

    def stream_factory():
        class MockStream:
            def read(self):
                return b"test content"

        return MockStream()

    def tool_returning_factory():
        return stream_factory

    tool = MockTool("stream_tool", func=tool_returning_factory)
    tools_map = build_tools_map([tool])

    result = tools_map["stream_tool"]()
    # The normalize_tool_result should handle stream factories
    # For now, just verify it doesn't crash
    assert result is not None
