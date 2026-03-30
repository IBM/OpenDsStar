"""Tests for ToolRegistry."""

from unittest.mock import Mock

import pytest

from OpenDsStar.experiments.utils.tool_registry import ToolRegistry, _RenamedTool


class TestToolRegistry:
    """Test ToolRegistry class."""

    def test_name_collision_adds_namespace_prefix(self):
        """Test that name collisions are resolved by adding namespace prefix."""
        from langchain_core.tools import BaseTool
        from pydantic import BaseModel, Field

        class TestInput(BaseModel):
            query: str = Field(description="Test query")

        class Tool1(BaseTool):
            name: str = "duplicate_name"
            description: str = "Tool 1"
            args_schema: type[BaseModel] = TestInput  # type: ignore

            def _run(self, query: str) -> str:
                return "result1"

        class Tool2(BaseTool):
            name: str = "duplicate_name"
            description: str = "Tool 2"
            args_schema: type[BaseModel] = TestInput  # type: ignore

            def _run(self, query: str) -> str:
                return "result2"

        registry = ToolRegistry()

        # Add first tool - keeps original name
        registry.add_all([Tool1()], namespace="namespace1")

        # Add second tool with same name - should be renamed with namespace prefix
        registry.add_all([Tool2()], namespace="namespace2")

        tools = registry.list()
        assert len(tools) == 2

        tool_names = [t.name for t in tools]
        assert "duplicate_name" in tool_names  # First tool keeps original name
        assert "namespace2.duplicate_name" in tool_names  # Second tool gets prefix

    def test_tool_without_name_raises_error(self):
        """Test that tool without name attribute raises ValueError."""
        registry = ToolRegistry()

        tool = Mock()
        tool.name = None  # No name

        with pytest.raises(ValueError, match="Tool missing .name"):
            registry.add_all([tool], namespace="test")

    def test_double_collision_raises_error(self):
        """Test that collision even after namespacing raises error."""
        from langchain_core.tools import BaseTool
        from pydantic import BaseModel, Field

        class TestInput(BaseModel):
            query: str = Field(description="Test query")

        class Tool1(BaseTool):
            name: str = "tool"
            description: str = "Tool 1"
            args_schema: type[BaseModel] = TestInput  # type: ignore

            def _run(self, query: str) -> str:
                return "result1"

        class Tool2(BaseTool):
            name: str = "tool"
            description: str = "Tool 2"
            args_schema: type[BaseModel] = TestInput  # type: ignore

            def _run(self, query: str) -> str:
                return "result2"

        class Tool3(BaseTool):
            name: str = "tool"
            description: str = "Tool 3"
            args_schema: type[BaseModel] = TestInput  # type: ignore

            def _run(self, query: str) -> str:
                return "result3"

        # Create a tool that will occupy the namespaced name
        class Tool4(BaseTool):
            name: str = "namespace3.tool"  # This will block Tool3's namespaced name
            description: str = "Tool 4"
            args_schema: type[BaseModel] = TestInput  # type: ignore

            def _run(self, query: str) -> str:
                return "result4"

        registry = ToolRegistry()
        registry.add_all([Tool1()], namespace="namespace1")  # Adds as "tool"
        registry.add_all(
            [Tool2()], namespace="namespace2"
        )  # Renamed to "namespace2.tool"
        registry.add_all([Tool4()], namespace="other")  # Adds as "namespace3.tool"

        # This should raise because "namespace3.tool" already exists
        with pytest.raises(
            ValueError, match="Tool name collision even after namespacing"
        ):
            registry.add_all([Tool3()], namespace="namespace3")

    def test_list_returns_copy(self):
        """Test that list() returns a new list each time (prevents external mutation)."""
        registry = ToolRegistry()

        tool = Mock()
        tool.name = "test_tool"
        tool.description = "Test"

        registry.add_all([tool], namespace="test")

        list1 = registry.list()
        list2 = registry.list()

        # Should be different list instances
        assert list1 is not list2
        # But contain same tools
        assert list1 == list2


class TestRenamedToolBehavior:
    """Test _RenamedTool behavior."""

    def test_renamed_tool_preserves_functionality(self):
        """Test that renamed tools preserve original functionality (critical for correctness)."""
        from langchain_core.tools import BaseTool
        from pydantic import BaseModel, Field

        class TestInput(BaseModel):
            query: str = Field(description="Test query")

        class TestTool(BaseTool):
            name: str = "test"
            description: str = "Test tool"
            args_schema: type[BaseModel] = TestInput

            def _run(self, query: str) -> str:
                return f"Result: {query}"

        tool = TestTool()
        renamed = _RenamedTool(inner=tool, name="renamed_test")

        # Name should be changed
        assert renamed.name == "renamed_test"
        # Description should be preserved
        assert renamed.description == "Test tool"
        # Functionality should be preserved
        result = renamed._run(query="test")
        assert result == "Result: test"


class TestToolRegistryIntegration:
    """Test ToolRegistry integration scenarios."""

    def test_realistic_multi_builder_workflow(self):
        """Test realistic workflow with multiple tool builders and collision handling."""
        from langchain_core.tools import BaseTool
        from pydantic import BaseModel, Field

        class TestInput(BaseModel):
            query: str = Field(description="Test query")

        class SearchTool1(BaseTool):
            name: str = "search"
            description: str = "Search tool"
            args_schema: type[BaseModel] = TestInput  # type: ignore

            def _run(self, query: str) -> str:
                return "search1"

        class CalculatorTool(BaseTool):
            name: str = "calculator"
            description: str = "Calculator tool"
            args_schema: type[BaseModel] = TestInput  # type: ignore

            def _run(self, query: str) -> str:
                return "calc"

        class SearchTool2(BaseTool):
            name: str = "search"  # Collision with SearchTool1!
            description: str = "Different search"
            args_schema: type[BaseModel] = TestInput  # type: ignore

            def _run(self, query: str) -> str:
                return "search2"

        class DatabaseTool(BaseTool):
            name: str = "database"
            description: str = "Database tool"
            args_schema: type[BaseModel] = TestInput  # type: ignore

            def _run(self, query: str) -> str:
                return "db"

        registry = ToolRegistry()

        # Builder 1 adds tools
        builder1_tools = [SearchTool1(), CalculatorTool()]
        registry.add_all(builder1_tools, namespace="builder1")

        # Builder 2 adds tools (with one collision)
        builder2_tools = [SearchTool2(), DatabaseTool()]
        registry.add_all(builder2_tools, namespace="builder2")

        tools = registry.list()
        assert len(tools) == 4

        tool_names = [t.name for t in tools]
        assert "search" in tool_names  # Original from builder1
        assert "builder2.search" in tool_names  # Renamed from builder2
        assert "calculator" in tool_names
        assert "database" in tool_names

    def test_three_way_collision_resolution(self):
        """Test handling three tools with same name from different namespaces."""
        from langchain_core.tools import BaseTool
        from pydantic import BaseModel, Field

        class TestInput(BaseModel):
            query: str = Field(description="Test query")

        class CommonTool1(BaseTool):
            name: str = "common"
            description: str = "Tool 1"
            args_schema: type[BaseModel] = TestInput  # type: ignore

            def _run(self, query: str) -> str:
                return "tool1"

        class CommonTool2(BaseTool):
            name: str = "common"
            description: str = "Tool 2"
            args_schema: type[BaseModel] = TestInput  # type: ignore

            def _run(self, query: str) -> str:
                return "tool2"

        class CommonTool3(BaseTool):
            name: str = "common"
            description: str = "Tool 3"
            args_schema: type[BaseModel] = TestInput  # type: ignore

            def _run(self, query: str) -> str:
                return "tool3"

        registry = ToolRegistry()
        registry.add_all([CommonTool1()], namespace="ns1")
        registry.add_all([CommonTool2()], namespace="ns2")
        registry.add_all([CommonTool3()], namespace="ns3")

        tools = registry.list()
        assert len(tools) == 3

        tool_names = [t.name for t in tools]
        assert "common" in tool_names  # First keeps original
        assert "ns2.common" in tool_names  # Second gets prefix
        assert "ns3.common" in tool_names  # Third gets prefix

    def test_renamed_tool_preserves_functionality_through_registry(self):
        """Test that tools renamed by registry still work correctly."""
        from langchain_core.tools import BaseTool
        from pydantic import BaseModel, Field

        class TestInput(BaseModel):
            query: str = Field(description="Test query")

        class OriginalTool(BaseTool):
            name: str = "original"
            description: str = "Original tool"
            args_schema: type[BaseModel] = TestInput  # type: ignore

            def _run(self, query: str) -> str:
                return "original_result"

        class DuplicateTool(BaseTool):
            name: str = "original"
            description: str = "Duplicate tool"
            args_schema: type[BaseModel] = TestInput  # type: ignore

            def _run(self, query: str) -> str:
                return "duplicate_result"

        registry = ToolRegistry()
        registry.add_all([OriginalTool()], namespace="ns1")
        registry.add_all([DuplicateTool()], namespace="ns2")

        tools = registry.list()

        # Find the renamed tool
        renamed = next(t for t in tools if t.name == "ns2.original")

        # Test that it still works correctly
        result = renamed._run(query="test")
        assert result == "duplicate_result"
