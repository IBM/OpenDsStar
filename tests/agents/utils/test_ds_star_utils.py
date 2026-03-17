"""Tests for ds_star_utils utility functions."""

import json
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, Field

from agents.ds_star.ds_star_state import DSState, DSStep
from agents.ds_star.ds_star_utils import (
    ToolExecutionError,
    add_event_to_trajectory,
    build_tools_map,
    format_tools_spec,
    safe_get,
    steps_to_plan_string,
)


class TestBuildToolsMap:
    """Test build_tools_map function."""

    def test_wraps_tool_with_invoke_method(self):
        """Test wrapping tool with .invoke method."""
        tool = Mock(spec=["name", "invoke"])
        tool.name = "test_tool"
        tool.invoke = Mock(return_value="result")

        tools_map = build_tools_map([tool])

        assert "test_tool" in tools_map
        result = tools_map["test_tool"](arg1="value1")
        tool.invoke.assert_called_once_with({"arg1": "value1"})
        assert result == "result"

    def test_wraps_tool_with_func_attribute(self):
        """Test wrapping tool with .func attribute."""

        def my_func(**kwargs):
            return f"called with {kwargs}"

        tool = Mock(spec=["name", "func"])
        tool.name = "test_tool"
        tool.func = my_func

        tools_map = build_tools_map([tool])

        assert "test_tool" in tools_map
        result = tools_map["test_tool"](arg1="value1")
        assert "called with" in result
        assert "arg1" in result

    def test_wraps_callable_tool(self):
        """Test wrapping callable tool."""

        class CallableTool:
            name = "test_tool"

            def __call__(self, **kwargs):
                return f"result: {kwargs}"

        my_tool = CallableTool()

        tools_map = build_tools_map([my_tool])

        assert "test_tool" in tools_map
        result = tools_map["test_tool"](arg1="value1")
        assert "result:" in result

    def test_raises_error_for_non_callable_tool(self):
        """Test that non-callable tool raises ToolExecutionError."""

        class NonCallableTool:
            name = "test_tool"
            invoke = None
            func = None

        tool = NonCallableTool()

        tools_map = build_tools_map([tool])

        with pytest.raises(ToolExecutionError, match="not callable"):
            tools_map["test_tool"]()

    def test_propagates_tool_execution_error(self):
        """Test that tool execution errors are wrapped in ToolExecutionError."""
        tool = Mock(spec=["name", "invoke"])
        tool.name = "test_tool"
        tool.invoke = Mock(side_effect=ValueError("Tool failed"))

        tools_map = build_tools_map([tool])

        with pytest.raises(ToolExecutionError, match="execution failed"):
            tools_map["test_tool"](arg1="value1")

    def test_uses_class_name_when_no_name_attribute(self):
        """Test using class name when tool has no name attribute."""

        class MyTool:
            def invoke(self, kwargs):
                return "result"

        tool = MyTool()

        tools_map = build_tools_map([tool])

        assert "MyTool" in tools_map

    def test_handles_multiple_tools(self):
        """Test handling multiple tools."""
        tool1 = Mock(spec=["name", "invoke"])
        tool1.name = "tool1"
        tool1.invoke = Mock(return_value="result1")

        tool2 = Mock(spec=["name", "invoke"])
        tool2.name = "tool2"
        tool2.invoke = Mock(return_value="result2")

        tools_map = build_tools_map([tool1, tool2])

        assert "tool1" in tools_map
        assert "tool2" in tools_map
        assert tools_map["tool1"]() == "result1"
        assert tools_map["tool2"]() == "result2"


class TestFormatToolsSpec:
    """Test format_tools_spec function."""

    def test_formats_tool_with_args_schema(self):
        """Tool with args_schema should include params with its fields."""

        class ToolInput(BaseModel):
            query: str = Field(description="Search query")
            limit: int = Field(default=10, description="Result limit")

        tool = Mock(spec=["name", "description", "args_schema"])
        tool.name = "search_tool"
        tool.description = "Search for information"
        tool.args_schema = ToolInput

        spec = json.loads(format_tools_spec([tool]))

        assert spec[0]["name"] == "search_tool"
        assert spec[0]["description"] == "Search for information"
        assert "query" in spec[0]["params"]
        assert "limit" in spec[0]["params"]

    def test_handles_tool_with_model_json_schema(self):
        """Test handling tool with model_json_schema method."""

        class ToolInput(BaseModel):
            text: str = Field(description="Input text")

        tool = Mock(spec=["name", "description", "args_schema"])
        tool.name = "process_tool"
        tool.description = "Process text"
        tool.args_schema = ToolInput

        spec = json.loads(format_tools_spec([tool]))

        assert spec[0]["name"] == "process_tool"
        assert "text" in spec[0]["params"]

    def test_supports_tools_without_params(self):
        """Tools with no args_schema (or no detectable params) should serialize with empty params."""
        tool = Mock(spec=["name", "description", "args_schema"])
        tool.name = "no_param_tool"
        tool.description = "A tool with no parameters"
        tool.args_schema = None

        spec = json.loads(format_tools_spec([tool]))

        assert spec == [
            {
                "name": "no_param_tool",
                "description": "A tool with no parameters",
                "params": {},
            }
        ]

    def test_handles_empty_description(self):
        """Test handling tool with empty description."""

        class ToolInput(BaseModel):
            arg: str

        tool = Mock(spec=["name", "description", "args_schema"])
        tool.name = "test_tool"
        tool.description = ""
        tool.args_schema = ToolInput

        spec = json.loads(format_tools_spec([tool]))

        assert spec[0]["name"] == "test_tool"
        assert spec[0]["description"] == ""
        assert "arg" in spec[0]["params"]

    def test_formats_multiple_tools(self):
        """Test formatting multiple tools."""

        class Tool1Input(BaseModel):
            query: str

        class Tool2Input(BaseModel):
            data: str

        tool1 = Mock(spec=["name", "description", "args_schema"])
        tool1.name = "tool1"
        tool1.description = "First tool"
        tool1.args_schema = Tool1Input

        tool2 = Mock(spec=["name", "description", "args_schema"])
        tool2.name = "tool2"
        tool2.description = "Second tool"
        tool2.args_schema = Tool2Input

        spec = json.loads(format_tools_spec([tool1, tool2]))

        assert [t["name"] for t in spec] == ["tool1", "tool2"]
        assert "query" in spec[0]["params"]
        assert "data" in spec[1]["params"]

    def test_output_is_valid_json(self):
        """Test that output is valid JSON."""

        class ToolInput(BaseModel):
            arg: str

        tool = Mock(spec=["name", "description", "args_schema"])
        tool.name = "test_tool"
        tool.description = "Test"
        tool.args_schema = ToolInput

        spec = format_tools_spec([tool])

        parsed = json.loads(spec)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["name"] == "test_tool"
        assert isinstance(parsed[0]["params"], dict)


class TestStepsToPlanString:
    """Test steps_to_plan_string function."""

    def test_formats_steps_with_plans(self):
        """Test formatting steps with plans."""
        steps = [
            DSStep(plan="First step"),
            DSStep(plan="Second step"),
            DSStep(plan="Third step"),
        ]

        result = steps_to_plan_string(steps)

        assert "Plan:" in result
        assert "0: First step" in result
        assert "1: Second step" in result
        assert "2: Third step" in result

    def test_handles_empty_steps(self):
        """Test handling empty steps list."""
        result = steps_to_plan_string([])

        assert "Plan:" in result
        assert "<plan is empty>" in result

    def test_handles_steps_without_plan(self):
        """Test handling steps without plan attribute."""

        class FakeStep:
            pass

        steps = [FakeStep(), FakeStep()]

        result = steps_to_plan_string(steps)

        assert "Plan:" in result
        assert "<plan is empty>" in result

    def test_handles_none_plan(self):
        """Test handling steps with None plan."""
        steps = [DSStep(plan="Step 1"), Mock(plan=None)]

        result = steps_to_plan_string(steps)

        assert "0: Step 1" in result
        # Step with None plan should be skipped


class TestAddEventToTrajectory:
    """Test add_event_to_trajectory function."""

    def test_adds_event_with_basic_info(self):
        """Test adding event with basic information."""
        state = DSState(
            user_query="test",
            tools={},
            output_max_length=1000,
            logs_max_length=1000,
            steps=[DSStep(plan="step 0")],
            steps_used=1,
        )

        add_event_to_trajectory(state, "test_node", action="test_action")

        assert len(state.trajectory) == 1
        event = state.trajectory[0]
        assert event["node"] == "test_node"
        assert event["action"] == "test_action"
        assert event["steps_used"] == 1
        assert "time" in event

    def test_adds_multiple_events(self):
        """Test adding multiple events."""
        state = DSState(
            user_query="test",
            tools={},
            output_max_length=1000,
            logs_max_length=1000,
            steps=[],
            steps_used=0,
        )

        add_event_to_trajectory(state, "node1", data="data1")
        add_event_to_trajectory(state, "node2", data="data2")

        assert len(state.trajectory) == 2
        assert state.trajectory[0]["node"] == "node1"
        assert state.trajectory[1]["node"] == "node2"

    def test_includes_last_step(self):
        """Test that last step is included in event as a dict."""
        state = DSState(
            user_query="test",
            tools={},
            output_max_length=1000,
            logs_max_length=1000,
            steps=[DSStep(plan="step 0"), DSStep(plan="step 1")],
            steps_used=2,
        )

        add_event_to_trajectory(state, "test_node")

        event = state.trajectory[0]
        # last_step should be a dict (converted from DSStep)
        assert isinstance(event["last_step"], dict)
        assert event["last_step"]["plan"] == "step 1"

    def test_handles_empty_steps(self):
        """Test handling state with no steps."""
        state = DSState(
            user_query="test",
            tools={},
            output_max_length=1000,
            logs_max_length=1000,
            steps=[],
            steps_used=0,
        )

        add_event_to_trajectory(state, "test_node")

        event = state.trajectory[0]
        assert event["last_step"] is None
        assert event["step_idx"] == 0

    def test_includes_custom_fields(self):
        """Test including custom fields in event."""
        state = DSState(
            user_query="test",
            tools={},
            output_max_length=1000,
            logs_max_length=1000,
            steps=[],
            steps_used=0,
        )

        add_event_to_trajectory(
            state,
            "test_node",
            custom_field="custom_value",
            another_field=42,
        )

        event = state.trajectory[0]
        assert event["custom_field"] == "custom_value"
        assert event["another_field"] == 42


class TestSafeGet:
    """Test safe_get function."""

    def test_gets_valid_index(self):
        """Test getting item at valid index."""
        lst = [1, 2, 3, 4, 5]

        assert safe_get(lst, 0) == 1
        assert safe_get(lst, 2) == 3
        assert safe_get(lst, -1) == 5

    def test_returns_default_for_out_of_bounds(self):
        """Test returning default for out of bounds index."""
        lst = [1, 2, 3]

        assert safe_get(lst, 10) is None
        assert safe_get(lst, 10, "default") == "default"
        assert safe_get(lst, -10, 0) == 0

    def test_handles_empty_list(self):
        """Test handling empty list."""
        lst = []

        assert safe_get(lst, 0) is None
        assert safe_get(lst, 0, "default") == "default"

    def test_handles_none_list(self):
        """Test handling None as list."""
        assert safe_get(None, 0) is None  # type: ignore[arg-type]
        assert safe_get(None, 0, "default") == "default"  # type: ignore[arg-type]

    def test_custom_default_value(self):
        """Test using custom default value."""
        lst = [1, 2, 3]

        assert safe_get(lst, 10, default=-1) == -1
        assert safe_get(lst, 10, default=[]) == []
        assert safe_get(lst, 10, default={}) == {}


class TestToolExecutionError:
    """Test ToolExecutionError exception."""

    def test_can_be_raised(self):
        """Test that ToolExecutionError can be raised."""
        with pytest.raises(ToolExecutionError):
            raise ToolExecutionError("Test error")

    def test_has_message(self):
        """Test that error message is preserved."""
        with pytest.raises(ToolExecutionError, match="Custom error message"):
            raise ToolExecutionError("Custom error message")

    def test_is_exception_subclass(self):
        """Test that ToolExecutionError is an Exception."""
        assert issubclass(ToolExecutionError, Exception)
