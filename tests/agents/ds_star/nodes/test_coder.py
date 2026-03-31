"""Tests for CoderNode logic and prompt building."""

from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from OpenDsStar.agents.ds_star.ds_star_state import CodeMode, DSState, DSStep
from OpenDsStar.agents.ds_star.nodes.coder import (
    CodeOutput,
    CoderNode,
    _collect_available_parameter_names,
    build_coder_prompt,
)


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = Mock()
    llm.with_structured_output = Mock(return_value=llm)
    return llm


@pytest.fixture
def coder_node(mock_llm):
    """Create a CoderNode instance for testing."""
    return CoderNode(
        system_prompt="test system",
        task_prompt="test task",
        tools_spec='[{"name": "test_tool", "description": "test"}]',
        llm=mock_llm,
    )


class TestCoderNodeErrorHandling:
    """Test error handling in CoderNode.__call__."""

    def test_skips_on_fatal_error(self, coder_node):
        """Test that coder skips when fatal_error exists."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[DSStep(plan="step 0")],
            fatal_error="Previous fatal error",
            steps_used=1,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = coder_node(state)

        assert result["fatal_error"] == "Previous fatal error"
        assert len(result["trajectory"]) == 1
        assert result["trajectory"][0]["skipped"] is True

    @patch("OpenDsStar.agents.ds_star.nodes.coder.invoke_structured_with_usage")
    def test_handles_validation_error(self, mock_invoke, coder_node):
        """Test handling of ValidationError."""
        mock_invoke.side_effect = ValidationError.from_exception_data(
            "test", [{"type": "missing", "loc": ("code",), "input": {}}]
        )

        state = DSState(
            user_query="test query",
            tools={},
            steps=[DSStep(plan="step 0")],
            fatal_error=None,
            steps_used=1,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = coder_node(state)

        assert result["fatal_error"] is not None
        assert "Coder schema validation failed" in result["fatal_error"]

    @patch("OpenDsStar.agents.ds_star.nodes.coder.invoke_structured_with_usage")
    def test_handles_general_exception(self, mock_invoke, coder_node):
        """Test handling of general Exception."""
        mock_invoke.side_effect = Exception("Test error")

        state = DSState(
            user_query="test query",
            tools={},
            steps=[DSStep(plan="step 0")],
            fatal_error=None,
            steps_used=1,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = coder_node(state)

        assert result["fatal_error"] is not None
        assert "Coder invocation failed" in result["fatal_error"]


class TestBuildCoderPrompt:
    """Test build_coder_prompt function."""

    def test_stepwise_mode_uses_prev_step_outputs(self):
        """Test STEPWISE mode mentions prev_step_outputs."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0", outputs={"result": "value"}),
                DSStep(plan="step 1"),
            ],
            code_mode=CodeMode.STEPWISE,
        )
        tools_spec = "[]"

        system_msg, user_msg = build_coder_prompt(state, tools_spec)
        full_prompt = system_msg + "\n" + user_msg

        assert "STEPWISE CODE GENERATION" in full_prompt
        assert "prev_step_outputs" in full_prompt
        assert "result" in full_prompt
        assert "ONLY the current step" in full_prompt

    def test_full_mode_generates_complete_script(self):
        """Test FULL mode generates complete script."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0"),
                DSStep(plan="step 1"),
            ],
            code_mode=CodeMode.FULL,
        )
        tools_spec = "[]"

        system_msg, user_msg = build_coder_prompt(state, tools_spec)
        full_prompt = system_msg + "\n" + user_msg

        assert "FULL-SCRIPT GENERATION" in full_prompt
        assert "ENTIRE PLAN" in full_prompt
        assert "Do NOT rely on prev_step_outputs" in full_prompt

    def test_includes_user_query_and_plan(self):
        """Test that prompt includes user query and plan."""
        state = DSState(
            user_query="What is the capital of France?",
            tools={},
            steps=[
                DSStep(plan="Search for France capital"),
                DSStep(plan="Format the answer"),
            ],
            code_mode=CodeMode.STEPWISE,
        )
        tools_spec = '[{"name": "search", "description": "Search tool"}]'

        system_msg, user_msg = build_coder_prompt(state, tools_spec)
        full_prompt = system_msg + "\n" + user_msg

        assert "What is the capital of France?" in full_prompt
        assert "0. Search for France capital" in full_prompt
        assert "1. Format the answer" in full_prompt
        assert "search" in full_prompt

    def test_includes_previous_code_when_exists(self):
        """Test that previous code is included when it exists."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0", code="previous_code = 'test'"),
            ],
            code_mode=CodeMode.STEPWISE,
        )
        tools_spec = "[]"

        system_msg, user_msg = build_coder_prompt(state, tools_spec)
        full_prompt = system_msg + "\n" + user_msg

        assert "Previous script:" in full_prompt
        assert "previous_code = 'test'" in full_prompt
        assert "Refine the previous script" in full_prompt

    def test_no_previous_code_when_none(self):
        """Test handling when no previous code exists."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[DSStep(plan="step 0")],
            code_mode=CodeMode.STEPWISE,
        )
        tools_spec = "[]"

        system_msg, user_msg = build_coder_prompt(state, tools_spec)
        full_prompt = system_msg + "\n" + user_msg

        assert "(None)" in full_prompt
        assert "Generate a NEW script" in full_prompt

    def test_stepwise_collects_parameters_from_previous_steps(self):
        """Test STEPWISE mode collects parameters from previous steps."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0", outputs={"data": [1, 2, 3]}),
                DSStep(plan="step 1", outputs={"result": 42}),
                DSStep(plan="step 2"),
            ],
            code_mode=CodeMode.STEPWISE,
        )
        tools_spec = "[]"

        system_msg, user_msg = build_coder_prompt(state, tools_spec)
        full_prompt = system_msg + "\n" + user_msg

        assert "data" in full_prompt
        assert "result" in full_prompt
        assert "Available parameters from previous steps" in full_prompt


class TestCollectAvailableParameterNames:
    """Test _collect_available_parameter_names function."""

    def test_collects_from_outputs(self):
        """Test collecting parameter names from outputs."""
        state = DSState(
            user_query="test",
            tools={},
            steps=[
                DSStep(plan="step 0", outputs={"a": 1, "b": 2}),
                DSStep(plan="step 1", outputs={"c": 3}),
                DSStep(plan="step 2"),  # Current step
            ],
        )

        result = _collect_available_parameter_names(state)

        assert "a" in result
        assert "b" in result
        assert "c" in result

    def test_collects_from_available_params(self):
        """Test collecting from available_params attribute."""
        step0 = DSStep(plan="step 0")
        setattr(step0, "available_params", ["x", "y"])
        step1 = DSStep(plan="step 1")

        state = DSState(
            user_query="test",
            tools={},
            steps=[step0, step1],
        )

        result = _collect_available_parameter_names(state)

        assert "x" in result
        assert "y" in result

    def test_returns_none_when_no_parameters(self):
        """Test returns 'None' when no parameters available."""
        state = DSState(
            user_query="test",
            tools={},
            steps=[DSStep(plan="step 0")],
        )

        result = _collect_available_parameter_names(state)

        assert result == "None"

    def test_excludes_current_step(self):
        """Test that current step (last) is excluded."""
        state = DSState(
            user_query="test",
            tools={},
            steps=[
                DSStep(plan="step 0", outputs={"a": 1}),
                DSStep(plan="step 1", outputs={"b": 2}),  # Current step
            ],
        )

        result = _collect_available_parameter_names(state)

        assert "a" in result
        assert "b" not in result

    def test_handles_exception_gracefully(self):
        """Test that exceptions in parameter collection are handled."""
        step0 = DSStep(plan="step 0")
        setattr(step0, "available_params", "not_a_list")  # Invalid type

        state = DSState(
            user_query="test",
            tools={},
            steps=[step0, DSStep(plan="step 1")],
        )

        # Should not raise exception
        result = _collect_available_parameter_names(state)
        assert isinstance(result, str)


class TestCodeOutput:
    """Test CodeOutput model."""

    def test_valid_code_output(self):
        """Test creating valid code output."""
        output = CodeOutput(code="result = call_tool('test')")

        assert output.code == "result = call_tool('test')"

    def test_code_can_be_multiline(self):
        """Test that code can be multiline."""
        code = """result = call_tool('test')
print(result)
outputs['result'] = result"""

        output = CodeOutput(code=code)

        assert "call_tool" in output.code
        assert "print" in output.code
        assert "outputs" in output.code


class TestCoderPromptEnvironmentBlock:
    """Test that coder prompt includes correct environment information."""

    def test_stepwise_mentions_no_imports(self):
        """Test STEPWISE mode mentions no imports allowed."""
        state = DSState(
            user_query="test",
            tools={},
            steps=[DSStep(plan="step 0")],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_coder_prompt(state, "[]")
        full_prompt = system_msg + "\n" + user_msg

        assert "No imports allowed" in full_prompt
        assert "import" in full_prompt.lower()

    def test_full_mentions_no_imports(self):
        """Test FULL mode mentions no imports allowed."""
        state = DSState(
            user_query="test",
            tools={},
            steps=[DSStep(plan="step 0")],
            code_mode=CodeMode.FULL,
        )

        system_msg, user_msg = build_coder_prompt(state, "[]")
        full_prompt = system_msg + "\n" + user_msg

        assert "No imports allowed" in full_prompt

    def test_mentions_preloaded_libraries(self):
        """Test that preloaded libraries are mentioned."""
        state = DSState(
            user_query="test",
            tools={},
            steps=[DSStep(plan="step 0")],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_coder_prompt(state, "[]")
        full_prompt = system_msg + "\n" + user_msg

        assert "numpy" in full_prompt
        assert "pandas" in full_prompt
        assert "scipy" in full_prompt

    def test_mentions_call_tool(self):
        """Test that tool calling is mentioned (tools are available as functions)."""
        state = DSState(
            user_query="test",
            tools={},
            steps=[DSStep(plan="step 0")],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_coder_prompt(state, "[]")
        full_prompt = system_msg + "\n" + user_msg

        # Tools are available as regular functions, not via call_tool()
        assert (
            "regular Python functions" in full_prompt
            or "Call them directly" in full_prompt
        )
