"""Tests for DebuggerNode prompt building - non-trivial functionality only."""

from unittest.mock import Mock, patch

import pytest

from OpenDsStar.agents.ds_star.ds_star_state import CodeMode, DSState, DSStep
from OpenDsStar.agents.ds_star.nodes.debugger import (
    CodeOutput,
    DebuggerNode,
    _build_debugger_prompt,
)


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = Mock()
    llm.with_structured_output = Mock(return_value=llm)
    return llm


@pytest.fixture
def debugger_node(mock_llm):
    """Create a DebuggerNode instance for testing."""
    return DebuggerNode(
        system_prompt="test system",
        task_prompt="test task",
        tools_spec='[{"name": "test_tool", "description": "test"}]',
        llm=mock_llm,
    )


class TestDebuggerPromptModeSpecificBehavior:
    """Test mode-specific prompt building in debugger."""

    def test_stepwise_mode_includes_all_step_logs(self):
        """Test STEPWISE mode includes logs from ALL steps, not just last."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    code="code0",
                    logs="Step 0 logs: loaded data",
                    outputs={"data": [1, 2, 3]},
                ),
                DSStep(
                    plan="step 1",
                    code="code1",
                    logs="Step 1 logs: processed data",
                    outputs={"result": 42},
                ),
                DSStep(
                    plan="step 2",
                    code="failing_code",
                    execution_error="NameError: undefined variable",
                    logs="Step 2 logs: attempted calculation",
                ),
            ],
            code_mode=CodeMode.STEPWISE,
        )
        tools_spec = "[]"

        system_msg, user_msg = _build_debugger_prompt(state, tools_spec)
        full_prompt = system_msg + "\n" + user_msg

        # Should include logs from ALL steps in STEPWISE mode
        assert "Step 0 logs:" in full_prompt
        assert "loaded data" in full_prompt
        assert "Step 1 logs:" in full_prompt
        assert "processed data" in full_prompt
        assert "Step 2 logs:" in full_prompt
        assert "attempted calculation" in full_prompt
        assert "MODE: STEPWISE" in full_prompt

    def test_full_mode_includes_only_last_step_logs(self):
        """Test FULL mode includes logs from LAST step only."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    code="code0",
                    logs="Step 0 logs: should not appear",
                    outputs={"data": [1, 2, 3]},
                ),
                DSStep(
                    plan="step 1",
                    code="failing_code",
                    execution_error="ValueError: invalid input",
                    logs="Step 1 logs: this should appear",
                ),
            ],
            code_mode=CodeMode.FULL,
        )
        tools_spec = "[]"

        system_msg, user_msg = _build_debugger_prompt(state, tools_spec)
        full_prompt = system_msg + "\n" + user_msg

        # Should only include last step logs in FULL mode
        assert "this should appear" in full_prompt
        assert "should not appear" not in full_prompt
        assert "MODE: FULL SCRIPT" in full_prompt

    def test_stepwise_mode_includes_all_step_outputs(self):
        """Test STEPWISE mode includes outputs from ALL steps."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    code="code0",
                    outputs={"data": [1, 2, 3], "count": 3},
                ),
                DSStep(
                    plan="step 1",
                    code="code1",
                    outputs={"processed": [2, 4, 6]},
                ),
                DSStep(
                    plan="step 2",
                    code="failing_code",
                    execution_error="IndexError: list index out of range",
                ),
            ],
            code_mode=CodeMode.STEPWISE,
        )
        tools_spec = "[]"

        system_msg, user_msg = _build_debugger_prompt(state, tools_spec)
        full_prompt = system_msg + "\n" + user_msg

        # Should include outputs from all steps
        assert "Step 0 outputs:" in full_prompt
        assert "data = [1, 2, 3]" in full_prompt
        assert "count = 3" in full_prompt
        assert "Step 1 outputs:" in full_prompt
        assert "processed = [2, 4, 6]" in full_prompt

    def test_full_mode_includes_only_last_step_outputs(self):
        """Test FULL mode includes outputs from LAST step only."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    code="code0",
                    outputs={"old_data": "should not appear"},
                ),
                DSStep(
                    plan="step 1",
                    code="failing_code",
                    execution_error="TypeError: unsupported operand",
                    outputs={"new_data": "should appear"},
                ),
            ],
            code_mode=CodeMode.FULL,
        )
        tools_spec = "[]"

        system_msg, user_msg = _build_debugger_prompt(state, tools_spec)
        full_prompt = system_msg + "\n" + user_msg

        # Should only include last step outputs
        assert "should appear" in full_prompt
        assert "should not appear" not in full_prompt

    def test_stepwise_mentions_prev_step_outputs(self):
        """Test STEPWISE mode mentions prev_step_outputs availability."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    code="code0",
                    outputs={"data": [1, 2, 3]},
                ),
                DSStep(
                    plan="step 1",
                    code="failing_code",
                    execution_error="KeyError: 'missing_key'",
                ),
            ],
            code_mode=CodeMode.STEPWISE,
        )
        tools_spec = "[]"

        system_msg, user_msg = _build_debugger_prompt(state, tools_spec)
        full_prompt = system_msg + "\n" + user_msg

        assert "prev_step_outputs" in full_prompt
        assert "Available prev_step_outputs keys:" in full_prompt
        assert "data" in full_prompt

    def test_full_mode_does_not_mention_prev_step_outputs(self):
        """Test FULL mode does NOT mention prev_step_outputs."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    code="failing_code",
                    execution_error="RuntimeError: test error",
                ),
            ],
            code_mode=CodeMode.FULL,
        )
        tools_spec = "[]"

        system_msg, user_msg = _build_debugger_prompt(state, tools_spec)
        full_prompt = system_msg + "\n" + user_msg

        assert "prev_step_outputs" not in full_prompt


class TestDebuggerPromptContextAggregation:
    """Test how debugger aggregates context from multiple steps."""

    def test_aggregates_outputs_from_multiple_steps_stepwise(self):
        """Test aggregation of outputs from multiple steps in STEPWISE mode."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0", outputs={"a": 1, "b": 2}),
                DSStep(plan="step 1", outputs={"c": 3}),
                DSStep(plan="step 2", outputs={"d": 4, "e": 5}),
                DSStep(
                    plan="step 3",
                    code="failing_code",
                    execution_error="Error",
                ),
            ],
            code_mode=CodeMode.STEPWISE,
        )
        tools_spec = "[]"

        system_msg, user_msg = _build_debugger_prompt(state, tools_spec)
        full_prompt = system_msg + "\n" + user_msg

        # All outputs should be present
        assert "a = 1" in full_prompt
        assert "b = 2" in full_prompt
        assert "c = 3" in full_prompt
        assert "d = 4" in full_prompt
        assert "e = 5" in full_prompt

    def test_handles_steps_with_no_outputs(self):
        """Test handling of steps that produced no outputs."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0"),  # No outputs
                DSStep(plan="step 1", outputs={"result": 42}),
                DSStep(
                    plan="step 2",
                    code="failing_code",
                    execution_error="Error",
                ),
            ],
            code_mode=CodeMode.STEPWISE,
        )
        tools_spec = "[]"

        system_msg, user_msg = _build_debugger_prompt(state, tools_spec)
        full_prompt = system_msg + "\n" + user_msg

        # Should handle gracefully - no crash, includes available outputs
        # Step 1 has outputs, so they should appear
        assert "result = 42" in full_prompt
        # Step 0 and 2 have no outputs, but the prompt aggregates all steps
        # The prompt shows outputs for steps that have them
        assert "Step 1 outputs:" in full_prompt

    def test_handles_steps_with_no_logs(self):
        """Test handling of steps with no logs."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0"),  # No logs
                DSStep(plan="step 1", logs="Some logs"),
                DSStep(
                    plan="step 2",
                    code="failing_code",
                    execution_error="Error",
                ),
            ],
            code_mode=CodeMode.STEPWISE,
        )
        tools_spec = "[]"

        system_msg, user_msg = _build_debugger_prompt(state, tools_spec)
        full_prompt = system_msg + "\n" + user_msg

        # Step 1 has logs, so they should appear
        assert "Some logs" in full_prompt
        assert "Step 1 logs:" in full_prompt
        # When no steps have logs, the aggregated section shows "(no logs captured)"
        # But here step 1 has logs, so we just verify the logs are included


class TestDebuggerPromptTruncation:
    """Test truncation behavior in debugger prompts."""

    def test_truncates_long_outputs(self):
        """Test that long outputs are truncated."""
        long_value = "x" * 10000
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    code="failing_code",
                    execution_error="Error",
                    outputs={"long_result": long_value},
                ),
            ],
            code_mode=CodeMode.STEPWISE,
            output_max_length=100,
        )
        tools_spec = "[]"

        system_msg, user_msg = _build_debugger_prompt(state, tools_spec)
        full_prompt = system_msg + "\n" + user_msg

        # Should be truncated
        assert "..." in full_prompt
        # Prompt should not be excessively long
        assert len(full_prompt) < 15000

    def test_truncates_long_logs(self):
        """Test that long logs are truncated."""
        long_logs = "log line\n" * 1000
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    code="failing_code",
                    execution_error="Error",
                    logs=long_logs,
                ),
            ],
            code_mode=CodeMode.STEPWISE,
            logs_max_length=200,
        )
        tools_spec = "[]"

        system_msg, user_msg = _build_debugger_prompt(state, tools_spec)
        full_prompt = system_msg + "\n" + user_msg

        # Should be truncated
        assert "..." in full_prompt
        assert len(full_prompt) < 10000


class TestDebuggerPromptInstructions:
    """Test mode-specific instructions in debugger prompts."""

    def test_stepwise_instructions_focus_on_current_step(self):
        """Test STEPWISE mode instructions focus on current step only."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0", outputs={"data": [1, 2, 3]}),
                DSStep(
                    plan="step 1",
                    code="failing_code",
                    execution_error="Error",
                ),
            ],
            code_mode=CodeMode.STEPWISE,
        )
        tools_spec = "[]"

        system_msg, user_msg = _build_debugger_prompt(state, tools_spec)
        full_prompt = system_msg + "\n" + user_msg

        assert "ONLY this step" in full_prompt
        assert "Do NOT re-implement previous steps" in full_prompt
        assert "You may read prev_step_outputs" in full_prompt

    def test_full_instructions_focus_on_entire_plan(self):
        """Test FULL mode instructions focus on entire plan."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0"),
                DSStep(
                    plan="step 1",
                    code="failing_code",
                    execution_error="Error",
                ),
            ],
            code_mode=CodeMode.FULL,
        )
        tools_spec = "[]"

        system_msg, user_msg = _build_debugger_prompt(state, tools_spec)
        full_prompt = system_msg + "\n" + user_msg

        assert "full script" in full_prompt.lower()
        assert "Ensure full plan works" in full_prompt


class TestDebuggerNodeDebugTriesCounter:
    """Test debug_tries counter increments correctly."""

    @patch("OpenDsStar.agents.ds_star.nodes.debugger.invoke_structured_with_usage")
    def test_increments_debug_tries_on_each_call(self, mock_invoke, debugger_node):
        """Test that debug_tries increments on each debugger call."""
        mock_invoke.return_value = (
            CodeOutput(code="fixed_code"),
            {"input_tokens": 100, "output_tokens": 50},
        )

        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    code="failing_code",
                    execution_error="Error",
                ),
            ],
            fatal_error=None,
            steps_used=1,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        # First call
        result = debugger_node(state)
        assert result["steps"][-1].debug_tries == 1

        # Second call (simulate another debug attempt)
        result["steps"][-1].execution_error = "Another error"
        result = debugger_node(result)
        assert result["steps"][-1].debug_tries == 2

    @patch("OpenDsStar.agents.ds_star.nodes.debugger.invoke_structured_with_usage")
    def test_debug_tries_starts_at_zero(self, mock_invoke, debugger_node):
        """Test that debug_tries starts at 0 for new steps."""
        mock_invoke.return_value = (
            CodeOutput(code="fixed_code"),
            {"input_tokens": 100, "output_tokens": 50},
        )

        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    code="failing_code",
                    execution_error="Error",
                ),
            ],
            fatal_error=None,
            steps_used=1,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        # Before debugger call, debug_tries should not exist or be 0
        assert getattr(state.steps[-1], "debug_tries", 0) == 0

        result = debugger_node(state)
        assert result["steps"][-1].debug_tries == 1


class TestDebuggerPromptErrorContext:
    """Test how debugger includes error context in prompts."""

    def test_includes_execution_error_in_prompt(self):
        """Test that execution error is prominently included."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    code="result = undefined_var + 1",
                    execution_error="NameError: name 'undefined_var' is not defined",
                ),
            ],
            code_mode=CodeMode.STEPWISE,
        )
        tools_spec = "[]"

        system_msg, user_msg = _build_debugger_prompt(state, tools_spec)
        full_prompt = system_msg + "\n" + user_msg

        assert "Execution error:" in full_prompt
        assert "NameError: name 'undefined_var' is not defined" in full_prompt

    def test_includes_failing_code_in_prompt(self):
        """Test that failing code is included for context."""
        failing_code = """result = call_tool('test')
data = result['missing_key']
print(data)"""

        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    code=failing_code,
                    execution_error="KeyError: 'missing_key'",
                ),
            ],
            code_mode=CodeMode.STEPWISE,
        )
        tools_spec = "[]"

        system_msg, user_msg = _build_debugger_prompt(state, tools_spec)
        full_prompt = system_msg + "\n" + user_msg

        assert "Failing script:" in full_prompt
        assert "call_tool('test')" in full_prompt
        assert "missing_key" in full_prompt


class TestDebuggerNodeSkipConditions:
    """Test conditions under which debugger skips execution."""

    def test_skips_when_no_execution_error(self, debugger_node):
        """Test debugger skips when there's no execution error."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    code="result = 42",
                    # No execution_error
                ),
            ],
            fatal_error=None,
            steps_used=1,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = debugger_node(state)

        # Should skip and add trajectory event
        assert len(result["trajectory"]) == 1
        assert result["trajectory"][0]["skipped"] is True
        assert result["trajectory"][0]["reason"] == "no_execution_error"

    def test_fails_when_no_code_to_debug(self, debugger_node):
        """Test debugger sets fatal error when there's no code."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    # No code
                    execution_error="Some error",
                ),
            ],
            fatal_error=None,
            steps_used=1,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = debugger_node(state)

        assert result["fatal_error"] == "No code to debug."


class TestDebuggerPromptEnvironmentInfo:
    """Test environment information in debugger prompts."""

    def test_mentions_no_imports_allowed(self):
        """Test that prompt mentions no imports are allowed."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    code="failing_code",
                    execution_error="Error",
                ),
            ],
            code_mode=CodeMode.STEPWISE,
        )
        tools_spec = "[]"

        system_msg, user_msg = _build_debugger_prompt(state, tools_spec)
        full_prompt = system_msg + "\n" + user_msg

        assert "No imports" in full_prompt

    def test_mentions_preloaded_libraries(self):
        """Test that preloaded libraries are mentioned."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    code="failing_code",
                    execution_error="Error",
                ),
            ],
            code_mode=CodeMode.STEPWISE,
        )
        tools_spec = "[]"

        system_msg, user_msg = _build_debugger_prompt(state, tools_spec)
        full_prompt = system_msg + "\n" + user_msg

        assert "numpy" in full_prompt
        assert "pandas" in full_prompt

    def test_mentions_call_tool_availability(self):
        """Test that tool calling is mentioned (tools are available as functions)."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    code="failing_code",
                    execution_error="Error",
                ),
            ],
            code_mode=CodeMode.STEPWISE,
        )
        tools_spec = "[]"

        system_msg, user_msg = _build_debugger_prompt(state, tools_spec)
        full_prompt = system_msg + "\n" + user_msg

        # Tools are available as regular functions, not via call_tool()
        assert (
            "regular Python functions" in full_prompt
            or "Call them directly" in full_prompt
        )
