"""Tests for PlannerNode logic and prompt building."""

from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from agents.ds_star.ds_star_state import CodeMode, DSState, DSStep
from agents.ds_star.nodes.planner import (
    PlannerNode,
    PlanOneStepOutput,
    build_planner_prompt,
    summarize_last_outputs_for_planner,
    summarize_step_for_planner,
)


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = Mock()
    llm.with_structured_output = Mock(return_value=llm)
    return llm


@pytest.fixture
def planner_node(mock_llm):
    """Create a PlannerNode instance for testing."""
    return PlannerNode(
        system_prompt="test system",
        task_prompt="test task",
        tools_spec='[{"name": "test_tool", "description": "test"}]',
        llm=mock_llm,
    )


class TestPlannerNodeStepReplacement:
    """Test step replacement logic in PlannerNode.__call__."""

    @patch("agents.ds_star.nodes.planner.invoke_structured_with_usage")
    def test_replaces_step_at_valid_fix_index(self, mock_invoke, planner_node):
        """Test replacing a step at valid fix_index."""
        mock_invoke.return_value = (
            PlanOneStepOutput(step="corrected step"),
            {"input_tokens": 100, "output_tokens": 50},
        )

        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0"),
                DSStep(plan="step 1"),
                DSStep(plan="step 2", router_action="fix_step", router_fix_index=1),
            ],
            fatal_error=None,
            steps_used=2,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = planner_node(state)

        # Step 1 should be replaced, step 2 should be removed
        assert len(result["steps"]) == 2
        assert result["steps"][0].plan == "step 0"
        assert result["steps"][1].plan == "corrected step"
        assert result["steps_used"] == 3

    @patch("agents.ds_star.nodes.planner.invoke_structured_with_usage")
    def test_fix_index_out_of_bounds_negative(self, mock_invoke, planner_node):
        """Test fix_index that is negative (clamped to 0)."""
        mock_invoke.return_value = (
            PlanOneStepOutput(step="new step"),
            {"input_tokens": 100, "output_tokens": 50},
        )

        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0"),
                DSStep(plan="step 1", router_action="fix_step", router_fix_index=-1),
            ],
            fatal_error=None,
            steps_used=1,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = planner_node(state)

        # Should replace step 0 (clamped from -1 to 0) and remove step 1
        assert len(result["steps"]) == 1
        assert result["steps"][0].plan == "new step"

    @patch("agents.ds_star.nodes.planner.invoke_structured_with_usage")
    def test_fix_index_out_of_bounds_too_large(self, mock_invoke, planner_node):
        """Test fix_index >= len(steps) (clamped to last step)."""
        mock_invoke.return_value = (
            PlanOneStepOutput(step="new step"),
            {"input_tokens": 100, "output_tokens": 50},
        )

        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0"),
                DSStep(plan="step 1", router_action="fix_step", router_fix_index=5),
            ],
            fatal_error=None,
            steps_used=1,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = planner_node(state)

        # Should replace step 1 (clamped from 5 to 1)
        assert len(result["steps"]) == 2
        assert result["steps"][1].plan == "new step"

    @patch("agents.ds_star.nodes.planner.invoke_structured_with_usage")
    def test_truncates_steps_after_replaced_step(self, mock_invoke, planner_node):
        """Test that steps after replaced step are removed."""
        mock_invoke.return_value = (
            PlanOneStepOutput(step="corrected step"),
            {"input_tokens": 100, "output_tokens": 50},
        )

        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0"),
                DSStep(plan="step 1"),
                DSStep(plan="step 2"),
                DSStep(plan="step 3", router_action="fix_step", router_fix_index=1),
            ],
            fatal_error=None,
            steps_used=3,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = planner_node(state)

        # Steps 2 and 3 should be removed, step 1 replaced
        assert len(result["steps"]) == 2
        assert result["steps"][0].plan == "step 0"
        assert result["steps"][1].plan == "corrected step"

    @patch("agents.ds_star.nodes.planner.invoke_structured_with_usage")
    def test_appends_when_no_router_action(self, mock_invoke, planner_node):
        """Test appending when no router_action."""
        mock_invoke.return_value = (
            PlanOneStepOutput(step="new step"),
            {"input_tokens": 100, "output_tokens": 50},
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

        result = planner_node(state)

        assert len(result["steps"]) == 2
        assert result["steps"][1].plan == "new step"

    @patch("agents.ds_star.nodes.planner.invoke_structured_with_usage")
    def test_appends_when_add_next_step(self, mock_invoke, planner_node):
        """Test appending when router_action is add_next_step."""
        mock_invoke.return_value = (
            PlanOneStepOutput(step="new step"),
            {"input_tokens": 100, "output_tokens": 50},
        )

        state = DSState(
            user_query="test query",
            tools={},
            steps=[DSStep(plan="step 0", router_action="add_next_step")],
            fatal_error=None,
            steps_used=1,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = planner_node(state)

        assert len(result["steps"]) == 2
        assert result["steps"][1].plan == "new step"

    @patch("agents.ds_star.nodes.planner.invoke_structured_with_usage")
    def test_steps_used_increments(self, mock_invoke, planner_node):
        """Test that steps_used counter increments correctly."""
        mock_invoke.return_value = (
            PlanOneStepOutput(step="new step"),
            {"input_tokens": 100, "output_tokens": 50},
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

        result = planner_node(state)

        assert result["steps_used"] == 2


class TestPlannerNodeErrorHandling:
    """Test error handling in PlannerNode.__call__."""

    def test_skips_on_fatal_error(self, planner_node):
        """Test that planner skips when fatal_error exists."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[],
            fatal_error="Previous fatal error",
            steps_used=0,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = planner_node(state)

        assert result["fatal_error"] == "Previous fatal error"
        assert len(result["trajectory"]) == 1
        assert result["trajectory"][0]["skipped"] is True

    @patch("agents.ds_star.nodes.planner.invoke_structured_with_usage")
    def test_handles_validation_error(self, mock_invoke, planner_node):
        """Test handling of ValidationError."""
        mock_invoke.side_effect = ValidationError.from_exception_data(
            "test", [{"type": "missing", "loc": ("step",), "input": {}}]
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

        result = planner_node(state)

        assert result["fatal_error"] is not None
        assert "Planner schema validation failed" in result["fatal_error"]

    @patch("agents.ds_star.nodes.planner.invoke_structured_with_usage")
    def test_handles_general_exception(self, mock_invoke, planner_node):
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

        result = planner_node(state)

        assert result["fatal_error"] is not None
        assert "Planner invocation failed" in result["fatal_error"]

    @patch("agents.ds_star.nodes.planner.invoke_structured_with_usage")
    def test_handles_empty_step_text(self, mock_invoke, planner_node):
        """Test handling of empty step text."""
        mock_invoke.return_value = (
            PlanOneStepOutput(step=""),
            {"input_tokens": 100, "output_tokens": 50},
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

        result = planner_node(state)

        assert result["fatal_error"] == "Planner produced empty step."


class TestBuildPlannerPrompt:
    """Test build_planner_prompt function."""

    def test_first_step_prompt(self):
        """Test prompt for first step."""
        state = DSState(
            user_query="What is the capital of France?",
            tools={},
            steps=[],
            code_mode=CodeMode.STEPWISE,
        )
        tools_spec = '[{"name": "search", "description": "Search tool"}]'

        system_msg, user_msg = build_planner_prompt(state, tools_spec)
        full_prompt = system_msg + "\n" + user_msg

        assert "What is the capital of France?" in full_prompt
        assert "There are no previous steps yet" in full_prompt
        assert "FIRST high-level analysis step" in full_prompt
        assert "search" in full_prompt

    def test_stepwise_mode_includes_outputs(self):
        """Test STEPWISE mode includes per-step outputs in history."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0", outputs={"result": "value"}),
                DSStep(plan="step 1", router_action="add_next_step"),
            ],
            code_mode=CodeMode.STEPWISE,
        )
        tools_spec = "[]"

        system_msg, user_msg = build_planner_prompt(state, tools_spec)
        full_prompt = system_msg + "\n" + user_msg

        assert "result = 'value'" in full_prompt
        assert "Outputs:" in full_prompt

    def test_full_mode_shows_last_outputs_explicitly(self):
        """Test FULL mode shows last step outputs explicitly."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0", outputs={"old": "data"}),
                DSStep(
                    plan="step 1",
                    outputs={"new": "data"},
                    router_action="add_next_step",
                ),
            ],
            code_mode=CodeMode.FULL,
        )
        tools_spec = "[]"

        system_msg, user_msg = build_planner_prompt(state, tools_spec)
        full_prompt = system_msg + "\n" + user_msg

        assert "Latest available outputs from the last step" in full_prompt
        assert "new = 'data'" in full_prompt

    def test_max_history_steps_truncation(self):
        """Test that history is truncated to max_history_steps."""
        steps = [DSStep(plan=f"step {i}") for i in range(10)]
        steps[-1].router_action = "add_next_step"

        state = DSState(
            user_query="test query",
            tools={},
            steps=steps,
            code_mode=CodeMode.STEPWISE,
        )
        tools_spec = "[]"

        system_msg, user_msg = build_planner_prompt(
            state, tools_spec, max_history_steps=3
        )
        full_prompt = system_msg + "\n" + user_msg

        # Should only show last 3 steps
        assert "Step 7" in full_prompt
        assert "Step 8" in full_prompt
        assert "Step 9" in full_prompt
        assert "Step 0" not in full_prompt
        assert "Only the last 3 steps are shown" in full_prompt

    def test_fix_step_mode_instructions(self):
        """Test fix_step mode provides correct instructions."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0"),
                DSStep(plan="step 1", router_action="fix_step", router_fix_index=0),
            ],
            code_mode=CodeMode.STEPWISE,
        )
        tools_spec = "[]"

        system_msg, user_msg = build_planner_prompt(state, tools_spec)
        full_prompt = system_msg + "\n" + user_msg

        assert "Step 0 is incorrect or needs revision" in full_prompt
        assert "propose a NEW corrected version of Step 0" in full_prompt
        assert "Do NOT add a new step" in full_prompt

    def test_add_next_step_mode_instructions(self):
        """Test add_next_step mode provides correct instructions."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[DSStep(plan="step 0", router_action="add_next_step")],
            code_mode=CodeMode.STEPWISE,
        )
        tools_spec = "[]"

        system_msg, user_msg = build_planner_prompt(state, tools_spec)
        full_prompt = system_msg + "\n" + user_msg

        assert "continue with a NEW step" in full_prompt
        assert "NEXT step that moves closer to answering" in full_prompt


class TestSummarizeStepForPlanner:
    """Test summarize_step_for_planner function."""

    def test_includes_outputs_when_requested(self):
        """Test that outputs are included when include_outputs=True."""
        step = DSStep(
            plan="test plan",
            outputs={"result": "test value"},
        )

        summary = summarize_step_for_planner(
            step, 0, include_outputs=True, output_max_length=500, logs_max_length=1000
        )

        assert "test plan" in summary
        assert "result = 'test value'" in summary

    def test_excludes_outputs_when_not_requested(self):
        """Test that outputs are excluded when include_outputs=False."""
        step = DSStep(
            plan="test plan",
            outputs={"result": "test value"},
        )

        summary = summarize_step_for_planner(
            step, 0, include_outputs=False, output_max_length=500, logs_max_length=1000
        )

        assert "test plan" in summary
        assert "result" not in summary

    def test_includes_execution_error(self):
        """Test that execution errors are included."""
        step = DSStep(
            plan="test plan",
            execution_error="Test error occurred",
        )

        summary = summarize_step_for_planner(
            step, 0, include_outputs=True, output_max_length=500, logs_max_length=1000
        )

        assert "Execution error: Test error occurred" in summary

    def test_truncates_long_outputs(self):
        """Test that long outputs are truncated."""
        long_value = "x" * 1000
        step = DSStep(
            plan="test plan",
            outputs={"result": long_value},
        )

        summary = summarize_step_for_planner(
            step, 0, include_outputs=True, output_max_length=100, logs_max_length=1000
        )

        assert "..." in summary
        assert len(summary) < 1000


class TestSummarizeLastOutputsForPlanner:
    """Test summarize_last_outputs_for_planner function."""

    def test_handles_no_steps(self):
        """Test handling of state with no steps."""
        state = DSState(user_query="test", tools={}, steps=[])

        result = summarize_last_outputs_for_planner(state, output_max_length=500)

        assert "No steps exist yet" in result

    def test_handles_no_outputs(self):
        """Test handling of last step with no outputs."""
        state = DSState(
            user_query="test",
            tools={},
            steps=[DSStep(plan="step 0")],
        )

        result = summarize_last_outputs_for_planner(state, output_max_length=500)

        assert "did not produce any outputs" in result

    def test_summarizes_last_step_outputs(self):
        """Test summarizing last step outputs."""
        state = DSState(
            user_query="test",
            tools={},
            steps=[
                DSStep(plan="step 0", outputs={"old": "data"}),
                DSStep(plan="step 1", outputs={"new": "data", "result": 42}),
            ],
        )

        result = summarize_last_outputs_for_planner(state, output_max_length=500)

        assert "new = 'data'" in result
        assert "result = 42" in result
        assert "old" not in result
