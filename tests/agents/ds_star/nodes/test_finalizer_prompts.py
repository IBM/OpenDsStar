"""Tests for FinalizerNode prompt building - non-trivial functionality."""

from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from OpenDsStar.agents.ds_star.ds_star_state import CodeMode, DSState, DSStep
from OpenDsStar.agents.ds_star.nodes.finalizer import (
    AnswerOutput,
    FinalizerNode,
    _collect_all_logs,
    _collect_all_outputs,
    build_finalizer_prompt,
)


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = Mock()
    llm.with_structured_output = Mock(return_value=llm)
    return llm


@pytest.fixture
def finalizer_node(mock_llm):
    """Create a FinalizerNode instance for testing."""
    return FinalizerNode(
        system_prompt="test system",
        task_prompt="test task",
        tools_spec='[{"name": "test_tool", "description": "test"}]',
        llm=mock_llm,
    )


class TestFinalizerPromptModeSpecificBehavior:
    """Test mode-specific prompt building in finalizer."""

    def test_stepwise_includes_all_outputs(self):
        """Test STEPWISE mode includes outputs from ALL steps."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0", outputs={"data0": [1, 2, 3]}),
                DSStep(plan="step 1", outputs={"data1": [4, 5, 6]}),
                DSStep(plan="step 2", outputs={"result": 42}),
            ],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_finalizer_prompt(state)
        full_prompt = system_msg + "\n" + user_msg

        # All outputs should be present
        assert "data0 = [1, 2, 3]" in full_prompt
        assert "data1 = [4, 5, 6]" in full_prompt
        assert "result = 42" in full_prompt
        assert "all steps" in full_prompt

    def test_full_mode_includes_only_last_outputs(self):
        """Test FULL mode includes outputs from LAST step only."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0", outputs={"old_data": "should not appear"}),
                DSStep(plan="step 1", outputs={"new_data": "should appear"}),
            ],
            code_mode=CodeMode.FULL,
        )

        system_msg, user_msg = build_finalizer_prompt(state)
        full_prompt = system_msg + "\n" + user_msg

        # Only last step outputs should be present
        assert "new_data" in full_prompt
        assert "old_data" not in full_prompt
        assert "last step only" in full_prompt

    def test_stepwise_includes_all_logs(self):
        """Test STEPWISE mode includes logs from ALL steps."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0", logs="Step 0 logs: loaded data"),
                DSStep(plan="step 1", logs="Step 1 logs: processed data"),
                DSStep(plan="step 2", logs="Step 2 logs: final result"),
            ],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_finalizer_prompt(state)
        full_prompt = system_msg + "\n" + user_msg

        # All logs should be present
        assert "Step 0 logs:" in full_prompt
        assert "loaded data" in full_prompt
        assert "Step 1 logs:" in full_prompt
        assert "processed data" in full_prompt
        assert "Step 2 logs:" in full_prompt
        assert "final result" in full_prompt
        assert "all steps" in full_prompt

    def test_full_mode_includes_only_last_logs(self):
        """Test FULL mode includes logs from LAST step only."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0", logs="Old logs: should not appear"),
                DSStep(plan="step 1", logs="New logs: should appear"),
            ],
            code_mode=CodeMode.FULL,
        )

        system_msg, user_msg = build_finalizer_prompt(state)
        full_prompt = system_msg + "\n" + user_msg

        # Only last step logs should be present
        assert "should appear" in full_prompt
        assert "should not appear" not in full_prompt
        assert "last step only" in full_prompt


class TestFinalizerPromptOutputCollection:
    """Test output collection logic."""

    def test_collect_all_outputs_merges_from_all_steps(self):
        """Test that _collect_all_outputs merges outputs from all steps."""
        state = DSState(
            user_query="test",
            tools={},
            steps=[
                DSStep(plan="step 0", outputs={"a": 1, "b": 2}),
                DSStep(plan="step 1", outputs={"c": 3}),
                DSStep(plan="step 2", outputs={"d": 4, "e": 5}),
            ],
        )

        result = _collect_all_outputs(state)

        assert result == {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}

    def test_collect_all_outputs_handles_overlapping_keys(self):
        """Test that later steps override earlier ones for same key."""
        state = DSState(
            user_query="test",
            tools={},
            steps=[
                DSStep(plan="step 0", outputs={"result": "old"}),
                DSStep(plan="step 1", outputs={"result": "new"}),
            ],
        )

        result = _collect_all_outputs(state)

        # Later step should override
        assert result == {"result": "new"}

    def test_collect_all_outputs_handles_no_outputs(self):
        """Test handling of steps with no outputs."""
        state = DSState(
            user_query="test",
            tools={},
            steps=[
                DSStep(plan="step 0"),
                DSStep(plan="step 1"),
            ],
        )

        result = _collect_all_outputs(state)

        assert result == {}


class TestFinalizerPromptLogCollection:
    """Test log collection logic."""

    def test_collect_all_logs_aggregates_from_all_steps(self):
        """Test that _collect_all_logs aggregates logs from all steps."""
        state = DSState(
            user_query="test",
            tools={},
            steps=[
                DSStep(plan="step 0", logs="Logs from step 0"),
                DSStep(plan="step 1", logs="Logs from step 1"),
                DSStep(plan="step 2", logs="Logs from step 2"),
            ],
        )

        result = _collect_all_logs(state, logs_max_length=1000)

        assert "Step 0 logs:" in result
        assert "Logs from step 0" in result
        assert "Step 1 logs:" in result
        assert "Logs from step 1" in result
        assert "Step 2 logs:" in result
        assert "Logs from step 2" in result

    def test_collect_all_logs_handles_no_logs(self):
        """Test handling when no steps have logs."""
        state = DSState(
            user_query="test",
            tools={},
            steps=[
                DSStep(plan="step 0"),
                DSStep(plan="step 1"),
            ],
        )

        result = _collect_all_logs(state, logs_max_length=1000)

        assert result == "(no logs captured)"

    def test_collect_all_logs_truncates_per_step(self):
        """Test that each step's logs are truncated individually."""
        long_logs = "x" * 1000
        state = DSState(
            user_query="test",
            tools={},
            steps=[
                DSStep(plan="step 0", logs=long_logs),
                DSStep(plan="step 1", logs=long_logs),
            ],
        )

        result = _collect_all_logs(state, logs_max_length=100)

        # Should have truncation markers
        assert "..." in result
        # Should not be excessively long
        assert len(result) < 500


class TestFinalizerPromptTruncation:
    """Test truncation behavior in finalizer prompts."""

    def test_truncates_long_outputs(self):
        """Test that long outputs are truncated."""
        long_value = "x" * 10000
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0", outputs={"huge_data": long_value}),
            ],
            code_mode=CodeMode.STEPWISE,
            output_max_length=100,
        )

        system_msg, user_msg = build_finalizer_prompt(state)
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
                DSStep(plan="step 0", logs=long_logs),
            ],
            code_mode=CodeMode.STEPWISE,
            logs_max_length=200,
        )

        system_msg, user_msg = build_finalizer_prompt(state)
        full_prompt = system_msg + "\n" + user_msg

        # Should be truncated
        assert "..." in full_prompt
        assert len(full_prompt) < 10000


class TestFinalizerPromptInstructions:
    """Test finalizer-specific instructions in prompts."""

    def test_includes_partial_answer_guidance(self):
        """Test that guidance for partial answers is included."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[DSStep(plan="step 0")],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_finalizer_prompt(state)
        full_prompt = system_msg + "\n" + user_msg

        assert "PARTIAL but accurate answer" in full_prompt
        assert "fewer results than requested" in full_prompt
        assert "partial" in full_prompt.lower()

    def test_includes_fatal_error_handling_guidance(self):
        """Test that fatal error handling guidance is included."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[DSStep(plan="step 0")],
            fatal_error="Test error",
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_finalizer_prompt(state)
        full_prompt = system_msg + "\n" + user_msg

        assert "Fatal error flag:" in full_prompt
        assert "Test error" in full_prompt
        assert "fatal error" in full_prompt.lower()

    def test_mentions_structured_output_format(self):
        """Test that AnswerOutput format is mentioned."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[DSStep(plan="step 0")],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_finalizer_prompt(state)
        full_prompt = system_msg + "\n" + user_msg

        assert "AnswerOutput" in full_prompt


class TestFinalizerPromptEdgeCases:
    """Test edge cases in finalizer prompt building."""

    def test_handles_no_steps(self):
        """Test handling when no steps have been executed."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_finalizer_prompt(state)
        full_prompt = system_msg + "\n" + user_msg

        assert "no steps executed" in full_prompt
        assert "no outputs" in full_prompt
        assert "no logs" in full_prompt

    def test_handles_steps_with_no_outputs(self):
        """Test handling of steps that produced no outputs."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0"),
                DSStep(plan="step 1"),
            ],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_finalizer_prompt(state)
        full_prompt = system_msg + "\n" + user_msg

        assert "no outputs produced" in full_prompt

    def test_handles_steps_with_no_logs(self):
        """Test handling of steps with no logs."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0"),
            ],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_finalizer_prompt(state)
        full_prompt = system_msg + "\n" + user_msg

        assert "no logs captured" in full_prompt


class TestFinalizerNodeBehavior:
    """Test FinalizerNode execution behavior."""

    @patch("OpenDsStar.agents.ds_star.nodes.finalizer.invoke_structured_with_usage")
    def test_sets_final_answer(self, mock_invoke, finalizer_node):
        """Test that final_answer is set correctly."""
        mock_invoke.return_value = (
            AnswerOutput(answer="The answer is 42"),
            {"input_tokens": 100, "output_tokens": 50},
        )

        state = DSState(
            user_query="test query",
            tools={},
            steps=[DSStep(plan="step 0", outputs={"result": 42})],
            fatal_error=None,
            steps_used=1,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = finalizer_node(state)

        assert result["final_answer"] == "The answer is 42"

    def test_handles_fatal_error_with_no_steps(self, finalizer_node):
        """Test handling of fatal error when no steps exist."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[],
            fatal_error="Critical error occurred",
            steps_used=0,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = finalizer_node(state)

        assert "Process terminated due to an error" in result["final_answer"]
        assert "Critical error occurred" in result["final_answer"]

    @patch("OpenDsStar.agents.ds_star.nodes.finalizer.invoke_structured_with_usage")
    def test_handles_validation_error(self, mock_invoke, finalizer_node):
        """Test handling of ValidationError."""
        mock_invoke.side_effect = ValidationError.from_exception_data(
            "test", [{"type": "missing", "loc": ("answer",), "input": {}}]
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

        result = finalizer_node(state)

        assert result["final_answer"] == "Unable to answer."
        assert "Finalizer schema validation failed" in result["fatal_error"]

    @patch("OpenDsStar.agents.ds_star.nodes.finalizer.invoke_structured_with_usage")
    def test_handles_general_exception(self, mock_invoke, finalizer_node):
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

        result = finalizer_node(state)

        assert result["final_answer"] == "Unable to answer."
        assert "Finalizer invocation failed" in result["fatal_error"]


class TestFinalizerPromptContextQuality:
    """Test that finalizer prompt provides sufficient context."""

    def test_includes_user_query(self):
        """Test that user query is included for context."""
        state = DSState(
            user_query="What are the top 5 products by revenue?",
            tools={},
            steps=[DSStep(plan="Query database", outputs={"products": [1, 2, 3]})],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_finalizer_prompt(state)
        full_prompt = system_msg + "\n" + user_msg

        assert "What are the top 5 products by revenue?" in full_prompt
        assert "User query:" in full_prompt

    def test_includes_plan_steps(self):
        """Test that all plan steps are included for context."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="Load data from database"),
                DSStep(plan="Process and filter data"),
                DSStep(plan="Calculate aggregates"),
            ],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_finalizer_prompt(state)
        full_prompt = system_msg + "\n" + user_msg

        assert "Load data from database" in full_prompt
        assert "Process and filter data" in full_prompt
        assert "Calculate aggregates" in full_prompt
        assert "Current plan" in full_prompt
