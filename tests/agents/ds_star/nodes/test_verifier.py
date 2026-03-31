"""Tests for VerifierNode logic and prompt building."""

from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from OpenDsStar.agents.ds_star.ds_star_state import CodeMode, DSState, DSStep
from OpenDsStar.agents.ds_star.nodes.verifier import (
    VerifierNode,
    VerifierOutput,
    _summarize_step_for_verifier,
    build_verifier_prompt,
)


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = Mock()
    llm.with_structured_output = Mock(return_value=llm)
    return llm


@pytest.fixture
def verifier_node(mock_llm):
    """Create a VerifierNode instance for testing."""
    return VerifierNode(
        system_prompt="test system",
        task_prompt="test task",
        tools_spec='[{"name": "test_tool", "description": "test"}]',
        llm=mock_llm,
    )


class TestVerifierNodeStateUpdates:
    """Test verifier state updates in VerifierNode.__call__."""

    @patch("OpenDsStar.agents.ds_star.nodes.verifier.invoke_structured_with_usage")
    def test_sets_verifier_sufficient_and_explanation(self, mock_invoke, verifier_node):
        """Test that verifier_sufficient and verifier_explanation are set correctly."""
        mock_invoke.return_value = (
            VerifierOutput(
                sufficient=True,
                explanation="All requirements met",
            ),
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

        result = verifier_node(state)

        last_step = result["steps"][-1]
        assert last_step.verifier_sufficient is True
        assert last_step.verifier_explanation == "All requirements met"

    @patch("OpenDsStar.agents.ds_star.nodes.verifier.invoke_structured_with_usage")
    def test_sets_sufficient_false_when_not_sufficient(
        self, mock_invoke, verifier_node
    ):
        """Test that verifier_sufficient is False when not sufficient."""
        mock_invoke.return_value = (
            VerifierOutput(
                sufficient=False,
                explanation="Need more data",
            ),
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

        result = verifier_node(state)

        last_step = result["steps"][-1]
        assert last_step.verifier_sufficient is False
        assert last_step.verifier_explanation == "Need more data"

    def test_skips_on_fatal_error(self, verifier_node):
        """Test that verifier skips when fatal_error exists."""
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

        result = verifier_node(state)

        assert result["fatal_error"] == "Previous fatal error"
        assert len(result["trajectory"]) == 1
        assert result["trajectory"][0]["skipped"] is True


class TestVerifierNodeErrorHandling:
    """Test error handling in VerifierNode.__call__."""

    @patch("OpenDsStar.agents.ds_star.nodes.verifier.invoke_structured_with_usage")
    def test_handles_validation_error(self, mock_invoke, verifier_node):
        """Test handling of ValidationError sets sufficient=False."""
        mock_invoke.side_effect = ValidationError.from_exception_data(
            "test", [{"type": "missing", "loc": ("sufficient",), "input": {}}]
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

        result = verifier_node(state)

        assert result["fatal_error"] is not None
        assert "Verifier schema validation failed" in result["fatal_error"]
        last_step = result["steps"][-1]
        assert last_step.verifier_sufficient is False

    @patch("OpenDsStar.agents.ds_star.nodes.verifier.invoke_structured_with_usage")
    def test_handles_general_exception(self, mock_invoke, verifier_node):
        """Test handling of general Exception sets sufficient=False."""
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

        result = verifier_node(state)

        assert result["fatal_error"] is not None
        assert "Verifier invocation failed" in result["fatal_error"]
        last_step = result["steps"][-1]
        assert last_step.verifier_sufficient is False


class TestBuildVerifierPrompt:
    """Test build_verifier_prompt function."""

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

        system_msg, user_msg = build_verifier_prompt(state, step_index=1)
        prompt = system_msg + "\n" + user_msg

        assert "What is the capital of France?" in prompt
        assert "0. Search for France capital" in prompt
        assert "1. Format the answer" in prompt

    def test_stepwise_mode_includes_all_outputs(self):
        """Test STEPWISE mode includes all step outputs."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    code="code0",
                    outputs={"result0": "value0"},
                ),
                DSStep(
                    plan="step 1",
                    code="code1",
                    outputs={"result1": "value1"},
                ),
            ],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_verifier_prompt(state, step_index=1)
        prompt = system_msg + "\n" + user_msg

        # Should show code and outputs for all steps
        assert "code0" in prompt
        assert "result0" in prompt
        assert "code1" in prompt
        assert "result1" in prompt
        assert "STEPWISE mode" in prompt

    def test_full_mode_shows_only_last_step(self):
        """Test FULL mode only shows last step execution."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    code="code0",
                    outputs={"old": "data"},
                ),
                DSStep(
                    plan="step 1",
                    code="code1",
                    outputs={"new": "data"},
                ),
            ],
            code_mode=CodeMode.FULL,
        )

        system_msg, user_msg = build_verifier_prompt(state, step_index=1)
        prompt = system_msg + "\n" + user_msg

        # Should show planned actions for all steps
        assert "Step 0 – Planned action:" in prompt
        assert "step 0" in prompt
        assert "Step 1 – Planned action:" in prompt
        assert "step 1" in prompt

        # Should show execution only for last step
        assert "FULL mode, last step only" in prompt
        assert "code1" in prompt
        assert "new" in prompt

        # Should NOT show old step's outputs in detail
        # (they might appear in plan text, but not in outputs section)
        assert prompt.count("code0") == 0 or "planned actions only" in prompt

    def test_handles_steps_with_no_outputs(self):
        """Test handling of steps with no outputs."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[DSStep(plan="step 0")],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_verifier_prompt(state, step_index=0)
        prompt = system_msg + "\n" + user_msg

        assert "No outputs produced" in prompt

    def test_handles_execution_errors(self):
        """Test handling of execution errors."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    execution_error="Test error occurred",
                ),
            ],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_verifier_prompt(state, step_index=0)
        prompt = system_msg + "\n" + user_msg

        assert "Test error occurred" in prompt


class TestSummarizeStepForVerifier:
    """Test _summarize_step_for_verifier function."""

    def test_includes_plan(self):
        """Test that plan is included in summary."""
        step = DSStep(plan="Test plan")

        summary = _summarize_step_for_verifier(
            step, 0, output_max_length=500, logs_max_length=1000
        )

        assert "Step 0" in summary
        assert "Test plan" in summary

    def test_includes_execution_error(self):
        """Test that execution error is included."""
        step = DSStep(
            plan="Test plan",
            execution_error="Error occurred",
        )

        summary = _summarize_step_for_verifier(
            step, 0, output_max_length=500, logs_max_length=1000
        )

        assert "Execution error: Error occurred" in summary

    def test_includes_outputs(self):
        """Test that outputs are included."""
        step = DSStep(
            plan="Test plan",
            outputs={"result": "value", "count": 42},
        )

        summary = _summarize_step_for_verifier(
            step, 0, output_max_length=500, logs_max_length=1000
        )

        assert "result = 'value'" in summary
        assert "count = 42" in summary

    def test_includes_verifier_info(self):
        """Test that verifier info is included."""
        step = DSStep(
            plan="Test plan",
            verifier_sufficient=True,
            verifier_explanation="All good",
        )

        summary = _summarize_step_for_verifier(
            step, 0, output_max_length=500, logs_max_length=1000
        )

        assert "Verifier: sufficient" in summary
        assert "All good" in summary

    def test_includes_router_info(self):
        """Test that router info is included."""
        step = DSStep(
            plan="Test plan",
            router_action="fix_step",
            router_fix_index=2,
            router_explanation="Need to fix",
        )

        summary = _summarize_step_for_verifier(
            step, 0, output_max_length=500, logs_max_length=1000
        )

        assert "Router action: fix_step" in summary
        assert "Fix index: 2" in summary
        assert "Need to fix" in summary

    def test_truncates_long_outputs(self):
        """Test that long outputs are truncated."""
        long_value = "x" * 1000
        step = DSStep(
            plan="Test plan",
            outputs={"result": long_value},
        )

        summary = _summarize_step_for_verifier(
            step, 0, output_max_length=100, logs_max_length=1000
        )

        assert "..." in summary
        assert len(summary) < 1000


# Removed TestCollectAllOutputs class - function no longer exists


class TestVerifierOutput:
    """Test VerifierOutput model."""

    def test_valid_sufficient_output(self):
        """Test creating valid sufficient output."""
        output = VerifierOutput(
            sufficient=True,
            explanation="All requirements met",
        )

        assert output.sufficient is True
        assert output.explanation == "All requirements met"

    def test_valid_not_sufficient_output(self):
        """Test creating valid not sufficient output."""
        output = VerifierOutput(
            sufficient=False,
            explanation="Need more data",
        )

        assert output.sufficient is False
        assert output.explanation == "Need more data"
