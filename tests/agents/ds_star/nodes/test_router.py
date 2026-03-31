"""Tests for RouterNode logic and prompt building."""

from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from OpenDsStar.agents.ds_star.ds_star_state import CodeMode, DSState, DSStep
from OpenDsStar.agents.ds_star.nodes.router import (
    RouteAction,
    RouterNode,
    RouterOutput,
    build_router_prompt,
)


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = Mock()
    llm.with_structured_output = Mock(return_value=llm)
    return llm


@pytest.fixture
def router_node(mock_llm):
    """Create a RouterNode instance for testing."""
    return RouterNode(
        system_prompt="test system",
        task_prompt="test task",
        tools_spec='[{"name": "test_tool", "description": "test"}]',
        llm=mock_llm,
    )


class TestRouterNodeDecisionMaking:
    """Test router decision making in RouterNode.__call__."""

    def test_skips_when_verifier_sufficient(self, router_node):
        """Test that router skips when verifier_sufficient is True."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[DSStep(plan="step 0", verifier_sufficient=True)],
            fatal_error=None,
            steps_used=1,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = router_node(state)

        assert len(result["trajectory"]) == 1
        assert result["trajectory"][0]["skipped"] is True
        assert result["trajectory"][0]["reason"] == "already_sufficient"

    def test_skips_on_fatal_error(self, router_node):
        """Test that router skips when fatal_error exists."""
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

        result = router_node(state)

        assert result["fatal_error"] == "Previous fatal error"
        assert len(result["trajectory"]) == 1
        assert result["trajectory"][0]["skipped"] is True

    @patch("OpenDsStar.agents.ds_star.nodes.router.invoke_structured_with_usage")
    def test_add_next_step_action(self, mock_invoke, router_node):
        """Test handling of add_next_step action."""
        mock_invoke.return_value = (
            RouterOutput(
                action=RouteAction.add_next_step,
                step_index=None,
                explanation="Need more analysis",
            ),
            {"input_tokens": 100, "output_tokens": 50},
        )

        state = DSState(
            user_query="test query",
            tools={},
            steps=[DSStep(plan="step 0", verifier_sufficient=False)],
            fatal_error=None,
            steps_used=1,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = router_node(state)

        last_step = result["steps"][-1]
        assert last_step.router_action == "add_next_step"
        assert last_step.router_fix_index is None
        assert last_step.router_explanation == "Need more analysis"

    @patch("OpenDsStar.agents.ds_star.nodes.router.invoke_structured_with_usage")
    def test_fix_step_action(self, mock_invoke, router_node):
        """Test handling of fix_step action."""
        mock_invoke.return_value = (
            RouterOutput(
                action=RouteAction.fix_step,
                step_index=0,
                explanation="Step 0 needs correction",
            ),
            {"input_tokens": 100, "output_tokens": 50},
        )

        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0"),
                DSStep(plan="step 1", verifier_sufficient=False),
            ],
            fatal_error=None,
            steps_used=2,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = router_node(state)

        last_step = result["steps"][-1]
        assert last_step.router_action == "fix_step"
        assert last_step.router_fix_index == 0
        assert last_step.router_explanation == "Step 0 needs correction"

    @patch("OpenDsStar.agents.ds_star.nodes.router.invoke_structured_with_usage")
    def test_fix_index_clamping_negative(self, mock_invoke, router_node):
        """Test that negative fix_index is clamped to 0."""
        mock_invoke.return_value = (
            RouterOutput(
                action=RouteAction.fix_step,
                step_index=-5,
                explanation="Fix needed",
            ),
            {"input_tokens": 100, "output_tokens": 50},
        )

        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0"),
                DSStep(plan="step 1", verifier_sufficient=False),
            ],
            fatal_error=None,
            steps_used=2,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = router_node(state)

        last_step = result["steps"][-1]
        assert last_step.router_fix_index == 0

    @patch("OpenDsStar.agents.ds_star.nodes.router.invoke_structured_with_usage")
    def test_fix_index_clamping_too_large(self, mock_invoke, router_node):
        """Test that fix_index >= len(steps) is clamped."""
        mock_invoke.return_value = (
            RouterOutput(
                action=RouteAction.fix_step,
                step_index=10,
                explanation="Fix needed",
            ),
            {"input_tokens": 100, "output_tokens": 50},
        )

        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0"),
                DSStep(plan="step 1", verifier_sufficient=False),
            ],
            fatal_error=None,
            steps_used=2,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = router_node(state)

        last_step = result["steps"][-1]
        # Should be clamped to len(steps) - 1 = 1
        assert last_step.router_fix_index == 1

    @patch("OpenDsStar.agents.ds_star.nodes.router.invoke_structured_with_usage")
    def test_none_fix_index_with_fix_step_raises_error(self, mock_invoke, router_node):
        """Test that None fix_index with fix_step action raises error."""
        mock_invoke.return_value = (
            RouterOutput(
                action=RouteAction.fix_step,
                step_index=None,
                explanation="Fix needed",
            ),
            {"input_tokens": 100, "output_tokens": 50},
        )

        state = DSState(
            user_query="test query",
            tools={},
            steps=[DSStep(plan="step 0", verifier_sufficient=False)],
            fatal_error=None,
            steps_used=1,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        with pytest.raises(Exception, match="action is fix_step but fix_idx is None"):
            router_node(state)


class TestRouterNodeErrorHandling:
    """Test error handling in RouterNode.__call__."""

    @patch("OpenDsStar.agents.ds_star.nodes.router.invoke_structured_with_usage")
    def test_handles_validation_error(self, mock_invoke, router_node):
        """Test handling of ValidationError."""
        mock_invoke.side_effect = ValidationError.from_exception_data(
            "test", [{"type": "missing", "loc": ("action",), "input": {}}]
        )

        state = DSState(
            user_query="test query",
            tools={},
            steps=[DSStep(plan="step 0", verifier_sufficient=False)],
            fatal_error=None,
            steps_used=1,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = router_node(state)

        assert result["fatal_error"] is not None
        assert "Router schema validation failed" in result["fatal_error"]

    @patch("OpenDsStar.agents.ds_star.nodes.router.invoke_structured_with_usage")
    def test_handles_general_exception(self, mock_invoke, router_node):
        """Test handling of general Exception."""
        mock_invoke.side_effect = Exception("Test error")

        state = DSState(
            user_query="test query",
            tools={},
            steps=[DSStep(plan="step 0", verifier_sufficient=False)],
            fatal_error=None,
            steps_used=1,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = router_node(state)

        assert result["fatal_error"] is not None
        assert "Router invocation failed" in result["fatal_error"]


class TestBuildRouterPrompt:
    """Test build_router_prompt function."""

    def test_includes_user_query_and_plan(self):
        """Test that prompt includes user query and plan."""
        state = DSState(
            user_query="What is the capital of France?",
            tools={},
            steps=[
                DSStep(plan="Search for France capital"),
                DSStep(plan="Format the answer", verifier_sufficient=False),
            ],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_router_prompt(state)
        prompt = system_msg + "\n" + user_msg

        assert "What is the capital of France?" in prompt
        assert "0. Search for France capital" in prompt
        assert "1. Format the answer" in prompt

    def test_stepwise_mode_shows_all_step_details(self):
        """Test STEPWISE mode shows code, outputs, and errors for all steps."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    code="code0",
                    outputs={"result": "value0"},
                    execution_error=None,
                ),
                DSStep(
                    plan="step 1",
                    code="code1",
                    outputs={"result": "value1"},
                    execution_error="error1",
                    verifier_sufficient=False,
                ),
            ],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_router_prompt(state)
        prompt = system_msg + "\n" + user_msg

        # Should show details for both steps
        assert "Step 0:" in prompt
        assert "code0" in prompt
        assert "result" in prompt
        assert "Step 1:" in prompt
        assert "code1" in prompt
        assert "error1" in prompt

    def test_full_mode_shows_only_plans_and_last_execution(self):
        """Test FULL mode shows only plans + last execution context."""
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
                    execution_error="error1",
                    verifier_sufficient=False,
                ),
            ],
            code_mode=CodeMode.FULL,
        )

        system_msg, user_msg = build_router_prompt(state)
        prompt = system_msg + "\n" + user_msg

        # Should show planned actions for all steps
        assert "Step 0 – Planned action:" in prompt
        assert "step 0" in prompt
        assert "Step 1 – Planned action:" in prompt
        assert "step 1" in prompt

        # Should show execution context only for last step
        assert "Current execution context (latest run):" in prompt
        assert "code1" in prompt
        assert "error1" in prompt

        # Should NOT show old step's code in detail
        assert "code0" not in prompt or "Code (latest):" in prompt

    def test_includes_verifier_info(self):
        """Test that verifier information is included."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    verifier_sufficient=False,
                    verifier_explanation="Not enough data",
                ),
            ],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_router_prompt(state)
        prompt = system_msg + "\n" + user_msg

        assert "sufficient=False" in prompt
        assert "Not enough data" in prompt

    def test_handles_missing_outputs(self):
        """Test handling of steps with no outputs."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0", verifier_sufficient=False),
            ],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_router_prompt(state)
        prompt = system_msg + "\n" + user_msg

        assert "No outputs produced" in prompt

    def test_handles_missing_errors(self):
        """Test handling of steps with no errors."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0", execution_error=None, verifier_sufficient=False),
            ],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_router_prompt(state)
        prompt = system_msg + "\n" + user_msg

        assert "(none)" in prompt


class TestRouterOutput:
    """Test RouterOutput model."""

    def test_valid_add_next_step(self):
        """Test creating valid add_next_step output."""
        output = RouterOutput(
            action=RouteAction.add_next_step,
            step_index=None,
            explanation="Need more steps",
        )

        assert output.action == RouteAction.add_next_step
        assert output.step_index is None
        assert output.explanation == "Need more steps"

    def test_valid_fix_step(self):
        """Test creating valid fix_step output."""
        output = RouterOutput(
            action=RouteAction.fix_step,
            step_index=2,
            explanation="Step 2 is wrong",
        )

        assert output.action == RouteAction.fix_step
        assert output.step_index == 2
        assert output.explanation == "Step 2 is wrong"

    def test_action_enum_values(self):
        """Test RouteAction enum values."""
        assert RouteAction.add_next_step.value == "add_next_step"
        assert RouteAction.fix_step.value == "fix_step"
