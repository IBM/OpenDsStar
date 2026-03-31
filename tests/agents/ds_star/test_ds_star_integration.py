"""Integration tests for DS-Star agent workflow."""

from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel, Field

from OpenDsStar.agents.ds_star.ds_star_graph import DSStarGraph
from OpenDsStar.agents.ds_star.ds_star_state import CodeMode


class MockToolInput(BaseModel):
    """Mock tool input schema."""

    query: str = Field(description="Query parameter")


@pytest.fixture
def mock_llm():
    """Create a mock LLM that returns structured outputs."""
    llm = Mock()
    llm.with_structured_output = Mock(return_value=llm)
    return llm


@pytest.fixture
def mock_tool():
    """Create a mock tool."""
    tool = Mock()
    tool.name = "search_tool"
    tool.description = "Search for information"
    tool.args_schema = MockToolInput
    tool.invoke = Mock(return_value="search result")
    return tool


class TestSuccessfulWorkflow:
    """Test successful execution workflows."""

    @patch("OpenDsStar.agents.ds_star.nodes.planner.invoke_structured_with_usage")
    @patch("OpenDsStar.agents.ds_star.nodes.coder.invoke_structured_with_usage")
    @patch("OpenDsStar.agents.ds_star.nodes.verifier.invoke_structured_with_usage")
    @patch("OpenDsStar.agents.ds_star.nodes.router.invoke_structured_with_usage")
    def test_simple_two_step_workflow(
        self, mock_router, mock_verifier, mock_coder, mock_planner, mock_llm, mock_tool
    ):
        """Test a simple successful 2-step workflow."""
        from OpenDsStar.agents.ds_star.nodes.coder import CodeOutput
        from OpenDsStar.agents.ds_star.nodes.planner import PlanOneStepOutput
        from OpenDsStar.agents.ds_star.nodes.router import RouteAction, RouterOutput
        from OpenDsStar.agents.ds_star.nodes.verifier import VerifierOutput

        # Mock planner to create 2 steps
        mock_planner.side_effect = [
            (
                PlanOneStepOutput(step="Search for data"),
                {"input_tokens": 100, "output_tokens": 50},
            ),
            (
                PlanOneStepOutput(step="Format the result"),
                {"input_tokens": 100, "output_tokens": 50},
            ),
        ]

        # Mock coder to generate code
        mock_coder.side_effect = [
            (
                CodeOutput(code="outputs['data'] = 'found'"),
                {"input_tokens": 100, "output_tokens": 50},
            ),
            (
                CodeOutput(code="outputs['result'] = 'found'"),
                {"input_tokens": 100, "output_tokens": 50},
            ),
        ]

        # Mock verifier - first not sufficient, then sufficient
        mock_verifier.side_effect = [
            (
                VerifierOutput(sufficient=False, explanation="Need formatting"),
                {"input_tokens": 100, "output_tokens": 50},
            ),
            (
                VerifierOutput(sufficient=True, explanation="Complete"),
                {"input_tokens": 100, "output_tokens": 50},
            ),
        ]

        # Mock router to add next step after first verification
        mock_router.return_value = (
            RouterOutput(
                action=RouteAction.add_next_step,
                step_index=None,
                explanation="Continue",
            ),
            {"input_tokens": 100, "output_tokens": 50},
        )

        graph = DSStarGraph(model=mock_llm, tools=[mock_tool], max_steps=5)

        result = graph.invoke({"user_query": "Find and format data"})

        # Should have 2 steps
        assert len(result["steps"]) == 2
        assert result["steps"][0].plan == "Search for data"
        assert result["steps"][1].plan == "Format the result"
        assert result["steps_used"] == 2


class TestMaxStepsTermination:
    """Test max_steps termination."""

    @patch("OpenDsStar.agents.ds_star.nodes.planner.invoke_structured_with_usage")
    @patch("OpenDsStar.agents.ds_star.nodes.coder.invoke_structured_with_usage")
    @patch("OpenDsStar.agents.ds_star.nodes.verifier.invoke_structured_with_usage")
    @patch("OpenDsStar.agents.ds_star.nodes.router.invoke_structured_with_usage")
    def test_terminates_at_max_steps(
        self, mock_router, mock_verifier, mock_coder, mock_planner, mock_llm, mock_tool
    ):
        """Test that workflow terminates at max_steps."""
        from OpenDsStar.agents.ds_star.nodes.coder import CodeOutput
        from OpenDsStar.agents.ds_star.nodes.planner import PlanOneStepOutput
        from OpenDsStar.agents.ds_star.nodes.router import RouteAction, RouterOutput
        from OpenDsStar.agents.ds_star.nodes.verifier import VerifierOutput

        # Mock planner to always create new steps
        mock_planner.return_value = (
            PlanOneStepOutput(step="Another step"),
            {"input_tokens": 100, "output_tokens": 50},
        )

        # Mock coder
        mock_coder.return_value = (
            CodeOutput(code="outputs['x'] = 1"),
            {"input_tokens": 100, "output_tokens": 50},
        )

        # Mock verifier to always say not sufficient
        mock_verifier.return_value = (
            VerifierOutput(sufficient=False, explanation="Not done"),
            {"input_tokens": 100, "output_tokens": 50},
        )

        # Mock router to always add next step
        mock_router.return_value = (
            RouterOutput(
                action=RouteAction.add_next_step,
                step_index=None,
                explanation="Continue",
            ),
            {"input_tokens": 100, "output_tokens": 50},
        )

        graph = DSStarGraph(model=mock_llm, tools=[mock_tool], max_steps=3)

        result = graph.invoke({"user_query": "Test query"})

        # Should stop at max_steps
        assert result["steps_used"] == 3
        assert result["fatal_error"] is not None
        assert "Max step limit reached" in result["fatal_error"]


class TestFixStepWorkflow:
    """Test fix_step workflow."""

    @patch("OpenDsStar.agents.ds_star.nodes.planner.invoke_structured_with_usage")
    @patch("OpenDsStar.agents.ds_star.nodes.coder.invoke_structured_with_usage")
    @patch("OpenDsStar.agents.ds_star.nodes.verifier.invoke_structured_with_usage")
    @patch("OpenDsStar.agents.ds_star.nodes.router.invoke_structured_with_usage")
    def test_fix_step_replaces_and_continues(
        self, mock_router, mock_verifier, mock_coder, mock_planner, mock_llm, mock_tool
    ):
        """Test that fix_step workflow replaces step and continues."""
        from OpenDsStar.agents.ds_star.nodes.coder import CodeOutput
        from OpenDsStar.agents.ds_star.nodes.planner import PlanOneStepOutput
        from OpenDsStar.agents.ds_star.nodes.router import RouteAction, RouterOutput
        from OpenDsStar.agents.ds_star.nodes.verifier import VerifierOutput

        # Planner: step 1, then corrected step 0, then step 1 again
        mock_planner.side_effect = [
            (
                PlanOneStepOutput(step="Wrong step"),
                {"input_tokens": 100, "output_tokens": 50},
            ),
            (
                PlanOneStepOutput(step="Corrected step"),
                {"input_tokens": 100, "output_tokens": 50},
            ),
            (
                PlanOneStepOutput(step="Final step"),
                {"input_tokens": 100, "output_tokens": 50},
            ),
        ]

        # Coder generates code for each
        mock_coder.return_value = (
            CodeOutput(code="outputs['x'] = 1"),
            {"input_tokens": 100, "output_tokens": 50},
        )

        # Verifier: not sufficient, not sufficient, then sufficient
        mock_verifier.side_effect = [
            (
                VerifierOutput(sufficient=False, explanation="Wrong approach"),
                {"input_tokens": 100, "output_tokens": 50},
            ),
            (
                VerifierOutput(sufficient=False, explanation="Need more"),
                {"input_tokens": 100, "output_tokens": 50},
            ),
            (
                VerifierOutput(sufficient=True, explanation="Done"),
                {"input_tokens": 100, "output_tokens": 50},
            ),
        ]

        # Router: fix step 0, then add next
        mock_router.side_effect = [
            (
                RouterOutput(
                    action=RouteAction.fix_step, step_index=0, explanation="Fix it"
                ),
                {"input_tokens": 100, "output_tokens": 50},
            ),
            (
                RouterOutput(
                    action=RouteAction.add_next_step,
                    step_index=None,
                    explanation="Continue",
                ),
                {"input_tokens": 100, "output_tokens": 50},
            ),
        ]

        graph = DSStarGraph(model=mock_llm, tools=[mock_tool], max_steps=5)

        result = graph.invoke({"user_query": "Test query"})

        # Should have 2 steps (step 0 was replaced)
        assert len(result["steps"]) == 2
        assert result["steps"][0].plan == "Corrected step"
        assert result["steps"][1].plan == "Final step"


class TestDebuggerWorkflow:
    """Test debugger workflow."""

    @patch("OpenDsStar.agents.ds_star.nodes.planner.invoke_structured_with_usage")
    @patch("OpenDsStar.agents.ds_star.nodes.coder.invoke_structured_with_usage")
    @patch("OpenDsStar.agents.ds_star.nodes.debugger.invoke_structured_with_usage")
    def test_debugger_fixes_execution_error(
        self, mock_debugger, mock_coder, mock_planner, mock_llm, mock_tool
    ):
        """Test that debugger fixes execution errors."""
        from OpenDsStar.agents.ds_star.nodes.coder import CodeOutput
        from OpenDsStar.agents.ds_star.nodes.planner import PlanOneStepOutput

        # Planner creates one step
        mock_planner.return_value = (
            PlanOneStepOutput(step="Calculate result"),
            {"input_tokens": 100, "output_tokens": 50},
        )

        # Coder first generates bad code, then debugger fixes it
        mock_coder.side_effect = [
            (
                CodeOutput(code="outputs['x'] = 1 / 0"),  # Will cause error
                {"input_tokens": 100, "output_tokens": 50},
            ),
        ]

        # Debugger fixes the code
        mock_debugger.return_value = (
            CodeOutput(code="outputs['x'] = 1"),  # Fixed code
            {"input_tokens": 100, "output_tokens": 50},
        )

        graph = DSStarGraph(model=mock_llm, tools=[mock_tool], max_steps=5)

        result = graph.invoke({"user_query": "Test query"})

        # Should have executed debugger
        assert len(result["steps"]) >= 1
        last_step = result["steps"][-1]
        # After debug, execution_error should be None or code should be fixed
        assert last_step.debug_tries >= 1


class TestFatalErrorPropagation:
    """Test fatal error propagation through nodes."""

    @patch("OpenDsStar.agents.ds_star.nodes.planner.invoke_structured_with_usage")
    def test_fatal_error_stops_execution(self, mock_planner, mock_llm, mock_tool):
        """Test that fatal error stops execution."""
        from pydantic import ValidationError

        # Planner raises validation error
        mock_planner.side_effect = ValidationError.from_exception_data(
            "test", [{"type": "missing", "loc": ("step",), "input": {}}]
        )

        graph = DSStarGraph(model=mock_llm, tools=[mock_tool], max_steps=5)

        result = graph.invoke({"user_query": "Test query"})

        # Should have fatal error
        assert result["fatal_error"] is not None
        assert "Planner schema validation failed" in result["fatal_error"]


class TestCodeModes:
    """Test different code modes."""

    @patch("OpenDsStar.agents.ds_star.nodes.planner.invoke_structured_with_usage")
    @patch("OpenDsStar.agents.ds_star.nodes.coder.invoke_structured_with_usage")
    @patch("OpenDsStar.agents.ds_star.nodes.verifier.invoke_structured_with_usage")
    def test_stepwise_mode(
        self, mock_verifier, mock_coder, mock_planner, mock_llm, mock_tool
    ):
        """Test STEPWISE code mode."""
        from OpenDsStar.agents.ds_star.nodes.coder import CodeOutput
        from OpenDsStar.agents.ds_star.nodes.planner import PlanOneStepOutput
        from OpenDsStar.agents.ds_star.nodes.verifier import VerifierOutput

        mock_planner.return_value = (
            PlanOneStepOutput(step="Test step"),
            {"input_tokens": 100, "output_tokens": 50},
        )

        mock_coder.return_value = (
            CodeOutput(code="outputs['x'] = 1"),
            {"input_tokens": 100, "output_tokens": 50},
        )

        mock_verifier.return_value = (
            VerifierOutput(sufficient=True, explanation="Done"),
            {"input_tokens": 100, "output_tokens": 50},
        )

        graph = DSStarGraph(model=mock_llm, tools=[mock_tool], max_steps=5)

        result = graph.invoke(
            {"user_query": "Test query", "code_mode": CodeMode.STEPWISE}
        )

        assert result["code_mode"] == CodeMode.STEPWISE

    @patch("OpenDsStar.agents.ds_star.nodes.planner.invoke_structured_with_usage")
    @patch("OpenDsStar.agents.ds_star.nodes.coder.invoke_structured_with_usage")
    @patch("OpenDsStar.agents.ds_star.nodes.verifier.invoke_structured_with_usage")
    def test_full_mode(
        self, mock_verifier, mock_coder, mock_planner, mock_llm, mock_tool
    ):
        """Test FULL code mode."""
        from OpenDsStar.agents.ds_star.nodes.coder import CodeOutput
        from OpenDsStar.agents.ds_star.nodes.planner import PlanOneStepOutput
        from OpenDsStar.agents.ds_star.nodes.verifier import VerifierOutput

        mock_planner.return_value = (
            PlanOneStepOutput(step="Test step"),
            {"input_tokens": 100, "output_tokens": 50},
        )

        mock_coder.return_value = (
            CodeOutput(code="outputs['x'] = 1"),
            {"input_tokens": 100, "output_tokens": 50},
        )

        mock_verifier.return_value = (
            VerifierOutput(sufficient=True, explanation="Done"),
            {"input_tokens": 100, "output_tokens": 50},
        )

        graph = DSStarGraph(model=mock_llm, tools=[mock_tool], max_steps=5)

        result = graph.invoke({"user_query": "Test query", "code_mode": CodeMode.FULL})

        assert result["code_mode"] == CodeMode.FULL
