"""Tests for DSStarGraph routing logic and state management."""

from unittest.mock import Mock

import pytest
from pydantic import BaseModel, Field

from OpenDsStar.agents.ds_star.ds_star_graph import DSStarGraph
from OpenDsStar.agents.ds_star.ds_star_results_prep import (
    prepare_result_from_graph_state_ds_star_agent,
)
from OpenDsStar.agents.ds_star.ds_star_state import CodeMode, DSState, DSStep


class MockToolInput(BaseModel):
    """Mock tool input schema for testing."""

    query: str = Field(description="Test query parameter")


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = Mock()
    llm.with_structured_output = Mock(return_value=llm)
    return llm


@pytest.fixture
def mock_tool():
    """Create a mock tool for testing."""
    tool = Mock()
    tool.name = "test_tool"
    tool.description = "A test tool"
    tool.args_schema = MockToolInput
    tool.invoke = Mock(return_value="test result")
    return tool


@pytest.fixture
def ds_star_graph(mock_llm, mock_tool):
    """Create a DSStarGraph instance for testing."""
    return DSStarGraph(
        model=mock_llm,
        tools=[mock_tool],
        max_steps=5,
        code_timeout=30,
    )


class TestRouteAfterExecute:
    """Test route_after_execute routing logic."""

    def test_route_to_finalizer_on_fatal_error(self, ds_star_graph):
        """Test routing to finalizer when fatal_error exists."""
        state = DSState(
            user_query="test",
            tools={},
            fatal_error="Fatal error occurred",
            steps=[DSStep(plan="step 1")],
        )
        result = ds_star_graph.route_after_execute(state)
        assert result == "n_finalizer"

    def test_route_to_finalizer_on_no_steps(self, ds_star_graph):
        """Test routing to finalizer when no steps exist."""
        state = DSState(user_query="test", tools={}, steps=[])
        result = ds_star_graph.route_after_execute(state)
        assert result == "n_finalizer"
        assert state.fatal_error == "No steps found after execution."

    def test_route_to_finalizer_on_max_debug_attempts(self, ds_star_graph):
        """Test routing to finalizer when max debug attempts reached."""
        state = DSState(
            user_query="test",
            tools={},
            steps=[DSStep(plan="step 1", debug_tries=5)],
        )
        result = ds_star_graph.route_after_execute(state)
        assert result == "n_finalizer"
        assert state.fatal_error == "Max debug attempts reached."

    def test_route_to_debugger_on_execution_error(self, ds_star_graph):
        """Test routing to debugger when execution_error exists."""
        state = DSState(
            user_query="test",
            tools={},
            steps=[
                DSStep(plan="step 1", execution_error="Error occurred", debug_tries=0)
            ],
        )
        result = ds_star_graph.route_after_execute(state)
        assert result == "n_debug"

    def test_route_to_verifier_on_success(self, ds_star_graph):
        """Test routing to verifier when no error."""
        state = DSState(
            user_query="test",
            tools={},
            steps=[DSStep(plan="step 1", execution_error=None)],
        )
        result = ds_star_graph.route_after_execute(state)
        assert result == "n_verify"


class TestRouteAfterVerify:
    """Test route_after_verify routing logic."""

    def test_route_to_finalizer_on_fatal_error(self, ds_star_graph):
        """Test routing to finalizer when fatal_error exists."""
        state = DSState(
            user_query="test",
            tools={},
            fatal_error="Fatal error",
            steps=[DSStep(plan="step 1")],
        )
        result = ds_star_graph.route_after_verify(state)
        assert result == "n_finalizer"

    def test_route_to_finalizer_when_sufficient(self, ds_star_graph):
        """Test routing to finalizer when verifier marks sufficient."""
        state = DSState(
            user_query="test",
            tools={},
            steps=[DSStep(plan="step 1", verifier_sufficient=True)],
        )
        result = ds_star_graph.route_after_verify(state)
        assert result == "n_finalizer"

    def test_route_to_finalizer_on_max_steps_reached(self, ds_star_graph):
        """Test routing to finalizer when max_steps reached."""
        state = DSState(
            user_query="test",
            tools={},
            steps=[DSStep(plan="step 1", verifier_sufficient=False)],
            steps_used=5,
            max_steps=5,
        )
        result = ds_star_graph.route_after_verify(state)
        assert result == "n_finalizer"
        assert state.fatal_error is not None
        assert "Max step limit reached" in state.fatal_error

    def test_route_to_finalizer_on_max_steps_exceeded(self, ds_star_graph):
        """Test routing to finalizer when steps_used > max_steps."""
        state = DSState(
            user_query="test",
            tools={},
            steps=[DSStep(plan="step 1", verifier_sufficient=False)],
            steps_used=6,
            max_steps=5,
        )
        result = ds_star_graph.route_after_verify(state)
        assert result == "n_finalizer"

    def test_route_to_router_when_not_sufficient(self, ds_star_graph):
        """Test routing to router when not sufficient and under max_steps."""
        state = DSState(
            user_query="test",
            tools={},
            steps=[DSStep(plan="step 1", verifier_sufficient=False)],
            steps_used=2,
            max_steps=5,
        )
        result = ds_star_graph.route_after_verify(state)
        assert result == "n_route"


class TestRouteAfterRoute:
    """Test route_after_route routing logic."""

    def test_route_to_finalizer_on_fatal_error(self, ds_star_graph):
        """Test routing to finalizer when fatal_error exists."""
        state = DSState(
            user_query="test",
            tools={},
            fatal_error="Fatal error",
            steps=[DSStep(plan="step 1")],
        )
        result = ds_star_graph.route_after_route(state)
        assert result == "n_finalizer"

    def test_route_to_planner_on_add_next_step(self, ds_star_graph):
        """Test routing to planner for add_next_step action."""
        state = DSState(
            user_query="test",
            tools={},
            steps=[DSStep(plan="step 1", router_action="add_next_step")],
        )
        result = ds_star_graph.route_after_route(state)
        assert result == "n_plan_one"

    def test_route_to_planner_on_fix_step(self, ds_star_graph):
        """Test routing to planner for fix_step action."""
        state = DSState(
            user_query="test",
            tools={},
            steps=[DSStep(plan="step 1", router_action="fix_step", router_fix_index=0)],
        )
        result = ds_star_graph.route_after_route(state)
        assert result == "n_plan_one"

    def test_route_to_finalizer_on_unknown_action(self, ds_star_graph):
        """Test routing to finalizer for unknown action."""
        state = DSState(
            user_query="test",
            tools={},
            steps=[DSStep(plan="step 1", router_action="unknown_action")],
        )
        result = ds_star_graph.route_after_route(state)
        assert result == "n_finalizer"

    def test_route_to_finalizer_on_no_action(self, ds_star_graph):
        """Test routing to finalizer when no router_action."""
        state = DSState(
            user_query="test",
            tools={},
            steps=[DSStep(plan="step 1", router_action=None)],
        )
        result = ds_star_graph.route_after_route(state)
        assert result == "n_finalizer"


class TestInitState:
    """Test _init_state validation and initialization."""

    def test_missing_user_query_raises_error(self, ds_star_graph):
        """Test that missing user_query raises ValueError."""
        with pytest.raises(ValueError, match="Missing required field: 'user_query'"):
            ds_star_graph._init_state({})

    def test_filters_invalid_fields(self, ds_star_graph):
        """Test that invalid fields are filtered from input_dict."""
        input_dict = {
            "user_query": "test",
            "invalid_field": "should be ignored",
            "max_steps": 10,
        }
        state = ds_star_graph._init_state(input_dict)
        assert state.user_query == "test"
        assert state.max_steps == 10
        assert not hasattr(state, "invalid_field")

    def test_default_tools_when_not_provided(self, ds_star_graph):
        """Test default tools={} when not provided."""
        input_dict = {"user_query": "test"}
        state = ds_star_graph._init_state(input_dict)
        assert state.tools == {}

    def test_preserves_valid_fields(self, ds_star_graph):
        """Test that valid DSState fields are preserved."""
        input_dict = {
            "user_query": "test",
            "max_steps": 10,
            "steps_used": 2,
            "code_mode": CodeMode.FULL,
        }
        state = ds_star_graph._init_state(input_dict)
        assert state.user_query == "test"
        assert state.max_steps == 10
        assert state.steps_used == 2
        assert state.code_mode == CodeMode.FULL


class TestInvoke:
    """Test invoke method validation."""

    def test_invoke_missing_user_query_raises_error(self, ds_star_graph):
        """Test that invoke without user_query raises ValueError."""
        with pytest.raises(ValueError, match="Missing required key: 'user_query'"):
            ds_star_graph.invoke({})


class TestPrepareResult:
    """Test prepare_result_from_graph_state_ds_star_agent."""

    def test_handles_empty_steps(self):
        """Test handling empty steps list."""
        state = {
            "user_query": "test",
            "tools": {},
            "steps": [],
            "final_answer": "answer",
            "fatal_error": None,
            "steps_used": 0,
            "max_steps": 5,
            "trajectory": [],
            "token_usage": [],
            "code_mode": CodeMode.STEPWISE,
        }
        result = prepare_result_from_graph_state_ds_star_agent(state)
        assert result["answer"] == "answer"
        assert result["verifier_sufficient"] is False
        assert result["execution_error"] == ""

    def test_aggregates_token_usage(self):
        """Test token usage aggregation."""
        state = {
            "user_query": "test",
            "tools": {},
            "steps": [DSStep(plan="step 1")],
            "final_answer": "answer",
            "fatal_error": None,
            "steps_used": 1,
            "max_steps": 5,
            "trajectory": [],
            "token_usage": [
                {"input_tokens": 100, "output_tokens": 50},
                {"input_tokens": 200, "output_tokens": 100},
            ],
            "code_mode": CodeMode.STEPWISE,
        }
        result = prepare_result_from_graph_state_ds_star_agent(state)
        assert result["input_tokens"] == 300
        assert result["output_tokens"] == 150
        assert result["num_llm_calls"] == 2

    def test_handles_non_serializable_trajectory(self):
        """Test handling non-serializable objects in trajectory.

        The _jsonify_and_truncate function converts non-serializable objects
        to strings using str() and truncates long strings to 1000 chars.
        """

        # Use a simple non-serializable object (a function) instead of Mock
        # to avoid infinite recursion issues with Mock's attribute access
        def non_serializable_func():
            return "test"

        state = {
            "user_query": "test",
            "tools": {},
            "steps": [DSStep(plan="step 1")],
            "final_answer": "answer",
            "fatal_error": None,
            "steps_used": 1,
            "max_steps": 5,
            "trajectory": [
                {"event": "test", "data": "string"},
                {"event": "test2", "obj": non_serializable_func},
            ],
            "token_usage": [],
            "code_mode": CodeMode.STEPWISE,
        }
        result = prepare_result_from_graph_state_ds_star_agent(state)

        # Verify trajectory is processed
        assert len(result["trajectory"]) == 2

        # First entry should remain unchanged (already serializable)
        assert result["trajectory"][0]["event"] == "test"
        assert result["trajectory"][0]["data"] == "string"

        # Second entry's function object should be converted to string
        assert result["trajectory"][1]["event"] == "test2"
        assert isinstance(result["trajectory"][1]["obj"], str)
        # Function objects stringify to something like "<function ...>", verify it's a string
        assert "function" in result["trajectory"][1]["obj"]

    def test_handles_none_values(self):
        """Test handling None values."""
        state = {
            "user_query": "test",
            "tools": {},
            "steps": [DSStep(plan="step 1")],
            "final_answer": None,
            "fatal_error": None,
            "steps_used": 1,
            "max_steps": 5,
            "trajectory": [],
            "token_usage": [],
            "code_mode": CodeMode.STEPWISE,
        }
        result = prepare_result_from_graph_state_ds_star_agent(state)
        assert result["answer"] == ""
        assert result["fatal_error"] == ""

    def test_extracts_last_step_info(self):
        """Test extraction of last step information."""
        state = {
            "user_query": "test",
            "tools": {},
            "steps": [
                DSStep(plan="step 1"),
                DSStep(
                    plan="step 2",
                    verifier_sufficient=True,
                    execution_error="test error",
                ),
            ],
            "final_answer": "answer",
            "fatal_error": None,
            "steps_used": 2,
            "max_steps": 5,
            "trajectory": [],
            "token_usage": [],
            "code_mode": CodeMode.STEPWISE,
        }
        result = prepare_result_from_graph_state_ds_star_agent(state)
        assert result["verifier_sufficient"] is True
        assert result["execution_error"] == "test error"


class TestUpdateToolsSpec:
    """Test update_tools_spec method."""

    def test_updates_all_nodes(self, ds_star_graph):
        """Test that update_tools_spec updates all nodes."""

        # Update tools_spec
        ds_star_graph.update_tools_spec()

        # Verify all nodes have the updated spec
        for node in ds_star_graph._nodes:
            assert node.tools_spec == ds_star_graph.tools_spec
