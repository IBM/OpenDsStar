"""Tests for ds_star_results_prep module."""

import json
from dataclasses import asdict

from agents.ds_star.ds_star_results_prep import (
    _normalize_trajectory_event,
    prepare_result_from_graph_state_ds_star_agent,
)
from agents.ds_star.ds_star_state import DSState, DSStep


class TestNormalizeTrajectoryEvent:
    """Test _normalize_trajectory_event function."""

    def test_normalizes_event_with_dict_last_step(self):
        """Test that dict last_step is preserved."""
        event = {
            "time": 123.456,
            "node": "n_plan_one",
            "last_step": {
                "plan": "Search files",
                "code": "print(1)",
                "execution_error": None,
            },
        }
        result = _normalize_trajectory_event(event)

        # Verify last_step remains a dict
        assert isinstance(result["last_step"], dict)
        assert result["last_step"]["plan"] == "Search files"
        assert result["last_step"]["code"] == "print(1)"
        assert result["last_step"]["execution_error"] is None

    def test_normalizes_event_with_none_last_step(self):
        """Test handling None last_step."""
        event = {
            "time": 123.456,
            "node": "entry",
            "last_step": None,
        }
        result = _normalize_trajectory_event(event)
        assert result["last_step"] is None


class TestPrepareResultFromGraphState:
    """Test prepare_result_from_graph_state_ds_star_agent function."""

    def test_trajectory_last_step_is_dict_not_string(self):
        """Test that trajectory events have last_step as dict, not string.

        This is the critical test - ensures saved state has proper JSON structure.
        """
        # Create a state with trajectory containing dict last_step (as created by add_event_to_trajectory)
        step1 = DSStep(plan="Search files", code="print(1)", execution_error=None)
        step2 = DSStep(
            plan="Process data",
            code="print(2)",
            outputs={"result": [1, 2, 3]},
            execution_error=None,
        )

        state = DSState(
            user_query="test query",
            tools={},
            steps=[step1, step2],
            final_answer="test answer",
            trajectory=[
                {
                    "time": 123.456,
                    "node": "n_plan_one",
                    "last_step": asdict(
                        step1
                    ),  # Dict (as created by add_event_to_trajectory)
                },
                {
                    "time": 124.567,
                    "node": "n_code",
                    "last_step": asdict(
                        step2
                    ),  # Dict (as created by add_event_to_trajectory)
                },
            ],
            token_usage=[{"input_tokens": 100, "output_tokens": 50}],
        )

        result = prepare_result_from_graph_state_ds_star_agent(state)

        # Verify trajectory events have last_step as dict
        assert len(result["trajectory"]) == 2

        for i, event in enumerate(result["trajectory"]):
            # Critical assertion: last_step must be a dict, not a string
            assert isinstance(event["last_step"], dict), (
                f"trajectory[{i}]['last_step'] is {type(event['last_step'])}, "
                f"expected dict. Value: {event['last_step']}"
            )

            # Verify it has expected fields
            assert "plan" in event["last_step"]
            assert "code" in event["last_step"]

        # Verify first event
        assert result["trajectory"][0]["last_step"]["plan"] == "Search files"
        assert result["trajectory"][0]["last_step"]["code"] == "print(1)"

        # Verify second event
        assert result["trajectory"][1]["last_step"]["plan"] == "Process data"
        assert result["trajectory"][1]["last_step"]["outputs"] == {"result": [1, 2, 3]}

    def test_result_is_json_serializable(self):
        """Test that entire result can be serialized to JSON."""
        step = DSStep(plan="test", code="print(1)")
        state = DSState(
            user_query="test",
            tools={},
            steps=[step],
            trajectory=[
                {
                    "time": 123.456,
                    "node": "n_plan_one",
                    "last_step": asdict(
                        step
                    ),  # Dict (as created by add_event_to_trajectory)
                }
            ],
        )

        result = prepare_result_from_graph_state_ds_star_agent(state)

        # Should not raise exception
        json_str = json.dumps(result)

        # Verify we can parse it back
        parsed = json.loads(json_str)
        assert parsed["trajectory"][0]["last_step"]["plan"] == "test"

    def test_state_snapshot_has_normalized_trajectory(self):
        """Test that state snapshot includes normalized trajectory."""
        step = DSStep(plan="test", code="print(1)")
        state = DSState(
            user_query="test",
            tools={},
            steps=[step],
            trajectory=[
                {
                    "time": 123.456,
                    "node": "n_plan_one",
                    "last_step": asdict(
                        step
                    ),  # Dict (as created by add_event_to_trajectory)
                }
            ],
        )

        result = prepare_result_from_graph_state_ds_star_agent(state)

        # Verify state snapshot has normalized trajectory
        assert "state" in result
        assert "trajectory" in result["state"]
        assert isinstance(result["state"]["trajectory"][0]["last_step"], dict)

    def test_last_step_field_is_dict(self):
        """Test that top-level last_step field is a dict."""
        step = DSStep(
            plan="final step",
            code="print('done')",
            verifier_sufficient=True,
            execution_error=None,
        )
        state = DSState(
            user_query="test",
            tools={},
            steps=[step],
            final_answer="answer",
        )

        result = prepare_result_from_graph_state_ds_star_agent(state)

        # Verify last_step is a dict
        assert isinstance(result["last_step"], dict)
        assert result["last_step"]["plan"] == "final step"
        assert result["last_step"]["verifier_sufficient"] is True
