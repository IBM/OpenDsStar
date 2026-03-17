"""Integration test to verify trajectory last_step is saved as dict, not string."""

import json

from agents.ds_star.ds_star_results_prep import (
    prepare_result_from_graph_state_ds_star_agent,
)
from agents.ds_star.ds_star_state import DSState, DSStep


def test_prepare_result_maintains_dict_last_step():
    """Test that prepare_result maintains dict format for trajectory last_step."""
    # Create a state with trajectory containing DSStep objects
    step1 = DSStep(plan="Step 1", code="print(1)")
    step2 = DSStep(plan="Step 2", code="print(2)", outputs={"result": 42})

    state = DSState(
        user_query="test",
        tools={},
        steps=[step1, step2],
        trajectory=[
            {
                "time": 123.0,
                "node": "n_plan_one",
                "last_step": step1,  # DSStep object
            },
            {
                "time": 124.0,
                "node": "n_code",
                "last_step": step2,  # DSStep object
            },
        ],
    )

    # Prepare result
    result = prepare_result_from_graph_state_ds_star_agent(state)

    # Verify trajectory has dict last_step
    assert len(result["trajectory"]) == 2

    for i, event in enumerate(result["trajectory"]):
        assert isinstance(event["last_step"], dict), (
            f"result['trajectory'][{i}]['last_step'] is {type(event['last_step'])}, "
            f"expected dict"
        )
        assert "plan" in event["last_step"]
        assert "code" in event["last_step"]

    # Verify JSON serialization works
    json_str = json.dumps(result)
    parsed = json.loads(json_str)

    # Verify after round-trip
    for i, event in enumerate(parsed["trajectory"]):
        assert isinstance(event["last_step"], dict)
        assert event["last_step"]["plan"] == f"Step {i+1}"


def test_state_snapshot_trajectory_is_dict():
    """Test that state snapshot in result has trajectory with dict last_step."""
    step = DSStep(plan="Test", code="x=1", outputs={"x": 1})

    state = DSState(
        user_query="test",
        tools={},
        steps=[step],
        trajectory=[
            {
                "time": 123.0,
                "node": "n_plan_one",
                "last_step": step,
            }
        ],
    )

    result = prepare_result_from_graph_state_ds_star_agent(state)

    # Check state snapshot
    assert "state" in result
    assert "trajectory" in result["state"]
    assert len(result["state"]["trajectory"]) == 1

    # Verify last_step in state snapshot is dict
    snapshot_event = result["state"]["trajectory"][0]
    assert isinstance(snapshot_event["last_step"], dict)
    assert snapshot_event["last_step"]["plan"] == "Test"
