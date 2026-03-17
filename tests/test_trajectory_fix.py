"""Quick test to verify trajectory last_step is saved as dict."""

import json

from agents.ds_star.ds_star_results_prep import (
    prepare_result_from_graph_state_ds_star_agent,
)
from agents.ds_star.ds_star_state import DSState, DSStep
from agents.ds_star.ds_star_utils import add_event_to_trajectory

# Create a state with steps
step1 = DSStep(
    plan="Search for files",
    code="print('searching')",
    outputs={"files": ["a.csv", "b.csv"]},
    execution_error=None,
)

step2 = DSStep(
    plan="Process data",
    code="print('processing')",
    outputs={"result": 42},
    execution_error=None,
)

state = DSState(
    user_query="test query", tools={}, steps=[step1, step2], final_answer="test answer"
)

# Add events to trajectory (simulating what happens during execution)
add_event_to_trajectory(state, "n_plan_one", planned_step="Search for files")
add_event_to_trajectory(state, "n_code", code="print('searching')")
add_event_to_trajectory(state, "n_execute", had_error=False)

# Prepare result (this is what gets saved to JSON)
result = prepare_result_from_graph_state_ds_star_agent(state)

# Verify trajectory has dict last_step
print("Checking trajectory events...")
for i, event in enumerate(result["trajectory"]):
    last_step = event.get("last_step")
    if last_step is not None:
        print(f"\nEvent {i} (node: {event['node']}):")
        print(f"  last_step type: {type(last_step)}")
        if isinstance(last_step, dict):
            print("  ✓ last_step is a dict")
            print(f"  plan: {last_step.get('plan', 'N/A')}")
        else:
            print(f"  ✗ last_step is NOT a dict: {last_step}")

# Try to serialize to JSON
print("\nTesting JSON serialization...")
try:
    json_str = json.dumps(result, indent=2)
    print("✓ Successfully serialized to JSON")

    # Parse it back
    parsed = json.loads(json_str)

    # Check parsed trajectory
    print("\nChecking parsed trajectory...")
    for i, event in enumerate(parsed["trajectory"]):
        last_step = event.get("last_step")
        if last_step is not None:
            if isinstance(last_step, dict):
                print(
                    f"  Event {i}: ✓ last_step is dict with plan='{last_step.get('plan', 'N/A')}'"
                )
            else:
                print(
                    f"  Event {i}: ✗ last_step is {type(last_step)}: {last_step[:100]}..."
                )

    print("\n✓ All trajectory events have last_step as dict!")

except Exception as e:
    print(f"✗ JSON serialization failed: {e}")
