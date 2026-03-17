"""Test that timeout works in non-main threads."""

import threading
import time
from typing import Any, Dict

from agents.ds_star.ds_star_execute_env import execute_user_code
from agents.ds_star.ds_star_state import CodeMode, DSState


def test_timeout_in_worker_thread():
    """Test that timeout works when execute_user_code is called from a worker thread."""

    # Code that will hang indefinitely
    hanging_code = """
import time
print("Starting infinite loop...")
while True:
    time.sleep(0.1)
"""

    state = DSState(
        user_query="test",
        tools={},
        steps=[],
        code_mode=CodeMode.STEPWISE,
    )

    result_container: Dict[str, Any] = {}

    def run_in_thread():
        """Run execute_user_code in a worker thread."""
        logs, outputs = execute_user_code(
            code=hanging_code,
            state=state,
            tools={},
            timeout=2,  # 2 second timeout
        )
        result_container["logs"] = logs
        result_container["outputs"] = outputs

    # Run in a worker thread (not main thread)
    thread = threading.Thread(target=run_in_thread)
    start_time = time.time()
    thread.start()
    thread.join(timeout=5)  # Give it max 5 seconds
    elapsed = time.time() - start_time

    # Verify thread completed (didn't hang)
    assert not thread.is_alive(), "Thread should have completed"

    # Verify it completed in reasonable time (around 2 seconds, not 5+)
    assert elapsed < 4, f"Should timeout around 2s, took {elapsed:.1f}s"

    # Verify we got a timeout error
    assert "_error" in result_container["outputs"], "Should have timeout error"
    assert (
        "timeout" in result_container["outputs"]["_error"].lower()
        or "exceeded" in result_container["outputs"]["_error"].lower()
    ), f"Error should mention timeout: {result_container['outputs']['_error']}"

    print(f"✓ Test passed: Timeout worked in worker thread (took {elapsed:.1f}s)")


def test_timeout_with_large_dataframe_operation():
    """Test timeout with expensive pandas operation (simulating the original issue)."""

    # Code that simulates expensive df.info() on large dataframe
    expensive_code = """
import pandas as pd
import numpy as np
import time

print("Creating large dataframe...")
# Create a large dataframe
df = pd.DataFrame(np.random.randn(1000000, 50))

print("Starting expensive operation...")
# Simulate expensive operation
for i in range(100):
    _ = df.describe()
    time.sleep(0.1)  # Make it take time

outputs['result'] = 'completed'
"""

    state = DSState(
        user_query="test",
        tools={},
        steps=[],
        code_mode=CodeMode.STEPWISE,
    )

    result_container: Dict[str, Any] = {}

    def run_in_thread():
        """Run execute_user_code in a worker thread."""
        logs, outputs = execute_user_code(
            code=expensive_code,
            state=state,
            tools={},
            timeout=3,  # 3 second timeout
        )
        result_container["logs"] = logs
        result_container["outputs"] = outputs

    thread = threading.Thread(target=run_in_thread)
    start_time = time.time()
    thread.start()
    thread.join(timeout=6)
    elapsed = time.time() - start_time

    assert not thread.is_alive(), "Thread should have completed"
    assert elapsed < 5, f"Should timeout around 3s, took {elapsed:.1f}s"

    # Should have timed out
    assert "_error" in result_container["outputs"], "Should have timeout error"

    print(
        f"✓ Test passed: Expensive operation timed out correctly (took {elapsed:.1f}s)"
    )


if __name__ == "__main__":
    print("Testing timeout fix...")
    print("\n1. Testing basic timeout in worker thread:")
    test_timeout_in_worker_thread()

    print("\n2. Testing timeout with expensive pandas operation:")
    test_timeout_with_large_dataframe_operation()

    print("\n✓ All tests passed!")

# Made with Bob
