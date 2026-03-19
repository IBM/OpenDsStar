"""Test that syntax errors during code validation are caught and routed to debugger."""

from agents.ds_star.ds_star_execute_env import execute_user_code
from agents.ds_star.ds_star_state import CodeMode, DSState


def test_syntax_error_during_validation():
    """Test that SyntaxError during ast.parse is caught and returned as execution error."""
    # Code with syntax error (unclosed bracket in f-string)
    malformed_code = """
import pandas as pd

def rank_longitude_values():
    outputs['result'] = 'test'
    print(f'Lowest 4 longitude values: {outputs[
"""

    # Create minimal state
    state = DSState(
        user_query="test question",
        tools={},
        code_mode=CodeMode.STEPWISE,
    )

    # Execute should catch SyntaxError and return it as execution error
    logs, outputs = execute_user_code(malformed_code, state, {}, timeout=5)

    # Verify error is captured
    assert isinstance(outputs, dict)
    assert "_error" in outputs
    assert "SyntaxError" in outputs["_error"]
    assert "_traceback" in outputs
    # Check for syntax error message (varies by Python version)
    error_text = outputs["_error"] + outputs["_traceback"]
    assert (
        "was never closed" in error_text
        or "unterminated string literal" in error_text
    )

    # Logs should be empty since code never executed
    assert logs == ""


def test_valid_code_still_works():
    """Test that valid code still executes normally after the fix."""
    valid_code = """
outputs['result'] = 42
print('Success')
"""

    state = DSState(
        user_query="test question",
        tools={},
        code_mode=CodeMode.STEPWISE,
    )

    logs, outputs = execute_user_code(valid_code, state, {}, timeout=5)

    # Verify successful execution
    assert isinstance(outputs, dict)
    assert "_error" not in outputs
    assert outputs.get("result") == 42
    assert "Success" in logs


def test_runtime_error_still_caught():
    """Test that runtime errors (not syntax errors) are still caught normally."""
    code_with_runtime_error = """
outputs['result'] = 1 / 0  # ZeroDivisionError
"""

    state = DSState(
        user_query="test question",
        tools={},
        code_mode=CodeMode.STEPWISE,
    )

    logs, outputs = execute_user_code(code_with_runtime_error, state, {}, timeout=5)

    # Verify runtime error is captured
    assert isinstance(outputs, dict)
    assert "_error" in outputs
    assert "division by zero" in outputs["_error"]
    assert "_traceback" in outputs
    assert "ZeroDivisionError" in outputs["_traceback"]


# Made with Bob
