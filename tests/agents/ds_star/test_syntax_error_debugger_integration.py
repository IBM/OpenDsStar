"""Integration test: verify syntax errors are caught and routed to debugger."""

from unittest.mock import Mock

import pytest

from OpenDsStar.agents.ds_star.ds_star_graph import DSStarGraph
from OpenDsStar.agents.ds_star.ds_star_state import CodeMode, DSState


@pytest.mark.integration
def test_syntax_error_routes_to_debugger():
    """Test that syntax errors during validation are caught and debugger can fix them."""

    # Mock LLM responses - need to mock the full chain: with_structured_output().with_config().invoke()
    mock_llm = Mock()

    # Create a mock that supports the chaining pattern
    mock_with_config = Mock()
    mock_with_structured = Mock()
    mock_with_structured.with_config = Mock(return_value=mock_with_config)
    mock_llm.with_structured_output = Mock(return_value=mock_with_structured)

    # Set up invoke responses
    mock_with_config.invoke.side_effect = [
        # Planner: Generate plan
        Mock(content="Step 0: Calculate result"),
        # Coder: Generate code with syntax error
        Mock(code='outputs["result"] = 42\nprint(f"Result: {outputs[")'),
        # Debugger: Fix the syntax error
        Mock(code='outputs["result"] = 42\nprint(f"Result: {outputs[\'result\']}")'),
        # Verifier: Check if sufficient
        Mock(sufficient=True, explanation="Task completed"),
        # Finalizer: Generate final answer
        Mock(answer="The result is 42"),
    ]

    # Create graph with mocked LLM
    graph = DSStarGraph(model=mock_llm, tools=[], code_timeout=30)

    # Create initial state
    state = DSState(
        user_query="Calculate 42",
        tools={},
        max_steps=3,
        max_debug_tries=2,
        code_mode=CodeMode.STEPWISE,
    )

    # Invoke the graph
    result = graph.invoke(state)

    # Verify the flow:
    # 1. Syntax error was caught during execution
    assert len(result["steps"]) >= 1
    first_step = result["steps"][0]

    # 2. Syntax error was caught and stored in failed_code_attempts
    assert len(first_step.failed_code_attempts) >= 1
    assert "SyntaxError" in first_step.failed_code_attempts[0]["error"]

    # 3. Debug tries counter was incremented (debugger ran)
    assert first_step.debug_tries >= 1

    # 4. After debugger fixed it, code executed successfully (no execution_error)
    assert first_step.execution_error is None

    # 5. LLM was called multiple times (planner, coder, debugger, verifier, finalizer)
    assert mock_with_config.invoke.call_count >= 4


@pytest.mark.integration
def test_syntax_error_in_assert_no_imports():
    """Test that syntax errors in assert_no_imports are caught."""

    mock_llm = Mock()

    # Create a mock that supports the chaining pattern
    mock_with_config = Mock()
    mock_with_structured = Mock()
    mock_with_structured.with_config = Mock(return_value=mock_with_config)
    mock_llm.with_structured_output = Mock(return_value=mock_with_structured)

    # Coder generates code that will fail ast.parse in assert_no_imports
    mock_with_config.invoke.side_effect = [
        # Planner
        Mock(content="Step 0: Test"),
        # Coder with syntax error
        Mock(code='if True\n    outputs["x"] = 1'),  # Missing colon
        # Debugger fixes it
        Mock(code='if True:\n    outputs["x"] = 1'),
        # Verifier
        Mock(sufficient=True, explanation="Done"),
        # Finalizer
        Mock(answer="Complete"),
    ]

    graph = DSStarGraph(model=mock_llm, tools=[], code_timeout=30)
    state = DSState(
        user_query="Test syntax error handling",
        tools={},
        max_steps=2,
        code_mode=CodeMode.STEPWISE,
    )

    result = graph.invoke(state)

    # Verify syntax error was caught and debugger ran
    assert len(result["steps"]) >= 1
    first_step = result["steps"][0]

    # Syntax error was caught and stored in failed_code_attempts
    assert len(first_step.failed_code_attempts) >= 1
    assert "SyntaxError" in first_step.failed_code_attempts[0]["error"]

    # Debugger ran and fixed it
    assert first_step.debug_tries >= 1

    # After fix, code executed successfully
    assert first_step.execution_error is None


# Made with Bob
