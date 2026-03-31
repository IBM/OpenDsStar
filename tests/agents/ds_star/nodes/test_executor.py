"""Tests for ExecutorNode logic."""

import pytest

from OpenDsStar.agents.ds_star.ds_star_state import CodeMode, DSState, DSStep
from OpenDsStar.agents.ds_star.nodes.executer import ExecutorNode


@pytest.fixture
def executor_node():
    """Create an ExecutorNode instance for testing."""
    return ExecutorNode(
        system_prompt="test system",
        task_prompt="test task",
        tools_spec='[{"name": "test_tool", "description": "test"}]',
        tools={},
        code_timeout=30,
    )


class TestExecutorNodeBasicBehavior:
    """Test basic executor behavior."""

    def test_skips_on_fatal_error(self, executor_node):
        """Test that executor skips when fatal_error exists."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[DSStep(plan="step 0", code="print('test')")],
            fatal_error="Previous fatal error",
            steps_used=1,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = executor_node(state)

        assert result["fatal_error"] == "Previous fatal error"
        assert len(result["trajectory"]) == 1
        assert result["trajectory"][0]["skipped"] is True

    def test_skips_when_no_code(self, executor_node):
        """Test that execution is skipped when step has no code."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[DSStep(plan="step 0", code=None)],
            fatal_error=None,
            steps_used=1,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = executor_node(state)

        last_step = result["steps"][-1]
        assert last_step.logs == "[SKIPPED] No code for this step."
        assert last_step.outputs.get("_skipped") is True
        assert last_step.execution_error is None


class TestExecutorNodeCodeExecution:
    """Test code execution behavior."""

    def test_executes_simple_code(self, executor_node):
        """Test executing simple code."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[DSStep(plan="step 0", code="outputs['result'] = 42")],
            fatal_error=None,
            steps_used=1,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = executor_node(state)

        last_step = result["steps"][-1]
        assert last_step.outputs.get("result") == 42
        assert last_step.execution_error is None

    def test_captures_execution_error(self, executor_node):
        """Test that execution errors are captured."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[DSStep(plan="step 0", code="x = 1 / 0")],
            fatal_error=None,
            steps_used=1,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = executor_node(state)

        last_step = result["steps"][-1]
        assert last_step.execution_error is not None
        assert "division by zero" in last_step.execution_error
        assert last_step.outputs.get("_error") is not None
        assert last_step.outputs.get("_traceback") is not None

    def test_provides_call_tool_function(self):
        """Test that tools are available as direct functions in execution."""

        def mock_tool(arg):
            return f"tool_result_{arg}"

        executor_node = ExecutorNode(tools={"test_tool": mock_tool})

        state = DSState(
            user_query="test query",
            tools={"test_tool": mock_tool},
            steps=[
                DSStep(
                    plan="step 0",
                    code="outputs['result'] = test_tool(arg='value')",
                )
            ],
            fatal_error=None,
            steps_used=1,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = executor_node(state)

        last_step = result["steps"][-1]
        assert last_step.outputs.get("result") == "tool_result_value"

    def test_provides_state_access(self, executor_node):
        """Test that state is accessible in execution."""
        state = DSState(
            user_query="What is 2+2?",
            tools={},
            steps=[
                DSStep(plan="step 0", code="outputs['query'] = state['user_query']")
            ],
            fatal_error=None,
            steps_used=1,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = executor_node(state)

        last_step = result["steps"][-1]
        assert last_step.outputs.get("query") == "What is 2+2?"


class TestExecutorNodeStepwiseMode:
    """Test STEPWISE mode specific behavior."""

    def test_provides_prev_step_outputs(self, executor_node):
        """Test that prev_step_outputs is available in STEPWISE mode."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0", outputs={"data": [1, 2, 3]}),
                DSStep(
                    plan="step 1",
                    code="outputs['sum'] = sum(prev_step_outputs['data'])",
                ),
            ],
            fatal_error=None,
            steps_used=2,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = executor_node(state)

        last_step = result["steps"][-1]
        assert last_step.outputs.get("sum") == 6

    def test_prev_step_outputs_aggregates_all_previous(self, executor_node):
        """Test that prev_step_outputs aggregates all previous steps."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0", outputs={"a": 1}),
                DSStep(plan="step 1", outputs={"b": 2}),
                DSStep(
                    plan="step 2",
                    code="outputs['total'] = prev_step_outputs['a'] + prev_step_outputs['b']",
                ),
            ],
            fatal_error=None,
            steps_used=3,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = executor_node(state)

        last_step = result["steps"][-1]
        assert last_step.outputs.get("total") == 3


class TestExecutorNodeFullMode:
    """Test FULL mode specific behavior."""

    def test_full_mode_executes_complete_script(self, executor_node):
        """Test that FULL mode executes the complete script."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0"),
                DSStep(plan="step 1"),
                DSStep(
                    plan="step 2",
                    code="""
data = [1, 2, 3, 4, 5]
total = sum(data)
outputs['result'] = total
""",
                ),
            ],
            fatal_error=None,
            steps_used=3,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.FULL,
        )

        result = executor_node(state)

        last_step = result["steps"][-1]
        assert last_step.outputs.get("result") == 15


class TestExecutorNodePreloadedLibraries:
    """Test that preloaded libraries are available."""

    def test_numpy_available(self, executor_node):
        """Test that numpy is available as np."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0", code="outputs['arr'] = list(np.array([1, 2, 3]))")
            ],
            fatal_error=None,
            steps_used=1,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = executor_node(state)

        last_step = result["steps"][-1]
        assert last_step.outputs.get("arr") == [1, 2, 3]

    def test_pandas_available(self, executor_node):
        """Test that pandas is available as pd."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    code="df = pd.DataFrame({'a': [1, 2]}); outputs['sum'] = df['a'].sum()",
                )
            ],
            fatal_error=None,
            steps_used=1,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = executor_node(state)

        last_step = result["steps"][-1]
        assert last_step.outputs.get("sum") == 3

    def test_math_available(self, executor_node):
        """Test that math module is available."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[DSStep(plan="step 0", code="outputs['pi'] = round(math.pi, 2)")],
            fatal_error=None,
            steps_used=1,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = executor_node(state)

        last_step = result["steps"][-1]
        assert last_step.outputs.get("pi") == 3.14


class TestExecutorNodeLogging:
    """Test that print statements are captured."""

    def test_captures_print_output(self, executor_node):
        """Test that print output is captured in logs."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    code="print('Processing data'); outputs['done'] = True",
                )
            ],
            fatal_error=None,
            steps_used=1,
            max_steps=5,
            trajectory=[],
            token_usage=[],
            code_mode=CodeMode.STEPWISE,
        )

        result = executor_node(state)

        last_step = result["steps"][-1]
        assert last_step.logs is not None
        assert "Processing data" in last_step.logs
