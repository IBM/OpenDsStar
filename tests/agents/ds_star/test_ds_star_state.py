"""Tests for DSState and DSStep - meaningful tests only."""

from agents.ds_star.ds_star_state import CodeMode, DSState, DSStep


class TestDSStepDictionaryBehavior:
    """Test DSStep dictionary-like behavior (important for LangGraph)."""

    def test_getitem_setitem_delitem(self):
        """Test dictionary-style operations work correctly."""
        step = DSStep(plan="Test plan", code="test code")

        # Test getitem
        assert step["plan"] == "Test plan"
        assert step["code"] == "test code"

        # Test setitem
        step["code"] = "new code"
        assert step.code == "new code"

        # Test delitem
        del step["code"]
        assert step.code is None


class TestCodeModeEnum:
    """Test CodeMode enum behavior."""

    def test_enum_from_string(self):
        """Test creating enum from string value (important for config)."""
        mode = CodeMode("stepwise")
        assert mode == CodeMode.STEPWISE

        mode = CodeMode("full")
        assert mode == CodeMode.FULL


class TestDSStateDictionaryBehavior:
    """Test DSState dictionary-like behavior (important for LangGraph)."""

    def test_getitem_setitem_delitem(self):
        """Test dictionary-style operations work correctly."""
        state = DSState(user_query="Test", tools={}, final_answer="Answer")

        # Test getitem
        assert state["user_query"] == "Test"
        assert state["final_answer"] == "Answer"

        # Test setitem
        state["final_answer"] = "New answer"
        assert state.final_answer == "New answer"

        # Test delitem
        del state["final_answer"]
        assert state.final_answer is None

    def test_steps_accumulation(self):
        """Test adding steps to state (important for workflow)."""
        state = DSState(user_query="Test", tools={})
        assert len(state.steps) == 0

        state.steps.append(DSStep(plan="Step 1"))
        assert len(state.steps) == 1

        state.steps.append(DSStep(plan="Step 2"))
        assert len(state.steps) == 2
        assert state.steps[0].plan == "Step 1"
        assert state.steps[1].plan == "Step 2"

    def test_trajectory_accumulation(self):
        """Test adding events to trajectory (important for debugging)."""
        state = DSState(user_query="Test", tools={})
        assert len(state.trajectory) == 0

        state.trajectory.append({"event": "start", "data": "test"})
        assert len(state.trajectory) == 1
        assert state.trajectory[0]["event"] == "start"

        state.trajectory.append({"event": "end", "data": "done"})
        assert len(state.trajectory) == 2

    def test_token_usage_accumulation(self):
        """Test adding token usage records (important for cost tracking)."""
        state = DSState(user_query="Test", tools={})
        assert len(state.token_usage) == 0

        state.token_usage.append({"input": 100, "output": 50})
        assert len(state.token_usage) == 1
        assert state.token_usage[0]["input"] == 100

        state.token_usage.append({"input": 200, "output": 100})
        total_input = sum(t["input"] for t in state.token_usage)
        assert total_input == 300
