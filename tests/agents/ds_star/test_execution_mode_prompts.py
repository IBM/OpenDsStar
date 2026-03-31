"""Tests to verify that prompts correctly reflect execution mode."""

from OpenDsStar.agents.ds_star.ds_star_state import CodeMode, DSState
from OpenDsStar.agents.ds_star.ds_star_utils import (
    build_execution_environment_instructions,
)


class TestExecutionModePrompts:
    """Test that prompts correctly reflect execution mode (stepwise vs full)."""

    def test_restricted_mode_prompt(self):
        """Test that restricted mode prompt mentions no imports."""
        state = DSState(
            user_query="test",
            tools={},
            code_mode=CodeMode.STEPWISE,
            steps=[],
        )

        instructions = build_execution_environment_instructions(
            state=state,
            mode="stepwise",
            include_prev_step_outputs=True,
        )

        # Should mention that imports are not allowed
        assert "No imports allowed" in instructions
        assert "import" in instructions.lower()

    def test_stepwise_mode_prompt(self):
        """Test that stepwise mode prompt mentions prev_step_outputs."""
        state = DSState(
            user_query="test",
            tools={},
            code_mode=CodeMode.STEPWISE,
            steps=[],
        )

        instructions = build_execution_environment_instructions(
            state=state,
            mode="stepwise",
            include_prev_step_outputs=True,
        )

        # Should mention prev_step_outputs in stepwise mode
        assert "prev_step_outputs" in instructions

    def test_full_mode_prompt(self):
        """Test that full mode prompt does not mention prev_step_outputs."""
        state = DSState(
            user_query="test",
            tools={},
            code_mode=CodeMode.FULL,
            steps=[],
        )

        instructions = build_execution_environment_instructions(
            state=state,
            mode="full",
            include_prev_step_outputs=False,
        )

        # Should NOT mention prev_step_outputs in full mode
        assert "prev_step_outputs" not in instructions
        # Should mention final results
        assert "final results" in instructions.lower()

    def test_preloaded_libraries_mentioned(self):
        """Test that instructions mention preloaded libraries."""
        state = DSState(
            user_query="test",
            tools={},
            code_mode=CodeMode.STEPWISE,
            steps=[],
        )

        instructions = build_execution_environment_instructions(
            state=state,
            mode="stepwise",
            include_prev_step_outputs=True,
        )

        # Should mention preloaded libraries
        assert "Preloaded libraries" in instructions
        assert "numpy" in instructions or "np" in instructions
        assert "pandas" in instructions or "pd" in instructions

    def test_filesystem_restrictions_with_tools(self):
        """Test that instructions warn against filesystem access when tools are available."""
        state = DSState(
            user_query="test",
            tools={},
            code_mode=CodeMode.STEPWISE,
            steps=[],
        )

        instructions = build_execution_environment_instructions(
            state=state,
            mode="stepwise",
            include_prev_step_outputs=True,
            include_tools_instructions=True,
        )

        # Should warn against direct file reading functions
        assert "no filesystem access" in instructions.lower()
        assert "pd.read_csv()" in instructions or "pd.read_parquet()" in instructions

        # Should direct users to check tool descriptions
        assert "use the provided tools" in instructions.lower()
        assert (
            "tool's description" in instructions.lower()
            or "tool description" in instructions.lower()
        )

    def test_filesystem_instructions_only_with_tools(self):
        """Test that filesystem warnings only appear when tools are enabled."""
        state = DSState(
            user_query="test",
            tools={},
            code_mode=CodeMode.STEPWISE,
            steps=[],
        )

        # With tools enabled
        with_tools = build_execution_environment_instructions(
            state=state,
            mode="stepwise",
            include_prev_step_outputs=True,
            include_tools_instructions=True,
        )

        # Without tools enabled
        without_tools = build_execution_environment_instructions(
            state=state,
            mode="stepwise",
            include_prev_step_outputs=True,
            include_tools_instructions=False,
        )

        # Filesystem warnings should only appear when tools are enabled
        assert "pd.read_csv()" in with_tools or "pd.read_parquet()" in with_tools
        assert "use the provided tools" in with_tools.lower()

        # Without tools, should have different message
        assert (
            "pd.read_csv()" not in without_tools
            and "pd.read_parquet()" not in without_tools
        )
