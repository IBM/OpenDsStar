"""Advanced tests for RouterNode prompt building - non-trivial functionality."""

from OpenDsStar.agents.ds_star.ds_star_state import CodeMode, DSState, DSStep
from OpenDsStar.agents.ds_star.nodes.router import build_router_prompt


class TestRouterPromptModeSpecificBehavior:
    """Test mode-specific prompt building differences in router."""

    def test_stepwise_includes_all_step_details(self):
        """Test STEPWISE mode includes detailed code/logs/outputs for all steps."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    code="code_step_0 = 'first'",
                    logs="Step 0 executed successfully",
                    outputs={"data0": [1, 2, 3]},
                ),
                DSStep(
                    plan="step 1",
                    code="code_step_1 = 'second'",
                    logs="Step 1 processed data",
                    outputs={"data1": [4, 5, 6]},
                ),
                DSStep(
                    plan="step 2",
                    code="code_step_2 = 'third'",
                    logs="Step 2 final processing",
                    outputs={"result": 42},
                    verifier_sufficient=False,
                ),
            ],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_router_prompt(state)
        full_prompt = system_msg + "\n" + user_msg

        # All steps should have detailed information
        assert "code_step_0" in full_prompt
        assert "Step 0 executed successfully" in full_prompt
        assert "data0" in full_prompt

        assert "code_step_1" in full_prompt
        assert "Step 1 processed data" in full_prompt
        assert "data1" in full_prompt

        assert "code_step_2" in full_prompt
        assert "Step 2 final processing" in full_prompt
        assert "result" in full_prompt

        assert "STEPWISE mode" in full_prompt

    def test_full_mode_shows_plans_plus_last_execution_only(self):
        """Test FULL mode shows all plans but only last step's execution details."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0 plan",
                    code="old_code_0",
                    logs="Old logs 0",
                    outputs={"old_data": "should not appear in detail"},
                ),
                DSStep(
                    plan="step 1 plan",
                    code="old_code_1",
                    logs="Old logs 1",
                    outputs={"old_result": "also should not appear"},
                ),
                DSStep(
                    plan="step 2 plan",
                    code="latest_code",
                    logs="Latest logs",
                    outputs={"latest_result": "should appear"},
                    execution_error="Latest error",
                    verifier_sufficient=False,
                ),
            ],
            code_mode=CodeMode.FULL,
        )

        system_msg, user_msg = build_router_prompt(state)
        full_prompt = system_msg + "\n" + user_msg

        # All plans should be visible
        assert "step 0 plan" in full_prompt
        assert "step 1 plan" in full_prompt
        assert "step 2 plan" in full_prompt

        # Only last step's execution details should be visible
        assert "latest_code" in full_prompt
        assert "Latest logs" in full_prompt
        assert "latest_result" in full_prompt
        assert "Latest error" in full_prompt

        # Old execution details should NOT be visible
        assert "old_code_0" not in full_prompt
        assert "old_code_1" not in full_prompt
        assert "Old logs 0" not in full_prompt
        assert "Old logs 1" not in full_prompt

        assert "Current execution context (latest run)" in full_prompt

    def test_stepwise_truncates_history_to_last_n_steps(self):
        """Test STEPWISE mode truncates detailed history to last N steps."""
        # Create many steps
        steps = [
            DSStep(
                plan=f"step {i}",
                code=f"code_{i}",
                logs=f"logs_{i}",
                outputs={f"out_{i}": i},
            )
            for i in range(25)
        ]
        steps[-1].verifier_sufficient = False

        state = DSState(
            user_query="test query",
            tools={},
            steps=steps,
            code_mode=CodeMode.STEPWISE,
        )

        # Default stepwise_detail_last_n is 20
        system_msg, user_msg = build_router_prompt(state, stepwise_detail_last_n=20)
        full_prompt = system_msg + "\n" + user_msg

        # Should include last 20 steps (indices 5-24) in detail
        assert "Step 24:" in full_prompt
        assert "Step 20:" in full_prompt
        assert "Step 10:" in full_prompt
        assert "Step 5:" in full_prompt

        # Should NOT include very early steps in detailed history
        # (they appear in plan list but not in detailed step blocks)
        assert "Step 0:" not in full_prompt
        assert "Step 1:" not in full_prompt
        assert "Step 2:" not in full_prompt
        assert "Step 3:" not in full_prompt
        assert "Step 4:" not in full_prompt

        assert "last 20 steps" in full_prompt


class TestRouterPromptTruncation:
    """Test truncation behavior in router prompts."""

    def test_truncates_long_outputs(self):
        """Test that long outputs are truncated."""
        long_output = "x" * 10000
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    code="code",
                    outputs={"huge_data": long_output},
                    verifier_sufficient=False,
                ),
            ],
            code_mode=CodeMode.STEPWISE,
            output_max_length=100,
        )

        system_msg, user_msg = build_router_prompt(state)
        full_prompt = system_msg + "\n" + user_msg

        # Should be truncated
        assert "..." in full_prompt
        # Prompt should not be excessively long
        assert len(full_prompt) < 15000

    def test_truncates_long_logs(self):
        """Test that long logs are truncated."""
        long_logs = "log line\n" * 1000
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    code="code",
                    logs=long_logs,
                    verifier_sufficient=False,
                ),
            ],
            code_mode=CodeMode.STEPWISE,
            logs_max_length=200,
        )

        system_msg, user_msg = build_router_prompt(state)
        full_prompt = system_msg + "\n" + user_msg

        # Should be truncated
        assert "..." in full_prompt
        assert len(full_prompt) < 10000


class TestRouterPromptVerifierIntegration:
    """Test how router prompt integrates verifier information."""

    def test_includes_verifier_sufficient_status(self):
        """Test that verifier sufficient status is included."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    verifier_sufficient=False,
                    verifier_explanation="Missing key information",
                ),
            ],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_router_prompt(state)
        full_prompt = system_msg + "\n" + user_msg

        assert "sufficient=False" in full_prompt
        assert "Missing key information" in full_prompt
        assert "Latest verifier judgment:" in full_prompt

    def test_includes_verifier_explanation(self):
        """Test that verifier explanation is properly formatted."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    verifier_sufficient=True,
                    verifier_explanation="All requirements met, data is complete",
                ),
            ],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_router_prompt(state)
        full_prompt = system_msg + "\n" + user_msg

        assert "sufficient=True" in full_prompt
        assert "All requirements met, data is complete" in full_prompt


class TestRouterPromptInstructions:
    """Test router-specific instructions in prompts."""

    def test_includes_decision_logic_guidance(self):
        """Test that decision logic guidance is included."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0", verifier_sufficient=False),
            ],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_router_prompt(state)
        full_prompt = system_msg + "\n" + user_msg

        assert "DECISION LOGIC GUIDANCE" in full_prompt
        assert "add_next_step" in full_prompt
        assert "fix_step" in full_prompt

    def test_includes_output_format_rules(self):
        """Test that output format rules are included."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0", verifier_sufficient=False),
            ],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_router_prompt(state)
        full_prompt = system_msg + "\n" + user_msg

        assert "OUTPUT RULES" in full_prompt
        assert "JSON" in full_prompt
        assert "RouterOutput" in full_prompt

    def test_mentions_step_index_requirement(self):
        """Test that step_index requirement for fix_step is mentioned."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0", verifier_sufficient=False),
            ],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_router_prompt(state)
        full_prompt = system_msg + "\n" + user_msg

        assert "step_index" in full_prompt
        assert "0-based index" in full_prompt


class TestRouterPromptEdgeCases:
    """Test edge cases in router prompt building."""

    def test_handles_steps_with_no_code(self):
        """Test handling of steps with no code."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0", verifier_sufficient=False),
            ],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_router_prompt(state)
        full_prompt = system_msg + "\n" + user_msg

        # Should handle gracefully
        assert "(none)" in full_prompt
        assert "Code:" in full_prompt

    def test_handles_steps_with_no_logs(self):
        """Test handling of steps with no logs."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0", code="some_code", verifier_sufficient=False),
            ],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_router_prompt(state)
        full_prompt = system_msg + "\n" + user_msg

        assert "(none)" in full_prompt
        assert "Logs" in full_prompt

    def test_handles_steps_with_no_outputs(self):
        """Test handling of steps with no outputs."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(plan="step 0", code="some_code", verifier_sufficient=False),
            ],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_router_prompt(state)
        full_prompt = system_msg + "\n" + user_msg

        assert "No outputs produced" in full_prompt

    def test_handles_steps_with_no_errors(self):
        """Test handling of steps with no execution errors."""
        state = DSState(
            user_query="test query",
            tools={},
            steps=[
                DSStep(
                    plan="step 0",
                    code="some_code",
                    execution_error=None,
                    verifier_sufficient=False,
                ),
            ],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_router_prompt(state)
        full_prompt = system_msg + "\n" + user_msg

        assert "Execution error:" in full_prompt
        assert "(none)" in full_prompt


class TestRouterPromptContextQuality:
    """Test that router prompt provides sufficient context for decision making."""

    def test_includes_all_plan_steps_for_context(self):
        """Test that all plan steps are included for context."""
        state = DSState(
            user_query="Analyze sales data",
            tools={},
            steps=[
                DSStep(plan="Load sales data from database"),
                DSStep(plan="Clean and preprocess the data"),
                DSStep(plan="Calculate monthly aggregates"),
                DSStep(plan="Generate visualization", verifier_sufficient=False),
            ],
            code_mode=CodeMode.FULL,
        )

        system_msg, user_msg = build_router_prompt(state)
        full_prompt = system_msg + "\n" + user_msg

        # All plan steps should be visible for context
        assert "Load sales data from database" in full_prompt
        assert "Clean and preprocess the data" in full_prompt
        assert "Calculate monthly aggregates" in full_prompt
        assert "Generate visualization" in full_prompt

        # Should show 0-based indices
        assert "0." in full_prompt
        assert "1." in full_prompt
        assert "2." in full_prompt
        assert "3." in full_prompt

    def test_includes_user_query_for_context(self):
        """Test that user query is included for decision context."""
        state = DSState(
            user_query="What are the top 5 products by revenue in Q4 2023?",
            tools={},
            steps=[
                DSStep(plan="Query database", verifier_sufficient=False),
            ],
            code_mode=CodeMode.STEPWISE,
        )

        system_msg, user_msg = build_router_prompt(state)
        full_prompt = system_msg + "\n" + user_msg

        assert "What are the top 5 products by revenue in Q4 2023?" in full_prompt
        assert "User query:" in full_prompt
