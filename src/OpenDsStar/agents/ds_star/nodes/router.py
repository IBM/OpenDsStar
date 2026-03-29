import logging
from enum import Enum
from typing import List, Optional, Type

from pydantic import BaseModel, Field, ValidationError

from OpenDsStar.agents.ds_star.ds_star_state import CodeMode, DSState
from OpenDsStar.agents.ds_star.ds_star_utils import (
    add_event_to_trajectory,
    invoke_structured_with_usage,
    truncate_text,
)
from OpenDsStar.agents.ds_star.nodes.base_node import BaseNode
from OpenDsStar.agents.utils.agents_utils import print_once

logger = logging.getLogger(__name__)


class RouteAction(str, Enum):
    add_next_step = "add_next_step"
    fix_step = "fix_step"


class RouterOutput(BaseModel):
    action: RouteAction = Field(..., description="add_next_step or fix_step")
    step_index: Optional[int] = Field(
        None,
        description="0-based index of the step to fix when action==fix_step. Ignore otherwise.",
    )
    explanation: str = Field(
        ..., description="Explain why you decided to take this action."
    )


def build_router_prompt(
    state: DSState,
    *,
    stepwise_detail_last_n: int = 20,
) -> tuple[str, str]:
    """
    Returns (system_message, user_message) so the caller can send a proper system role.
    Produces a structured RouterOutput.
    """
    if not state.steps:
        system_instructions = (
            "You are the Router agent in a DS-STAR-style data science system.\n"
            "You decide whether to add a next step or fix a prior step.\n"
            "Output RouterOutput as JSON."
        )
        user_message = (
            f"User query:\n{state.user_query.strip()}\n\n"
            "No steps exist; you cannot route meaningfully.\n"
            'Return: {"action":"add_next_step","step_index":null,"explanation":"No steps yet."}'
        )
        return system_instructions, user_message

    output_max_length = state.output_max_length
    logs_max_length = state.logs_max_length
    last_step = state.steps[-1]
    code_mode = state.code_mode

    # --- Always show user query ---
    user_block = f"User query:\n{state.user_query.strip()}\n"

    # --- Always show ALL plan steps (0-based) ---
    plan_lines = [f"{i}. {step.plan.strip()}" for i, step in enumerate(state.steps)]
    plan_block = "Current plan steps (0-based indices):\n" + "\n".join(plan_lines)

    # --- Mode-dependent visibility of code/outputs/errors/logs ---
    if code_mode == CodeMode.STEPWISE:
        # Avoid prompt bloat: include detailed history only for last N steps
        start_i = max(0, len(state.steps) - stepwise_detail_last_n)

        per_step_blocks: List[str] = []
        for i in range(start_i, len(state.steps)):
            s = state.steps[i]
            code_text = s.code or "(none)"
            logs_text = (
                truncate_text(s.logs.strip(), logs_max_length) if s.logs else "(none)"
            )
            outputs_text = (
                str(s.outputs) if s.outputs else "No outputs produced in this step."
            )
            outputs_text = truncate_text(outputs_text, output_max_length)
            error_text = s.execution_error or "(none)"

            per_step_blocks.append(
                f"Step {i}:\n"
                f"  Planned action:\n{s.plan.strip()}\n\n"
                f"  Code:\n{code_text}\n\n"
                f"  Logs (print outputs):\n{logs_text}\n\n"
                f"  Outputs:\n{outputs_text}\n\n"
                f"  Execution error:\n{error_text}\n"
            )

        history_block = (
            f"Recent step history with code, logs, outputs, and errors "
            f"(last {len(per_step_blocks)} steps; STEPWISE mode):\n"
            + "\n\n".join(per_step_blocks)
        )

    else:
        # FULL mode: show planned actions for all steps + last step execution context
        history_lines = [
            f"Step {i} – Planned action:\n{s.plan.strip()}"
            for i, s in enumerate(state.steps)
        ]
        history_block = "Step history (planned actions):\n" + "\n\n".join(history_lines)

        last_code = last_step.code or "(none)"
        last_logs = (
            truncate_text(last_step.logs.strip(), logs_max_length)
            if last_step.logs
            else "(none)"
        )
        last_outputs = (
            str(last_step.outputs) if last_step.outputs else "No outputs produced."
        )
        last_outputs = truncate_text(last_outputs, output_max_length)
        last_error = last_step.execution_error or "(none)"

        history_block += (
            "\n\nCurrent execution context (latest run):\n"
            f"Code (latest):\n{last_code}\n\n"
            f"Logs (print outputs, latest):\n{last_logs}\n\n"
            f"Outputs (latest):\n{last_outputs}\n\n"
            f"Execution error (latest):\n{last_error}\n"
        )

    # --- Verifier info from last step ---
    verifier_info = (
        f"sufficient={last_step.verifier_sufficient}, "
        f"explanation={repr(last_step.verifier_explanation)}"
    )

    system_instructions = (
        "You are the Router agent in a DS-STAR-style data science system.\n"
        "Your task is to decide how the planning process should proceed, "
        "based on the verifier's latest judgment and the current plan and results.\n\n"
        "You must output a structured object of type RouterOutput.\n"
        "RouterOutput has exactly these fields:\n"
        '  - action: one of "add_next_step" or "fix_step"\n'
        '  - step_index: 0-based index of the step to fix (ONLY when action == "fix_step"; otherwise null)\n'
        "  - explanation: a short natural-language explanation for your decision\n\n"
        "OUTPUT RULES:\n"
        "- You MUST output a SINGLE JSON object that is directly parseable into RouterOutput.\n"
        "- Do NOT wrap the JSON in markdown, backticks, or prose.\n"
        "- Do NOT output any text before or after the JSON.\n"
        '- The JSON must contain EXACTLY the fields: "action", "step_index", "explanation".\n\n'
        "DECISION LOGIC GUIDANCE:\n"
        '- If the verifier says the solution is sufficient, choose: {"action": "add_next_step", ...}\n'
        "  (The outer loop will stop; this is just a sentinel.)\n"
        '- If the reasoning is mostly correct but incomplete → choose "add_next_step".\n'
        '- If a specific earlier step appears incorrect → choose "fix_step" and specify its index.\n'
    )

    user_message = (
        f"{user_block}\n\n"
        f"{plan_block}\n\n"
        f"{history_block}\n\n"
        f"Latest verifier judgment:\n{verifier_info}\n\n"
        "Now provide ONLY the JSON object for RouterOutput."
    )

    logger.info(
        "router prompt length: %d", len(system_instructions) + len(user_message)
    )
    return system_instructions, user_message


class RouterNode(BaseNode):
    structured_output_schema: Optional[Type[BaseModel]] = RouterOutput

    def __call__(self, state: DSState) -> DSState:
        """Router decides next action when task is not sufficient."""
        # Defensive: if something upstream passed a dict, fail loudly and early.
        if not isinstance(state, DSState):
            raise TypeError(
                f"RouterNode expected DSState, got {type(state).__name__}. "
                "Fix the caller to pass DSState, or refactor this node to use dict access."
            )

        if state.fatal_error:
            add_event_to_trajectory(
                state, "n_route", skipped=True, fatal_error=state.fatal_error
            )
            return state

        if not state.steps:
            state.fatal_error = "Router called with no steps executed."
            add_event_to_trajectory(
                state, "n_route", skipped=True, fatal_error=state.fatal_error
            )
            return state

        last_step = state.steps[-1]
        i = len(state.steps) - 1

        # If verifier already marked sufficient on the last step, skip routing
        if last_step.verifier_sufficient:
            add_event_to_trajectory(
                state, "n_route", skipped=True, reason="already_sufficient"
            )
            return state

        system_msg, user_msg = build_router_prompt(state)
        print_once("ds_star_router_prompt", f"DS Star Router prompt:\n{user_msg}")

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        try:
            response, usage = invoke_structured_with_usage(
                self.llm_with_output, messages, run_name="n_route"
            )
            state.token_usage.append(usage)

            action = response.action.value
            fix_idx = response.step_index
            explanation = response.explanation
        except ValidationError as ve:
            state.fatal_error = f"Router schema validation failed: {ve}"
            logger.error("Fatal error routing at step %s: %s", i, state.fatal_error)
            add_event_to_trajectory(state, "n_route", fatal_error=state.fatal_error)
            return state
        except Exception as e:
            state.fatal_error = f"Router invocation failed: {e}"
            logger.error("Fatal error routing at step %s: %s", i, state.fatal_error)
            add_event_to_trajectory(state, "n_route", fatal_error=state.fatal_error)
            return state

        if action == RouteAction.add_next_step.value:
            last_step.router_action = RouteAction.add_next_step.value
            last_step.router_fix_index = None
            last_step.router_explanation = explanation
            logger.info("Router: add_next_step, explanation: %s", explanation)
            add_event_to_trajectory(
                state,
                "n_route",
                decision=RouteAction.add_next_step.value,
                explanation=explanation,
                fix_index=None,
            )
            return state

        if action == RouteAction.fix_step.value:
            if fix_idx is None:
                raise Exception("action is fix_step but fix_idx is None")

            original_fix_idx = fix_idx
            clamped_fix_idx = max(0, min(int(fix_idx), len(state.steps) - 1))
            if clamped_fix_idx != original_fix_idx:
                logger.warning(
                    "Router requested step_index=%s but clamped to %s (steps length=%s)",
                    original_fix_idx,
                    clamped_fix_idx,
                    len(state.steps),
                )

            last_step.router_action = RouteAction.fix_step.value
            last_step.router_fix_index = clamped_fix_idx
            last_step.router_explanation = explanation

            logger.info(
                "Router: fix_step at index %s, explanation: %s",
                clamped_fix_idx,
                explanation,
            )
            add_event_to_trajectory(
                state,
                "n_route",
                decision=RouteAction.fix_step.value,
                explanation=explanation,
                fix_index=clamped_fix_idx,
            )
            return state

        state.fatal_error = f"Unknown router action: {action}"
        add_event_to_trajectory(state, "n_route", fatal_error=state.fatal_error)
        return state
