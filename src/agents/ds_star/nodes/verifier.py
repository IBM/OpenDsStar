import logging
from typing import List, Optional, Type

from pydantic import BaseModel, Field, ValidationError

from agents.ds_star.ds_star_state import CodeMode, DSState, DSStep
from agents.ds_star.ds_star_utils import (
    add_event_to_trajectory,
    invoke_structured_with_usage,
    truncate_text,
)
from agents.ds_star.nodes.base_node import BaseNode
from agents.utils.agents_utils import print_once

logger = logging.getLogger(__name__)


class VerifierOutput(BaseModel):
    sufficient: bool = Field(
        ..., description="Whether the overall task is now sufficiently answered."
    )
    explanation: str = Field(
        ...,
        description="Explain why you decided that the task is sufficiently answered or not.",
    )


def _summarize_step_for_verifier(
    step: DSStep,
    index: int,
    output_max_length: int,
    logs_max_length: int,
) -> str:
    parts: List[str] = [f"Step {index} – Planned action:\n{step.plan.strip()}"]

    if step.logs:
        logs_text = truncate_text(step.logs.strip(), logs_max_length)
        parts.append(f"  Logs (print outputs):\n{logs_text}")

    if step.execution_error:
        parts.append(f"  Execution error: {step.execution_error.strip()}")
    elif step.outputs:
        out_summaries: List[str] = []
        for k, v in step.outputs.items():
            v_repr = truncate_text(repr(v), output_max_length)
            out_summaries.append(f"{k} = {v_repr}")
        if out_summaries:
            parts.append("  Outputs:\n    " + "\n    ".join(out_summaries))

    if step.verifier_sufficient is not None:
        suff = "sufficient" if step.verifier_sufficient else "NOT sufficient"
        parts.append(f"  Verifier: {suff}")
        if step.verifier_explanation:
            parts.append(
                f"    Verifier explanation: {step.verifier_explanation.strip()}"
            )

    if step.router_action:
        parts.append(f"  Router action: {step.router_action}")
        if step.router_fix_index is not None:
            parts.append(f"    Fix index: {step.router_fix_index}")
        if step.router_explanation:
            parts.append(f"    Router explanation: {step.router_explanation.strip()}")

    return "\n".join(parts)


def build_verifier_prompt(
    state: DSState,
    step_index: int,
    *,
    stepwise_detail_last_n: int = 20,
) -> tuple[str, str]:
    """
    Returns (system_message, user_message) so the caller can send a proper system role.
    """
    if not state.steps:
        # Should generally not happen, but keep it safe.
        system_instructions = (
            "You are the Verifier agent in a DS-STAR-style system.\n"
            "Your task is to decide whether the current state is SUFFICIENT to answer the user query.\n"
            "Output: VerifierOutput(sufficient=<bool>, explanation=<str>)."
        )
        user_message = (
            f"User query:\n{state.user_query.strip()}\n\n"
            "No steps have been executed yet, so the task is not answered.\n"
            "Return VerifierOutput(sufficient=False, explanation=...)."
        )
        return system_instructions, user_message

    step = state.steps[step_index]
    code_mode = state.code_mode
    output_max_length = state.output_max_length
    logs_max_length = state.logs_max_length

    system_instructions = (
        "You are the Verifier agent in a DS-STAR-style system.\n"
        "Your task is to decide whether the current plan and its execution are "
        "SUFFICIENT to answer the user query.\n\n"
        "You must output a structured judgment: VerifierOutput(sufficient=<bool>, explanation=<str>).\n\n"
        "Guidelines:\n"
        "- Mark sufficient=True only if the outputs correctly and directly answer the user query.\n"
        "- If important information is missing and an additional reasonable step could fill the gap, "
        "mark sufficient=False.\n"
        "- If the underlying data is inherently incomplete (for example, fewer items exist than requested), "
        "but the answer clearly explains this limitation and returns all available results, "
        "you may mark sufficient=True.\n"
    )

    user_block = f"User query:\n{state.user_query.strip()}\n"

    plan_lines = [f"{i}. {s.plan.strip()}" for i, s in enumerate(state.steps)]
    plan_block = "Current plan (0-based steps):\n" + "\n".join(plan_lines)

    # Compact history summary is useful in both modes
    history_lines = [
        _summarize_step_for_verifier(s, i, output_max_length, logs_max_length)
        for i, s in enumerate(state.steps)
    ]
    history_block = "Step history and outcomes:\n" + "\n\n".join(history_lines)

    if code_mode == CodeMode.STEPWISE:
        # In STEPWISE mode, avoid prompt bloat: include *detailed* dump for last N steps only.
        start_i = max(0, len(state.steps) - stepwise_detail_last_n)
        per_step_blocks: List[str] = []
        for i in range(start_i, len(state.steps)):
            s = state.steps[i]
            code_text = s.code or "(none)"
            logs_text = (
                truncate_text(s.logs.strip(), logs_max_length) if s.logs else "(none)"
            )
            outputs_text = str(s.outputs) if s.outputs else "No outputs produced."
            outputs_text = truncate_text(outputs_text, output_max_length)
            error_text = s.execution_error or "(none)"

            per_step_blocks.append(
                f"Step {i} – Code and results:\n"
                f"Code:\n{code_text}\n\n"
                f"Logs (print outputs):\n{logs_text}\n\n"
                f"Outputs:\n{outputs_text}\n\n"
                f"Execution error:\n{error_text}\n"
            )

        detail_block = (
            f"Code + logs + outputs + errors (last {len(per_step_blocks)} steps; STEPWISE mode):\n"
            + "\n\n".join(per_step_blocks)
        )
    else:
        # FULL mode: focus on last step execution
        last_code = step.code or "(none)"
        last_logs = (
            truncate_text(step.logs.strip(), logs_max_length) if step.logs else "(none)"
        )
        last_outputs = str(step.outputs) if step.outputs else "No outputs produced."
        last_outputs = truncate_text(last_outputs, output_max_length)
        last_error = step.execution_error or "(none)"

        detail_block = (
            "Final execution context (FULL mode, last step only):\n"
            f"Code (last step):\n{last_code}\n\n"
            f"Logs (print outputs, last step):\n{last_logs}\n\n"
            f"Outputs (last step):\n{last_outputs}\n\n"
            f"Execution error (last step):\n{last_error}\n"
        )

    user_message = (
        f"{user_block}\n\n"
        f"{plan_block}\n\n"
        f"{history_block}\n\n"
        f"{detail_block}\n\n"
        "Now decide whether the current overall state is sufficient to answer the query. "
        "Return your judgment as VerifierOutput(sufficient=<bool>, explanation=<str>)."
    )

    logger.info(
        "verifier prompt length: %d", len(system_instructions) + len(user_message)
    )
    return system_instructions, user_message


class VerifierNode(BaseNode):
    structured_output_schema: Optional[Type[BaseModel]] = VerifierOutput

    def __call__(self, state: DSState) -> DSState:
        # Defensive: if something upstream passed a dict, fail loudly and early.
        if not isinstance(state, DSState):
            raise TypeError(
                f"VerifierNode expected DSState, got {type(state).__name__}. "
                "Fix the caller to pass DSState, or refactor this node to use dict access."
            )

        if state.fatal_error:
            add_event_to_trajectory(
                state, "n_verify", skipped=True, fatal_error=state.fatal_error
            )
            return state

        # If no steps exist, we cannot verify completion.
        if not state.steps:
            state.fatal_error = "Verifier called with no steps executed."
            add_event_to_trajectory(
                state, "n_verify", skipped=True, fatal_error=state.fatal_error
            )
            return state

        # Check if max_steps reached before verification
        if state.steps_used >= state.max_steps:
            logger.warning(
                "Max steps reached (%s >= %s)", state.steps_used, state.max_steps
            )
            state.fatal_error = f"Max step limit reached ({state.max_steps})"
            add_event_to_trajectory(
                state, "n_verify", skipped=True, fatal_error=state.fatal_error
            )
            return state

        last_index = len(state.steps) - 1
        last_step = state.steps[last_index]

        system_msg, user_msg = build_verifier_prompt(state, step_index=last_index)

        # Helpful for debugging, but ensure print_once key is stable
        print_once("ds_star_verifier_prompt", f"DS Star Verifier prompt:\n{user_msg}")

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        try:
            response, usage = invoke_structured_with_usage(
                self.llm_with_output, messages, run_name="n_verify"
            )
            state.token_usage.append(usage)

            last_step.verifier_sufficient = bool(response.sufficient)
            last_step.verifier_explanation = str(response.explanation)

            add_event_to_trajectory(
                state,
                "n_verify",
                sufficient=last_step.verifier_sufficient,
                execution_error=bool(last_step.execution_error),
                explanation=last_step.verifier_explanation,
            )
        except ValidationError as ve:
            state.fatal_error = f"Verifier schema validation failed: {ve}"
            last_step.verifier_sufficient = False
            last_step.verifier_explanation = str(ve)
            add_event_to_trajectory(
                state,
                "n_verify",
                fatal_error=state.fatal_error,
                sufficient=False,
                explanation=last_step.verifier_explanation,
            )
        except Exception as e:
            state.fatal_error = f"Verifier invocation failed: {e}"
            last_step.verifier_sufficient = False
            last_step.verifier_explanation = str(e)
            add_event_to_trajectory(
                state,
                "n_verify",
                fatal_error=state.fatal_error,
                sufficient=False,
                explanation=last_step.verifier_explanation,
            )

        logger.info(
            "Verifier sufficient: %s (execution_error: %s)\nexplanation: %s",
            last_step.verifier_sufficient,
            bool(last_step.execution_error),
            last_step.verifier_explanation,
        )
        return state
