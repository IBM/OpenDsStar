import logging
from typing import Optional, Tuple, Type

from pydantic import BaseModel, Field, ValidationError

from agents.ds_star.ds_star_state import CodeMode, DSState, DSStep
from agents.ds_star.ds_star_utils import (
    add_event_to_trajectory,
    build_execution_environment_instructions,
    invoke_structured_with_usage,
    truncate_text,
)
from agents.ds_star.nodes.base_node import BaseNode
from agents.ds_star.nodes.coder import (
    _collect_available_parameter_names,
)
from agents.utils.agents_utils import print_once

logger = logging.getLogger(__name__)


class CodeOutput(BaseModel):
    code: str = Field(
        ...,
        description=(
            "Corrected Python code produced by the Debugger. "
            "STEPWISE: corrected code for current step only. "
            "FULL: corrected full script."
        ),
    )


def _build_debugger_prompt(state: DSState, tools_specs: str) -> Tuple[str, str]:
    if not state.steps:
        return ("You are the Debugger agent.", "No steps exist. Output: pass")

    last_step: DSStep = state.steps[-1]
    if not last_step.code:
        raise AssertionError("Debugger expects code in the last step.")
    if not last_step.execution_error:
        raise AssertionError("Debugger expects an execution error in the last step.")

    k = len(state.steps) - 1
    logs_max_length = state.logs_max_length
    output_max_length = state.output_max_length

    # ---------- shared ----------
    user_block = f"User query:\n{state.user_query.strip()}\n"

    plan_block = "Current plan:\n" + "\n".join(
        f"{i}. {s.plan.strip()}" for i, s in enumerate(state.steps)
    )

    step_block = f"Current step (k={k}):\n{k}. {last_step.plan.strip()}\n"
    tools_block = f"Available tools:\n{tools_specs}\n"

    # Build failed attempts block showing all previous failed codes and errors
    failed_attempts_block = ""
    if last_step.failed_code_attempts:
        failed_attempts_block = "Previous failed attempts for this step:\n"
        for idx, attempt in enumerate(last_step.failed_code_attempts, 1):
            failed_attempts_block += f"\n--- Attempt {idx} ---\n"
            failed_attempts_block += f"Code:\n{attempt['code']}\n"
            failed_attempts_block += f"Error:\n{attempt['error']}\n"
        failed_attempts_block += "\n"

    code_block = f"Failing script:\n{last_step.code}\n"
    error_block = f"Execution error:\n{last_step.execution_error.strip()}\n"

    # ==========================================================
    # SINGLE MODE SWITCH
    # ==========================================================
    if state.code_mode == CodeMode.STEPWISE:
        # -------- logs: ALL steps --------
        log_chunks = []
        for i, s in enumerate(state.steps):
            if s.logs:
                log_chunks.append(
                    f"Step {i} logs:\n{truncate_text(str(s.logs), logs_max_length)}"
                )
        logs_block = (
            "Execution logs (all steps):\n"
            + ("\n\n".join(log_chunks) if log_chunks else "(no logs captured)")
            + "\n\n"
        )

        # -------- outputs: ALL steps --------
        out_chunks = []
        for i, s in enumerate(state.steps):
            outs = getattr(s, "outputs", None)
            if isinstance(outs, dict) and outs:
                # compact, deterministic: key=value per line
                lines = [
                    f"{k} = {truncate_text(str(v), output_max_length)}"
                    for k, v in outs.items()
                ]
                out_chunks.append(f"Step {i} outputs:\n" + "\n".join(lines))
        outputs_block = (
            "Step outputs (all steps):\n"
            + ("\n\n".join(out_chunks) if out_chunks else "(no outputs captured)")
            + "\n\n"
        )

        system_msg = (
            "You are the Debugger agent.\n"
            "MODE: STEPWISE.\n"
            "Fix code for the CURRENT step only.\n"
        )

        params_block = (
            "Available prev_step_outputs keys:\n"
            f"{_collect_available_parameter_names(state)}\n"
        )

        env_block = build_execution_environment_instructions(
            state, mode=state.code_mode.value, include_prev_step_outputs=True
        )

        rules = (
            "- Fix root cause.\n"
            "- Implement ONLY this step.\n"
            "- Do NOT re-implement previous steps.\n"
            "- You may read prev_step_outputs.\n"
            "- Store this step’s artifacts in outputs.\n"
            "- Output ONLY corrected Python code.\n"
        )

    else:
        # -------- logs: LAST step only --------
        logs_block = (
            "Execution logs (last step only):\n"
            + (
                truncate_text(str(last_step.logs), logs_max_length)
                if last_step.logs
                else "(no logs captured)"
            )
            + "\n\n"
        )

        # -------- outputs: LAST step only --------
        last_outs = getattr(last_step, "outputs", None)
        if isinstance(last_outs, dict) and last_outs:
            lines = [
                f"{k} = {truncate_text(str(v), output_max_length)}"
                for k, v in last_outs.items()
            ]
            outputs_block = (
                "Step outputs (last step only):\n" + "\n".join(lines) + "\n\n"
            )
        else:
            outputs_block = "Step outputs (last step only):\n(no outputs captured)\n\n"

        system_msg = (
            "You are the Debugger agent.\n"
            "MODE: FULL SCRIPT.\n"
            "Fix the full script implementing all steps.\n"
        )

        params_block = ""

        env_block = build_execution_environment_instructions(
            state, mode=state.code_mode.value, include_prev_step_outputs=False
        )

        rules = (
            "- Fix root cause.\n"
            "- Ensure full plan works.\n"
            "- Store final artifacts in outputs.\n"
            "- Output ONLY corrected Python code.\n"
        )

    user_msg = (
        f"{user_block}\n"
        f"{plan_block}\n\n"
        f"{step_block}\n"
        f"{params_block}\n"
        f"{tools_block}\n"
        f"{env_block}\n"
        f"{rules}\n\n"
        f"{outputs_block}"
        f"{logs_block}"
        f"{failed_attempts_block}"
        f"{error_block}\n"
        f"{code_block}\n"
        "Return ONLY corrected Python code:\n"
    )

    logger.info("debugger prompt length: %d", len(system_msg) + len(user_msg))
    return system_msg, user_msg


class DebuggerNode(BaseNode):
    structured_output_schema: Optional[Type[BaseModel]] = CodeOutput

    def __call__(self, state: DSState) -> DSState:
        if not isinstance(state, DSState):
            raise TypeError("DebuggerNode expects DSState")

        if state.fatal_error:
            add_event_to_trajectory(
                state, "n_debug", skipped=True, fatal_error=state.fatal_error
            )
            return state

        if not state.steps:
            state.fatal_error = "Debugger called with no steps."
            add_event_to_trajectory(state, "n_debug", fatal_error=state.fatal_error)
            return state

        last_step: DSStep = state.steps[-1]
        last_step.debug_tries = int(getattr(last_step, "debug_tries", 0)) + 1

        if not last_step.code:
            state.fatal_error = "No code to debug."
            add_event_to_trajectory(state, "n_debug", fatal_error=state.fatal_error)
            return state

        if not last_step.execution_error:
            add_event_to_trajectory(
                state, "n_debug", skipped=True, reason="no_execution_error"
            )
            return state

        # Store the current failed code and error before attempting to fix it
        if last_step.code and last_step.execution_error:
            last_step.failed_code_attempts.append(
                {"code": last_step.code, "error": last_step.execution_error}
            )

        system_msg, user_msg = _build_debugger_prompt(state, self.tools_spec or "")
        print_once("ds_star_debugger_prompt", f"DS Star Debugger prompt:\n{user_msg}")

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        try:
            response, usage = invoke_structured_with_usage(
                self.llm_with_output, messages, run_name="n_debug"
            )
            state.token_usage.append(usage)

            last_step.code = response.code
            last_step.execution_error = None

            i = len(state.steps) - 1

            logger.info(
                "Debugger generated code for step %s:\n%s",
                i,
                response.code,
            )

            add_event_to_trajectory(
                state, "n_debug", corrected_code=last_step.code, usage=usage
            )

        except ValidationError as ve:
            state.fatal_error = f"Debugger schema validation failed: {ve}"
            add_event_to_trajectory(state, "n_debug", fatal_error=state.fatal_error)

        except Exception as e:
            state.fatal_error = f"Debugger invocation failed: {e}"
            add_event_to_trajectory(state, "n_debug", fatal_error=state.fatal_error)

        return state
