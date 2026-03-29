import logging

from pydantic import BaseModel, Field, ValidationError

from agents.ds_star.ds_star_state import CodeMode, DSState
from agents.ds_star.ds_star_utils import (
    add_event_to_trajectory,
    build_execution_environment_instructions,
    invoke_structured_with_usage,
)
from agents.ds_star.nodes.base_node import BaseNode
from agents.utils.agents_utils import print_once
from agents.utils.safe_imports import get_safe_builtins, get_safe_scientific_env

logger = logging.getLogger(__name__)


class CodeOutput(BaseModel):
    code: str = Field(
        ...,
        description="Python code for the current plan step only. No backticks, no markdown, no explanations.",
    )


def _collect_available_parameter_names(state: DSState) -> str:
    names = set()
    for step in state.steps[:-1]:
        if getattr(step, "available_params", None):
            try:
                names.update(step.available_params)
            except Exception:
                pass
        if isinstance(getattr(step, "outputs", None), dict):
            names.update(step.outputs.keys())
    if not names:
        return "None"
    return ", ".join(sorted(str(n) for n in names))


def fix_name(package_name: str) -> str:
    if package_name == "np":
        return "numpy (as np)"
    if package_name == "pd":
        return "pandas (as pd)"
    return str(package_name)


def get_available_packages_list() -> list[str]:
    available_packages = list(get_safe_builtins().keys()) + list(
        get_safe_scientific_env().keys()
    )
    return [fix_name(name) for name in available_packages]


def build_coder_prompt(state: DSState, tools_specs: str) -> tuple[str, str]:
    """Returns (system_message, user_message) so the caller can send a proper system role.
    """
    if not state.steps:
        system_msg = (
            "You are the Coder agent.\n"
            "Generate Python code to implement the current plan step(s).\n"
            "Output CodeOutput(code=<python>) only."
        )
        user_msg = (
            f"User query:\n{state.user_query.strip()}\n\n"
            "No plan steps exist yet. Output minimal valid Python code that does nothing.\n"
            "Return valid Python code (e.g., 'pass')."
        )
        return system_msg, user_msg

    k = len(state.steps) - 1
    current_step = state.steps[-1]
    prev_code = current_step.code

    user_block = f"User query:\n{state.user_query.strip()}\n"

    plan_block = "Current plan (0-based steps):\n" + "\n".join(
        f"{i}. {s.plan.strip()}" for i, s in enumerate(state.steps)
    )

    tools_block = f"Available tools (call directly as functions):\n{tools_specs}\n"

    step_block = f"Latest step (index k = {k}):\n{k}. {current_step.plan.strip()}\n"

    code_block = f"Previous script:\n{prev_code if prev_code else '(None)'}\n"

    # ===============================================================
    # STEPWISE MODE
    # ===============================================================
    if state.code_mode == CodeMode.STEPWISE:
        params_block = (
            "Available parameters from previous steps (via prev_step_outputs):\n"
            f"{_collect_available_parameter_names(state)}\n"
        )

        system_instructions = (
            "You are the Coder agent.\n"
            "SYSTEM MODE: STEPWISE CODE GENERATION.\n"
            "Generate Python code that implements ONLY the current step (k).\n"
            "Do NOT re-implement earlier steps.\n"
            "You may use values from prev_step_outputs, using prev_step_outputs[var_name].\n"
            "You may not use local variables from the previous script if they weren't explicitly saved in outputs.\n"
            "Persist structured results via outputs, and use print() for human-readable traces.\n"
        )

        refinement = (
            "Refine the previous script for step k."
            if prev_code
            else "Generate a NEW script that implements ONLY this step."
        )

        env_block = build_execution_environment_instructions(
            state, mode=state.code_mode.value, include_prev_step_outputs=True
        )

        guidelines = (
            "Guidelines:\n"
            "- Implement ONLY step k.\n"
            "- Inspect structure to understand available columns.\n"
            "- For UNIQUE/DISTINCT queries: Always use .nunique() for DataFrames or len(set()) for lists.\n"
            "- For RANKING/TOP-N queries: group-by before ranking. Never assume first N rows are top N.\n"
            "- For BOOLEAN questions: Carefully check negations and use .any()/.all() appropriately.\n"
            "- Always verify the data type and format of your final answer matches the question intent.\n"
            "- Persist structured results in outputs (dict).\n"
            "- Use print() to communicate important intermediate results and final answer to later agents.\n"
            "- NEVER use locals(), globals(), vars(), or similar introspection functions.\n"
            "- To check if a variable exists from previous steps, check if it's in prev_step_outputs or use try-except.\n"
            "- Output ONLY raw Python code. No backticks, no markdown.\n"
            "  If you create a Plotly figure (e.g. via px or go), store it in outputs['figure']; "
            "the finalizer will automatically render it in the user-facing answer.\n"
        )

        user_msg = (
            f"{refinement}\n\n"
            f"{user_block}\n"
            f"{plan_block}\n\n"
            f"{step_block}\n"
            f"{params_block}\n"
            f"{tools_block}\n"
            f"{env_block}\n"
            f"{guidelines}\n"
            f"{code_block}\n"
            "Now output ONLY the COMPLETE Python script for this step:\n"
        )
        logger.info("coder prompt length: %d", len(system_instructions) + len(user_msg))
        return system_instructions, user_msg

    # ===============================================================
    # FULL-SCRIPT MODE
    # ===============================================================

    system_instructions = (
        "You are the Coder agent.\n"
        "SYSTEM MODE: FULL-SCRIPT GENERATION.\n"
        "Generate Python code that implements the ENTIRE PLAN (steps 0..k) as one script.\n"
        "Do NOT rely on prev_step_outputs (ignored in this mode).\n"
        "Persist structured final results via outputs, and use print() for human-readable traces.\n"
    )

    refinement = (
        "Refine the previous FULL script to match the complete plan."
        if prev_code
        else "Generate a NEW holistic script implementing ALL steps."
    )

    env_block = build_execution_environment_instructions(
        state, mode=state.code_mode.value, include_prev_step_outputs=False
    )

    guidelines = (
        "Guidelines:\n"
        "- Implement ALL steps in order.\n"
        "- Inspect structure to understand available columns.\n"
        "- For UNIQUE/DISTINCT queries: Always use .nunique() for DataFrames or len(set()) for lists.\n"
        "  * Verify count with print statement.\n"
        "- For RANKING/TOP-N queries: group-by before ranking. Never assume first N rows are top N.\n"
        "- For BOOLEAN questions: Carefully check negations and use .any()/.all() appropriately.\n"
        "- Always verify the data type and format of your final answer matches the question intent.\n"
        "- Persist structured final results in outputs (dict).\n"
        "- Use print() to communicate important intermediate results and final answer to later agents.\n"
        "- NEVER use locals(), globals(), vars(), or similar introspection functions.\n"
        "- To check if a variable exists, use try-except blocks instead of introspection.\n"
        "- Output ONLY raw Python code. No backticks, no markdown.\n"
        "- Persist structured final results in outputs (dict). You may store large objects if useful.\n"
        "  If you create a Plotly figure, put it in outputs['figure'] so the finalizer can render it.\n"
    )
    user_msg = (
        f"{refinement}\n\n"
        f"{user_block}\n"
        f"{plan_block}\n\n"
        f"{step_block}\n"
        f"{tools_block}\n"
        f"{env_block}\n"
        f"{guidelines}\n"
        f"{code_block}\n"
        "Now output ONLY the COMPLETE Python script implementing the entire plan:\n"
    )

    logger.info("coder prompt length: %d", len(system_instructions) + len(user_msg))
    return system_instructions, user_msg


class CoderNode(BaseNode):
    structured_output_schema: type[BaseModel] | None = CodeOutput

    def __call__(self, state: DSState) -> DSState:
        # Defensive: if something upstream passed a dict, fail loudly and early.
        if not isinstance(state, DSState):
            raise TypeError(
                f"CoderNode expected DSState, got {type(state).__name__}. "
                "Fix the caller to pass DSState, or refactor this node to use dict access."
            )

        if state.fatal_error:
            add_event_to_trajectory(
                state, "n_code", skipped=True, fatal_error=state.fatal_error
            )
            return state

        if not state.steps:
            state.fatal_error = "Coder called with no steps planned."
            add_event_to_trajectory(
                state, "n_code", skipped=True, fatal_error=state.fatal_error
            )
            return state

        system_msg, user_msg = build_coder_prompt(state, self.tools_spec or "")
        print_once("ds_star_coder_prompt", f"DS Star Coder prompt:\n{user_msg}")

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        try:
            response, usage = invoke_structured_with_usage(
                self.llm_with_output, messages, run_name="n_code"
            )
            state.token_usage.append(usage)

            current_step = state.steps[-1]
            i = len(state.steps) - 1

            current_step.code = response.code
            logger.info("Coder generated code for step %s:\n%s", i, response.code)

        except ValidationError as ve:
            i = len(state.steps) - 1
            state.fatal_error = f"Coder schema validation failed: {ve}"
            logger.error(
                "Fatal error generating code for step %s: %s", i, state.fatal_error
            )
            add_event_to_trajectory(state, "n_code", fatal_error=state.fatal_error)
            return state

        except Exception as e:
            i = len(state.steps) - 1
            state.fatal_error = f"Coder invocation failed: {e}"
            logger.error(
                "Fatal error generating code for step %s: %s", i, state.fatal_error
            )
            add_event_to_trajectory(state, "n_code", fatal_error=state.fatal_error)
            return state

        add_event_to_trajectory(state, "n_code", code=state.steps[-1].code, usage=usage)
        return state
