"""Code generator for analyzer agent - simplified single-shot generation."""

import logging
import re
from typing import Any, Dict

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from OpenDsStar.agents.ds_star.ds_star_utils import (
    build_execution_environment_instructions,
)

logger = logging.getLogger(__name__)


class CodeOutput(BaseModel):
    """Schema for structured code output."""

    code: str = Field(..., description="Complete Python code for analyzing the file.")


def _build_analysis_prompt(filename: str) -> str:
    """Build the analysis prompt with execution environment instructions."""
    env_instructions = build_execution_environment_instructions(
        state=None,
        mode="stepwise",
        include_prev_step_outputs=False,
        include_tools_instructions=False,
        logs_max_length=0,  # No print output tracking for analyzer
    )

    return f"""You are an expert data analyst.
Generate a Python code that loads and describes the content of {filename}.

# Requirements
- The file can be both unstructured or structured data.
- If there are too many structured data, print out just few examples.
- Print out essential information. For example, print out all the column names.
- The Python code should print out the content of {filename}.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Important: You should not include dummy contents since we will debug if error occurs.
- Do not use try: and except: to prevent error. I will debug it later.

# {env_instructions}

# Store any important results in the global 'outputs' dictionary for later retrieval.

Output format:
- Your response will be parsed into CodeOutput.code.
- You MUST output ONLY the COMPLETE Python script as plain text.
- Do NOT include backticks, markdown fences, JSON, or any surrounding text.
- Output ONLY raw Python code. No backticks, no markdown."""


class CoderNode:
    """Generates Python code for file analysis."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        # Create structured output LLM
        if hasattr(llm, "with_structured_output"):
            self.llm_with_output = llm.with_structured_output(CodeOutput)
        else:
            raise TypeError(
                "Provided llm does not support `with_structured_output(...)`."
            )

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code for analyzing the file."""
        if state["fatal_error"]:
            state["trajectory"].append(
                {"node": "n_code", "skipped": True, "fatal_error": state["fatal_error"]}
            )
            return state

        filename = state["filename"]
        prompt = _build_analysis_prompt(filename)
        messages = [{"role": "user", "content": prompt}]

        logger.info(f"Generating analysis code for: {filename}")

        try:
            response = self.llm_with_output.invoke(messages)

            # Extract code from structured output and remove any import statements
            code_without_imports = re.sub(
                r"^\s*(import|from)\s+.*$", "", response.code, flags=re.MULTILINE
            )
            state["code"] = code_without_imports
            logger.info("Code generated successfully")

            state["trajectory"].append(
                {"node": "n_code", "code_length": len(state["code"])}
            )

        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            state["fatal_error"] = f"Code generation failed: {e}"
            state["trajectory"].append({"node": "n_code", "error": str(e)})

        return state
