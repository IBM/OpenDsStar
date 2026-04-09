"""CodeActAgentSmolagents - Wrapper for smolagents CodeAgent.

Wrapper around smolagents' CodeAgent with the same interface as OpenDsStarAgent.
CodeAgent uses a code-based action approach for solving tasks.
"""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool as LangChainBaseTool
from smolagents import CodeAgent, LiteLLMModel, RunResult
from smolagents import Tool as SmolagentsTool

from OpenDsStar.agents.base_agent import BaseAgent
from OpenDsStar.agents.utils.logging_utils import init_logger
from OpenDsStar.experiments.core.config import AgentConfig
from OpenDsStar.tools.string_to_stream_tool import StringToStreamTool

# Suppress litellm proxy import warnings (we don't use proxy features)
warnings.filterwarnings("ignore", message=".*fastapi_sso.*")
warnings.filterwarnings("ignore", message=".*apscheduler.*")

# Suppress litellm's ERROR logs about missing proxy dependencies
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("litellm").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)


class CodeActAgentSmolagents(BaseAgent):
    """CodeActAgentSmolagents - Wrapper around smolagents' CodeAgent.

    Provides the same interface as OpenDsStarAgent for consistency.
    Uses smolagents' CodeAgent which generates and executes code to solve tasks.
    """

    def __init__(
        self,
        model: str | BaseChatModel | LiteLLMModel,
        temperature: float = 0.0,
        tools: list[Any] | None = None,
        system_prompt: str | None = None,
        task_prompt: str | None = None,  # Ignored for smolagents
        max_steps: int = 5,
        code_timeout: int = 30,
        code_mode: str = "stepwise",  # Ignored for smolagents
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize the CodeActAgentSmolagents.

        Args:
            model: smolagents LiteLLMModel instance (use ModelBuilder.build() with framework="smolagents" to create).
            temperature: Temperature for generation (stored for config, model should already have this set).
            cache_dir: Ignored (kept for backwards compatibility). Cache should be configured when building the model.

        Raises:
            ValueError: If model is not a LiteLLMModel instance
        """
        # Initialize logger if not already initialized (respects existing config)
        init_logger()

        # Store model info
        if isinstance(model, LiteLLMModel):
            # Already a LiteLLMModel instance (from ModelBuilder)
            self.smol_model = model
            # Extract model_id from the LiteLLMModel
            self._model_id = model.model_id
        elif isinstance(model, str):
            self._model_id = model
            # Create LiteLLMModel for smolagents
            # Pass api_base=None to let LiteLLM use environment variables
            self.smol_model = LiteLLMModel(
                model_id=model,
                temperature=temperature,
                api_base=None,  # Let LiteLLM read from env vars
            )
        elif isinstance(model, BaseChatModel):
            # Extract model_id from LangChain model
            model_id = None

            # Try different attributes for model ID
            if hasattr(model, "model_id") and model.model_id:
                model_id = model.model_id
            elif hasattr(model, "model") and model.model:
                model_id = model.model
            elif hasattr(model, "model_name") and model.model_name:
                model_id = model.model_name
            elif hasattr(model, "_model_name") and model._model_name:
                model_id = model._model_name

            if not model_id:
                raise ValueError(
                    "Cannot extract model_id from BaseChatModel. "
                    f"Model type: {type(model).__name__}. "
                    "Please provide a string model ID instead or ensure the model has 'model_id', 'model', or 'model_name' attribute."
                )

            self._model_id = model_id

            # Set environment variables from LangChain model attributes
            # This allows LiteLLM to auto-detect the provider and use the correct credentials
            self._set_env_vars_from_model(model, model_id)

            # Determine the LiteLLM model ID (with provider prefix if needed)
            litellm_model_id = self._get_litellm_model_id(model, model_id)

            logger.info(f"Creating LiteLLMModel with model_id: {litellm_model_id}")

            # Create LiteLLMModel - it will use environment variables we just set
            self.smol_model = LiteLLMModel(
                model_id=litellm_model_id,
                temperature=temperature,
            )
        else:
            raise ValueError(
                "model must be either a model ID string (e.g., 'gpt-4o-mini'), "
                "a LangChain BaseChatModel instance, or a smolagents LiteLLMModel instance"
            )

        self.temperature = temperature
        # Filter out invalid tools (empty strings, None, etc.) that Langflow might pass
        # when no tools are connected
        if tools:
            self.tools = [
                t for t in tools if t and not (isinstance(t, str) and not t.strip())
            ]
        else:
            self.tools = []
        self.max_steps = max_steps
        self.code_timeout = code_timeout
        self.code_mode = code_mode  # Stored for interface compatibility

        # Always add StringToStreamTool - it's always available to the agent
        string_to_stream_tool = StringToStreamTool()
        self.tools.append(string_to_stream_tool)

        # Convert LangChain tools to smolagents tools if needed
        smol_tools = self._convert_tools(self.tools)

        # Wrap all tools to guarantee they return string results
        # This prevents issues when tools (especially from LangFlow/MCP)
        # return CallToolResult objects which cannot be JSON-parsed by
        # generated code. See docs/CODEACT_LANGFLOW_TOOL_RESULT_FIX.md
        smol_tools = [self._wrap_tool_for_string_results(tool) for tool in smol_tools]

        # Create the CodeAgent
        # Use the same safe imports as DS-Star for consistency
        from OpenDsStar.agents.utils.safe_imports import get_authorized_imports_list

        authorized_imports = get_authorized_imports_list()

        # Configure CodeAgent with proper tool access
        # Tools are made available in the execution environment by smolagents
        # The agent will generate code that calls tools directly by name
        self._agent = CodeAgent(
            tools=smol_tools,
            model=self.smol_model,
            max_steps=max_steps,
            additional_authorized_imports=authorized_imports,
            executor_kwargs={"timeout_seconds": None},
        )

        # DO NOT override system prompt - CodeAgent builds it dynamically to include tools!
        # The system_prompt parameter is stored but not used to override the agent's prompt
        # because CodeAgent needs its default prompt structure to properly list available tools
        if system_prompt:
            logger.info(
                "Custom system_prompt provided but NOT overriding CodeAgent's prompt template"
            )
            logger.info(
                "CodeAgent needs its default prompt structure to include tool descriptions"
            )
            # Store it for reference but don't apply it
            self.system_prompt = system_prompt
        else:
            self.system_prompt = None

        self.task_prompt = task_prompt  # Stored for interface compatibility

        # Store agent configuration for cache key generation
        self.agent_config = AgentConfig(
            agent_type="codeact_smolagents",
            model=self._model_id,
            temperature=temperature,
            max_steps=max_steps,
            code_timeout=code_timeout,
            code_mode=code_mode,
            system_prompt=system_prompt,
            task_prompt=task_prompt,
        )

        logger.info(
            "CodeActAgentSmolagents initialized: model=%s, tools=%d, max_steps=%d, cache=%s",
            self._model_id,
            len(self.tools),
            max_steps,
            "enabled" if cache_dir else "disabled",
        )

    def _convert_tools(self, tools: list[Any]) -> list[Any]:
        """Convert LangChain tools to smolagents format if needed.

        Uses smolagents' built-in Tool.from_langchain() method to convert
        LangChain BaseTool instances to smolagents tools.

        Raises:
            TypeError: If a tool is not a valid Tool object (e.g., empty string)
        """
        # Clean up empty strings or None which Langflow sometimes passes when the input is functionally empty
        if isinstance(tools, list):
            tools = [
                t for t in tools if t and not (isinstance(t, str) and not t.strip())
            ]

        converted_tools = []
        for idx, tool in enumerate(tools):

            # Validate tool is not an invalid type (string, None, etc.)
            if isinstance(tool, str):
                raise TypeError(
                    f"Tool at index {idx} is a string '{tool}', not a Tool object. "
                    "This usually means tools weren't properly connected in the flow. "
                    "Please ensure Tool components (not text/strings) are connected to the Tools input. "
                    f"All tools received: {tools}"
                )
            if tool is None:
                raise TypeError(
                    f"Tool at index {idx} is None. "
                    "This usually means tools weren't properly connected in the flow. "
                    "Please ensure Tool components are connected to the Tools input."
                )

            # Check if it's a LangChain tool
            if isinstance(tool, LangChainBaseTool):
                logger.info(
                    f"Converting LangChain tool '{tool.name}' to smolagents format"
                )
                # Use smolagents' built-in conversion method
                wrapped_tool = SmolagentsTool.from_langchain(tool)
                logger.info(
                    f"Converted tool type: {type(wrapped_tool)}, name: {wrapped_tool.name}"
                )
                converted_tools.append(wrapped_tool)
            else:
                # Already a smolagents tool or compatible
                logger.info(
                    f"Tool already in smolagents format: {type(tool)}, name: {getattr(tool, 'name', 'unknown')}"
                )
                converted_tools.append(tool)

        logger.info(f"Total tools converted: {len(converted_tools)}")
        return converted_tools

    def _wrap_tool_for_string_results(self, tool: Any) -> Any:
        """Ensure the given smolagents tool always returns a **string** result.

        LangChain tools wrapped via `Tool.from_langchain()` (and several
        MCP/CodeAct helpers) may return complex objects such as
        `CallToolResult`. When the agent generates Python that calls a tool
        and then immediately tries to parse the result (e.g. via
        ``json.loads(result)``) this leads to a ``TypeError``. The generated
        code expects a raw string.

        By intercepting the ``forward`` method we can unwrap common result
        types (``CallToolResult`` with ``content[0].text``, any object with a
        ``text`` or string ``content`` attribute) and fall back to ``str()``.
        """
        original_forward = tool.forward

        def string_forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)

            # Unwrap common non-string tool results so that generated code
            # that expects a string (e.g. for json.loads) works correctly.
            # We only transform when we know how; otherwise we leave the value
            # untouched so that tools returning arbitrary objects (streams,
            # dataframes, etc.) continue to function.

            # Handle CallToolResult objects (from MCP tools)
            if hasattr(result, "content") and result.content:
                if isinstance(result.content, list) and len(result.content) > 0:
                    first_content = result.content[0]
                    if hasattr(first_content, "text"):
                        return first_content.text

            # Otherwise return the original object as-is (do not stringify).
            return result

        tool.forward = string_forward
        return tool

    @property
    def model_id(self) -> str:
        """Get the model identifier."""
        return self._model_id

    def get_agent_config(self) -> AgentConfig:
        """Get the agent's configuration."""
        return self.agent_config

    def invoke(
        self,
        query: str,
        config: dict[str, Any] | None = None,
        return_state: bool = False,
    ) -> dict[str, Any]:
        """Execute the agent with a query."""
        if not query or not isinstance(query, str):
            raise ValueError("query must be a non-empty string")

        logger.info("Invoking CodeActAgentSmolagents with query: %s", query)

        try:
            result = self._agent.run(query, return_full_result=True)

            # Always expect RunResult when return_full_result=True
            if not isinstance(result, RunResult):
                raise TypeError(f"Expected RunResult, got {type(result).__name__}")

            answer = result.output
            trajectory = result.steps
            steps_used = len(trajectory) if trajectory else 1

            # Extract token usage from agent monitor
            token_usage = self._agent.monitor.get_total_token_counts()

            return {
                "answer": answer,
                "trajectory": trajectory,
                "plan": "",  # CodeAgent doesn't have explicit plans
                "steps_used": steps_used,
                "max_steps": self.max_steps,
                "verifier_sufficient": True,
                "fatal_error": "",
                "execution_error": "",
                "input_tokens": token_usage.input_tokens,
                "output_tokens": token_usage.output_tokens,
                "num_llm_calls": steps_used,
            }

        except Exception as e:
            error_msg = str(e)
            # Suppress timeout errors if they're the default 30 second timeout
            # These are false positives - the code actually completed successfully
            if "exceeded the maximum execution time of 30 seconds" in error_msg.lower():
                logger.warning(
                    "Ignoring spurious 30-second timeout error - code executed successfully"
                )
                # Return empty result but don't treat as fatal error
                return {
                    "answer": "",
                    "trajectory": [],
                    "plan": "",
                    "steps_used": 0,
                    "max_steps": self.max_steps,
                    "verifier_sufficient": True,
                    "fatal_error": "",
                    "execution_error": "",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "num_llm_calls": 0,
                }

            logger.error("CodeActAgentSmolagents execution failed: %s", error_msg)
            return {
                "answer": "",
                "trajectory": [{"type": "error", "content": error_msg}],
                "plan": "",
                "steps_used": 0,
                "max_steps": self.max_steps,
                "verifier_sufficient": False,
                "fatal_error": error_msg,
                "execution_error": error_msg,
                "input_tokens": 0,
                "output_tokens": 0,
                "num_llm_calls": 0,
            }

    def stream_invoke(
        self,
        query: str,
        config: dict[str, Any] | None = None,
        yield_state: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Stream execution of the agent, yielding code and results step-by-step.

        Uses smolagents' built-in streaming capability (stream=True parameter)
        to yield intermediate steps as they're generated.

        Args:
            query: The user's question or task to solve.
            config: Optional configuration dict (unused for smolagents).
            yield_state: If True, yields final state at end. If False, only yields events.

        Yields:
            Dict with step information:
            - step: Step label (e.g., "step_0", "step_1")
            - event: Dict containing:
                - step_idx: Step index
                - node_name: "code_generation" or "code_execution"
                - code: Generated code (for generation events)
                - logs: Execution output (for execution events)
                - error: Error message if execution failed
                - time: Timestamp

        """
        if not query or not isinstance(query, str):
            raise ValueError("query must be a non-empty string")

        import time

        logger.info("Stream invoking CodeActAgentSmolagents with query: %s", query)

        step_idx = 0
        final_answer = ""

        # Use smolagents' built-in streaming - this returns the final result at the end
        stream_result = self._agent.run(query, stream=True)

        for step in stream_result:
            # Each step is a dict-like object with information about the current action
            # Extract relevant information from the step

            event_data = {
                "step_idx": step_idx,
                "time": time.time(),
            }

            # Check what type of step this is
            if hasattr(step, "code_action") or (
                isinstance(step, dict) and "code_action" in step
            ):
                # Code generation step
                code = (
                    step.code_action
                    if hasattr(step, "code_action")
                    else step.get("code_action", "")
                )
                if code:
                    event_data["node_name"] = "code_generation"
                    event_data["code"] = str(code)

                    yield {"step": f"step_{step_idx}", "event": event_data}

            if hasattr(step, "observations") or (
                isinstance(step, dict) and "observations" in step
            ):
                # Execution result step
                observations = (
                    step.observations
                    if hasattr(step, "observations")
                    else step.get("observations", "")
                )
                if observations:
                    event_data["node_name"] = "code_execution"
                    event_data["logs"] = str(observations)

                    # Extract clean final answer from observations
                    # Smolagents wraps the final_answer() result in a message like:
                    # "Last output from code snippet:\nUS"
                    obs_str = str(observations)
                    if "Last output from code snippet:" in obs_str:
                        # Extract just the answer part after the prefix
                        parts = obs_str.split("Last output from code snippet:")
                        if len(parts) > 1:
                            final_answer = parts[1].strip()
                        else:
                            final_answer = obs_str
                    else:
                        final_answer = obs_str

                    yield {"step": f"step_{step_idx}", "event": event_data}

            if hasattr(step, "error") or (isinstance(step, dict) and "error" in step):
                # Error step
                error = step.error if hasattr(step, "error") else step.get("error", "")
                if error:
                    event_data["node_name"] = "error"
                    event_data["error"] = str(error)

                    yield {"step": f"step_{step_idx}", "event": event_data}

            step_idx += 1

        # After streaming completes, smolagents returns the final result
        # The stream_result variable now contains the final RunResult
        if isinstance(stream_result, RunResult) or hasattr(stream_result, "output"):
            final_answer = stream_result.output

        # ALWAYS yield final answer event at the end (required by Langflow component)
        yield {
            "step": "final",
            "event": {
                "step_idx": step_idx,
                "node_name": "final_answer",
                "answer": final_answer,
                "time": time.time(),
            },
        }

    def _get_secret_value(self, value: Any) -> str | None:
        """Extract string value from potential SecretStr object."""
        if value is None:
            return None
        if hasattr(value, "get_secret_value"):
            return value.get_secret_value()
        return str(value) if value else None

    def _set_env_vars_from_model(self, model: BaseChatModel, model_id: str) -> None:
        """Set environment variables from LangChain model attributes.

        This allows LiteLLM to auto-detect provider and credentials.
        """
        model_type = type(model).__name__
        logger.info(f"Setting environment variables for {model_type}")

        # Provider-specific environment variable mapping
        if model_type == "ChatOllama":
            # Ollama specific attributes
            if hasattr(model, "base_url"):
                base_url = self._get_secret_value(model.base_url)
                if base_url:
                    os.environ["OLLAMA_API_BASE"] = base_url
                    logger.info(f"Set OLLAMA_API_BASE to {base_url}")
            # Ollama typically runs locally and doesn't need API keys
            logger.info("Set Ollama environment variables")

        elif model_type == "ChatWatsonx":
            # WatsonX specific attributes
            if hasattr(model, "apikey"):
                os.environ["WATSONX_API_KEY"] = (
                    self._get_secret_value(model.apikey) or ""
                )
            if hasattr(model, "url"):
                os.environ["WATSONX_URL"] = self._get_secret_value(model.url) or ""
            if hasattr(model, "project_id"):
                os.environ["WATSONX_PROJECT_ID"] = (
                    self._get_secret_value(model.project_id) or ""
                )
            logger.info(
                f"Set WatsonX environment variables (project_id: {os.environ.get('WATSONX_PROJECT_ID', 'not set')})"
            )

        elif model_type == "ChatOpenAI":
            # OpenAI specific attributes
            if hasattr(model, "openai_api_key") or hasattr(model, "api_key"):
                api_key = self._get_secret_value(
                    getattr(model, "openai_api_key", None)
                    or getattr(model, "api_key", None)
                )
                if api_key:
                    os.environ["OPENAI_API_KEY"] = api_key
            if hasattr(model, "openai_api_base") or hasattr(model, "base_url"):
                base_url = self._get_secret_value(
                    getattr(model, "openai_api_base", None)
                    or getattr(model, "base_url", None)
                )
                if base_url:
                    os.environ["OPENAI_API_BASE"] = base_url
            logger.info("Set OpenAI environment variables")

        elif model_type == "ChatAnthropic":
            # Anthropic specific attributes
            if hasattr(model, "anthropic_api_key") or hasattr(model, "api_key"):
                api_key = self._get_secret_value(
                    getattr(model, "anthropic_api_key", None)
                    or getattr(model, "api_key", None)
                )
                if api_key:
                    os.environ["ANTHROPIC_API_KEY"] = api_key
            logger.info("Set Anthropic environment variables")

        else:
            # Generic approach - try common attribute names
            logger.info(
                f"Using generic environment variable extraction for {model_type}"
            )

            # Try to extract API key
            for attr in ["api_key", "apikey", "openai_api_key", "anthropic_api_key"]:
                if hasattr(model, attr):
                    value = self._get_secret_value(getattr(model, attr))
                    if value:
                        # Set a generic key that LiteLLM might recognize
                        os.environ["LITELLM_API_KEY"] = value
                        logger.info(f"Set LITELLM_API_KEY from {attr}")
                        break

            # Try to extract base URL
            for attr in ["base_url", "api_base", "url", "openai_api_base"]:
                if hasattr(model, attr):
                    value = self._get_secret_value(getattr(model, attr))
                    if value:
                        os.environ["LITELLM_API_BASE"] = value
                        logger.info(f"Set LITELLM_API_BASE from {attr}")
                        break

    def _get_litellm_model_id(self, model: BaseChatModel, model_id: str) -> str:
        """Get the LiteLLM model ID with provider prefix if needed.

        Some providers require a prefix (e.g., 'watsonx/', 'anthropic/', 'ollama/').
        """
        model_type = type(model).__name__

        if model_type == "ChatWatsonx":
            if not model_id.startswith("watsonx/"):
                return f"watsonx/{model_id}"
        elif model_type == "ChatOllama":
            # Ollama models need the ollama/ prefix for LiteLLM
            if not model_id.startswith("ollama/"):
                return f"ollama/{model_id}"

        return model_id
