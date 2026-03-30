"""ReactAgentSmolagents - Wrapper for smolagents ToolCallingAgent.

Wrapper around smolagents' ToolCallingAgent with the same interface as OpenDsStarAgent.
ToolCallingAgent uses a ReAct-style approach with tool calling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from OpenDsStar.agents.base_agent import BaseAgent
from OpenDsStar.experiments.core.config import AgentConfig
from langchain_core.tools import BaseTool as LangChainBaseTool
from smolagents import RunResult, ToolCallingAgent
from smolagents import Tool as SmolagentsTool

logger = logging.getLogger(__name__)


class ReactAgentSmolagents(BaseAgent):
    """ReactAgentSmolagents - Wrapper around smolagents' ToolCallingAgent.

    Provides the same interface as OpenDsStarAgent for consistency.
    Uses smolagents' ToolCallingAgent which follows a ReAct-style reasoning approach.
    """

    def __init__(
        self,
        model: Any,  # smolagents LiteLLMModel instance
        temperature: float = 0.0,
        tools: list[Any] | None = None,
        system_prompt: str | None = None,
        task_prompt: str | None = None,  # Ignored for smolagents
        max_steps: int = 5,
        code_timeout: int = 30,  # Ignored for smolagents
        code_mode: str = "stepwise",  # Ignored for smolagents
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize the ReactAgentSmolagents.

        Args:
            model: smolagents LiteLLMModel instance (use ModelBuilder.build() with framework="smolagents" to create).
            temperature: Temperature for generation (stored for config, model should already have this set).
            cache_dir: Ignored (kept for backwards compatibility). Cache should be configured when building the model.

        Raises:
            ValueError: If model is invalid
        """
        # Import here to avoid requiring smolagents if not used
        try:
            from smolagents import LiteLLMModel
        except ImportError:
            raise ImportError(
                "smolagents is required for ReactAgentSmolagents. "
                "Install it with: pip install smolagents"
            )

        # Validate model type
        if not isinstance(model, LiteLLMModel):
            raise ValueError(
                f"model must be a smolagents LiteLLMModel instance, got: {type(model)}"
            )

        self.smol_model = model
        self._model_id = getattr(model, "model_id", "unknown")
        logger.info(f"Using LiteLLMModel for model_id: {self._model_id}")

        self.temperature = temperature
        
        # Clean up empty strings or None which Langflow sometimes passes when the input is functionally empty
        _tools = tools or []
        if isinstance(_tools, list):
            _tools = [t for t in _tools if t and not (isinstance(t, str) and not t.strip())]
        self.tools = _tools
        
        self.max_steps = max_steps
        self.code_timeout = code_timeout  # Stored for interface compatibility
        self.code_mode = code_mode  # Stored for interface compatibility

        # Convert LangChain tools to smolagents tools if needed
        smol_tools = self._convert_tools(self.tools)

        # Create the ToolCallingAgent
        self._agent = ToolCallingAgent(
            tools=smol_tools,
            model=self.smol_model,
            max_steps=max_steps,
        )

        # Override system prompt if provided
        if system_prompt:
            self._agent.system_prompt = system_prompt

        self.system_prompt = system_prompt
        self.task_prompt = task_prompt  # Stored for interface compatibility

        # Store agent configuration for cache key generation
        self.agent_config = AgentConfig(
            agent_type="react_smolagents",
            model=self._model_id,
            temperature=temperature,
            max_steps=max_steps,
            code_timeout=code_timeout,
            code_mode=code_mode,
            system_prompt=system_prompt,
            task_prompt=task_prompt,
        )

        logger.info(
            "ReactAgentSmolagents initialized: model=%s, tools=%d, max_steps=%d, cache=%s",
            self._model_id,
            len(self.tools),
            max_steps,
            "enabled" if cache_dir else "disabled",
        )

    def _convert_tools(self, tools: list[Any]) -> list[Any]:
        """Convert LangChain tools to smolagents format if needed.

        Uses smolagents' built-in Tool.from_langchain() method to convert
        LangChain BaseTool instances to smolagents tools.
        """
        converted_tools = []
        for tool in tools:
            # Check if it's a LangChain tool
            if isinstance(tool, LangChainBaseTool):
                logger.info(
                    f"Converting LangChain tool '{tool.name}' to smolagents format"
                )
                # Use smolagents' built-in conversion method
                wrapped_tool = SmolagentsTool.from_langchain(tool)
                converted_tools.append(wrapped_tool)
            else:
                # Already a smolagents tool or compatible
                converted_tools.append(tool)

        return converted_tools

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
        """Execute the agent with a query.
        """
        if not query or not isinstance(query, str):
            raise ValueError("query must be a non-empty string")

        logger.info("Invoking ReactAgentSmolagents with query: %s", query)

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
                "plan": "",  # ToolCallingAgent doesn't have explicit plans
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
            logger.error("ReactAgentSmolagents execution failed: %s", str(e))
            return {
                "answer": "",
                "trajectory": [{"type": "error", "content": str(e)}],
                "plan": "",
                "steps_used": 0,
                "max_steps": self.max_steps,
                "verifier_sufficient": False,
                "fatal_error": str(e),
                "execution_error": str(e),
                "input_tokens": 0,
                "output_tokens": 0,
                "num_llm_calls": 0,
            }
