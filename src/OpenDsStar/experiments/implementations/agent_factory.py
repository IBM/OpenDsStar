from __future__ import annotations

from enum import Enum
from typing import Any, List

from langchain_core.tools import BaseTool

from OpenDsStar.agents import (
    CodeActAgentSmolagents,
    OpenDsStarAgent,
    ReactAgentLangchain,
    ReactAgentSmolagents,
)

from ..interfaces.agent_builder import Agent, AgentBuilder


class AgentType(str, Enum):
    """Supported agent types."""

    DS_STAR = "ds_star"
    REACT_LANGCHAIN = "react_langchain"
    CODEACT_SMOLAGENTS = "codeact_smolagents"
    REACT_SMOLAGENTS = "react_smolagents"

    # Backwards compatibility alias
    REACT = "react_langchain"  # Maps to REACT_LANGCHAIN


class AgentFactory(AgentBuilder):
    """
    Agent factory that creates agents based on configuration.

    This builder reads the agent_type from the context configuration and
    creates the appropriate agent type (DS-Star, React, CodeAct, etc.).
    """

    def __init__(
        self,
        agent_type: AgentType | None = None,
        model: str | None = None,
        max_steps: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the agent factory.

        Args:
            agent_type: Override agent type (if None, uses context config)
            model: Override model (if None, uses context config)
            max_steps: Override max_steps (if None, uses context config)
            temperature: Override temperature (if None, uses context config)
            **kwargs: Additional agent-specific parameters
        """
        self._agent_type_override = agent_type
        self._model_override = model
        self._max_steps_override = max_steps
        self._temperature_override = temperature
        self._extra_kwargs = kwargs

    def _resolve_config(self, ctx) -> dict[str, Any]:
        """
        Resolve agent configuration from context and overrides.

        Args:
            ctx: Pipeline context with configuration

        Returns:
            Dictionary with resolved configuration
        """
        agent_config = ctx.config.agent_config

        if agent_config is None:
            # Fallback to defaults if no agent config
            agent_type = self._agent_type_override or AgentType.DS_STAR
            model = self._model_override or "watsonx/mistralai/mistral-medium-2505"
            max_steps = self._max_steps_override or 10
            temperature = self._temperature_override or 0.0
            system_prompt = None
            max_debug_tries = 5
            code_timeout = 30
            code_mode = "stepwise"
            output_max_length = 500
            logs_max_length = 20000
            task_prompt = None
        else:
            # Use config values with overrides taking precedence
            agent_type = self._agent_type_override or agent_config.agent_type
            model = self._model_override or agent_config.model
            max_steps = self._max_steps_override or agent_config.max_steps
            temperature = self._temperature_override or agent_config.temperature
            system_prompt = agent_config.system_prompt
            max_debug_tries = agent_config.max_debug_tries
            code_timeout = agent_config.code_timeout
            code_mode = agent_config.code_mode
            output_max_length = agent_config.output_max_length
            logs_max_length = agent_config.logs_max_length
            task_prompt = agent_config.task_prompt

        return {
            "agent_type": agent_type,
            "model": model,
            "max_steps": max_steps,
            "temperature": temperature,
            "system_prompt": system_prompt,
            "max_debug_tries": max_debug_tries,
            "code_timeout": code_timeout,
            "code_mode": code_mode,
            "output_max_length": output_max_length,
            "logs_max_length": logs_max_length,
            "task_prompt": task_prompt,
        }

    def build_agent(self, ctx, tools) -> Agent:
        """
        Build an agent using the factory based on configuration.

        Args:
            ctx: Pipeline context with configuration
            tools: List of tools for the agent

        Returns:
            An agent instance (OpenDsStarAgent, ReactAgent, etc.)
        """
        import logging

        logger = logging.getLogger(__name__)

        # Resolve configuration
        config = self._resolve_config(ctx)

        logger.info(
            f"building_agent agent_type={config['agent_type']} model={config['model']} "
            f"max_steps={config['max_steps']} num_tools={len(tools)}"
        )

        # Merge extra_kwargs with agent-specific params, avoiding duplicates
        agent_kwargs = {
            "max_debug_tries": config["max_debug_tries"],
            "code_timeout": config["code_timeout"],
            "code_mode": config["code_mode"],
            "output_max_length": config["output_max_length"],
            "logs_max_length": config["logs_max_length"],
            "task_prompt": config["task_prompt"],
            "cache_dir": ctx.config.cache_dir,  # Pass cache_dir from context
        }
        # Update with extra_kwargs, allowing them to override
        agent_kwargs.update(self._extra_kwargs)

        # Create agent using internal factory method
        agent = self._create_agent(
            agent_type=config["agent_type"],  # type: ignore
            model=config["model"],
            tools=list(tools),
            temperature=config["temperature"],
            max_steps=config["max_steps"],
            system_prompt=config["system_prompt"],
            **agent_kwargs,
        )

        return agent  # type: ignore[return-value]

    @staticmethod
    def _create_agent(
        agent_type: AgentType,
        model: str,
        tools: List[BaseTool],
        temperature: float = 0.0,
        max_steps: int = 5,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> (
        OpenDsStarAgent
        | ReactAgentLangchain
        | CodeActAgentSmolagents
        | ReactAgentSmolagents
    ):
        """
        Create an agent based on the specified type.

        Args:
            agent_type: Type of agent to create
            model: Model identifier
            tools: List of tools for the agent
            temperature: Temperature for generation
            max_steps: Maximum steps/iterations
            system_prompt: Optional system prompt
            **kwargs: Additional agent-specific parameters

        Returns:
            An instance of the requested agent type

        Raises:
            ValueError: If agent_type is not recognized
        """
        if agent_type == AgentType.DS_STAR:
            return AgentFactory._create_ds_star_agent(
                model=model,
                tools=tools,
                temperature=temperature,
                max_steps=max_steps,
                system_prompt=system_prompt,
                **kwargs,
            )
        elif agent_type in (AgentType.REACT, AgentType.REACT_LANGCHAIN):
            return AgentFactory._create_react_langchain_agent(
                model=model,
                tools=tools,
                temperature=temperature,
                max_steps=max_steps,
                system_prompt=system_prompt,
                **kwargs,
            )
        elif agent_type == AgentType.CODEACT_SMOLAGENTS:
            return AgentFactory._create_codeact_smolagents_agent(
                model=model,
                tools=tools,
                temperature=temperature,
                max_steps=max_steps,
                system_prompt=system_prompt,
                **kwargs,
            )
        elif agent_type == AgentType.REACT_SMOLAGENTS:
            return AgentFactory._create_react_smolagents_agent(
                model=model,
                tools=tools,
                temperature=temperature,
                max_steps=max_steps,
                system_prompt=system_prompt,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown agent_type: {agent_type}. Supported types: {[t.value for t in AgentFactory.get_supported_types()]}"
            )

    @staticmethod
    def _create_ds_star_agent(
        model: str,
        tools: List[BaseTool],
        temperature: float,
        max_steps: int,
        system_prompt: str | None,
        code_timeout: int,
        code_mode: str,
        output_max_length: int,
        logs_max_length: int,
        task_prompt: str | None = None,
        max_debug_tries: int = 5,
        cache_dir: Any = None,
        **kwargs: Any,
    ) -> OpenDsStarAgent:
        """Create a DS-Star agent."""
        from pathlib import Path

        from OpenDsStar.agents.utils.model_builder import ModelBuilder

        # Build model instance
        model_instance, _ = ModelBuilder.build(
            model=model,
            temperature=temperature,
            cache_dir=Path(cache_dir) if cache_dir else None,
            framework="langchain",
        )

        return OpenDsStarAgent(
            model=model_instance,
            temperature=temperature,
            tools=tools,
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            max_steps=max_steps,
            max_debug_tries=max_debug_tries,
            code_timeout=code_timeout,
            code_mode=code_mode,
            output_max_length=output_max_length,
            logs_max_length=logs_max_length,
            cache_dir=cache_dir,
        )

    @staticmethod
    def _create_react_langchain_agent(
        model: str,
        tools: List[BaseTool],
        temperature: float,
        max_steps: int,
        system_prompt: str | None,
        **kwargs: Any,
    ) -> ReactAgentLangchain:
        """Create a React agent using LangChain."""
        from OpenDsStar.agents.utils.model_builder import ModelBuilder

        # Build model instance (no cache for React agent)
        model_instance, _ = ModelBuilder.build(
            model=model,
            temperature=temperature,
            cache_dir=None,
            framework="langchain",
        )

        return ReactAgentLangchain(
            model=model_instance,
            temperature=temperature,
            tools=tools,
            system_prompt=system_prompt,
            max_steps=max_steps,
        )

    @staticmethod
    def _create_codeact_smolagents_agent(
        model: str,
        tools: List[BaseTool],
        temperature: float,
        max_steps: int,
        system_prompt: str | None,
        code_timeout: int = 30,
        cache_dir: Any = None,
        **kwargs: Any,
    ) -> CodeActAgentSmolagents:
        """Create a CodeAct agent using smolagents."""
        from pathlib import Path

        from OpenDsStar.agents.utils.model_builder import ModelBuilder

        # Build LiteLLMModel using framework parameter
        model_instance, _ = ModelBuilder.build(
            model=model,
            temperature=temperature,
            cache_dir=Path(cache_dir) if cache_dir else None,
            framework="smolagents",
        )

        return CodeActAgentSmolagents(
            model=model_instance,
            temperature=temperature,
            tools=tools,
            system_prompt=system_prompt,
            max_steps=max_steps,
            code_timeout=code_timeout,
            cache_dir=Path(cache_dir) if cache_dir else None,
        )

    @staticmethod
    def _create_react_smolagents_agent(
        model: str,
        tools: List[BaseTool],
        temperature: float,
        max_steps: int,
        system_prompt: str | None,
        cache_dir: Any = None,
        **kwargs: Any,
    ) -> ReactAgentSmolagents:
        """Create a React agent using smolagents."""
        from pathlib import Path

        from OpenDsStar.agents.utils.model_builder import ModelBuilder

        # Build LiteLLMModel using framework parameter
        model_instance, _ = ModelBuilder.build(
            model=model,
            temperature=temperature,
            cache_dir=Path(cache_dir) if cache_dir else None,
            framework="smolagents",
        )

        return ReactAgentSmolagents(
            model=model_instance,
            temperature=temperature,
            tools=tools,
            system_prompt=system_prompt,
            max_steps=max_steps,
            cache_dir=Path(cache_dir) if cache_dir else None,
        )

    @staticmethod
    def get_supported_types() -> List[AgentType]:
        """Get list of supported agent types."""
        return [
            AgentType.DS_STAR,
            AgentType.REACT_LANGCHAIN,
            AgentType.CODEACT_SMOLAGENTS,
            AgentType.REACT_SMOLAGENTS,
        ]


# Backwards compatibility alias
FlexibleAgentBuilder = AgentFactory
