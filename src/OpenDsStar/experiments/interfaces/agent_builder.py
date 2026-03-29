"""
Interface for building agents.

This module defines the interface that all agent builders must implement.
Agent builders are responsible for creating and configuring agents with
the appropriate tools and settings for a specific experiment.

The separation between AgentBuilder (interface) and agent implementations
allows experiments to be decoupled from specific agent implementations,
improving modularity and testability.

Note: The Agent protocol is satisfied by agents.BaseAgent and all its subclasses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, Sequence

from langchain_core.tools import BaseTool

from ..core.context import PipelineContext


class Agent(Protocol):
    """
    Protocol for agents with invoke method.

    This protocol defines the minimal interface that any agent must implement
    to be compatible with the experiment pipeline. Agents can have additional
    methods and attributes, but must at least provide an invoke() method.

    Note: All agents inheriting from OpenDsStar.agents.BaseAgent automatically satisfy
    this protocol, as BaseAgent provides a more comprehensive interface that
    includes invoke() and additional methods.

    Example:
        from agents import BaseAgent

        class MyAgent(BaseAgent):
            def invoke(self, query: str, **kwargs) -> dict[str, Any]:
                # Agent logic here
                return {"answer": "result"}

            @property
            def model_id(self) -> str:
                return "my-model"
    """

    def invoke(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """
        Execute the agent with the given query.

        Args:
            query: The user's question or task to solve
            **kwargs: Additional arguments for agent execution
                     (e.g., config, return_state, etc.)

        Returns:
            Dictionary containing at least an 'answer' key with the result.
            May include additional keys like 'trajectory', 'metadata', etc.

            For agents inheriting from BaseAgent, the return dict includes:
                - answer: The final answer (str)
                - trajectory: List of reasoning steps (list)
                - plan: Execution plan (str)
                - steps_used: Number of steps taken (int)
                - max_steps: Maximum steps allowed (int)
                - verifier_sufficient: Whether answer is sufficient (bool)
                - fatal_error: Fatal errors if any (str)
                - execution_error: Execution errors if any (str)
                - input_tokens: Input tokens used (int)
                - output_tokens: Output tokens used (int)
                - num_llm_calls: Number of LLM calls (int)
        """
        ...


class AgentBuilder(ABC):
    """
    Interface for building an agent with tools.

    Agent builders are responsible for:
    1. Creating an agent instance
    2. Configuring the agent with appropriate parameters
    3. Providing the agent with tools
    4. Providing agent configuration for reproducibility

    This interface allows experiments to create agents without knowing
    the specific implementation details, promoting loose coupling.

    Example:
        class MyAgentBuilder(AgentBuilder):
            def __init__(self, model: str, temperature: float):
                self.model = model
                self.temperature = temperature

            def build_agent(self, ctx, tools):
                from my_agents import MyAgent
                return MyAgent(
                    model=self.model,
                    temperature=self.temperature,
                    tools=tools
                )

            def get_agent_config(self) -> Dict[str, Any]:
                return {
                    "agent_type": "my_agent",
                    "model": self.model,
                    "temperature": self.temperature
                }
    """

    @abstractmethod
    def build_agent(
        self,
        ctx: PipelineContext,
        tools: Sequence[BaseTool],
    ) -> Agent:
        """
        Build an agent configured with the given tools.

        This method is called by the experiment pipeline after tools have
        been created. The builder should instantiate and configure an agent
        with the provided tools.

        Args:
            ctx: Pipeline context with config.
                 Use ctx.config for configuration.
            tools: LangChain BaseTool instances to provide to the agent.
                   These tools have been created by ToolBuilders and are
                   ready for use.

        Returns:
            Configured agent with invoke() method ready for execution.
            The agent must implement the Agent protocol (have an invoke method).

        Raises:
            ImportError: If agent implementation cannot be imported
            ValueError: If configuration is invalid

        Example:
            import logging

            logger = logging.getLogger(__name__)

            def build_agent(self, ctx, tools):
                logger.info(f"building_agent tool_count={len(tools)}")

                from agents import OpenDsStarAgent

                agent = OpenDsStarAgent(
                    model=self.model,
                    temperature=self.temperature,
                    tools=list(tools),
                    max_steps=self.max_steps,
                )

                logger.info("agent_built agent_type=OpenDsStarAgent")
                return agent
        """
        raise NotImplementedError
