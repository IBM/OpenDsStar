"""
Base Agent Interface - Abstract base class for all agent implementations.

This module defines the common interface that all agents (ReactAgent, OpenDsStarAgent, etc.)
must implement to ensure consistency across different agent types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from experiments.core.config import AgentConfig


class BaseAgent(ABC):
    """
    Abstract base class for all agent implementations.

    All agents must implement the invoke method and maintain consistent
    initialization parameters and return types.
    """

    def __init__(
        self,
        model: Any,
        temperature: float = 0.0,
        tools: list[Any] | None = None,
        system_prompt: str | None = None,
        task_prompt: str | None = None,
        max_steps: int = 5,
        code_timeout: int = 30,
        code_mode: str = "stepwise",
    ) -> None:
        """
        Initialize the agent.

        Args:
            model: Model instance (BaseChatModel for LangChain agents, LiteLLMModel for smolagents).
                Use ModelBuilder.build() to create the appropriate model instance.
            temperature: Temperature for generation (stored for config, model should already have this set).
            tools: List of tools the agent can use. Defaults to empty list.
            system_prompt: Optional system prompt to guide the agent's behavior.
            task_prompt: Optional task-specific prompt.
            max_steps: Maximum number of steps the agent can take. Defaults to 5.
            code_timeout: Timeout in seconds for code execution. Defaults to 30.
            code_mode: Code execution mode - either "stepwise" or "full". Defaults to "stepwise".
                Note: Some agents may not use all parameters.

        Raises:
            ValueError: If model or other parameters are invalid.
        """
        pass

    @abstractmethod
    def invoke(
        self,
        query: str,
        config: dict[str, Any] | None = None,
        return_state: bool = False,
    ) -> dict[str, Any]:
        """
        Execute the agent with the given query.

        Args:
            query: The user's question or task to solve.
            config: Optional configuration dict for the agent execution.
                Example: {"configurable": {"thread_id": "my_thread"}, "recursion_limit": 1000}
            return_state: If True, returns the full internal state object. If False (default),
                returns a cleaned result dict with answer, trajectory, and metrics.

        Returns:
            Dictionary containing at minimum:
                - answer: The final answer to the query (str)
                - trajectory: List of events showing agent's reasoning process (list)
                - plan: String representation of the execution plan (str)
                - steps_used: Number of steps actually used (int)
                - max_steps: Maximum steps allowed (int)
                - verifier_sufficient: Whether the answer is sufficient (bool)
                - fatal_error: Any fatal errors encountered (str, empty if none)
                - execution_error: Any execution errors in the last step (str, empty if none)
                - input_tokens: Total input tokens used (int)
                - output_tokens: Total output tokens used (int)
                - num_llm_calls: Number of LLM API calls made (int)

            Additional fields may be included depending on the agent implementation.

        Raises:
            ValueError: If query is empty or None.
            Exception: If agent execution fails.
        """
        pass

    @property
    @abstractmethod
    def model_id(self) -> str:
        """
        Get the model identifier.

        Returns:
            String identifier of the model being used.
        """
        pass

    @abstractmethod
    def get_agent_config(self) -> "AgentConfig":
        """
        Get the agent's configuration.

        This method must be implemented by all agents to provide their
        configuration for cache key generation and serialization.

        Returns:
            AgentConfig object containing all agent configuration parameters.
        """
        pass
