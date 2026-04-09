"""ReactAgentLangchain - Wrapper for a LangChain ReAct-style agent graph.

Simple wrapper around LangGraph's create_react_agent with the same interface as OpenDsStarAgent.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from OpenDsStar.agents.base_agent import BaseAgent
from OpenDsStar.agents.utils.logging_utils import init_logger
from OpenDsStar.experiments.core.config import AgentConfig

logger = logging.getLogger(__name__)


class ReactAgentLangchain(BaseAgent):
    """ReactAgentLangchain - Wrapper around a LangChain agent graph.

    Provides the same interface as OpenDsStarAgent for consistency.
    Uses LangChain's create_react_agent function.
    """

    def __init__(
        self,
        model: BaseChatModel,
        temperature: float = 0.0,
        tools: list[Any] | None = None,
        system_prompt: str | None = None,
        task_prompt: str | None = None,  # Ignored for React agent
        max_steps: int = 5,
        code_timeout: int = 30,  # Ignored for React agent
        code_mode: str = "stepwise",  # Ignored for React agent
    ) -> None:
        """Initialize the ReactAgentLangchain.

        Args:
            model: LangChain BaseChatModel instance (use ModelBuilder.build() to create).
            temperature: Temperature for generation (stored for config, model should already have this set).

        Raises:
            ValueError: If model is not a BaseChatModel instance
        """
        # Initialize logger if not already initialized
        init_logger()

        # Validate model type
        if not isinstance(model, BaseChatModel):
            raise ValueError(
                f"model must be a LangChain BaseChatModel instance, got: {type(model)}"
            )

        self.model = model
        self._model_id = getattr(model, "model", model.__class__.__name__)
        self.temperature = temperature

        # Clean up empty strings or None which Langflow sometimes passes when the input is functionally empty
        _tools = tools or []
        if isinstance(_tools, list):
            _tools = [
                t for t in _tools if t and not (isinstance(t, str) and not t.strip())
            ]
        self.tools = _tools

        self.max_steps = max_steps
        self.code_timeout = code_timeout  # Stored for interface compatibility
        self.code_mode = code_mode  # Stored for interface compatibility

        self.system_prompt = system_prompt or (
            "You are a helpful AI assistant. "
            "You can use tools to help answer questions and solve problems."
        )
        self.task_prompt = task_prompt  # Stored for interface compatibility

        # Store agent configuration for cache key generation
        self.agent_config = AgentConfig(
            agent_type="react_langchain",
            model=self._model_id,
            temperature=temperature,
            max_steps=max_steps,
            code_timeout=code_timeout,
            code_mode=code_mode,
            system_prompt=self.system_prompt,
            task_prompt=task_prompt,
        )

        # Create the agent using LangGraph's create_react_agent
        # This function expects: model, tools, and optional prompt
        self._graph = create_react_agent(
            model=self.model,
            tools=self.tools,
            prompt=self.system_prompt,
        )

        logger.info(
            "ReactAgentLangchain initialized: model=%s, tools=%d, max_steps=%d",
            self._model_id,
            len(self.tools),
            max_steps,
        )

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

        # Set default config
        if config is None:
            config = {
                "configurable": {"thread_id": "default"},
                "recursion_limit": self.max_steps * 2,  # allow tool calls
            }
        else:
            config = dict(config)  # avoid mutating caller dict
            config.setdefault("recursion_limit", self.max_steps * 2)

        logger.info("Invoking React Agent with query: %s", query)

        state = self._graph.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config,
        )

        if return_state:
            return state

        messages = state.get("messages", [])

        # Final answer: last AI message that is not a tool-call message
        answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                answer = msg.content or ""
                break

        # Build trajectory (similar format to OpenDsStarAgent)
        trajectory: list[dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                trajectory.append({"type": "human", "content": msg.content})

            elif isinstance(msg, AIMessage):
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    trajectory.append({"type": "tool_call", "tool_calls": tool_calls})
                else:
                    trajectory.append({"type": "response", "content": msg.content})

            elif isinstance(msg, ToolMessage):
                trajectory.append(
                    {
                        "type": "tool_result",
                        "tool_call_id": msg.tool_call_id,
                        "name": msg.name,
                        "content": msg.content,
                    }
                )
            else:
                # Fallback for any other message types
                trajectory.append(
                    {"type": "message", "content": getattr(msg, "content", str(msg))}
                )

        num_llm_calls = sum(1 for m in messages if isinstance(m, AIMessage))

        # Extract token usage from messages
        input_tokens, output_tokens = self._extract_token_usage(messages)

        return {
            "answer": answer,
            "messages": messages,  # Include raw messages for compatibility
            "trajectory": trajectory,
            "plan": "",  # ReAct agent doesn't have explicit plans by default
            "steps_used": num_llm_calls,
            "max_steps": self.max_steps,
            "verifier_sufficient": True,
            "fatal_error": "",
            "execution_error": "",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "num_llm_calls": num_llm_calls,
        }

    def _extract_token_usage(self, messages: list[Any]) -> tuple[int, int]:
        """Extract token usage from LangChain messages.

        LangChain messages may contain usage_metadata or response_metadata
        with token information.

        Returns:
            Tuple of (input_tokens, output_tokens)
        """
        total_input = 0
        total_output = 0

        for msg in messages:
            # Check for usage_metadata (newer LangChain)
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                usage = msg.usage_metadata
                if isinstance(usage, dict):
                    total_input += usage.get("input_tokens", 0)
                    total_output += usage.get("output_tokens", 0)

            # Check for response_metadata (older LangChain / LiteLLM)
            if hasattr(msg, "response_metadata") and msg.response_metadata:
                metadata = msg.response_metadata
                if isinstance(metadata, dict):
                    # Check for token_usage in response_metadata
                    token_usage = metadata.get("token_usage", {})
                    if isinstance(token_usage, dict):
                        total_input += token_usage.get(
                            "prompt_tokens", 0
                        ) or token_usage.get("input_tokens", 0)
                        total_output += token_usage.get(
                            "completion_tokens", 0
                        ) or token_usage.get("output_tokens", 0)

        return total_input, total_output
