"""OpenDsStarAgent - Main user-facing agent class.

This module provides a simple interface for users to interact with
the agent implementation.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from langchain_core.language_models import BaseChatModel

from OpenDsStar.agents.base_agent import BaseAgent
from OpenDsStar.agents.ds_star.ds_star_graph import DSStarGraph
from OpenDsStar.agents.ds_star.ds_star_results_prep import (
    prepare_result_from_graph_state_ds_star_agent,
)
from OpenDsStar.agents.ds_star.ds_star_state import CodeMode
from OpenDsStar.agents.utils.logging_utils import init_logger
from OpenDsStar.experiments.core.config import AgentConfig

logger = logging.getLogger(__name__)


class OpenDsStarAgent(BaseAgent):
    def stream_invoke(
        self,
        query: str,
        config: dict[str, Any] | None = None,
        yield_state: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Generator version of invoke: yields after each trajectory event/state.

        Args:
            query: The user's question or task to solve.
            config: Optional LangGraph configuration dict. Callers can pass trajectory_callback
                in config for invoke(), while stream_invoke() internally wires its own.
            yield_state: If True, yields the full DSState after completion. If False, yields event dicts.

        Yields:
            When yield_state=False: Dict with {"step": label, "event": event_data}
            When yield_state=True: Individual events followed by final DSState

        Note:
            This method runs the graph in a background thread and streams events via a queue.
            It ensures all events are drained before termination by relying on a stop_signal sentinel.
        """
        if not query or not isinstance(query, str):
            raise ValueError("query must be a non-empty string")

        import threading
        import uuid
        from queue import Empty, Queue

        # Generate unique session ID to prevent cross-contamination between concurrent runs
        session_id = str(uuid.uuid4())
        event_queue = Queue()
        stop_signal = object()

        def trajectory_callback(event):
            # Wrap event with node/step info for UI and add session ID
            node = event.get("node") if isinstance(event, dict) else None
            step_idx = event.get("step_idx") if isinstance(event, dict) else None
            etype = event.get("type") if isinstance(event, dict) else None
            # Provide a label: prefer node, else step_idx, else type, else unknown
            label = node or (
                f"step_{step_idx}" if step_idx is not None else etype or "unknown"
            )
            event_queue.put({"session_id": session_id, "step": label, "event": event})

        # Shallow copy to avoid mutating caller's config
        config = (
            dict(config)
            if config is not None
            else {
                "configurable": {"thread_id": "default"},
                "recursion_limit": 1000,
            }
        )
        # Don't set trajectory_callback in config to avoid duplication

        def run_graph():
            try:
                state = self._graph.invoke(
                    input_data={
                        "user_query": query,
                        "max_steps": self.max_steps,
                        "code_mode": self.code_mode,
                        "output_max_length": self.output_max_length,
                        "logs_max_length": self.logs_max_length,
                        "trajectory_callback": trajectory_callback,
                    },
                    config=config,
                )
                if yield_state:
                    event_queue.put(state)
                event_queue.put(stop_signal)
            except Exception as e:
                event_queue.put(e)
                event_queue.put(stop_signal)

        thread = threading.Thread(target=run_graph, daemon=True)
        thread.start()

        # Consume until we see the stop signal
        while True:
            try:
                item = event_queue.get(timeout=0.5)
            except Empty:
                continue

            if item is stop_signal:
                break

            if isinstance(item, Exception):
                raise item

            yield item

    def __init__(
        self,
        model: BaseChatModel,
        temperature: float = 0.0,
        tools: list[Any] | None = None,
        system_prompt: str | None = None,
        task_prompt: str | None = None,
        max_steps: int = 5,
        max_debug_tries: int = 5,
        code_timeout: int = 30,
        code_mode: str = "stepwise",
        output_max_length: int = 500,
        logs_max_length: int = 1000,
        cache_dir: Any = None,
    ) -> None:
        """Initialize the OpenDsStarAgent.

        Args:
            model: LangChain BaseChatModel instance (use ModelBuilder.build() to create).
            temperature: Temperature for generation (stored for config, model should already have this set).
            tools: List of LangChain tools the agent can use. Defaults to empty list.
            system_prompt: Optional system prompt to guide the agent's behavior.
                Defaults to a helpful data science assistant prompt.
            task_prompt: Optional task-specific prompt.
            max_steps: Maximum number of planning steps the agent can take. Defaults to 5.
            max_debug_tries: Maximum number of debug attempts per step. Defaults to 5.
            code_timeout: Timeout in seconds for code execution. Defaults to 30.
            code_mode: Code execution mode - either "stepwise" (execute each step separately)
                or "full" (execute all steps together). Defaults to "stepwise".
            output_max_length: Maximum length for output truncation in prompts. Defaults to 500.
            logs_max_length: Maximum length for log truncation in prompts. Defaults to 1000.
            cache_dir: Ignored (kept for backwards compatibility). Cache should be configured when building the model.

        Raises:
            ValueError: If model is not a BaseChatModel instance or code_mode is not recognized.
        """
        # Initialize logger if not already initialized
        # This ensures INFO messages are visible when using the package after pip install
        init_logger()
        
        # Validate model type first
        if not isinstance(model, BaseChatModel):
            raise ValueError(
                f"model must be a LangChain BaseChatModel instance, got: {type(model)}"
            )

        # Validate code_mode
        if code_mode not in ("stepwise", "full"):
            raise ValueError(
                f"code_mode must be either 'stepwise' or 'full', got: {code_mode}"
            )

        # Use the provided model instance directly
        self.model = model
        self._model_id = getattr(model, "model", model.__class__.__name__)

        self.code_mode = CodeMode(code_mode)
        self.temperature = temperature

        # Clean up tools list - remove empty strings or None which Langflow sometimes passes when the input is functionally empty
        _tools = tools or []
        if isinstance(_tools, list):
            _tools = [
                t for t in _tools if t and not (isinstance(t, str) and not t.strip())
            ]

        self.tools = _tools

        self.max_steps = max_steps
        self.max_debug_tries = max_debug_tries
        self.code_timeout = code_timeout
        self.output_max_length = output_max_length
        self.logs_max_length = logs_max_length

        # TODO: check with Yoav if needed
        # Set default system prompt if not provided
        if system_prompt is None:
            system_prompt = (
                "You are a helpful data science assistant. "
                "You can break down complex queries into steps, write code, and provide answers."
            )

        self.system_prompt = system_prompt
        self.task_prompt = task_prompt

        # Store agent configuration for cache key generation
        self.agent_config = AgentConfig(
            agent_type="ds_star",
            model=self._model_id,
            temperature=temperature,
            max_steps=max_steps,
            code_timeout=code_timeout,
            code_mode=code_mode,
            output_max_length=output_max_length,
            logs_max_length=logs_max_length,
            system_prompt=system_prompt,
            task_prompt=task_prompt,
        )

        # Initialize the underlying graph
        self._graph = DSStarGraph(
            model=self.model,
            tools=self.tools,
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            max_steps=max_steps,
            max_debug_tries=max_debug_tries,
            code_timeout=code_timeout,
        )

        logger.info(
            f"OpenDsStarAgent initialized: model={self._model_id}, "
            f"tools={len(self.tools)}, max_steps={max_steps}, "
            f"code_mode={code_mode}, code_timeout={code_timeout}s, "
            f"output_max_length={output_max_length}, logs_max_length={logs_max_length}"
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
    ) -> dict[str, Any] | Any:
        """Execute the agent with the given query.

        Args:
            query: The user's question or task to solve.
            config: Optional LangGraph configuration dict. If not provided, uses defaults.
                Example: {"configurable": {"thread_id": "my_thread"}, "recursion_limit": 1000}
            return_state: If True, returns the full DSState object. If False (default),
                returns a cleaned result dict with answer, trajectory, and metrics.

        Returns:
            Dictionary containing:
                - answer: The final answer to the query
                - trajectory: List of events showing agent's reasoning process
                - plan: String representation of the execution plan
                - steps_used: Number of steps actually used
                - max_steps: Maximum steps allowed
                - verifier_sufficient: Whether the verifier deemed the answer sufficient
                - fatal_error: Any fatal errors encountered (empty string if none)
                - execution_error: Any execution errors in the last step
                - input_tokens: Total input tokens used
                - output_tokens: Total output tokens used
                - num_llm_calls: Number of LLM API calls made

        Raises:
            ValueError: If query is empty or None.
            Exception: If graph execution fails.
        """
        # Update tools if needed - need to find a better soloution(?)
        tools_updated = False
        for tool in self.tools:
            if hasattr(tool, "update_description"):
                tool.update_description(query)
                tools_updated = True

        # If any tool descriptions were updated, regenerate the tools_spec
        # and propagate it to all nodes
        if tools_updated:
            self._graph.update_tools_spec()

        if not query or not isinstance(query, str):
            raise ValueError("query must be a non-empty string")

        # Set default config if not provided
        if config is None:
            config = {
                "configurable": {"thread_id": "default"},
                "recursion_limit": 1000,
            }

        logger.info(f"Invoking agent with query: {query}")

        try:
            # Invoke the graph
            # Allow callers to pass a `trajectory_callback` inside `config` so
            # the low-level LangGraph state will call it for each trajectory event.
            trajectory_callback = None
            if isinstance(config, dict) and "trajectory_callback" in config:
                trajectory_callback = config.get("trajectory_callback")

            state = self._graph.invoke(
                input_data={
                    "user_query": query,
                    "max_steps": self.max_steps,
                    "code_mode": self.code_mode,
                    "output_max_length": self.output_max_length,
                    "logs_max_length": self.logs_max_length,
                    "trajectory_callback": trajectory_callback,
                },
                config=config,
            )

            # Return full state if requested
            if return_state:
                return state

            # Otherwise, prepare a clean result dict
            result = prepare_result_from_graph_state_ds_star_agent(state)

            # If trajectory_callback was used (streaming), clear trajectory from result
            # to avoid duplicates (events were already streamed via callback)
            if trajectory_callback is not None:
                result["trajectory"] = []
                logger.debug(
                    "Cleared trajectory from result (already streamed via callback)"
                )

            logger.info(
                f"Agent completed: {result['steps_used']} steps, "
                f"{result['num_llm_calls']} LLM calls"
            )

            return result

        except Exception as e:
            logger.error(f"Error during agent invocation: {e}", exc_info=True)
            raise
