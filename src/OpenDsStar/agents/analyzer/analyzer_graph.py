"""Analyzer agent graph using LangGraph."""

from __future__ import annotations

import logging
from dataclasses import fields
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph

from OpenDsStar.agents.analyzer.analyzer_state import AnalyzerState
from OpenDsStar.agents.analyzer.nodes.coder import CoderNode
from OpenDsStar.agents.analyzer.nodes.debugger import DebuggerNode
from OpenDsStar.agents.analyzer.nodes.executer import ExecutorNode
from OpenDsStar.agents.analyzer.nodes.finalizer import FinalizerNode

logger = logging.getLogger(__name__)


class AnalyzerGraph:
    """
    Simple analyzer agent that generates code to analyze a file,
    executes it, and debugs if needed.

    Uses DS*Star components (CoderNode, DebuggerNode, ExecutorNode) via adapters.

    Graph flow:
        coder -> executor -> [debugger -> executor] (if error & retries left) -> finalizer
    """

    def __init__(
        self,
        llm: BaseChatModel,
        code_timeout: int = 30,
        max_debug_tries: int = 3,
    ) -> None:
        self.llm = llm
        self.code_timeout = code_timeout
        self.max_debug_tries = max_debug_tries

        # Initialize nodes (these are adapters that use DS*Star nodes internally)
        self.coder = CoderNode(llm=self.llm)
        self.executor = ExecutorNode(code_timeout=self.code_timeout)
        self.debugger = DebuggerNode(llm=self.llm)
        self.finalizer = FinalizerNode()

        # Build the graph
        self.graph = self._build_graph()

    def route_after_execute(self, state: AnalyzerState) -> str:
        """Route after execution: either debug or finalize."""
        if state.fatal_error:
            return "n_finalizer"

        # If error and haven't exceeded max debug tries
        if state.execution_error:
            if state.debug_tries < state.max_debug_tries:
                return "n_debug"
            else:
                logger.warning(f"Max debug tries reached ({state.max_debug_tries})")
                return "n_finalizer"

        # No error - go to finalizer
        return "n_finalizer"

    def _build_graph(self) -> Any:
        """Build and compile the LangGraph state graph."""
        g = StateGraph(AnalyzerState)

        # Add nodes
        g.add_node("n_code", self.coder)
        g.add_node("n_execute", self.executor)
        g.add_node("n_debug", self.debugger)
        g.add_node("n_finalizer", self.finalizer)

        # Define edges
        g.set_entry_point("n_code")
        g.add_edge("n_code", "n_execute")

        # After execute -> conditional routing
        g.add_conditional_edges(
            "n_execute",
            self.route_after_execute,
            {
                "n_debug": "n_debug",
                "n_finalizer": "n_finalizer",
            },
        )

        # Debugger always goes back to execute
        g.add_edge("n_debug", "n_execute")

        # Finalizer ends the graph
        g.add_edge("n_finalizer", END)

        return g.compile()

    def _init_state(self, input_dict: dict) -> AnalyzerState:
        """Initialize state from input dictionary."""
        if "filename" not in input_dict:
            raise ValueError("Missing required field: 'filename'")

        valid_fields = {f.name for f in fields(AnalyzerState)}
        init_kwargs = {k: v for k, v in input_dict.items() if k in valid_fields}

        # Set max_debug_tries if not provided
        if "max_debug_tries" not in init_kwargs:
            init_kwargs["max_debug_tries"] = self.max_debug_tries

        return AnalyzerState(**init_kwargs)

    def invoke(self, input_dict: dict, config: Optional[dict] = None) -> AnalyzerState:
        """
        Execute the analyzer graph with the given input.

        Args:
            input_dict: Must contain 'filename' at minimum
            config: Optional LangGraph config dict

        Returns:
            Final AnalyzerState after graph execution
        """
        if "filename" not in input_dict:
            raise ValueError("Missing required key: 'filename'")

        state = self._init_state(input_dict)

        state.trajectory.append({"node": "entry", "note": "graph_start"})
        return self.graph.invoke(state, config=config)


def prepare_result_from_graph_state_analyzer_agent(state: dict) -> dict:
    """
    Extract and format key results from the final graph state.

    Args:
        state: The final AnalyzerState dict returned by graph.invoke()

    Returns:
        A clean dict with answer, trajectory, metrics, etc.
    """
    state = AnalyzerState(**state)

    input_tokens = sum(tu.get("input_tokens", 0) for tu in state.token_usage)
    output_tokens = sum(tu.get("output_tokens", 0) for tu in state.token_usage)

    return {
        "answer": state.final_answer,
        "logs": state.logs,
        "outputs": state.outputs,
        "code": state.code,
        "trajectory": state.trajectory,
        "debug_tries": state.debug_tries,
        "max_debug_tries": state.max_debug_tries,
        "execution_error": state.execution_error,
        "fatal_error": state.fatal_error or "",
        "success": not bool(state.execution_error or state.fatal_error),
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "num_llm_calls": int(len(state.token_usage)),
    }
