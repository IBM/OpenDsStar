# DS-STAR-style loop using LangGraph
from __future__ import annotations

import logging
from dataclasses import fields
from typing import Any, Callable, Dict, List, Optional, cast

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph

from OpenDsStar.agents.ds_star.ds_star_state import DSState
from OpenDsStar.agents.ds_star.ds_star_utils import (
    add_event_to_trajectory,
    build_tools_map,
    format_tools_spec,
)
from OpenDsStar.agents.ds_star.nodes.coder import CoderNode
from OpenDsStar.agents.ds_star.nodes.debugger import DebuggerNode
from OpenDsStar.agents.ds_star.nodes.executer import ExecutorNode
from OpenDsStar.agents.ds_star.nodes.finalizer import FinalizerNode
from OpenDsStar.agents.ds_star.nodes.planner import PlannerNode
from OpenDsStar.agents.ds_star.nodes.router import RouterNode
from OpenDsStar.agents.ds_star.nodes.verifier import VerifierNode

logger = logging.getLogger(__name__)


class DSStarGraph:
    def __init__(
        self,
        model: BaseChatModel,
        tools: List[Any],
        system_prompt: Optional[str] = None,
        task_prompt: Optional[str] = None,
        max_steps: int = 5,
        max_debug_tries: int = 5,
        code_timeout: int = 30,
    ) -> None:
        self.model = model
        self.max_steps = max_steps
        self.max_debug_tries = max_debug_tries
        self.code_timeout = code_timeout
        self.system_prompt = system_prompt
        self.task_prompt = task_prompt

        self.tools_list = tools
        self.tools: Dict[str, Callable[..., Any]] = build_tools_map(self.tools_list)

        # Tool spec string for all nodes
        self.tools_spec = format_tools_spec(self.tools_list)

        # Nodes
        self.planner = PlannerNode(
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            tools_spec=self.tools_spec,
            llm=self.model,
        )
        self.coder = CoderNode(
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            tools_spec=self.tools_spec,
            llm=self.model,
        )
        self.executor = ExecutorNode(
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            tools_spec=self.tools_spec,
            tools=self.tools,
            code_timeout=code_timeout,
        )
        self.debugger = DebuggerNode(
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            tools_spec=self.tools_spec,
            llm=self.model,
        )
        self.verifier = VerifierNode(
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            tools_spec=self.tools_spec,
            llm=self.model,
        )
        self.router = RouterNode(
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            tools_spec=self.tools_spec,
            llm=self.model,
        )
        self.finalizer = FinalizerNode(
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            tools_spec=self.tools_spec,
            llm=self.model,
        )

        # Store all nodes for easy iteration (e.g., updating tools_spec)
        self._nodes = [
            self.planner,
            self.coder,
            self.executor,
            self.debugger,
            self.verifier,
            self.router,
            self.finalizer,
        ]

        self.graph = self._build_graph()

    def update_tools_spec(self) -> None:
        """
        Regenerate tools_spec from the current tool descriptions
        and update all nodes with the new spec.
        """
        self.tools_spec = format_tools_spec(self.tools_list)
        for node in self._nodes:
            node.tools_spec = self.tools_spec

    # ------ Routing helpers ------
    def _get_last_step(self, state: DSState) -> Any:
        """Return last step object or None. Centralized to avoid repeated footguns."""
        return state.steps[-1] if state.steps else None

    def route_after_execute(self, state: DSState) -> str:
        """Route after executing the code: debugger on error, otherwise verifier."""
        if state.fatal_error:
            return "n_finalizer"

        last_step = self._get_last_step(state)
        if last_step is None:
            state.fatal_error = "No steps found after execution."
            return "n_finalizer"

        if getattr(last_step, "debug_tries", 0) >= state.max_debug_tries:
            state.fatal_error = "Max debug attempts reached."
            return "n_finalizer"

        if getattr(last_step, "execution_error", None):
            return "n_debug"

        return "n_verify"

    def route_after_verify(self, state: DSState) -> str:
        """Route after verification: finalize if sufficient, else router."""
        if state.fatal_error:
            return "n_finalizer"

        last_step = self._get_last_step(state)
        if last_step is None:
            state.fatal_error = "No steps found after verification."
            return "n_finalizer"

        if bool(getattr(last_step, "verifier_sufficient", False)):
            return "n_finalizer"

        # Safety net (also should be enforced in verifier)
        if state.steps_used >= state.max_steps:
            logger.warning(
                "Max steps reached (%s >= %s)", state.steps_used, state.max_steps
            )
            state.fatal_error = f"Max step limit reached ({state.max_steps})"
            return "n_finalizer"

        return "n_route"

    def route_after_route(self, state: DSState) -> str:
        """Route after router decision: back to planning or finalize."""
        if state.fatal_error:
            return "n_finalizer"

        last_step = self._get_last_step(state)
        if last_step is None:
            state.fatal_error = "No steps found after routing."
            return "n_finalizer"

        action = getattr(last_step, "router_action", None)
        if action in {"add_next_step", "fix_step"}:
            return "n_plan_one"

        return "n_finalizer"

    # ------ Graph builder ------
    def _build_graph(self) -> Any:
        """Build and compile the LangGraph state graph."""
        g: StateGraph = StateGraph(DSState)

        g.add_node("n_plan_one", self.planner)
        g.add_node("n_code", self.coder)
        g.add_node("n_execute", self.executor)
        g.add_node("n_debug", self.debugger)
        g.add_node("n_verify", self.verifier)
        g.add_node("n_route", self.router)
        g.add_node("n_finalizer", self.finalizer)

        g.set_entry_point("n_plan_one")
        g.add_edge("n_plan_one", "n_code")
        g.add_edge("n_code", "n_execute")

        g.add_conditional_edges(
            "n_execute",
            self.route_after_execute,
            {
                "n_debug": "n_debug",
                "n_verify": "n_verify",
                "n_finalizer": "n_finalizer",
            },
        )

        # Debugger always goes back to execute (with corrected code)
        g.add_edge("n_debug", "n_execute")

        g.add_conditional_edges(
            "n_verify",
            self.route_after_verify,
            {
                "n_finalizer": "n_finalizer",
                "n_route": "n_route",
            },
        )

        g.add_conditional_edges(
            "n_route",
            self.route_after_route,
            {
                "n_plan_one": "n_plan_one",
                "n_finalizer": "n_finalizer",
            },
        )

        g.add_edge("n_finalizer", END)
        return g.compile()

    # ------ Public: initialization and invocation ------
    def _init_state(self, input_dict: Dict[str, Any]) -> DSState:
        if "user_query" not in input_dict:
            raise ValueError("Missing required field: 'user_query'")

        valid_fields = {f.name for f in fields(DSState)}
        init_kwargs = {k: v for k, v in input_dict.items() if k in valid_fields}

        # Defaults
        init_kwargs.setdefault("tools", {})
        init_kwargs.setdefault("max_steps", self.max_steps)
        init_kwargs.setdefault("max_debug_tries", self.max_debug_tries)

        return DSState(**init_kwargs)

    def invoke(
        self, input_data: Dict[str, Any] | DSState, config: Optional[dict] = None
    ) -> DSState:
        """
        Execute the DS-STAR graph with the given input.

        Args:
            input_data: Either a dict containing 'user_query' or a DSState object.
            config: Optional LangGraph config dict.

        Returns:
            Final DSState after graph execution.
        """
        # Handle both dict and DSState inputs
        if isinstance(input_data, DSState):
            state = input_data
        else:
            if "user_query" not in input_data:
                raise ValueError("Missing required key: 'user_query'")
            state = self._init_state(input_data)

        add_event_to_trajectory(state, "entry", note="graph_start")
        return cast(DSState, self.graph.invoke(state, config=config))
