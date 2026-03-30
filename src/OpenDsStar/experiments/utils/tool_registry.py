"""Tool registry for managing tool name collisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from langchain_core.tools import BaseTool


@dataclass
class ToolRegistry:
    """
    Collect tools from builders, handle collisions deterministically.
    Strategy:
      - If name collision: prefix with "{namespace}.{name}"
    """

    _tools_by_name: Dict[str, BaseTool] = field(default_factory=dict)

    def add_all(self, tools: list[BaseTool], namespace: str) -> None:
        """
        Add tools to registry with namespace collision handling.

        Args:
            tools: List of LangChain BaseTool instances to add
            namespace: Namespace for collision resolution

        Raises:
            ValueError: If tool is missing name or collision cannot be resolved
        """
        for t in tools:
            name = getattr(t, "name", None)
            if not name:
                raise ValueError(f"Tool missing .name in namespace={namespace}: {t}")
            if name in self._tools_by_name:
                new_name = f"{namespace}.{name}"
                if new_name in self._tools_by_name:
                    raise ValueError(
                        f"Tool name collision even after namespacing: {new_name}"
                    )
                self._tools_by_name[new_name] = _RenamedTool(inner=t, name=new_name)
            else:
                self._tools_by_name[name] = t

    def list(self) -> List[BaseTool]:
        """Get all registered tools."""
        return list(self._tools_by_name.values())


class _RenamedTool(BaseTool):
    """Wrapper for a tool with a renamed name."""

    inner: BaseTool

    def __init__(self, inner: BaseTool, name: str):
        """Initialize renamed tool wrapper."""
        super().__init__(name=name, description=inner.description, inner=inner)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the inner tool."""
        return self.inner._run(*args, **kwargs)
