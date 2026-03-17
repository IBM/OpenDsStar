"""
Standalone MCP Integration for DS Star Agent.

This module provides complete MCP integration without relying on langflow,
creating LangChain-compatible tools from MCP servers.
"""

import asyncio
import logging
import threading
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, create_model

from tools.mcp_client_standalone import (
    MCPToolInfo,
    StandaloneMCPClient,
    create_mcp_client_from_config,
)

logger = logging.getLogger(__name__)

# Global registry to keep MCP clients and their event loops alive
_mcp_clients: Dict[str, StandaloneMCPClient] = {}
_mcp_loops: Dict[str, asyncio.AbstractEventLoop] = {}
_mcp_threads: Dict[str, threading.Thread] = {}
_client_lock = threading.Lock()


class MCPToolWrapper:
    """
    Wrapper that converts an MCP tool into a LangChain StructuredTool.

    This makes MCP tools compatible with any LangChain-based agent
    (DS Star, React, etc.).
    """

    def __init__(
        self,
        tool_info: MCPToolInfo,
        client: StandaloneMCPClient,
        tool_key: str,
    ):
        """
        Initialize the MCP tool wrapper.

        Args:
            tool_info: Information about the MCP tool
            client: The MCP client to use for calling the tool
            tool_key: Full key for the tool (server_name_tool_name)
        """
        self.tool_info = tool_info
        self.client = client
        self.tool_key = tool_key

    def _create_pydantic_model(self) -> type[BaseModel]:
        """Create a Pydantic model from the tool's input schema."""
        properties = self.tool_info.input_schema.get("properties", {})
        required = self.tool_info.input_schema.get("required", [])

        # Build field definitions
        fields = {}
        for prop_name, prop_info in properties.items():
            prop_type = prop_info.get("type", "string")

            # Map JSON schema types to Python types
            type_map = {
                "string": str,
                "integer": int,
                "number": float,
                "boolean": bool,
                "array": list,
                "object": dict,
            }
            python_type = type_map.get(prop_type, str)

            # Make optional if not required
            if prop_name not in required:
                python_type = Optional[python_type]
                fields[prop_name] = (python_type, None)
            else:
                fields[prop_name] = (python_type, ...)

        # Create dynamic Pydantic model
        model_name = f"{self.tool_info.name.title().replace('_', '')}Input"
        return create_model(model_name, **fields)

    def _sync_func(self, **kwargs) -> str:
        """Synchronous function that runs the async MCP tool."""
        # Always run in a separate thread to avoid event loop conflicts
        return self._run_in_thread(kwargs)

    def _run_in_thread(self, kwargs: Dict[str, Any]) -> str:
        """Run async call using the client's event loop."""
        # Find the event loop for this client
        client_id = None
        with _client_lock:
            for cid, client in _mcp_clients.items():
                if client == self.client:
                    client_id = cid
                    break

        if client_id is None or client_id not in _mcp_loops:
            raise RuntimeError("MCP client event loop not found")

        loop = _mcp_loops[client_id]

        # Schedule the coroutine on the client's event loop
        future = asyncio.run_coroutine_threadsafe(self._async_func(**kwargs), loop)

        # Wait for result (with timeout)
        try:
            return future.result(timeout=60)
        except Exception as e:
            logger.error(f"Error calling MCP tool: {e}")
            raise

    async def _async_func(self, **kwargs) -> str:
        """Async function that calls the MCP tool."""
        try:
            result = await self.client.call_tool(self.tool_key, kwargs)
            # Ensure we return a string, not a CallToolResult object
            if isinstance(result, str):
                return result
            # Handle any object by converting to string
            return str(result)
        except Exception as e:
            logger.error(f"Error calling MCP tool '{self.tool_info.name}': {e}")
            raise

    def to_langchain_tool(self) -> StructuredTool:
        """Convert this MCP tool to a LangChain StructuredTool."""
        # Create Pydantic model for input validation
        args_schema = self._create_pydantic_model()

        # Create LangChain StructuredTool
        return StructuredTool(
            name=self.tool_info.name,
            description=self.tool_info.description
            or f"MCP tool: {self.tool_info.name}",
            func=self._sync_func,
            coroutine=self._async_func,
            args_schema=args_schema,
        )


def create_langchain_tools_from_mcp(
    mcp_servers: Dict[str, Dict[str, Any]], client_id: str = "default"
) -> List[StructuredTool]:
    """
    Create LangChain-compatible tools from MCP server configurations.

    This is the main entry point for MCP integration. Returns LangChain
    StructuredTool instances that can be used with any LangChain agent.

    The MCP client connection is kept alive in a global registry to ensure
    tools can be called multiple times. Use cleanup_mcp_clients() to close
    connections when done.

    Args:
        mcp_servers: Dictionary mapping server names to configurations.
                    Example:
                    {
                        "math": {
                            "command": "python",
                            "args": ["math_server.py"]
                        },
                        "weather": {
                            "url": "http://localhost:8000/mcp",
                            "headers": {"Authorization": "Bearer token"}
                        }
                    }
        client_id: Identifier for this client instance (default: "default")

    Returns:
        List of LangChain StructuredTool instances
    """
    if not mcp_servers:
        return []

    try:
        with _client_lock:
            # Check if client already exists
            if client_id in _mcp_clients:
                logger.info(f"Reusing existing MCP client '{client_id}'")
                client = _mcp_clients[client_id]
            else:
                # Create and connect MCP client in a background thread with persistent event loop
                logger.info(f"Connecting to {len(mcp_servers)} MCP servers")

                # Containers for thread communication
                client_container: List[Optional[StandaloneMCPClient]] = [None]
                exception_container: List[Optional[Exception]] = [None]
                ready_event = threading.Event()

                def run_event_loop():
                    """Run event loop in background thread."""
                    loop = None
                    try:
                        # Create new event loop for this thread
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        _mcp_loops[client_id] = loop

                        # Connect to MCP servers
                        async def connect():
                            client = await create_mcp_client_from_config(mcp_servers)
                            client_container[0] = client
                            _mcp_clients[client_id] = client
                            ready_event.set()

                            # Keep loop running
                            while client_id in _mcp_clients:
                                await asyncio.sleep(0.1)

                        loop.run_until_complete(connect())

                    except Exception as e:
                        exception_container[0] = e
                        ready_event.set()
                    finally:
                        if loop:
                            loop.close()

                # Start background thread
                thread = threading.Thread(target=run_event_loop, daemon=True)
                thread.start()
                _mcp_threads[client_id] = thread

                # Wait for connection
                ready_event.wait(timeout=30)

                if exception_container[0]:
                    raise exception_container[0]

                client = client_container[0]
                if client is None:
                    raise RuntimeError("Failed to create MCP client")

        # Get all tools
        tool_infos = client.get_tool_list()
        logger.info(f"Found {len(tool_infos)} tools across all servers")

        # Create LangChain tools
        langchain_tools = []
        for tool_key, tool_info in client.tools.items():
            wrapper = MCPToolWrapper(tool_info, client, tool_key)
            langchain_tool = wrapper.to_langchain_tool()
            langchain_tools.append(langchain_tool)
            logger.debug(
                f"Created LangChain tool: {tool_info.name} from {tool_info.server_name}"
            )

        logger.info(f"Created {len(langchain_tools)} LangChain tools from MCP servers")
        return langchain_tools

    except Exception as e:
        logger.error(f"Failed to create LangChain tools from MCP servers: {e}")
        return []


def cleanup_mcp_clients(client_id: Optional[str] = None) -> None:
    """
    Clean up MCP client connections and stop background threads.

    Args:
        client_id: Specific client to cleanup, or None to cleanup all
    """
    with _client_lock:
        clients_to_cleanup = [client_id] if client_id else list(_mcp_clients.keys())

        for cid in clients_to_cleanup:
            if cid in _mcp_clients:
                try:
                    # Remove client to signal thread to stop
                    _mcp_clients.pop(cid)

                    # Stop event loop if it exists
                    if cid in _mcp_loops:
                        loop = _mcp_loops.pop(cid)
                        if loop.is_running():
                            loop.call_soon_threadsafe(loop.stop)

                    # Wait for thread to finish
                    if cid in _mcp_threads:
                        thread = _mcp_threads.pop(cid)
                        thread.join(timeout=5)

                    logger.info(f"Cleaned up MCP client '{cid}'")

                except Exception as e:
                    logger.error(f"Error cleaning up client '{cid}': {e}")


def validate_mcp_config(config: Dict[str, Any]) -> bool:
    """
    Validate an MCP server configuration.

    Args:
        config: Server configuration dictionary

    Returns:
        True if valid, False otherwise
    """
    # Must have either url (HTTP) or command (stdio)
    has_url = "url" in config
    has_command = "command" in config

    if not (has_url or has_command):
        logger.error("Config must have either 'url' or 'command'")
        return False

    if has_url and has_command:
        logger.warning("Config has both 'url' and 'command', will use 'url'")

    # Validate stdio config
    if has_command and not isinstance(config.get("args", []), list):
        logger.error("'args' must be a list")
        return False

    # Validate HTTP config
    if has_url and not isinstance(config["url"], str):
        logger.error("'url' must be a string")
        return False

    return True


def validate_mcp_configs(mcp_servers: Dict[str, Dict[str, Any]]) -> bool:
    """
    Validate all MCP server configurations.

    Args:
        mcp_servers: Dictionary of server configurations

    Returns:
        True if all valid, False otherwise
    """
    if not mcp_servers:
        return True

    all_valid = True
    for server_name, config in mcp_servers.items():
        if not validate_mcp_config(config):
            logger.error(f"Invalid config for server '{server_name}'")
            all_valid = False

    return all_valid


# Helper functions for creating configs (for convenience)


def create_stdio_config(
    command: str,
    args: List[str],
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Create a stdio transport configuration.

    Args:
        command: Command to run (e.g., "python", "node")
        args: Arguments for the command
        env: Optional environment variables

    Returns:
        Configuration dictionary
    """
    config = {
        "command": command,
        "args": args,
    }
    if env:
        config["env"] = env
    return config


def create_http_config(
    url: str,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Create an HTTP transport configuration.

    Args:
        url: HTTP URL of the MCP server
        headers: Optional HTTP headers (e.g., for authentication)

    Returns:
        Configuration dictionary
    """
    config: Dict[str, Any] = {"url": url}
    if headers:
        config["headers"] = headers
    return config
