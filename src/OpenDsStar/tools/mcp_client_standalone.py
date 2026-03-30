"""
Standalone MCP Client for DS Star Agent.

This module provides a standalone MCP client that doesn't rely on langflow,
using the official MCP Python SDK directly.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MCPToolInfo:
    """Information about an MCP tool."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str


class StandaloneMCPClient:
    """
    Standalone MCP client that connects to MCP servers and provides tools.

    Supports both stdio and HTTP transports using the official MCP SDK.
    """

    def __init__(self):
        """Initialize the MCP client."""
        self.sessions: Dict[str, Any] = {}
        self.tools: Dict[str, MCPToolInfo] = {}

    async def connect_stdio(
        self,
        server_name: str,
        command: str,
        args: List[str],
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Connect to an MCP server using stdio transport.

        Args:
            server_name: Name to identify this server
            command: Command to run (e.g., "python", "node")
            args: Arguments for the command
            env: Optional environment variables
        """
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            logger.info(f"Connecting to MCP server '{server_name}' via stdio")

            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=env or {},
            )

            # Create stdio client
            stdio = stdio_client(server_params)
            read, write = await stdio.__aenter__()

            # Create session
            session = ClientSession(read, write)
            await session.__aenter__()

            # Initialize the session
            await session.initialize()

            # Store session
            self.sessions[server_name] = {
                "session": session,
                "stdio": stdio,
                "transport": "stdio",
            }

            # List and store tools
            await self._load_tools(server_name, session)

            logger.info(
                f"Connected to '{server_name}' with {len([t for t in self.tools.values() if t.server_name == server_name])} tools"
            )

        except Exception as e:
            logger.error(f"Failed to connect to stdio server '{server_name}': {e}")
            raise

    async def connect_http(
        self,
        server_name: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Connect to an MCP server using HTTP transport.

        Args:
            server_name: Name to identify this server
            url: HTTP URL of the server
            headers: Optional HTTP headers
        """
        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client

            logger.info(f"Connecting to MCP server '{server_name}' via HTTP")

            # Create SSE client
            sse = sse_client(url, headers=headers or {})
            read, write = await sse.__aenter__()

            # Create session
            session = ClientSession(read, write)
            await session.__aenter__()

            # Initialize the session
            await session.initialize()

            # Store session
            self.sessions[server_name] = {
                "session": session,
                "sse": sse,
                "transport": "http",
            }

            # List and store tools
            await self._load_tools(server_name, session)

            logger.info(
                f"Connected to '{server_name}' with {len([t for t in self.tools.values() if t.server_name == server_name])} tools"
            )

        except Exception as e:
            logger.error(f"Failed to connect to HTTP server '{server_name}': {e}")
            raise

    async def _load_tools(self, server_name: str, session: Any) -> None:
        """Load tools from a connected session."""
        try:
            # List available tools
            tools_result = await session.list_tools()

            for tool in tools_result.tools:
                tool_key = f"{server_name}_{tool.name}"
                self.tools[tool_key] = MCPToolInfo(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema,
                    server_name=server_name,
                )
                logger.debug(f"Loaded tool: {tool_key}")

        except Exception as e:
            logger.error(f"Failed to load tools from '{server_name}': {e}")
            raise

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Any:
        """
        Call an MCP tool.

        Args:
            tool_name: Name of the tool (format: "server_name_tool_name")
            arguments: Arguments to pass to the tool

        Returns:
            Tool result
        """
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        tool_info = self.tools[tool_name]
        server_name = tool_info.server_name

        if server_name not in self.sessions:
            raise ValueError(f"Server '{server_name}' not connected")

        session = self.sessions[server_name]["session"]

        try:
            # Call the tool
            result = await session.call_tool(tool_info.name, arguments)

            # Extract content from result
            if hasattr(result, "content") and result.content:
                # Return first content item's text
                if len(result.content) > 0:
                    first_content = result.content[0]
                    if hasattr(first_content, "text"):
                        return first_content.text
                    return str(first_content)

            return str(result)

        except Exception as e:
            logger.error(f"Failed to call tool '{tool_name}': {e}")
            raise

    async def disconnect(self, server_name: Optional[str] = None) -> None:
        """
        Disconnect from MCP server(s).

        Args:
            server_name: Specific server to disconnect, or None for all
        """
        servers_to_disconnect = (
            [server_name] if server_name else list(self.sessions.keys())
        )

        for name in servers_to_disconnect:
            if name in self.sessions:
                try:
                    session_info = self.sessions[name]
                    session = session_info["session"]

                    # Exit session
                    await session.__aexit__(None, None, None)

                    # Exit transport
                    if "stdio" in session_info:
                        await session_info["stdio"].__aexit__(None, None, None)
                    elif "sse" in session_info:
                        await session_info["sse"].__aexit__(None, None, None)

                    del self.sessions[name]
                    logger.info(f"Disconnected from '{name}'")

                except Exception as e:
                    logger.error(f"Error disconnecting from '{name}': {e}")

    def get_tool_list(self) -> List[MCPToolInfo]:
        """Get list of all available tools."""
        return list(self.tools.values())

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


async def create_mcp_client_from_config(
    mcp_servers: Dict[str, Dict[str, Any]],
) -> StandaloneMCPClient:
    """
    Create and connect an MCP client from server configurations.

    Args:
        mcp_servers: Dictionary mapping server names to configurations

    Returns:
        Connected MCP client
    """
    client = StandaloneMCPClient()

    for server_name, config in mcp_servers.items():
        try:
            if "url" in config:
                # HTTP transport
                await client.connect_http(
                    server_name=server_name,
                    url=config["url"],
                    headers=config.get("headers"),
                )
            elif "command" in config:
                # stdio transport
                await client.connect_stdio(
                    server_name=server_name,
                    command=config["command"],
                    args=config.get("args", []),
                    env=config.get("env"),
                )
            else:
                logger.warning(
                    f"Invalid config for server '{server_name}': missing 'url' or 'command'"
                )

        except Exception as e:
            logger.error(f"Failed to connect to server '{server_name}': {e}")
            continue

    return client
