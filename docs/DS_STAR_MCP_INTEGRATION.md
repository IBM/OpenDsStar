# DS Star MCP Integration Guide

This guide explains how to integrate MCP (Model Context Protocol) tools with the DS Star agent and other LangChain-based agents.

## Overview

The MCP integration converts MCP tools into **LangChain StructuredTools**, making them compatible with:
- DS Star agent
- React agent (LangChain)
- Any other LangChain-based agent

## Quick Start

### 1. Install MCP Dependencies

```bash
pip install -e ".[mcp]"
```

### 2. Basic Usage Pattern

```python
from tools.mcp_integration_standalone import (
    create_stdio_config,
    create_langchain_tools_from_mcp,
)
from agents.ds_star.open_ds_star_agent import OpenDsStarAgent

# 1. Configure MCP servers
mcp_servers = {
    "math": create_stdio_config(
        command="python",
        args=["examples/mcp_servers/math_server.py"]
    )
}

# 2. Convert to LangChain tools
mcp_tools = create_langchain_tools_from_mcp(mcp_servers)

# 3. Use with any LangChain agent
agent = OpenDsStarAgent(
    model="ollama/phi4:latest",
    tools=mcp_tools
)

result = agent.invoke("What is 15 * 23?")
```

## Configuration

### Local MCP Server (stdio)

For MCP servers running as local processes:

```python
from tools.mcp_integration_standalone import create_stdio_config

config = create_stdio_config(
    command="python",           # Command to run
    args=["server.py"],        # Arguments
    env={"VAR": "value"}       # Optional environment variables
)
```

### Remote MCP Server (HTTP)

For MCP servers accessible via HTTP:

```python
from tools.mcp_integration_standalone import create_http_config

config = create_http_config(
    url="http://localhost:8000/mcp",
    headers={"Authorization": "Bearer token"}  # Optional headers
)
```

### Multiple Servers

You can connect to multiple MCP servers at once:

```python
mcp_servers = {
    "math": create_stdio_config(
        command="python",
        args=["math_server.py"]
    ),
    "weather": create_http_config(
        url="http://localhost:8000/mcp"
    ),
}

# All tools from all servers will be available
mcp_tools = create_langchain_tools_from_mcp(mcp_servers)
```

## Using with Different Agents

### DS Star Agent

```python
from agents.ds_star.open_ds_star_agent import OpenDsStarAgent

agent = OpenDsStarAgent(
    model="ollama/phi4:latest",
    tools=mcp_tools
)
```

### React Agent (LangChain)

```python
from agents.react_langchain.react_agent_langchain import ReactAgentLangchain

agent = ReactAgentLangchain(
    model="ollama/phi4:latest",
    tools=mcp_tools
)
```

### Any LangChain Agent

Since MCP tools are converted to LangChain StructuredTools, they work with any agent that accepts LangChain tools:

```python
from langchain.agents import create_react_agent

agent = create_react_agent(
    llm=llm,
    tools=mcp_tools,
    prompt=prompt
)
```

## API Reference

### `create_langchain_tools_from_mcp(mcp_servers)`

Main entry point for MCP integration.

**Parameters:**
- `mcp_servers` (Dict[str, Dict[str, Any]]): Dictionary mapping server names to configurations

**Returns:**
- `List[StructuredTool]`: List of LangChain StructuredTool instances

**Example:**
```python
mcp_tools = create_langchain_tools_from_mcp({
    "math": create_stdio_config(command="python", args=["math.py"])
})
```

### `create_stdio_config(command, args, env=None)`

Create configuration for local MCP server.

**Parameters:**
- `command` (str): Command to execute
- `args` (List[str]): Command arguments
- `env` (Optional[Dict[str, str]]): Environment variables

**Returns:**
- `Dict[str, Any]`: Server configuration

### `create_http_config(url, headers=None)`

Create configuration for remote MCP server.

**Parameters:**
- `url` (str): HTTP endpoint URL
- `headers` (Optional[Dict[str, str]]): HTTP headers

**Returns:**
- `Dict[str, Any]`: Server configuration

## How It Works

1. **MCP Client**: Connects to MCP servers (stdio or HTTP)
2. **Tool Discovery**: Discovers available tools from each server
3. **Wrapper Creation**: Wraps each MCP tool in `MCPToolWrapper`
4. **LangChain Conversion**: Converts to LangChain `StructuredTool`
5. **Agent Usage**: Tools can be used with any LangChain agent

### Async Support

MCP tools are inherently async. The wrapper provides both sync and async interfaces:

- **Sync interface** (`func`): Runs async code in a thread
- **Async interface** (`coroutine`): Native async execution

LangChain agents automatically use the appropriate interface.

## Examples

See [`examples/ds_star_mcp_example.py`](../examples/ds_star_mcp_example.py) for complete examples:

1. Local MCP server (stdio)
2. Remote MCP server (HTTP)
3. Using same tools with multiple agents

## Troubleshooting

### Event Loop Conflicts

If you encounter event loop errors, the wrapper handles this automatically by running async code in a separate thread.

### Tool Not Found

Ensure your MCP server is running and accessible:

```python
# For stdio servers, check the command and args
# For HTTP servers, verify the URL is accessible
```

### Import Errors

Make sure MCP dependencies are installed:

```bash
pip install -e ".[mcp]"
```

## Creating MCP Servers

To create your own MCP server, see the official MCP documentation:
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [FastMCP](https://github.com/jlowin/fastmcp)

Example servers are in [`examples/mcp_servers/`](../examples/mcp_servers/).
