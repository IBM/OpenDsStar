"""
Tests for standalone MCP integration with DS Star agent.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from tools.mcp_client_standalone import MCPToolInfo
from tools.mcp_integration_standalone import (
    MCPToolWrapper,
    create_http_config,
    create_stdio_config,
    validate_mcp_config,
    validate_mcp_configs,
)


class TestConfigCreation:
    """Test configuration helper functions."""

    def test_create_stdio_config(self):
        """Test creating stdio configuration."""
        config = create_stdio_config(
            command="python",
            args=["server.py"],
            env={"VAR": "value"},
        )

        assert config["command"] == "python"
        assert config["args"] == ["server.py"]
        assert config["env"] == {"VAR": "value"}

    def test_create_stdio_config_no_env(self):
        """Test creating stdio configuration without env."""
        config = create_stdio_config(
            command="node",
            args=["server.js"],
        )

        assert config["command"] == "node"
        assert config["args"] == ["server.js"]
        assert "env" not in config

    def test_create_http_config(self):
        """Test creating HTTP configuration."""
        config = create_http_config(
            url="http://localhost:8000/mcp",
            headers={"Authorization": "Bearer token"},
        )

        assert config["url"] == "http://localhost:8000/mcp"
        assert config["headers"] == {"Authorization": "Bearer token"}

    def test_create_http_config_no_headers(self):
        """Test creating HTTP configuration without headers."""
        config = create_http_config(
            url="http://localhost:8000/mcp",
        )

        assert config["url"] == "http://localhost:8000/mcp"
        assert "headers" not in config


class TestConfigValidation:
    """Test configuration validation."""

    def test_validate_stdio_config(self):
        """Test validating stdio configuration."""
        config = {
            "command": "python",
            "args": ["server.py"],
        }
        assert validate_mcp_config(config) is True

    def test_validate_http_config(self):
        """Test validating HTTP configuration."""
        config = {
            "url": "http://localhost:8000/mcp",
        }
        assert validate_mcp_config(config) is True

    def test_validate_config_missing_both(self):
        """Test validation fails when both url and command are missing."""
        config = {
            "args": ["server.py"],
        }
        assert validate_mcp_config(config) is False

    def test_validate_config_invalid_args(self):
        """Test validation fails with invalid args type."""
        config = {
            "command": "python",
            "args": "server.py",  # Should be list
        }
        assert validate_mcp_config(config) is False

    def test_validate_config_invalid_url(self):
        """Test validation fails with invalid url type."""
        config = {
            "url": 123,  # Should be string
        }
        assert validate_mcp_config(config) is False

    def test_validate_multiple_configs(self):
        """Test validating multiple configurations."""
        configs = {
            "math": {"command": "python", "args": ["math.py"]},
            "weather": {"url": "http://localhost:8000/mcp"},
        }
        assert validate_mcp_configs(configs) is True

    def test_validate_multiple_configs_with_invalid(self):
        """Test validation fails when one config is invalid."""
        configs = {
            "math": {"command": "python", "args": ["math.py"]},
            "invalid": {"args": ["server.py"]},  # Missing command/url
        }
        assert validate_mcp_configs(configs) is False


class TestMCPToolWrapper:
    """Test MCP tool wrapper for LangChain."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock MCP client."""
        client = Mock()
        client.call_tool = AsyncMock(return_value="test result")
        return client

    @pytest.fixture
    def tool_info(self):
        """Create test tool info."""
        return MCPToolInfo(
            name="test_tool",
            description="A test tool",
            input_schema={
                "type": "object",
                "properties": {
                    "arg1": {"type": "string", "description": "First argument"},
                    "arg2": {"type": "integer", "description": "Second argument"},
                },
                "required": ["arg1"],
            },
            server_name="test_server",
        )

    @pytest.fixture
    def mcp_wrapper(self, tool_info, mock_client):
        """Create an MCP tool wrapper for testing."""
        return MCPToolWrapper(
            tool_info=tool_info,
            client=mock_client,
            tool_key="test_server_test_tool",
        )

    def test_wrapper_initialization(self, mcp_wrapper):
        """Test wrapper is initialized correctly."""
        assert mcp_wrapper.tool_info.name == "test_tool"
        assert mcp_wrapper.tool_info.description == "A test tool"
        assert mcp_wrapper.tool_key == "test_server_test_tool"

    def test_to_langchain_tool(self, mcp_wrapper):
        """Test conversion to LangChain StructuredTool."""
        langchain_tool = mcp_wrapper.to_langchain_tool()

        assert langchain_tool.name == "test_tool"
        assert langchain_tool.description == "A test tool"
        assert callable(langchain_tool.func)
        assert langchain_tool.coroutine is not None
        assert langchain_tool.args_schema is not None

    def test_async_func(self, mcp_wrapper, mock_client):
        """Test async function via sync wrapper."""
        # Test the sync function which internally calls the async function
        # This is how the tool will actually be used in practice
        import asyncio

        # Create a simple event loop to test the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                mcp_wrapper._async_func(arg1="value1", arg2=42)
            )

            assert result == "test result"
            mock_client.call_tool.assert_called_once_with(
                "test_server_test_tool", {"arg1": "value1", "arg2": 42}
            )
        finally:
            loop.close()


class TestIntegration:
    """Integration tests for MCP with LangChain."""

    def test_import_standalone_modules(self):
        """Test that standalone modules can be imported."""
        from tools import mcp_client_standalone, mcp_integration_standalone

        assert hasattr(mcp_client_standalone, "StandaloneMCPClient")
        assert hasattr(mcp_integration_standalone, "MCPToolWrapper")
        assert hasattr(mcp_integration_standalone, "create_langchain_tools_from_mcp")

    def test_config_helpers_available(self):
        """Test that config helper functions are available."""
        from tools.mcp_integration_standalone import (
            create_http_config,
            create_stdio_config,
        )

        assert callable(create_stdio_config)
        assert callable(create_http_config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
