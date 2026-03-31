"""
Test token tracking for all agent types.

Verifies that agents properly extract and report token usage.
"""

from unittest.mock import Mock, patch

from OpenDsStar.agents.codeact_smolagents.codeact_agent_smolagents import (
    CodeActAgentSmolagents,
)
from OpenDsStar.agents.react_langchain.react_agent_langchain import ReactAgentLangchain
from OpenDsStar.agents.react_smolagents.react_agent_smolagents import (
    ReactAgentSmolagents,
)


class TestTokenTracking:
    """Test token tracking across different agent implementations."""

    @patch("OpenDsStar.agents.codeact_smolagents.codeact_agent_smolagents.CodeAgent")
    def test_codeact_uses_monitor_for_tokens(self, mock_code_agent):
        """Test CodeActAgentSmolagents uses monitor for token tracking."""
        from smolagents import LiteLLMModel

        mock_model = LiteLLMModel(model_id="gpt-4o-mini")
        mock_agent_instance = Mock()
        mock_agent_instance.monitor = Mock()
        mock_agent_instance.monitor.get_total_token_counts = Mock(
            return_value=Mock(input_tokens=0, output_tokens=0)
        )
        mock_code_agent.return_value = mock_agent_instance

        agent = CodeActAgentSmolagents(model=mock_model, max_steps=3)

        # Verify agent has monitor
        assert hasattr(agent._agent, "monitor")
        assert hasattr(agent._agent.monitor, "get_total_token_counts")

        # Check initial token counts are zero
        token_usage = agent._agent.monitor.get_total_token_counts()
        assert token_usage.input_tokens == 0
        assert token_usage.output_tokens == 0

    @patch("OpenDsStar.agents.react_smolagents.react_agent_smolagents.ToolCallingAgent")
    def test_react_smolagents_uses_monitor_for_tokens(self, mock_tool_agent):
        """Test ReactAgentSmolagents uses monitor for token tracking."""
        from smolagents import LiteLLMModel

        mock_model = LiteLLMModel(model_id="gpt-4o-mini")
        mock_agent_instance = Mock()
        mock_agent_instance.monitor = Mock()
        mock_agent_instance.monitor.get_total_token_counts = Mock(
            return_value=Mock(input_tokens=0, output_tokens=0)
        )
        mock_tool_agent.return_value = mock_agent_instance

        agent = ReactAgentSmolagents(model=mock_model, max_steps=3)

        # Verify agent has monitor
        assert hasattr(agent._agent, "monitor")
        assert hasattr(agent._agent.monitor, "get_total_token_counts")

        # Check initial token counts are zero
        token_usage = agent._agent.monitor.get_total_token_counts()
        assert token_usage.input_tokens == 0
        assert token_usage.output_tokens == 0

    def test_react_langchain_extract_token_usage_usage_metadata(self):
        """Test ReactAgentLangchain extracts tokens from usage_metadata."""
        from langchain_core.language_models.fake_chat_models import FakeChatModel

        mock_model = FakeChatModel()
        agent = ReactAgentLangchain(model=mock_model, max_steps=3)

        # Mock message with usage_metadata
        msg1 = Mock()
        msg1.usage_metadata = {"input_tokens": 180, "output_tokens": 90}

        msg2 = Mock()
        msg2.usage_metadata = {"input_tokens": 220, "output_tokens": 110}

        messages = [msg1, msg2]

        input_tokens, output_tokens = agent._extract_token_usage(messages)

        assert input_tokens == 400  # 180 + 220
        assert output_tokens == 200  # 90 + 110

    def test_react_langchain_extract_token_usage_response_metadata(self):
        """Test ReactAgentLangchain extracts tokens from response_metadata."""
        from langchain_core.language_models.fake_chat_models import FakeChatModel

        mock_model = FakeChatModel()
        agent = ReactAgentLangchain(model=mock_model, max_steps=3)

        # Mock message with response_metadata
        msg = Mock()
        msg.usage_metadata = None
        msg.response_metadata = {
            "token_usage": {"prompt_tokens": 250, "completion_tokens": 125}
        }

        messages = [msg]

        input_tokens, output_tokens = agent._extract_token_usage(messages)

        assert input_tokens == 250
        assert output_tokens == 125

    def test_token_tracking_with_no_usage_data(self):
        """Test agents handle missing usage data gracefully."""
        from langchain_core.language_models.fake_chat_models import FakeChatModel

        mock_model = FakeChatModel()
        react_lc_agent = ReactAgentLangchain(model=mock_model, max_steps=3)

        # Empty messages
        assert react_lc_agent._extract_token_usage([]) == (0, 0)

        # Messages with no usage data
        msg_no_usage = Mock()
        msg_no_usage.usage_metadata = None
        msg_no_usage.response_metadata = None
        assert react_lc_agent._extract_token_usage([msg_no_usage]) == (0, 0)


# Made with Bob
