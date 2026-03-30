"""Test that cache_dir flows through AgentFactory to OpenDsStarAgent."""

import tempfile
from pathlib import Path


def test_cache_dir_flows_to_agent():
    """Test that cache_dir is passed from AgentFactory to OpenDsStarAgent."""
    # Reset cache state
    from OpenDsStar.agents.utils.cache_manager import CacheManager

    CacheManager.clear()

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "test_cache"

        # Create a mock context with cache_dir
        from langchain_core.tools import tool

        from OpenDsStar.experiments.core.config import AgentConfig
        from OpenDsStar.experiments.core.context import PipelineConfig, PipelineContext
        from OpenDsStar.experiments.implementations.agent_factory import (
            AgentFactory,
            AgentType,
        )

        @tool
        def dummy_tool(query: str) -> str:
            """A dummy tool for testing."""
            return "dummy result"

        # Create config with cache_dir
        config = PipelineConfig(
            run_id="test_run",
            cache_dir=cache_dir,
            agent_config=AgentConfig(
                agent_type=AgentType.DS_STAR,
                model="watsonx/mistralai/mistral-medium-2505",
                max_steps=2,
                temperature=0.0,
                max_debug_tries=5,
                code_timeout=30,
                code_mode="stepwise",
                output_max_length=500,
                logs_max_length=20000,
            ),
        )

        ctx = PipelineContext(config=config)

        # Build agent using AgentFactory
        agent_builder = AgentFactory(agent_type=AgentType.DS_STAR)
        agent = agent_builder.build_agent(ctx, [dummy_tool])

        # Verify cache was configured
        model_cache_dir = (
            cache_dir / "litellm_cache_watsonx_mistralai_mistral-medium-2505"
        )
        assert (
            model_cache_dir.exists()
        ), f"Model cache directory should exist: {model_cache_dir}"

        cache_db = model_cache_dir / "text_cache.db"
        assert cache_db.exists(), f"Cache database should exist: {cache_db}"

        print(f"✓ Cache directory created: {cache_dir}")
        print(f"✓ Model cache directory: {model_cache_dir}")
        print(f"✓ Cache database: {cache_db}")
        print(f"✓ Agent created with cache_dir: {agent}")
        print(
            "\n✅ Test passed! cache_dir flows correctly from AgentFactory to OpenDsStarAgent."
        )


if __name__ == "__main__":
    test_cache_dir_flows_to_agent()
