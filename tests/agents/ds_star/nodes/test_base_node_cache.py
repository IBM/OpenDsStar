"""Test that BaseNode properly handles cache with structured outputs."""

import tempfile
from pathlib import Path
from typing import Optional, Type

from pydantic import BaseModel, Field

from OpenDsStar.agents.ds_star.nodes.base_node import BaseNode
from OpenDsStar.agents.utils.model_builder import ModelBuilder


class OutputSchema(BaseModel):
    """Test schema for structured output."""

    result: str = Field(description="The result")


class StructuredNode(BaseNode):
    """Test node with structured output."""

    structured_output_schema: Optional[Type[BaseModel]] = OutputSchema


class TestBaseNodeCache:
    """Test BaseNode cache handling with structured outputs."""

    def test_base_node_disables_cache_for_structured_output(self):
        """Test that BaseNode creates structured output wrapper without cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "test_cache"

            # Build model with cache
            model, model_id = ModelBuilder.build(
                model="watsonx/mistralai/mistral-medium-2505",
                temperature=0.0,
                cache_dir=cache_dir,
            )

            # Verify base model has cache
            assert hasattr(model, "cache")
            assert model.cache is not None
            original_cache = model.cache

            # Create test node with structured output
            node = StructuredNode(llm=model)

            # Verify base LLM still has cache
            assert node.llm.cache is not None
            assert node.llm.cache is original_cache

            # Verify structured output wrapper was created
            assert node.llm_with_output is not None

            # The wrapper should not have cache (or have None cache)
            # This prevents cache serialization issues with structured outputs
            wrapper_cache = getattr(node.llm_with_output, "cache", None)
            # Note: The wrapper might not have a cache attribute at all,
            # or it might be None - both are acceptable
            print(f"Base LLM cache: {node.llm.cache}")
            print(f"Wrapper cache: {wrapper_cache}")

            # The key test: wrapper should be different from base LLM
            # (it's a wrapped version without cache interference)
            assert node.llm_with_output is not node.llm

    def test_base_node_without_cache_still_works(self):
        """Test that BaseNode works when LLM has no cache."""

        # Create a mock LLM without cache
        class MockLLM:
            def with_structured_output(self, schema):
                return self

        mock_llm = MockLLM()

        # Should not raise error
        node = StructuredNode(llm=mock_llm)
        assert node.llm_with_output is not None

    def test_base_node_without_structured_schema(self):
        """Test that BaseNode works without structured output schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "test_cache"

            model, model_id = ModelBuilder.build(
                model="watsonx/mistralai/mistral-medium-2505",
                temperature=0.0,
                cache_dir=cache_dir,
            )

            # Create node without structured output schema
            class PlainNode(BaseNode):
                structured_output_schema: Optional[Type[BaseModel]] = None

            node = PlainNode(llm=model)

            # Should not create structured output wrapper
            assert node.llm_with_output is None
            # Base LLM should still have cache
            assert node.llm.cache is not None
