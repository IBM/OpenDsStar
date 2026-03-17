"""Simple provider registry using dict of transformers."""

from __future__ import annotations

import logging
from typing import Callable

from agents.utils.model_config import ModelConfig

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Registry for model string transformers."""

    _transformers: dict[str, Callable[[ModelConfig], ModelConfig]] = {}

    @classmethod
    def register(
        cls, prefix: str, transformer: Callable[[ModelConfig], ModelConfig]
    ) -> None:
        """
        Register a prefix transformer.

        Args:
            prefix: Model string prefix to match (e.g., "tpm/")
            transformer: Function that transforms ModelConfig
        """
        cls._transformers[prefix] = transformer
        logger.info(f"Registered provider for prefix: {prefix}")

    @classmethod
    def apply(cls, config: ModelConfig) -> ModelConfig:
        """
        Apply transformer if prefix matches.

        Args:
            config: Model configuration to transform

        Returns:
            Transformed ModelConfig (or original if no match)
        """
        for prefix, transformer in cls._transformers.items():
            if config.model_id.startswith(prefix):
                logger.debug(f"Applying {prefix} transformer to {config.model_id}")
                return transformer(config)
        return config

    @classmethod
    def clear(cls) -> None:
        """Clear all providers (useful for testing)."""
        cls._transformers.clear()
        logger.debug("Cleared all providers")


# Note: TPM provider has been moved to the new CustomAPIProvider system.
# See src/agents/utils/providers/custom_api_provider.py
# This old transformer-based registry is kept for backward compatibility
# but no longer auto-registers TPM.
