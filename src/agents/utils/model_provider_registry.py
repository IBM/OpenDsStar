"""
Model Provider Registry - Manages custom model providers.

This module provides a registry for registering and managing custom model
providers that extend ModelBuilder's capabilities.
"""

from __future__ import annotations

import logging
from typing import Optional

from agents.utils.model_provider import ModelProvider

logger = logging.getLogger(__name__)


class ModelProviderRegistry:
    """
    Registry for managing custom model providers.

    Providers are checked in registration order, so register more specific
    providers before more general ones.
    """

    def __init__(self):
        """Initialize empty provider registry."""
        self._providers: list[ModelProvider] = []

    def register(self, provider: ModelProvider) -> None:
        """
        Register a custom model provider.

        Args:
            provider: ModelProvider instance to register

        Raises:
            ValueError: If provider is already registered
        """
        if provider in self._providers:
            raise ValueError(f"Provider {provider.name} is already registered")

        self._providers.append(provider)
        logger.info("Registered model provider: %s", provider.name)

    def unregister(self, provider: ModelProvider) -> None:
        """
        Unregister a model provider.

        Args:
            provider: ModelProvider instance to unregister
        """
        if provider in self._providers:
            self._providers.remove(provider)
            logger.info("Unregistered model provider: %s", provider.name)

    def get_provider(self, model_string: str) -> Optional[ModelProvider]:
        """
        Find a provider that can handle the given model string.

        Args:
            model_string: Model identifier string

        Returns:
            First provider that can handle the model, or None if no match
        """
        for provider in self._providers:
            if provider.can_handle(model_string):
                logger.debug(
                    "Provider %s will handle model: %s",
                    provider.name,
                    model_string,
                )
                return provider
        return None

    def list_providers(self) -> list[str]:
        """
        List all registered provider names.

        Returns:
            List of provider names in registration order
        """
        return [provider.name for provider in self._providers]

    def clear(self) -> None:
        """Clear all registered providers."""
        self._providers.clear()
        logger.debug("Cleared all model providers")


# Global registry instance
_global_registry = ModelProviderRegistry()


def get_global_registry() -> ModelProviderRegistry:
    """
    Get the global model provider registry.

    Returns:
        Global ModelProviderRegistry instance
    """
    return _global_registry
