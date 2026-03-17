"""
Utility for setting up model providers in experiments.

This module provides convenience functions for registering custom model providers
that need to be available to ModelBuilder.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def setup_custom_api_provider() -> None:
    """
    Register Custom API provider with ModelBuilder.

    This function registers the Custom API provider, enabling ModelBuilder to
    handle models with custom prefixes configured via environment variables.
    This should be called once at the start of experiment main functions.

    Configuration via environment variables:
        - CUSTOM_API_PREFIX: Model prefix (default: "custom")
        - CUSTOM_API_BASE: API base URL (required)
        - CUSTOM_API_KEY: API key (required)
        - CUSTOM_API_PROVIDER: LiteLLM provider type (default: "openai")
        - CUSTOM_API_NAME: Display name (default: "CustomAPI")

    This function is idempotent - calling it multiple times is safe.

    Example:
        >>> from experiments.utils.model_provider_setup import setup_custom_api_provider
        >>> setup_custom_api_provider()
        >>> # Now ModelBuilder can handle custom API models
    """
    from agents.utils.providers.custom_api_provider import CustomAPIProvider

    CustomAPIProvider.register()
    logger.debug("Custom API provider setup complete")
