"""
Model Providers - External model building implementations.

This package contains custom model providers that can be registered
with ModelBuilder to extend its capabilities.
"""

from agents.utils.providers.custom_api_provider import CustomAPIProvider

__all__ = ["CustomAPIProvider"]
