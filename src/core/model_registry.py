"""
Model Registry - Centralized model identifiers.

This module provides a registry of available models with their identifiers.
All model references should use this registry instead of hardcoded strings.
"""


class ModelRegistry:
    """Registry of available models with their identifiers."""

    # LLM Models
    WX_MISTRAL_MEDIUM = "watsonx/mistralai/mistral-medium-2505"
    WX_MISTRAL_SMALL = "watsonx/mistralai/mistral-small-3-1-24b-instruct-2503"
    WX_LLAMA_MAVERICK = "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    WX_GPT_OSS_120B = "watsonx/gpt-oss-120b"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    # CLAUDE_3_SONNET = "anthropic/claude-3-sonnet-20240229"
    TPM_GEMINI_2_0_FLASH = "tpm/GCP/gemini-2.0-flash"
    TPM_GEMINI_2_5_FLASH = "tpm/GCP/gemini-2.5-flash"
    TPM_GEMINI_2_5_PRO = "tpm/GCP/gemini-2.5-pro"
    # TPM_GEMINI_3_0_PRO_PREVIEW = "tpm/gcp/gemini-3-pro-preview"
    # TPM_CLAUDE_3_7_SONNET = "tpm/GCP/claude-3-7-sonnet"
    TPM_CLAUDE_4_SONNET = "tpm/GCP/claude-4-sonnet"

    # Embedding Models
    GRANITE_EMBEDDING = "ibm-granite/granite-embedding-english-r2"

    @classmethod
    def get_model_id(cls, model_alias: str) -> str:
        """
        Convert a short model alias to its full model identifier.

        Args:
            model_alias: Short model name (e.g., "wx_mistral_medium", "gpt_4o")

        Returns:
            Full model identifier (e.g., "watsonx/mistralai/mistral-medium-2505")
            If not found in registry, returns the input alias unchanged

        Examples:
            >>> ModelRegistry.get_model_id("wx_mistral_medium")
            'watsonx/mistralai/mistral-medium-2505'
            >>> ModelRegistry.get_model_id("gpt_4o")
            'gpt-4o'
            >>> ModelRegistry.get_model_id("watsonx/mistralai/mistral-medium-2505")
            'watsonx/mistralai/mistral-medium-2505'  # Already full ID, returns as-is
        """
        # Normalize the alias to uppercase for comparison
        alias_upper = model_alias.upper()

        # Check all class attributes for matching alias
        for attr_name in dir(cls):
            # Skip private/magic attributes and methods
            if attr_name.startswith("_") or callable(getattr(cls, attr_name)):
                continue

            # Check if attribute name matches the alias
            if attr_name == alias_upper:
                attr_value = getattr(cls, attr_name)
                if isinstance(attr_value, str):
                    return attr_value

        # If not found, check if it's already a full model ID
        # (i.e., if it matches any of the registered values)
        for attr_name in dir(cls):
            if attr_name.startswith("_") or callable(getattr(cls, attr_name)):
                continue

            attr_value = getattr(cls, attr_name)
            if isinstance(attr_value, str) and attr_value == model_alias:
                return model_alias  # Already a full ID

        # If still not found, return the alias unchanged
        # This allows for custom model IDs not in the registry
        return model_alias

    @classmethod
    def get_model_name(cls, model_string: str) -> str:
        """
        Convert a model string to its short name.

        Args:
            model_string: Full model identifier (e.g., "watsonx/mistralai/mistral-medium-2505")

        Returns:
            Short model name (e.g., "wx_mistral_medium")
            If not found in registry, returns a sanitized version of the model string

        Examples:
            >>> ModelRegistry.get_model_name("watsonx/mistralai/mistral-medium-2505")
            'wx_mistral_medium'
            >>> ModelRegistry.get_model_name("gpt-4o")
            'gpt_4o'
        """
        # Check all class attributes for matching model string
        for attr_name in dir(cls):
            # Skip private/magic attributes and methods
            if attr_name.startswith("_") or callable(getattr(cls, attr_name)):
                continue

            attr_value = getattr(cls, attr_name)
            # Check if it's a string constant that matches
            if isinstance(attr_value, str) and attr_value == model_string:
                return attr_name.lower()

        # If not found, create a sanitized name from the model string
        # Replace special characters with underscores and convert to lowercase
        sanitized = model_string.replace("/", "_").replace("-", "_").replace(".", "_")
        return sanitized.lower()
