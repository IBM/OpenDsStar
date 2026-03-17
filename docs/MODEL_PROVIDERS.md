# Model Providers - Extensible Model Building

The ModelBuilder supports a modular provider system that allows you to register custom model builders for specific model types or prefixes.

## Overview

The provider system consists of:
- **ModelProvider**: Abstract interface for custom providers
- **ModelProviderRegistry**: Registry for managing providers
- **ModelBuilder**: Uses registered providers to build models

## Built-in Providers

### CustomAPIProvider (Generic Provider)

The `CustomAPIProvider` is a generic, configurable provider that reads all settings from environment variables. This makes it suitable for any custom API endpoint without hardcoding credentials.

**Configuration via Environment Variables:**

```bash
# .env file
CUSTOM_API_PREFIX=myapi              # Model prefix (default: "custom")
CUSTOM_API_BASE=https://api.example.com  # API base URL (required)
CUSTOM_API_KEY=sk-your-key           # Direct API key (optional)
CUSTOM_API_KEY_ENV=MY_API_KEY        # Or reference another env var (optional)
CUSTOM_API_PROVIDER=openai           # LiteLLM provider type (default: "openai")
CUSTOM_API_NAME=MyAPI                # Display name (default: "CustomAPI")
```

**Usage:**

```python
from pathlib import Path
from agents.utils.model_builder import ModelBuilder
from agents.utils.providers.custom_api_provider import CustomAPIProvider

# Register provider (reads config from env vars)
CustomAPIProvider.register()

# Now you can use your custom models
model, model_id = ModelBuilder.build(
    "myapi/my-model-name",
    temperature=0.0,
    cache_dir=Path("./cache")
)
```

**IBM TPM Example:**

For IBM users with access to Third-Party Models:

```bash
# .env file
CUSTOM_API_PREFIX=tpm
CUSTOM_API_BASE=https://ete-litellm.ai-models.vpc-int.res.ibm.com
CUSTOM_API_KEY_ENV=THIRD_PARTY_MODELS_API_KEY
THIRD_PARTY_MODELS_API_KEY=sk-your-ibm-key
CUSTOM_API_PROVIDER=openai
CUSTOM_API_NAME=TPM
```

Then use: `model, model_id = ModelBuilder.build("tpm/GCP/gemini-2.5-flash", ...)`

## Creating Custom Providers

### Step 1: Implement ModelProvider Interface

```python
from agents.utils.model_provider import ModelProvider
from langchain_core.caches import BaseCache
from langchain_core.language_models import BaseChatModel
from langchain_litellm import ChatLiteLLM

class MyCustomProvider(ModelProvider):
    """Provider for my custom model service."""

    @property
    def name(self) -> str:
        return "MyCustomProvider"

    def can_handle(self, model_string: str) -> bool:
        """Check if this provider handles the model."""
        return model_string.startswith("mycustom/")

    def build_model(
        self,
        model_string: str,
        temperature: float,
        cache: BaseCache,
    ) -> tuple[BaseChatModel, str]:
        """Build the model instance."""
        # Remove prefix
        model_id = model_string[9:]  # Remove "mycustom/"

        # Build your custom model
        model = ChatLiteLLM(
            model=model_id,
            temperature=temperature,
            api_base="https://my-api.example.com",
            api_key="my-api-key",
            cache=cache,
        )

        return model, model_id
```

### Step 2: Register Your Provider

```python
from agents.utils.model_provider_registry import get_global_registry

# Get the global registry
registry = get_global_registry()

# Register your provider
registry.register(MyCustomProvider())

# Now ModelBuilder can handle your models
model, model_id = ModelBuilder.build(
    "mycustom/my-model-name",
    cache_dir=Path("./cache")
)
```

### Step 3: (Optional) Auto-register on Import

Create a setup module that auto-registers your provider:

```python
# my_package/setup_providers.py
from agents.utils.model_provider_registry import get_global_registry
from my_package.my_provider import MyCustomProvider

def setup_my_providers():
    registry = get_global_registry()
    registry.register(MyCustomProvider())

# Auto-register on import
setup_my_providers()
```

Then import it in your main module:

```python
# my_package/__init__.py
import my_package.setup_providers  # noqa: F401
```

## Provider Registry API

### Registering Providers

```python
from agents.utils.model_provider_registry import get_global_registry

registry = get_global_registry()

# Register a provider
registry.register(MyProvider())

# Unregister a provider
registry.unregister(my_provider_instance)

# List all registered providers
provider_names = registry.list_providers()
print(provider_names)  # ['TPM', 'MyCustomProvider']

# Clear all providers (useful for testing)
registry.clear()
```

### Provider Resolution Order

Providers are checked in **registration order**. Register more specific providers before more general ones:

```python
# Register specific provider first
registry.register(SpecificProvider())  # Handles "specific/model"

# Then general provider
registry.register(GeneralProvider())   # Handles "general/*"
```

## Example: Custom Provider with Authentication

```python
import os
from agents.utils.model_provider import ModelProvider
from langchain_core.caches import BaseCache
from langchain_core.language_models import BaseChatModel
from langchain_litellm import ChatLiteLLM

class SecureAPIProvider(ModelProvider):
    """Provider with custom authentication."""

    def __init__(self, api_base: str, api_key_env: str = "MY_API_KEY"):
        self.api_base = api_base
        self.api_key_env = api_key_env

    @property
    def name(self) -> str:
        return "SecureAPI"

    def can_handle(self, model_string: str) -> bool:
        return model_string.startswith("secure/")

    def build_model(
        self,
        model_string: str,
        temperature: float,
        cache: BaseCache,
    ) -> tuple[BaseChatModel, str]:
        model_id = model_string[7:]  # Remove "secure/"

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(f"{self.api_key_env} not set")

        model = ChatLiteLLM(
            model=model_id,
            temperature=temperature,
            api_base=self.api_base,
            api_key=api_key,
            cache=cache,
        )

        return model, model_id

# Usage
registry = get_global_registry()
registry.register(SecureAPIProvider(
    api_base="https://secure-api.example.com",
    api_key_env="SECURE_API_KEY"
))

model, model_id = ModelBuilder.build(
    "secure/my-secure-model",
    cache_dir=Path("./cache")
)
```

## Testing Custom Providers

```python
import tempfile
from pathlib import Path
from agents.utils.model_builder import ModelBuilder
from agents.utils.model_provider_registry import get_global_registry

def test_my_provider():
    # Register provider
    registry = get_global_registry()
    registry.register(MyCustomProvider())

    try:
        # Test building model
        with tempfile.TemporaryDirectory() as tmpdir:
            model, model_id = ModelBuilder.build(
                "mycustom/test-model",
                cache_dir=Path(tmpdir)
            )

            assert model is not None
            assert model_id == "test-model"
    finally:
        # Clean up
        registry.clear()
```

## Migration Guide

### Before (Hardcoded Custom Logic)

```python
# Old: Custom provider logic was hardcoded in ModelBuilder
model, model_id = ModelBuilder.build(
    "custom_prefix/provider/model-name",
    cache_dir=Path("./cache")
)
```

### After (Modular Providers)

```python
# New: CustomAPI is a registered provider (auto-registered via setup)
# Same API, but now extensible!
from experiments.utils import setup_custom_api_provider
setup_custom_api_provider()  # Registers provider from env vars

model, model_id = ModelBuilder.build(
    "custom_prefix/provider/model-name",
    cache_dir=Path("./cache")
)

# Add your own providers
from agents.utils.model_provider_registry import get_global_registry
registry = get_global_registry()
registry.register(MyCustomProvider())

# Now your custom models work too
model, model_id = ModelBuilder.build(
    "mycustom/my-model",
    cache_dir=Path("./cache")
)
```

**No breaking changes** - existing code continues to work!

## Best Practices

1. **Prefix Convention**: Use a unique prefix for your provider (e.g., `mycompany/`, `custom/`)
2. **Error Handling**: Validate API keys and configuration in `build_model()`
3. **Logging**: Use Python logging to help debug provider issues
4. **Testing**: Test your provider in isolation before registering globally
5. **Documentation**: Document required environment variables and configuration

## See Also

- `src/agents/utils/model_provider.py` - Provider interface
- `src/agents/utils/model_provider_registry.py` - Registry implementation
- `src/agents/utils/providers/custom_api_provider.py` - CustomAPI provider example
- `src/agents/utils/model_builder.py` - Main builder class
