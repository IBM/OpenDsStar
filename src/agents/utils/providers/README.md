# Custom Model Providers

This directory contains custom model provider implementations that extend ModelBuilder's capabilities.

## Overview

The provider system allows you to register custom model builders for specific model prefixes or types. This is useful for:
- Connecting to internal/proprietary APIs
- Adding support for new model providers
- Customizing authentication or configuration

## CustomAPIProvider

The `CustomAPIProvider` is a generic, configurable provider that reads all settings from environment variables. This makes it suitable for open-source projects without hardcoded credentials.

### Configuration

All configuration is done via environment variables:

| Environment Variable | Description | Default | Required |
|---------------------|-------------|---------|----------|
| `CUSTOM_API_PREFIX` | Model prefix to handle (e.g., "myapi") | `"custom"` | No |
| `CUSTOM_API_BASE` | Base URL for the API | None | **Yes** |
| `CUSTOM_API_KEY` | API key for authentication | None | **Yes** |
| `CUSTOM_API_PROVIDER` | LiteLLM provider type | `"openai"` | No |
| `CUSTOM_API_NAME` | Display name for the provider | `"CustomAPI"` | No |

### Usage Example

**1. Configure via environment variables:**

```bash
# .env file
CUSTOM_API_PREFIX=myapi
CUSTOM_API_BASE=https://api.example.com
CUSTOM_API_KEY=sk-your-secret-key
CUSTOM_API_PROVIDER=openai
CUSTOM_API_NAME=MyAPI
```

**2. Register the provider:**

```python
from agents.utils.providers.custom_api_provider import CustomAPIProvider

# Register with default settings (reads from env vars)
CustomAPIProvider.register()
```

**3. Use with ModelBuilder:**

```python
from pathlib import Path
from agents.utils.model_builder import ModelBuilder

# Now you can use models with your custom prefix
model, model_id = ModelBuilder.build(
    "myapi/my-model-name",
    temperature=0.0,
    cache_dir=Path("./cache")
)
```

### IBM TPM Example

For IBM users with access to Third-Party Models (TPM):

```bash
# .env file
CUSTOM_API_PREFIX=tpm
CUSTOM_API_BASE=https://ete-litellm.ai-models.vpc-int.res.ibm.com
CUSTOM_API_KEY=sk-your-ibm-key
CUSTOM_API_PROVIDER=openai
CUSTOM_API_NAME=TPM
```

Then use models like: `tpm/GCP/gemini-2.5-flash`

## Creating Your Own Provider

To create a custom provider, implement the `ModelProvider` interface:

```python
from agents.utils.model_provider import ModelProvider
from langchain_core.caches import BaseCache
from langchain_core.language_models import BaseChatModel

class MyCustomProvider(ModelProvider):
    @property
    def name(self) -> str:
        return "MyProvider"

    def can_handle(self, model_string: str) -> bool:
        return model_string.startswith("myprovider/")

    def build_model(
        self,
        model_string: str,
        temperature: float,
        cache: BaseCache,
    ) -> tuple[BaseChatModel, str]:
        # Your custom model building logic
        model_id = model_string[11:]  # Remove "myprovider/"
        # ... build and return model
        return model_instance, model_id
```

Then register it:

```python
from agents.utils.model_provider_registry import get_global_registry

registry = get_global_registry()
registry.register(MyCustomProvider())
```

## Best Practices

1. **Use unique prefixes** - Choose a prefix that won't conflict with other providers
2. **Validate configuration** - Check required settings in `build_model()`
3. **Use environment variables** - Keep credentials out of code
4. **Log appropriately** - Use Python logging for debugging
5. **Handle errors gracefully** - Provide clear error messages

## See Also

- [MODEL_PROVIDERS.md](../../../../docs/MODEL_PROVIDERS.md) - Detailed provider documentation
- [model_provider.py](../model_provider.py) - Provider interface
- [model_provider_registry.py](../model_provider_registry.py) - Registry implementation
