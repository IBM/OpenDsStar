import logging
from typing import Any, Optional, Type

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class BaseNode(BaseModel):
    """
    Base class for DS-STAR graph nodes.
    Handles:
    - LLM initialization
    - Structured output creation
    - Common prompt context fields (system_prompt, task_prompt, tools_spec)
    - Basic event logging helpers
    """

    model_config = {"arbitrary_types_allowed": True}

    # Optional prompts/spec injected into system messages
    system_prompt: Optional[str] = None
    task_prompt: Optional[str] = None
    tools_spec: Optional[str] = None

    # LLMs
    llm: Any = None
    llm_with_output: Any = None

    # Must be defined by subclasses
    structured_output_schema: Optional[Type[BaseModel]] = None

    # --- Pydantic v2 hook ---
    def model_post_init(self, __context: Any) -> None:
        """Attach structured-output model if subclass defined it."""
        schema = self.structured_output_schema
        if self.llm is not None and schema is not None:
            if hasattr(self.llm, "with_structured_output"):
                # Use the LLM with cache - structured output wrapper will handle it
                self.llm_with_output = self.llm.with_structured_output(schema)
            else:
                raise TypeError(
                    "Provided llm does not support `with_structured_output(...)`."
                )
