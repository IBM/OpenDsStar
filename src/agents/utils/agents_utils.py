import logging

from langchain_core.language_models import BaseChatModel
from langchain_litellm import ChatLiteLLM

logger = logging.getLogger(__name__)

seen_labels = set()


def print_once(label, msg):
    if label not in seen_labels:
        seen_labels.add(label)
        logger.info(msg)


def to_langchain_model(llm: str | BaseChatModel) -> BaseChatModel:
    """Convert a model name or instance to a BaseChatModel instance."""

    if isinstance(llm, str):
        llm = ChatLiteLLM(model=llm)
    return llm
