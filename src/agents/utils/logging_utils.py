import logging
from pathlib import Path

import litellm

logger_initialized = False


def get_current_log_level():
    from logging import getLogger

    root_logger = getLogger()
    return root_logger.handlers[0].level


def replace_file_handler(logger, file_name):
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()
    new_handler = logging.FileHandler(file_name)
    new_handler.setLevel(logger.handlers[0].level)
    new_handler.setFormatter(logger.handlers[0].formatter)
    logger.addHandler(new_handler)


def init_logger(
    level: int = logging.INFO, suffix="", file_name: str | Path | None = None
):
    from logging import Formatter, StreamHandler, getLogger

    global logger_initialized

    root_logger = getLogger()
    if logger_initialized:
        current_handler_level = get_current_log_level()
        if level != current_handler_level:
            current_handler_level = logging.getLevelName(current_handler_level)
            level = logging.getLevelName(level)
            root_logger.warning(
                f"Logger already initialized to level '{current_handler_level}', "
                f"skipping second initialization to level '{level}'."
            )
        if file_name:
            replace_file_handler(root_logger, file_name)
        return

    # This sets the default logging level for all loggers under "root".
    # The level of a specific logger can still be set explicitly by using
    # logging.getLogger("logger name").setLevel(level)
    root_logger.setLevel(logging.INFO)
    logger_initialized = True

    log_formatter = Formatter(
        "%(asctime)s [%(levelname)s] %(filename)s %(lineno)d: %(message)s [%(process)d:%(threadName)s] "
        + suffix
    )

    console_handler = StreamHandler()
    console_handler.setFormatter(log_formatter)
    # Use the provided level for the handler, this affects all messages to the console,
    # even from loggers that have a level different from what is specified here.
    # For example, if 'level' here is logging.INFO, no DEBUG messages will be shown,
    # even from loggers with their level was set explicitly to DEBUG.
    console_handler.setLevel(level)
    root_logger.handlers = []
    root_logger.addHandler(console_handler)
    if file_name:
        Path(file_name).touch(exist_ok=True)
        file_handler = logging.FileHandler(file_name)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
    root_logger.info(f"Logger initialized to level '{logging.getLevelName(level)}'.")

    # Configure vLLM logger to propagate logs and remove custom handlers
    vllm_logger = getLogger("vllm")
    vllm_logger.handlers = []  # Remove any vLLM-specific handlers
    vllm_logger.propagate = True  # Ensure logs propagate to root logger
    vllm_logger.setLevel(level)  # Align with the provided level

    # Disable OpenAI, httpx, and watsonx.ai sdk logging
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("ibm_watsonx_ai.wml_resource").setLevel(logging.ERROR)
    logging.getLogger("ibm_watsonx_ai.client").setLevel(logging.ERROR)
    logging.getLogger("genai.extensions.langchain.llm").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(
        logging.ERROR
    )
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    litellm.suppress_debug_info = True
