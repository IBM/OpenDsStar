"""
Simple QA Loop for DS-Star Agent
Run interactive queries or process a question file.
"""

import argparse
import logging
from typing import Optional

from dotenv import load_dotenv
from langchain_litellm import ChatLiteLLM

from OpenDsStar.agents.ds_star.ds_star_graph import DSStarGraph
from OpenDsStar.agents.utils.logging_utils import init_logger

# Load environment variables from .env file
load_dotenv()
init_logger()
logger = logging.getLogger(__name__)


def get_litellm_model(
    model: str = "watsonx/mistralai/mistral-medium-2505",
    temperature: float = 0.0,
    **kwargs,
) -> ChatLiteLLM:
    """
    Create a LiteLLM chat model instance.

    Supports multiple providers through LiteLLM:
    - OpenAI: "gpt-4", "watsonx/mistralai/mistral-medium-2505", "gpt-3.5-turbo"
    - Anthropic: "claude-3-opus-20240229", "claude-3-sonnet-20240229"
    - Azure: "azure/gpt-4"
    - Google: "gemini/gemini-pro"
    - WatsonX: "watsonx/meta-llama/llama-3-3-70b-instruct"
    - Ollama: "ollama/llama3.2" (local models)
    - And many more...

    Args:
        model: Model identifier (e.g., "gpt-4", "claude-3-opus-20240229")
        temperature: Temperature for generation (0.0 = deterministic)
        **kwargs: Additional arguments passed to ChatLiteLLM

    Returns:
        ChatLiteLLM instance
    """
    return ChatLiteLLM(model=model, temperature=temperature, **kwargs)


def setup_agent(
    model: str = "watsonx/mistralai/mistral-medium-2505",
    temperature: float = 0.0,
    tools: list | None = None,
    max_steps: int = 5,
    max_debug_tries: int = 5,
    code_timeout: int = 30,
) -> DSStarGraph:
    """
    Initialize the DS-Star agent.

    Args:
        model: LiteLLM model identifier (e.g., "gpt-4", "claude-3-sonnet-20240229")
        temperature: Temperature for generation (0.0 = deterministic)
        tools: List of langchain tools
        max_steps: Maximum planning steps
        max_debug_tries: Maximum debug attempts per step
        code_timeout: Timeout for code execution

    Returns:
        Configured DSStarGraph instance
    """
    # Create LLM instance
    llm = get_litellm_model(model=model, temperature=temperature)

    if tools is None:
        tools = []

    system_prompt = (
        "You are a helpful data science assistant. "
        "You can break down complex queries into steps, write code, and provide answers."
    )

    agent = DSStarGraph(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        task_prompt=None,
        max_steps=max_steps,
        max_debug_tries=max_debug_tries,
        code_timeout=code_timeout,
    )

    return agent


def run_single_query(
    agent: DSStarGraph, query: str, config: Optional[dict] = None
) -> dict:
    """
    Run a single query through the agent.

    Args:
        agent: DSStarGraph instance
        query: User question/query
        config: Optional LangGraph config

    Returns:
        Result dictionary with answer and metadata
    """
    if config is None:
        config = {
            "configurable": {"thread_id": "default"},
            "recursion_limit": 1000,
        }

    logger.info(f"Processing query: {query}")

    try:
        result = agent.invoke(input_dict={"user_query": query}, config=config)

        # Result is a dict, extract key information
        output = {
            "query": query,
            "answer": result.get("final_answer"),
            "steps_used": result.get("steps_used", 0),
            "max_steps": result.get("max_steps", 0),
            "fatal_error": result.get("fatal_error"),
            "trajectory": result.get("trajectory", []),
        }

        return output

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return {
            "query": query,
            "answer": None,
            "error": str(e),
        }


def interactive_mode(agent: DSStarGraph):
    """
    Run agent in interactive QA mode.
    """
    print("\n" + "=" * 60)
    print("DS-Star Agent - Interactive Mode")
    print("Type your questions below. Type 'exit' or 'quit' to stop.")
    print("=" * 60 + "\n")

    # Use memory saver for conversation continuity
    config = {
        "configurable": {"thread_id": "interactive_session"},
        "recursion_limit": 1000,
    }

    while True:
        try:
            query = input("\n🤔 Question: ").strip()

            if query.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye! 👋")
                break

            if not query:
                continue

            result = run_single_query(agent, query, config)

            print(f"\n💡 Answer: {result.get('answer', 'No answer generated')}")

            if result.get("fatal_error"):
                print(f"⚠️  Error: {result['fatal_error']}")

            print(
                f"📊 Steps used: {result.get('steps_used', 0)}/{result.get('max_steps', 0)}"
            )

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye! 👋")
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}", exc_info=True)
            print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="DS-Star Agent - Simple QA Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum number of planning steps (default: 5)",
    )

    parser.add_argument(
        "--code-timeout",
        type=int,
        default=30,
        help="Timeout for code execution in seconds (default: 30)",
    )

    parser.add_argument(
        "--tools",
        nargs="+",
        default=[],
        help="List of tools to enable (e.g., dataset_sql vector_store)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="watsonx/mistralai/mistral-medium-2505",
        help="LiteLLM model identifier (default: gpt-4o-mini). Examples: gpt-4, claude-3-sonnet-20240229, gemini/gemini-pro",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (default: 0.0)",
    )

    args = parser.parse_args()

    # Setup agent
    logger.info(f"Initializing DS-Star agent with model: {args.model}")
    agent = setup_agent(
        model=args.model,
        temperature=args.temperature,
        tool_names=args.tools,
        max_steps=args.max_steps,
        code_timeout=args.code_timeout,
    )
    logger.info("Agent initialized successfully")

    interactive_mode(agent)


if __name__ == "__main__":
    main()
