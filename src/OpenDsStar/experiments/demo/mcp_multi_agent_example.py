"""
Multi-Agent MCP Example - Run DS-Star, ReAct, and CodeAct with DataBench MCP tools

Usage:
    # Run with default DataBench MCP server (stdio)
    python -m src.experiments.demo.mcp_multi_agent_example

    # Run with HTTP MCP server (start server first in separate terminal)
    python -m src.experiments.demo.create_mcp_tools --http
    python -m src.experiments.demo.mcp_multi_agent_example \
        --mcp-server http://localhost:8000/mcp

    # Run specific agents only
    python -m src.experiments.demo.mcp_multi_agent_example \
        --agents ds_star react_langchain

    # Custom MCP server path (stdio)
    python -m src.experiments.demo.mcp_multi_agent_example \
        --mcp-server path/to/mcp_server.py
"""

import argparse
import logging
import os
import sys
from typing import List

from OpenDsStar.core.model_registry import ModelRegistry
from OpenDsStar.experiments.implementations.agent_factory import AgentFactory, AgentType
from OpenDsStar.tools.mcp_integration_standalone import (
    cleanup_mcp_clients,
    create_http_config,
    create_langchain_tools_from_mcp,
    create_stdio_config,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default queries - one question per DataBench dataset (10 total)
# Queries are phrased to encourage agents to first discover the relevant dataset
DEFAULT_QUERIES = [
    "Find data about wealthy individuals and determine if the person with the highest net worth is self-made.",  # 001_Forbes
    "Locate property data for London and check if all properties are in the same neighbourhood.",  # 006_London
    "Search for FIFA player statistics and identify if any players have a greater overall score than their potential score.",  # 007_Fifa
    "Find information about roller coasters and determine if the oldest one in the data was still operating.",  # 013_Roller
    "Look for Airbnb rental listings and check if there's a property with exactly 5 bedrooms.",  # 014_Airbnb
    "Search for real estate property data and determine if any properties have a price over 1,000,000.",  # 020_Real
    "Find professional demographics data and verify if 'USA' is the most common entry in the 'Geographies' column.",  # 030_Professionals
    "Locate speed dating participant data and check if the youngest participant has met their match.",  # 040_Speed
    "Search for product review data and determine if there are more reviews with rating 5 from 'GB' than 'US'.",  # 058_US
    "Find food product information and check if there are any vegan products in the data.",  # 070_OpenFoodFacts
]


def setup_mcp_tools(mcp_server: str) -> List:
    """
    Set up MCP tools from server path or URL.

    Args:
        mcp_server: Path to MCP server script (stdio) or HTTP URL

    Returns:
        List of LangChain tools
    """
    # Detect if it's HTTP URL or local path
    if mcp_server.startswith("http://") or mcp_server.startswith("https://"):
        logger.info(f"Connecting to HTTP MCP server: {mcp_server}")
        mcp_servers = {"databench": create_http_config(url=mcp_server)}
    else:
        if not os.path.exists(mcp_server):
            raise FileNotFoundError(f"MCP server not found: {mcp_server}")
        logger.info(f"Starting stdio MCP server: {mcp_server}")
        mcp_servers = {
            "databench": create_stdio_config(command="python", args=[mcp_server])
        }

    # Convert to LangChain tools
    mcp_tools = create_langchain_tools_from_mcp(mcp_servers)
    logger.info(f"Created {len(mcp_tools)} MCP tools")

    return mcp_tools


def run_agent(
    agent_type: AgentType, query: str, tools: List, model: str, max_steps: int
) -> dict:
    """Run a single agent and return results."""
    print(f"\n{'='*70}")
    print(f"{agent_type.value.upper()} Agent")
    print(f"{'='*70}")

    try:
        # Create agent
        agent = AgentFactory.create_agent(
            agent_type=agent_type,
            model=model,
            tools=tools,
            max_steps=max_steps,
            temperature=0.0,
        )

        # Run agent
        result = agent.invoke(query)

        # Display results
        print(f"✓ Answer: {result.get('answer', 'N/A')}")
        print(f"  Steps: {result.get('steps_used', 'N/A')}")

        return result

    except Exception as e:
        logger.error(f"{agent_type.value} failed: {e}")
        print(f"✗ Error: {e}")
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple agents with MCP tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--mcp-server",
        type=str,
        help="Path to MCP server script. Default: create_mcp_tools.py in demo folder",
    )
    parser.add_argument(
        "--agents",
        type=str,
        nargs="+",
        default=["ds_star", "react_langchain", "codeact_smolagents"],
        choices=[
            "ds_star",
            "react_langchain",
            "codeact_smolagents",
            "react_smolagents",
        ],
        help="Agent types to run (default: all 3 main types)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=ModelRegistry.WX_MISTRAL_MEDIUM,
        help="Model identifier",
    )
    parser.add_argument(
        "--max-steps", type=int, default=5, help="Maximum steps (default: 5)"
    )

    args = parser.parse_args()

    try:
        print("\n" + "=" * 70)
        print("MCP Multi-Agent Example - DataBench")
        print("=" * 70)

        # Default to create_mcp_tools.py in demo folder if not specified
        if not args.mcp_server:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            args.mcp_server = os.path.join(script_dir, "create_mcp_tools.py")
            logger.info(f"Using default MCP server: {args.mcp_server}")

        print(f"MCP Server: {args.mcp_server}")
        print(f"Model: {args.model}")
        print(f"Agents: {', '.join(args.agents)}")

        # Setup MCP tools
        mcp_tools = setup_mcp_tools(args.mcp_server)
        print("\nAvailable tools:")
        for tool in mcp_tools:
            print(f"  - {tool.name}: {tool.description}")

        # Run each agent on all queries
        for agent_str in args.agents:
            agent_type = AgentType(agent_str)

            print(f"\n{'='*70}")
            print(f"AGENT: {agent_type.value.upper()}")
            print(f"{'='*70}")

            for query in DEFAULT_QUERIES:
                print(f"\nQuery: {query}")
                run_agent(
                    agent_type=agent_type,
                    query=query,
                    tools=mcp_tools,
                    model=args.model,
                    max_steps=args.max_steps,
                )

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted")
        return 130

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

    finally:
        # Always cleanup MCP clients
        print(f"\n{'='*70}")
        try:
            cleanup_mcp_clients()
            print("Cleanup complete")
        except Exception as cleanup_error:
            logger.warning(f"Cleanup error (non-fatal): {cleanup_error}")


if __name__ == "__main__":
    sys.exit(main())
