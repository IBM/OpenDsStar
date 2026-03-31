"""
Create MCP Tools for DataBench Datasets using FastMCP

This module provides 3 MCP tools for DataBench datasets:
1. search_files - Search for relevant datasets using semantic search with VectorStoreTool
2. get_file_content - Get the actual dataset content as a stream
3. save_dataframe - Save a DataFrame (as CSV string) to the local 'files' output folder

Previous versions also exposed a ``plotly`` image tool; that helper has
been removed.  Agents should now simply save any generated Plotly figure in
``outputs['figure']`` and rely on the DS-Star finalizer to render it.

Uses the existing VectorStoreTool infrastructure for consistent summary generation.
"""

import io
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from fastmcp import FastMCP

from OpenDsStar.experiments.core.types import Document
from OpenDsStar.tools import VectorStoreTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastMCP server
mcp = FastMCP("databench-server")


# NOTE: explicit plotly tool has been removed.  Agents should simply
# store any generated Plotly figure object in ``outputs['figure']``.  The
# DS-Star finalizer (see ``src/agents/ds_star/nodes/finalizer.py``) will
# detect that value, render it to PNG using Plotly/Kaleido, and append a
# data-URI image to the final answer.  This keeps all plotting logic on the
# finalizer side and avoids the need for an MCP round-trip.


# Global state for VectorStoreTool and dataset mappings
_vector_tool: Optional[VectorStoreTool] = None
_dataset_paths: Dict[str, str] = {}


def initialize_vector_db(
    base_path: str = "databench_subset", cache_dir: Optional[str] = None
):
    """
    Initialize the vector database using VectorStoreTool for consistent summary generation.

    Args:
        base_path: Base directory containing datasets
        cache_dir: Directory for caching vector store (if None, disables cache)
    """
    global _vector_tool, _dataset_paths

    if _vector_tool is not None:
        logger.info("Vector tool already initialized")
        return

    logger.info(f"Initializing vector tool from {base_path}")

    # Find all datasets
    if not os.path.exists(base_path):
        error_msg = f"Base path not found: {base_path}. Please ensure DataBench datasets are downloaded."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Get all dataset directories
    dataset_dirs = [
        d
        for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d)) and not d.startswith(".")
    ]

    logger.info(f"Found {len(dataset_dirs)} datasets")

    # Create Document objects for VectorStoreTool
    corpus = []
    for dataset_id in sorted(dataset_dirs):
        csv_path = os.path.join(base_path, dataset_id, f"{dataset_id}.csv")

        if not os.path.exists(csv_path):
            logger.warning(f"CSV not found for {dataset_id}")
            continue

        # Store path mapping
        _dataset_paths[dataset_id] = csv_path

        # Create Document with stream factory
        def make_stream_factory(path):
            def stream_factory():
                return open(path, "rb")

            return stream_factory

        doc = Document(
            document_id=dataset_id,
            path=csv_path,
            mime_type="text/csv",
            stream_factory=make_stream_factory(csv_path),
            extra_metadata={"dataset_id": dataset_id, "filename": f"{dataset_id}.csv"},
        )
        corpus.append(doc)

        logger.info(f"Processed {dataset_id}")

    if not corpus:
        logger.error("No datasets found to index")
        return

    if not cache_dir:
        logger.info("No cache_dir specified: caching is DISABLED for VectorStoreTool.")

    # Create VectorStoreTool with the corpus (uses existing caching and chunking)
    logger.info("Creating VectorStoreTool with corpus...")
    _vector_tool = VectorStoreTool(
        corpus=corpus,
        cache_dir=Path(cache_dir) if cache_dir else None,
        name="search_databench",
        embedding_model="ibm-granite/granite-embedding-english-r2",
        chunk_size=1000,
        chunk_overlap=200,
        experiment_name="databench",
    )

    logger.info(f"Vector tool initialized with {len(corpus)} datasets")


@mcp.tool()
def search_files(query: str) -> str:
    """
    Search for datasets. Returns JSON string that must be parsed.

    Usage pattern:
        # In current step:
        result = search_files(query="your query")
        data = json.loads(result)
        outputs["search_result"] = data

        # To get first dataset ID (WITHOUT quotes):
        dataset_id = data["datasets"][0]["id"]  # This is a string like: 006_London
        # NOT: dataset_id = "006_London"  # Wrong - don't add extra quotes!

    Returns:
        JSON string: {"datasets": [{"id": "001_Forbes", "summary": "..."}, ...], "count": N}
        CRITICAL: Parse with json.loads(result), then access data["datasets"][0]["id"] directly
    """
    top_k = 5  # Fixed to 5 results
    global _vector_tool

    if _vector_tool is None:
        # Initialize on first use
        initialize_vector_db()

    if _vector_tool is None:
        return json.dumps({"datasets": [], "count": 0})

    try:
        # Use VectorStoreTool's search functionality
        results = _vector_tool._run(query, top_k=top_k)

        if not results:
            logger.info(f"No relevant datasets found for query: {query}")
            return json.dumps({"datasets": [], "count": 0})

        # Extract dataset info with summaries
        datasets_info = []
        seen_ids = set()

        # Access the underlying vector_db to get metadata and content
        if hasattr(_vector_tool, "vector_db") and _vector_tool.vector_db:
            search_results = _vector_tool.vector_db.similarity_search(query, k=top_k)
            for doc in search_results:
                if hasattr(doc, "metadata") and "dataset_id" in doc.metadata:
                    dataset_id = doc.metadata["dataset_id"]
                    if dataset_id not in seen_ids:  # Avoid duplicates
                        seen_ids.add(dataset_id)
                        # Get summary from document content (first 200 chars)
                        summary = (
                            doc.page_content
                            if hasattr(doc, "page_content")
                            else f"Dataset {dataset_id}"
                        )
                        datasets_info.append({"id": dataset_id, "summary": summary})

        logger.info(
            f"Found {len(datasets_info)} relevant dataset(s) for query: {query[:50]}..."
        )

        # Return as JSON object string
        return json.dumps(
            {"datasets": datasets_info, "count": len(datasets_info)}, indent=2
        )

    except Exception as e:
        logger.error(f"Error during search: {e}")
        return json.dumps({"datasets": [], "count": 0})


@mcp.tool()
def get_file_content(dataset_id: str) -> str:
    """
    Get CSV content for a dataset. Pass the dataset ID WITHOUT extra quotes.

    Usage pattern:
        # Get ID from parsed search results (already stored in outputs):
        search_data = prev_step_outputs["search_result"]
        dataset_id = search_data["datasets"][0]["id"]  # This is already a string!

        # Load CSV:
        csv_content = get_file_content(dataset_id=dataset_id)
        df = pd.read_csv(StringIO(csv_content))

    Args:
        dataset_id: Dataset ID string like 006_London

    Returns:
        CSV string for pd.read_csv(StringIO(result))
    """
    global _dataset_paths

    if not _dataset_paths:
        # Initialize on first use
        initialize_vector_db()

    # Validate dataset_id is a string and not something weird like "[" or a list
    if not isinstance(dataset_id, str):
        error_msg = f"Error: dataset_id must be a string, got {type(dataset_id).__name__}: {dataset_id}"
        logger.error(error_msg)
        return error_msg

    # Check if it looks like the agent is trying to use the search result incorrectly
    if (
        dataset_id in ["[", "]", "[0]", "{", "}"]
        or dataset_id.startswith("[")
        or dataset_id.startswith("{")
    ):
        error_msg = (
            f"Error: Invalid dataset_id '{dataset_id}'. "
            f"It looks like you're trying to use the search_files result incorrectly. "
            f"The search_files tool returns a JSON object with dataset information. "
            f"\nExample: "
            f"\n  import json"
            f"\n  result = search_files(query='...')"
            f"\n  data = json.loads(result)  # Parse JSON"
            f"\n  datasets = data['datasets']  # Get datasets list"
            f"\n  first_dataset = datasets[0]  # Get first dataset"
            f"\n  dataset_id = first_dataset['id']  # Extract ID"
            f"\n  content = get_file_content(dataset_id=dataset_id)"
            f"\nAvailable datasets: {', '.join(_dataset_paths.keys())}"
        )
        logger.error(error_msg)
        return error_msg

    if dataset_id not in _dataset_paths:
        error_msg = f"Error: Dataset not found: {dataset_id}\nAvailable datasets: {', '.join(_dataset_paths.keys())}"
        logger.error(error_msg)
        return error_msg

    try:
        csv_path = _dataset_paths[dataset_id]

        # Read CSV file directly as string to avoid any pandas formatting issues
        with open(csv_path, "r", encoding="utf-8") as f:
            csv_content = f.read()

        # Verify it's valid CSV by trying to parse it
        df = pd.read_csv(io.StringIO(csv_content))

        logger.info(
            f"Retrieved content for dataset: {dataset_id} ({len(df)} rows, {len(df.columns)} columns)"
        )
        logger.debug(
            f"CSV content length: {len(csv_content)} chars, first 200 chars: {csv_content[:200]}"
        )

        # Return pure CSV string that can be directly loaded with pandas
        return csv_content

    except Exception as e:
        error_msg = f"Error retrieving content for {dataset_id}: {e}"
        logger.error(error_msg)
        return error_msg


@mcp.tool()
def save_dataframe(df_csv: str, name: str) -> str:
    """
    Save a DataFrame (provided as a CSV string) to the 'files' output folder.

    If a file with the given name already exists, a numeric suffix is appended
    (e.g. my_data_1.csv, my_data_2.csv, ...) so existing files are never overwritten.

    Usage pattern:
        # Convert your DataFrame to CSV and pass it here:
        import io
        csv_str = df.to_csv(index=False)
        result = save_dataframe(df_csv=csv_str, name="my_output")
        # result contains the path of the saved file

    Args:
        df_csv: CSV string representation of the DataFrame (e.g. df.to_csv(index=False))
        name:   Base filename (without .csv extension) for the saved file

    Returns:
        Path of the saved CSV file, or an error message if saving failed.
    """
    try:
        # Determine the output directory (sibling 'files/' next to this module)
        files_dir = Path(__file__).parent / "files"
        files_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize name: strip whitespace and remove the .csv extension if already present
        base_name = name.strip()
        if base_name.lower().endswith(".csv"):
            base_name = base_name[:-4]

        # Find a non-conflicting filename
        target_path = files_dir / f"{base_name}.csv"
        suffix = 1
        while target_path.exists():
            target_path = files_dir / f"{base_name}_{suffix}.csv"
            suffix += 1

        # Validate that df_csv is parseable before writing
        df = pd.read_csv(io.StringIO(df_csv))
        logger.info(
            f"Saving DataFrame with {len(df)} rows and {len(df.columns)} columns "
            f"to {target_path}"
        )

        target_path.write_text(df_csv, encoding="utf-8")

        logger.info(f"DataFrame saved successfully to {target_path}")
        return str(target_path)

    except Exception as e:
        error_msg = f"Error saving DataFrame to files folder: {e}"
        logger.error(error_msg)
        return error_msg


if __name__ == "__main__":
    import argparse
    import socket

    def find_free_port(start_port=8000, max_attempts=100):
        """Find an available port starting from start_port."""
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    return port
            except OSError:
                continue
        raise RuntimeError(
            f"Could not find free port in range {start_port}-{start_port + max_attempts}"
        )

    parser = argparse.ArgumentParser(
        description="Run DataBench MCP Server with 3 tools: search_files, get_file_content, save_dataframe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--http", action="store_true", help="Run as HTTP server instead of stdio"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host for HTTP server (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for HTTP server (default: auto-select starting from 8000)",
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default=None,
        help="Base directory containing datasets (default: src/experiments/demo/databench_subset)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching vector store (default: disabled)",
    )

    args = parser.parse_args()

    # Set default base_path relative to this module if not specified
    if args.base_path is None:
        module_dir = Path(__file__).parent
        args.base_path = str(module_dir / "databench_subset")

    # Initialize vector DB before starting server
    logger.info(f"Initializing vector database from: {args.base_path}")
    initialize_vector_db(base_path=args.base_path, cache_dir=args.cache_dir)

    if args.http:
        # Find available port if not specified
        if args.port is None:
            args.port = find_free_port(8000)

        # Run as HTTP server using FastMCP's built-in HTTP support
        server_url = f"http://{args.host}:{args.port}/sse"

        print("\n" + "=" * 70)
        print(
            "DataBench MCP HTTP Server (3 tools: search_files, get_file_content, save_dataframe)"
        )
        print("=" * 70)
        print(f"Server URL: {server_url}")
        print("\nPass this URL to the multi-agent example:")
        print("  python -m src.experiments.demo.mcp_multi_agent_example \\")
        print(f"    --mcp-server {server_url}")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 70 + "\n")

        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        # Run as stdio server (default)
        logger.info("Starting stdio MCP server...")
        mcp.run()
