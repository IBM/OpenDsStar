"""Shared tools for file-based benchmarks (KramaBench, DataBench, etc.)."""

from __future__ import annotations

import logging
import re
from io import BytesIO
from pathlib import PurePosixPath
from typing import Any, BinaryIO, Callable, Literal

import pandas as pd
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _clean_query(query: str) -> str:
    """Remove quotes and normalize whitespace in a search query."""
    return re.sub(r'["\']', "", query).strip()


def _suffix(path: str) -> str:
    """Return the lowercase file suffix, including the leading dot."""
    return PurePosixPath(path).suffix.lower()


def _filename(path: str) -> str:
    """Return the file name from a full path."""
    return PurePosixPath(path).name


def _infer_format(path: str) -> str | None:
    """Infer the file format from the path suffix."""
    ext = _suffix(path)

    if ext == ".parquet":
        return "parquet"
    if ext in {".csv", ".tsv"}:
        return "csv"
    if ext in {".xls", ".xlsx", ".xlsm", ".ods"}:
        return "excel"
    if ext == ".json":
        return "json"
    if ext in {".pkl", ".pickle"}:
        return "pickle"
    if ext in {".txt", ".md", ".log"}:
        return "text"

    return None


class MilvusSearchInput(BaseModel):
    """Input schema for file search."""

    query: str = Field(..., description="Search query describing the file(s) needed")
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of candidate files to return",
    )


class MilvusSearchTool(BaseTool):
    """
    Search for relevant files using semantic similarity over file descriptions.
    """

    name: str = "search_files"
    description: str = """
Search for relevant files in the corpus using semantic search.

INPUT:
- query (str): Describe the file(s) you need.
- top_k (int, optional): Number of candidate files to return. Default: 5.

RETURNS:
- List[Dict[str, str]]
- Each result contains:
  - "path": exact full file path
  - "filename": file name only
  - "description": AI-generated description of the file content

IMPORTANT:
- Read ALL returned descriptions before choosing a file.
- Do not assume the first result is always the right dataset.
- Use "path", not "filename", when opening a file.
- The description may contain useful schema hints such as column names,
  date ranges, and file structure.

RECOMMENDED WORKFLOW:
1. Call search_files(query)
2. Review every description
3. Choose the most relevant path(s)
4. Use get_file_info(path) if needed
5. Prefer load_dataframe(path) for tabular files
6. Use get_file_content(path) only for low-level/manual reading
"""

    args_schema: type[MilvusSearchInput] = MilvusSearchInput

    vector_db: Any = None

    def __init__(self, vector_db: Any):
        super().__init__()
        self.vector_db = vector_db

    def _run(self, query: str, top_k: int = 5) -> list[dict[str, str]]:
        cleaned_query = _clean_query(query)

        if self.vector_db is None:
            logger.error("Vector database not initialized")
            return []

        try:
            results = self.vector_db.similarity_search(cleaned_query, k=top_k)
        except Exception:
            logger.exception("Error during file search for query: %s", cleaned_query)
            return []

        if not results:
            logger.info("No relevant files found for query: %s", cleaned_query)
            return []

        file_results: list[dict[str, str]] = []
        for doc in results:
            metadata = getattr(doc, "metadata", None)
            if not metadata or "file_path" not in metadata:
                continue

            path = str(metadata["file_path"])
            file_results.append(
                {
                    "path": path,
                    "filename": str(metadata.get("filename", _filename(path))),
                    "description": str(doc.page_content),
                }
            )

        logger.info(
            "Found %d relevant file(s) for query: %s",
            len(file_results),
            cleaned_query[:80],
        )
        return file_results


class FileInfoInput(BaseModel):
    """Input schema for file info."""

    path: str = Field(..., description="Exact full path from search_files results")


class FileInfoTool(BaseTool):
    """
    Inspect a file path and return lightweight metadata.
    """

    name: str = "get_file_info"
    description: str = """
Inspect a file path and return basic metadata.

INPUT:
- path (str): Exact full path from search_files results

RETURNS:
- Dict with:
  - "path"
  - "filename"
  - "extension"
  - "likely_format"
  - "is_tabular"

USE THIS TOOL TO:
- Decide whether the file is parquet / csv / excel / json / pickle / text
- Choose the correct reader before opening the file
- Avoid mistakes like reading a .parquet file with pd.read_csv(...)
"""

    args_schema: type[FileInfoInput] = FileInfoInput

    def _run(self, path: str) -> dict[str, Any]:
        likely_format = _infer_format(path)
        return {
            "path": path,
            "filename": _filename(path),
            "extension": _suffix(path),
            "likely_format": likely_format,
            "is_tabular": likely_format in {"parquet", "csv", "excel", "json"},
        }


class FileContentInput(BaseModel):
    """Input schema for file content retrieval."""

    path: str = Field(..., description="Exact full path from search_files results")


class FileContentTool(BaseTool):
    """
    Retrieve file content as a binary stream.

    This is a low-level tool. Prefer load_dataframe(path) for tabular data.
    """

    name: str = "get_file_content"
    description: str = """
Retrieve the content of a specific file as a BytesIO stream.

INPUT:
- path (str): Must be the exact "path" from search_files results.
  Do NOT use "filename".

RETURN VALUE:
- BinaryIO: A BytesIO stream ready to use with pandas readers or other file operations.

CRITICAL RULE:
Choose the reader that matches the file extension.

COMMON PATTERNS:
- .parquet -> pd.read_parquet(get_file_content(path))
- .csv     -> pd.read_csv(get_file_content(path))
- .tsv     -> pd.read_csv(get_file_content(path), sep='\\t')
- .xlsx    -> pd.read_excel(get_file_content(path))
- .json    -> pd.read_json(get_file_content(path)) or json.loads(get_file_content(path).read())
- .txt     -> get_file_content(path).read().decode("utf-8")
- .pkl     -> pickle.load(get_file_content(path))

IMPORTANT WARNINGS:
- Do NOT use pd.read_csv(...) for .parquet files.
- Do NOT decode binary files (parquet, excel, pickle) as UTF-8 text.
- The stream is ready to use directly - no need to call it as a function.

PREFERRED WORKFLOW:
- Prefer load_dataframe(path) for tabular files.
- Use get_file_info(path) first if unsure which reader to use.
"""

    args_schema: type[FileContentInput] = FileContentInput

    path_to_bytes_factory: dict[str, Callable[[], bytes]] = {}

    def __init__(self, path_to_bytes_factory: dict[str, Callable[[], bytes]]):
        super().__init__()
        self.path_to_bytes_factory = path_to_bytes_factory

    def _run(self, path: str) -> BinaryIO:
        logger.info("Retrieving content for file: %s", path)

        bytes_factory = self.path_to_bytes_factory.get(path)
        if bytes_factory is None:
            raise ValueError(f"File not found in corpus: {path}")

        return BytesIO(bytes_factory())


def _auto_fix_csv(raw_bytes: bytes, sep: str = ",") -> pd.DataFrame:
    """Try to load a CSV that has metadata/comment lines before the real header.

    Strategy: scan the first 50 lines and find the first line whose field count
    matches the most common field count (i.e. the data rows).  Use that line as
    the header and skip everything above it.
    """
    text = raw_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()

    # Count fields per line for the first 50 lines
    counts: list[int] = []
    for line in lines[:50]:
        counts.append(len(line.split(sep)))

    if not counts:
        raise ValueError("CSV file is empty")

    # Most common field count is likely the data width
    from collections import Counter

    most_common_width = Counter(counts).most_common(1)[0][0]

    # Find first line with that width — treat it as the header
    skip = 0
    for i, c in enumerate(counts):
        if c == most_common_width:
            skip = i
            break

    logger.info("auto_fix_csv: skipping %d lines, detected %d columns", skip, most_common_width)
    return pd.read_csv(
        BytesIO(raw_bytes),
        sep=sep,
        skiprows=skip,
        on_bad_lines="skip",
    )


class LoadDataFrameInput(BaseModel):
    """Input schema for dataframe loading."""

    path: str = Field(..., description="Exact full path from search_files results")
    format: Literal["auto", "parquet", "csv", "excel", "json", "pickle"] = Field(
        default="auto",
        description="File format to load. Use 'auto' to infer from the extension.",
    )
    auto_fix_csv: bool = Field(
        default=False,
        description=(
            "If True and CSV parsing fails, automatically retry by "
            "skipping bad lines and detecting the header row. "
            "Use when a CSV has metadata/comment lines before the actual table."
        ),
    )


class LoadDataFrameTool(BaseTool):
    """
    Load a tabular file into a pandas DataFrame.

    This is the safest tool for structured data files.
    """

    name: str = "load_dataframe"
    description: str = """
Load a tabular file into a pandas DataFrame.

INPUT:
- path (str): Exact full path from search_files results
- format (str, optional): One of:
  - "auto" (default)
  - "parquet"
  - "csv"
  - "excel"
  - "json"
  - "pickle"
- auto_fix_csv (bool, optional): Default False. If True and CSV parsing fails,
  automatically retry by skipping metadata/comment lines and detecting the real
  header row. Use this when a CSV file has non-tabular lines before the data.

BEHAVIOR:
- "auto" infers the reader from the file extension.
- Supported mappings:
  - .parquet -> pd.read_parquet
  - .csv     -> pd.read_csv
  - .tsv     -> pd.read_csv(..., sep='\\t')
  - .xls/.xlsx/.xlsm/.ods -> pd.read_excel
  - .json    -> pd.read_json
  - .pkl/.pickle -> pickle.load (must contain a pandas DataFrame)

WHY USE THIS TOOL:
- Prefer this tool for tabular data instead of manual stream handling.
- It prevents common mistakes such as reading Parquet with pd.read_csv(...).
- If you get a CSV parsing error, retry with auto_fix_csv=True.
"""

    args_schema: type[LoadDataFrameInput] = LoadDataFrameInput

    path_to_bytes_factory: dict[str, Callable[[], bytes]] = {}

    def __init__(self, path_to_bytes_factory: dict[str, Callable[[], bytes]]):
        super().__init__()
        self.path_to_bytes_factory = path_to_bytes_factory

    def _run(
        self,
        path: str,
        format: Literal["auto", "parquet", "csv", "excel", "json", "pickle"] = "auto",
        auto_fix_csv: bool = False,
    ) -> pd.DataFrame:
        import pickle

        logger.info("Loading dataframe from file: %s (format=%s)", path, format)

        bytes_factory = self.path_to_bytes_factory.get(path)
        if bytes_factory is None:
            raise ValueError(f"File not found in corpus: {path}")

        chosen_format = format
        if chosen_format == "auto":
            inferred = _infer_format(path)
            if inferred not in {"parquet", "csv", "excel", "json", "pickle"}:
                raise ValueError(
                    f"Could not infer a supported tabular format from path: {path}"
                )
            chosen_format = inferred

        stream = BytesIO(bytes_factory())
        ext = _suffix(path)

        if chosen_format == "parquet":
            return pd.read_parquet(stream)

        if chosen_format == "csv":
            sep = "\t" if ext == ".tsv" else ","
            try:
                return pd.read_csv(stream, sep=sep)
            except pd.errors.ParserError:
                if not auto_fix_csv:
                    raise
                logger.info("CSV parse failed for %s, retrying with auto_fix_csv", path)
                return _auto_fix_csv(bytes_factory(), sep=sep)

        if chosen_format == "excel":
            return pd.read_excel(stream)

        if chosen_format == "json":
            return pd.read_json(stream)

        if chosen_format == "pickle":
            obj = pickle.load(stream)
            if not isinstance(obj, pd.DataFrame):
                raise TypeError(
                    f"Pickle file did not contain a pandas DataFrame: {path} "
                    f"(got {type(obj)!r})"
                )
            return obj

        raise ValueError(f"Unsupported format {chosen_format!r} for file {path!r}")


def build_file_tools(
    *,
    vector_db: Any,
    path_to_bytes_factory: dict[str, Callable[[], bytes]],
) -> list[BaseTool]:
    """
    Build the standard tool set for file-based benchmarks.
    """
    return [
        MilvusSearchTool(vector_db=vector_db),
        FileInfoTool(),
        FileContentTool(path_to_bytes_factory=path_to_bytes_factory),
        LoadDataFrameTool(path_to_bytes_factory=path_to_bytes_factory),
    ]
