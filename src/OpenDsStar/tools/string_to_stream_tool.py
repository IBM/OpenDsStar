"""
StringToStreamTool - Convert string to StringIO stream.

This tool allows agents to convert string data into a StringIO stream object,
enabling parsing of encoded files without requiring direct file I/O access.
"""

from io import StringIO

from smolagents import Tool


class StringToStreamTool(Tool):
    """
    Tool that converts a string into a StringIO stream object.

    This is useful for parsing encoded files (CSV, JSON, etc.) from string data
    without requiring file system access. The agent can then use standard parsing
    libraries on the returned stream.
    """

    name = "string_to_stream"
    description = """
    Converts a string into a StringIO stream object that can be used with file-reading functions.

    This is useful when you have file content as a string (e.g., from an API, database, or encoded data)
    and need to parse it using libraries that expect a file-like object (pandas.read_csv, json.load, etc.).

    Args:
        content (str): The string content to convert to a stream

    Returns:
        StringIO: A StringIO object containing the string data, positioned at the start

    Example usage in code:
        # Convert CSV string to stream and parse with pandas
        csv_data = "name,age\\nAlice,30\\nBob,25"
        stream = string_to_stream(csv_data)
        df = pd.read_csv(stream)

        # Convert JSON string to stream and parse
        json_data = '{"key": "value"}'
        stream = string_to_stream(json_data)
        data = json.load(stream)

    NOTE:
        do not import StringIO!

        incorrect:
            import pandas as pd
            from io import StringIO
            stream = string_to_stream(csv_content)
            df = pd.read_csv(stream)

        correct:
            import pandas as pd
            # string_to_stream handles the StringIO import
            stream = string_to_stream(csv_content)
            df = pd.read_csv(stream)
    """

    inputs = {
        "content": {
            "type": "string",
            "description": "The string content to convert into a StringIO stream",
        }
    }

    output_type = "any"  # Returns StringIO object

    def forward(self, content: str) -> StringIO:
        """
        Convert string to StringIO stream.

        Args:
            content: The string content to convert

        Returns:
            StringIO object containing the content, positioned at start
        """
        if not isinstance(content, str):
            raise TypeError(f"content must be a string, got {type(content).__name__}")

        # Create StringIO object and ensure it's positioned at the start
        stream = StringIO(content)
        stream.seek(0)
        return stream


# For LangChain compatibility, create a LangChain tool wrapper
try:
    from langchain_core.tools import StructuredTool

    def string_to_stream_langchain(content: str) -> StringIO:
        """
        Convert a string into a StringIO stream object.

        Args:
            content: The string content to convert to a stream

        Returns:
            StringIO object containing the string data
        """
        if not isinstance(content, str):
            raise TypeError(f"content must be a string, got {type(content).__name__}")

        stream = StringIO(content)
        stream.seek(0)
        return stream

    # Create LangChain tool
    StringToStreamLangChainTool = StructuredTool.from_function(
        func=string_to_stream_langchain,
        name="string_to_stream",
        description=(
            "Converts a string into a StringIO stream object that can be used with file-reading functions. "
            "Useful for parsing CSV, JSON, or other file formats from string data without file system access."
        ),
    )

except ImportError:
    # LangChain not available, only smolagents tool will work
    StringToStreamLangChainTool = None
