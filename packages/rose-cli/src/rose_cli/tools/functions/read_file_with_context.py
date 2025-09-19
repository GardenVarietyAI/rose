"""Read file tool with code context for actors."""

from pathlib import Path
from typing import Any

from agents import RunContextWrapper, function_tool


@function_tool
def read_file_with_context(ctx: RunContextWrapper[Any], path: str) -> str:
    """Read the contents of a file with additional context for code review.

    Args:
        path: The path to the file to read.
    """
    try:
        full_path = Path(path).resolve()

        if full_path.exists() and full_path.is_file():
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Get file extension for context
            extension = full_path.suffix
            lines = content.count("\n") + 1

            return f"File: {full_path}\nType: {extension}\nLines: {lines}\n\nContent:\n{content}"
        else:
            return f"Error: File not found at {full_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"
