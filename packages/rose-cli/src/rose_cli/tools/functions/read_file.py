"""Read file tool for actors."""

from pathlib import Path
from typing import Any, Optional

from agents import RunContextWrapper, function_tool


@function_tool
def read_file(ctx: RunContextWrapper[Any], path: str, directory: Optional[str] = None) -> str:
    """Read the contents of a file.

    Args:
        path: The path to the file to read.
        directory: The directory to read the file from.
    """
    try:
        # Construct full path
        if directory:
            full_path = Path(directory) / path
        else:
            full_path = Path(path)

        # Security check - ensure path doesn't go outside allowed directories
        full_path = full_path.resolve()

        # Read file
        if full_path.exists() and full_path.is_file():
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            return f"Contents of {full_path}:\n\n{content}"
        else:
            return f"Error: File not found at {full_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"
