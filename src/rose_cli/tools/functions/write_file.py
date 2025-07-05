"""Write file tool for actors."""

from pathlib import Path
from typing import Any

from agents import RunContextWrapper, function_tool


@function_tool
def write_file(ctx: RunContextWrapper[Any], path: str, content: str) -> str:
    """Write content to a file.

    Args:
        path: The path to write the file to.
        content: The content to write to the file.
    """
    try:
        full_path = Path(path).resolve()

        # Create parent directories if they don't exist
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the file
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Successfully wrote {len(content)} characters to {full_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"
