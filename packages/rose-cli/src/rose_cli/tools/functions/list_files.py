"""List files tool for actors."""

from pathlib import Path
from typing import Any

from agents import RunContextWrapper, function_tool


@function_tool
def list_files(ctx: RunContextWrapper[Any], directory: str = ".") -> str:
    """List files in a directory.

    Args:
        directory: The directory to list files from.
    """
    try:
        path = Path(directory).resolve()
        if path.exists() and path.is_dir():
            files = []
            for item in path.iterdir():
                if item.is_file():
                    files.append(f"{item.name}")
                elif item.is_dir():
                    files.append(f"{item.name}/")

            if files:
                return f"Files in {path}:\n" + "\n".join(sorted(files))
            else:
                return f"No files found in {path}"
        else:
            return f"Error: Directory not found at {path}"
    except Exception as e:
        return f"Error listing files: {str(e)}"
