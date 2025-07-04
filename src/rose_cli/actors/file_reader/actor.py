from pathlib import Path
from typing import Any, Dict

from agents import (
    Agent,
    RunContextWrapper,
    Runner,
    function_tool,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from jinja2 import Environment, FileSystemLoader

from rose_cli.utils import get_async_client


@function_tool
def read_file(ctx: RunContextWrapper[Any], path: str, directory: str | None = None) -> str:
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


class FileReaderActor:
    """Agent that can read files and list directories."""

    def __init__(self, model: str = "qwen2.5-0.5b") -> None:
        client = get_async_client()
        set_default_openai_client(client)
        set_tracing_disabled(True)
        set_default_openai_api("responses")

        # Load instructions from Jinja template
        template_dir = Path(__file__).parent
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        template = env.get_template("instructions.jinja2")

        # Prepare tool information for the template
        tools_info = [
            {"name": "read_file", "description": "Read the contents of a file"},
            {"name": "list_files", "description": "List files in a directory"},
        ]

        instructions = template.render(tools=tools_info)

        self.agent = Agent(
            name="FileReader",
            model=model,
            instructions=instructions,
            tools=[read_file, list_files],
        )

    def run(self, query: str) -> Dict[str, Any]:
        """Execute the agent with the given query."""
        try:
            result = Runner.run_sync(self.agent, query)
            return {"response": result.final_output, "success": True}
        except Exception as e:
            return {"response": f"Error processing query: {str(e)}", "success": False}
