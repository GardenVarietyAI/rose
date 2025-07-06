from pathlib import Path
from typing import Any, Dict

from agents import (
    Agent,
    Runner,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from jinja2 import Environment, FileSystemLoader

from rose_cli.tools.functions import list_files, read_file
from rose_cli.utils import get_async_client


class FileReaderActor:
    """Agent that can read files and list directories."""

    def __init__(self, model: str = "Qwen/Qwen2.5-1.5B-Instruct") -> None:
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
