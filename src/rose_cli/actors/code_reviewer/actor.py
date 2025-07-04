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
def read_file(ctx: RunContextWrapper[Any], path: str) -> str:
    """Read the contents of a file for code review.

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


@function_tool
def write_file(ctx: RunContextWrapper[Any], path: str, content: str) -> str:
    """Write refactored code to a file.

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


@function_tool
def analyze_code_metrics(ctx: RunContextWrapper[Any], path: str) -> str:
    """Analyze basic code metrics for a file.

    Args:
        path: The path to the file to analyze.
    """
    try:
        full_path = Path(path).resolve()

        if not full_path.exists():
            return f"Error: File not found at {full_path}"

        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")
        total_lines = len(lines)
        non_empty_lines = len([line for line in lines if line.strip()])

        # Count imports
        import_lines = [line for line in lines if line.strip().startswith(("import ", "from "))]

        # Count functions and classes (basic)
        function_count = content.count("def ")
        class_count = content.count("class ")

        # Check line lengths
        long_lines = [i + 1 for i, line in enumerate(lines) if len(line) > 120]

        metrics = f"""Code Metrics for {full_path.name}:
- Total lines: {total_lines}
- Non-empty lines: {non_empty_lines}
- Import statements: {len(import_lines)}
- Functions: {function_count}
- Classes: {class_count}
- Lines exceeding 120 characters: {len(long_lines)}"""

        if long_lines[:5]:  # Show first 5 long lines
            metrics += f"\n- Long lines at: {', '.join(map(str, long_lines[:5]))}"
            if len(long_lines) > 5:
                metrics += f" (and {len(long_lines) - 5} more)"

        return metrics
    except Exception as e:
        return f"Error analyzing file: {str(e)}"


class CodeReviewerActor:
    """Agent that reviews and refactors code."""

    def __init__(self, model: str = "qwen-coder") -> None:
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
            {"name": "read_file", "description": "Read a code file for review"},
            {"name": "analyze_code_metrics", "description": "Analyze basic code metrics"},
            {"name": "write_file", "description": "Write refactored code to a file"},
        ]

        instructions = template.render(tools=tools_info)

        self.agent = Agent(
            name="CodeReviewer",
            model=model,
            instructions=instructions,
            tools=[read_file, analyze_code_metrics, write_file],
        )

    def run(self, query: str) -> Dict[str, Any]:
        """Execute the agent with the given query."""
        try:
            result = Runner.run_sync(self.agent, query)
            return {"response": result.final_output, "success": True}
        except Exception as e:
            return {"response": f"Error processing query: {str(e)}", "success": False}
