"""Code analysis tool for actors."""

from pathlib import Path
from typing import Any

from agents import RunContextWrapper, function_tool


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
