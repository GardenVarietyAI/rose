import subprocess
import sys
from typing import Any, Dict, Optional, Tuple


async def intercept_code_interpreter_tool_call(
    tool_call: Dict[str, Any], assistant_id: Optional[str] = None
) -> Optional[Tuple[str, str]]:
    """Run Python code if this is a code_interpreter tool call."""
    if tool_call.get("tool") != "code_interpreter":
        return None

    code = tool_call.get("arguments", {}).get("code", "")
    if not code:
        return ("code_interpreter", "Error: No code provided")

    try:
        # Run the code in a subprocess
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)

        output = result.stdout
        if result.stderr:
            output += f"\n{result.stderr}"

        return ("code_interpreter", output or "Code executed successfully")

    except subprocess.TimeoutExpired:
        return ("code_interpreter", "Error: Code execution timed out")
    except Exception as e:
        return ("code_interpreter", f"Error: {str(e)}")
