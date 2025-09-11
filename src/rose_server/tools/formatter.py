"""Tool formatting for LLM prompts."""

import logging
from pathlib import Path
from typing import Any, List, Optional

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)
template_dir = Path(__file__).parent / "prompts"
jinja_env = Environment(loader=FileSystemLoader(str(template_dir)), trim_blocks=True, lstrip_blocks=True)


def _has_tool_results(messages: List[Any]) -> bool:
    """Check if conversation contains tool result messages."""
    for msg in messages:
        if hasattr(msg, "role") and msg.role == "tool":
            return True
        elif isinstance(msg, dict) and msg.get("role") == "tool":
            return True
    return False


def format_tools_for_prompt(
    tools: List[Any],
    messages: Optional[List[Any]] = None,
    assistant_id: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> str:
    """Format tools into XML prompt instructions for LLMs.

    Args:
        tools: List of tool definitions
        assistant_id: Optional assistant ID for context-specific instructions
        user_agent: User agent string to detect different client types
    Returns:
        Formatted prompt string with tool instructions
    """
    if not tools:
        return ""
    logger.info(f"Formatting {len(tools)} tools for prompt")
    tool_list = []
    for tool in tools:
        if hasattr(tool, "type"):
            tool_type = tool.type
        elif isinstance(tool, dict):
            tool_type = tool.get("type")
            # Handle OpenAI agents format (name, description, input_schema/params_json_schema)
            if not tool_type and "name" in tool and "description" in tool:
                tool_type = "function"
        else:
            continue
        if tool_type == "function":
            if isinstance(tool, dict):
                if "function" in tool:
                    func = tool["function"]
                    name = func.get("name", "")
                    description = func.get("description", "")
                    parameters = func.get("parameters", {})
                else:
                    name = tool.get("name", "")
                    description = tool.get("description", "")
                    parameters = tool.get("parameters", {})
            elif hasattr(tool, "name"):
                name = tool.name
                description = getattr(tool, "description", "")
                parameters = tool.parameters
            elif hasattr(tool, "function"):
                func = tool.function
                name = func.name
                description = func.description
                parameters = func.parameters
            else:
                continue
            tool_list.append(
                {"name": name, "description": description, "parameters": parameters if parameters is not None else {}}
            )
    if not tool_list:
        return ""

    # Choose appropriate instructions based on conversation state
    if messages and _has_tool_results(messages):
        template_name = "tool_response.jinja2"
        logger.info("Using tool response instructions - tool results detected")
    else:
        template_name = "tool_calling.jinja2"
        logger.info("Using tool calling template - no tool results detected")

    template = jinja_env.get_template(template_name)
    render_args = {
        "tools": tool_list,
    }

    rendered = template.render(**render_args)
    logger.debug(f"Rendered tool prompt:\n{rendered[:500]}...")  # Log first 500 chars
    return rendered


def format_function_output(output: str, exit_code: int = 0, model: str = "gpt-4") -> str:
    """Format function call output for the model to understand.

    Args:
        output: The output from the function call
        exit_code: Exit code of the command (0 = success)
        model: Model name for token counting
    Returns:
        Formatted string to add to the conversation
    """
    # Simple truncation for very large outputs
    if len(output) > 10000:
        output = output[:9500] + "\n... (output truncated)"
    template = jinja_env.get_template("function_output.jinja2")
    return template.render(output=output, exit_code=exit_code)
