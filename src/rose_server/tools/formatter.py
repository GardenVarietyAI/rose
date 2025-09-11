"""Tool formatting for LLM prompts."""

import logging
from pathlib import Path
from typing import Any, List, Optional

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)
template_dir = Path(__file__).parent / "prompts"
jinja_env = Environment(loader=FileSystemLoader(str(template_dir)), trim_blocks=True, lstrip_blocks=True)


def _most_recent_is_tool_result(messages: List[Any]) -> bool:
    """Check if the most recent message is a tool result."""
    if not messages:
        return False

    last_msg = messages[-1]
    if hasattr(last_msg, "role") and last_msg.role == "tool":
        return True
    elif isinstance(last_msg, dict) and last_msg.get("role") == "tool":
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

    if messages and _most_recent_is_tool_result(messages):
        template_name = "tool_response.jinja2"
        logger.info("Using tool response instructions")
    else:
        template_name = "tool_calling.jinja2"
        logger.info("Using tool calling template")

    template = jinja_env.get_template(template_name)
    render_args = {
        "tools": tool_list,
    }

    rendered = template.render(**render_args)
    return rendered
