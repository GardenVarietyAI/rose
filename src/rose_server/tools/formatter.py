"""Tool formatting for LLM prompts."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader
from openai.types.beta.file_search_tool import FileSearchTool
from openai.types.beta.function_tool import FunctionTool
from openai.types.shared_params.function_definition import FunctionDefinition

from rose_server.tools.toolbox import Tool

logger = logging.getLogger(__name__)
template_dir = Path(__file__).parent / "prompts"
jinja_env = Environment(loader=FileSystemLoader(str(template_dir)), trim_blocks=True, lstrip_blocks=True)


def format_tools_for_prompt(
    tools: List[Any], assistant_id: Optional[str] = None, user_agent: Optional[str] = None
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
        elif tool_type in ["retrieval", "file_search"]:
            tool_list.append(
                {
                    "name": "file_search",
                    "description": "Search through attached documents",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string", "description": "search query string"}},
                        "required": ["query"],
                    },
                }
            )
    if not tool_list:
        return ""

    template = jinja_env.get_template("tool_calling.jinja2")
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


def validate_tools(tools: List[Dict[str, Any]]) -> List[Tool]:
    """Validate and parse tool definitions.

    Args:
        tools: List of tool dictionaries from API requests
    Returns:
        List of validated Tool objects
    Raises:
        ValueError: If tool type is unknown
    """
    validated: List[Tool] = []
    for tool_dict in tools:
        tool_type = tool_dict.get("type")
        if tool_type == "function":
            validated.append(FunctionTool(**tool_dict))
        elif tool_type == "file_search":
            validated.append(FileSearchTool(type="file_search"))
        else:
            logger.warning(f"Unknown tool type: {tool_type}, treating as custom function tool")
            validated.append(
                FunctionTool(
                    type="function",
                    function=FunctionDefinition(
                        name=tool_type or "unknown",
                        description=f"Custom {tool_type} tool",
                        parameters={"type": "object", "properties": {}, "required": []},
                    ),
                )
            )
    return validated
