"""Tool formatting for LLM prompts."""

import logging
from pathlib import Path
from typing import List, Optional

from jinja2 import Environment, FileSystemLoader
from openai.types.beta.code_interpreter_tool import CodeInterpreterTool
from openai.types.beta.file_search_tool import FileSearchTool
from openai.types.beta.function_tool import FunctionTool

from rose_core.config.service import MAX_TOOL_OUTPUT_TOKENS

from .chunker import chunk_tool_output
from .toolbox import BUILTIN_TOOLS, Tool

logger = logging.getLogger(__name__)
template_dir = Path(__file__).parent / "prompts"
jinja_env = Environment(loader=FileSystemLoader(str(template_dir)), trim_blocks=True, lstrip_blocks=True)


def format_tools_for_prompt(tools: List, assistant_id: Optional[str] = None, user_agent: Optional[str] = None) -> str:
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
    has_retrieval = False
    for tool in tools:
        if hasattr(tool, "type"):
            tool_type = tool.type
        elif isinstance(tool, dict):
            tool_type = tool.get("type")
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
            has_retrieval = True
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
        elif tool_type == "code_interpreter":
            tool_list.append(
                {
                    "name": "code_interpreter",
                    "description": "Execute Python code to solve problems",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string", "description": "Python code to execute"}},
                        "required": ["code"],
                    },
                }
            )
        elif tool_type == "web_search":
            tool_list.append(
                {
                    "name": "web_search",
                    "description": "Search the web for current information",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string", "description": "Search query"}},
                        "required": ["query"],
                    },
                }
            )
    if not tool_list:
        return ""
    is_agents_sdk = user_agent and "Agents/Python" in user_agent
    has_shell_tools = any(tool["name"] in ["shell", "bash", "execute"] for tool in tool_list)
    if is_agents_sdk:
        template_name = "agent_tool_instructions.jinja2"
    elif has_shell_tools:
        template_name = "tool_instructions.jinja2"
    else:
        template_name = "function_tools.jinja2"
    template = jinja_env.get_template(template_name)
    render_args = {
        "tools": tool_list,
        "has_retrieval": has_retrieval,
        "assistant_id": assistant_id,
    }
    if template_name in ["tool_instructions.jinja2", "agent_tool_instructions.jinja2"]:
        render_args["example_tool"] = "shell"
        render_args["example_command"] = "cat README.md"

    rendered = template.render(**render_args)
    logger.info(f"Using template: {template_name}")
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
    if len(output) > 10000:
        chunked_output, was_truncated = chunk_tool_output(output, MAX_TOOL_OUTPUT_TOKENS, model)
        if was_truncated:
            output = chunked_output
    template = jinja_env.get_template("function_output.jinja2")
    return template.render(output=output, exit_code=exit_code)


def validate_tools(tools: List[dict]) -> List[Tool]:
    """Validate and parse tool definitions.

    Args:
        tools: List of tool dictionaries from API requests
    Returns:
        List of validated Tool objects
    Raises:
        ValueError: If tool type is unknown
    """
    validated = []
    for tool_dict in tools:
        tool_type = tool_dict.get("type")
        if tool_type == "function":
            validated.append(FunctionTool(**tool_dict))
        elif tool_type == "code_interpreter":
            validated.append(CodeInterpreterTool(type="code_interpreter"))
        elif tool_type == "file_search":
            validated.append(FileSearchTool(type="file_search"))
        elif tool_type in BUILTIN_TOOLS:
            validated.append(
                FunctionTool(
                    type="function",
                    function={"name": tool_type, "description": f"Built-in {tool_type} tool", "parameters": {}},
                )
            )
        else:
            raise ValueError(f"Unknown tool type: {tool_type}")
    return validated
