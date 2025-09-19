import json
from typing import Any, Dict, List, Optional


def format_tools_for_system_prompt(tools: Optional[List[Dict[str, Any]]]) -> str:
    """Format tools into Hermes-style system prompt."""

    if not tools:
        return ""

    formatted_tools: List[Dict[str, Any]] = []

    for tool in tools:
        if tool.get("type") != "function":
            continue

        if "function" in tool and isinstance(tool["function"], dict):
            fn = tool["function"]
            name = fn.get("name")
            description = fn.get("description", "")
            parameters = fn.get("parameters", {})
        else:
            name = tool.get("name")
            description = tool.get("description", "")
            parameters = tool.get("parameters", {})

        if not name:
            continue

        formatted_tools.append(
            {
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        )

    if not formatted_tools:
        return ""

    prompt = (
        "You have access to the following functions:\n\n"
        "<tools>"
        f"{json.dumps(formatted_tools, indent=2)}\n\n"
        "</tools>"
        "To use a function, respond ONLY with a tool call block, no additional prose before or after.\n"
        "Format strictly as follows:\n"
        "<tool_call>\n"
        '{"name": "function_name", "arguments": {}}\n'
        "</tool_call>\n"
    )

    return prompt
