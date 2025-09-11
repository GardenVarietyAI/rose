import ast
import json
import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Precompiled regex patterns
CODE_BLOCK_PATTERN = re.compile(r"```(?:xml|json)?\s*(.*?)\s*```", re.DOTALL)
TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _strip_markdown(text: str) -> str:
    """Strip markdown code blocks if present."""
    # First try triple backticks
    if "```" in text:
        match = CODE_BLOCK_PATTERN.search(text)
        if match:
            logger.info("Stripped markdown code blocks from tool call")
            return match.group(1)

    # Also strip single backticks around the entire content
    if text.startswith("`") and text.endswith("`"):
        logger.info("Stripped single backticks from tool call")
        return text[1:-1]

    return text


def _parse_args(args_element: ET.Element, tool_name: str) -> Dict[str, Any]:
    """Parse arguments from XML element."""
    args_dict: Dict[str, Any] = {}

    # Parse all child elements
    for child in args_element:
        if child.text:
            args_dict[child.tag] = child.text

    return args_dict


def parse_xml_tool_call(
    reply: str, available_tools: Optional[List[Any]] = None
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Expected format: <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>
    """
    working_reply = _strip_markdown(reply)

    # Look for tool_call tags
    match = TOOL_CALL_PATTERN.search(working_reply)
    if not match:
        return None, reply

    original_call = match.group(0)
    inner_content = match.group(1).strip()

    # Check if there are multiple tool calls
    all_matches = TOOL_CALL_PATTERN.findall(working_reply)
    if len(all_matches) > 1:
        logger.warning(f"Found {len(all_matches)} tool calls in response, only processing the first...")

    # Python dict parsing
    try:
        tool_call_data = ast.literal_eval(inner_content)
        if isinstance(tool_call_data, dict) and "name" in tool_call_data:
            parsed_call = {
                "tool": tool_call_data["name"],
                "arguments": tool_call_data.get("arguments", {}),
            }

            cleaned_reply = _strip_markdown(reply.replace(original_call, "").strip())

            logger.info(f"Parsed dict tool call: {parsed_call}")
            logger.info(f"Remaining reply text after tool call removal: '{cleaned_reply}'")
            return parsed_call, cleaned_reply

    except Exception:
        pass

    # JSON parsing
    try:
        tool_call_data = json.loads(inner_content)
        if isinstance(tool_call_data, dict) and "name" in tool_call_data:
            parsed_call = {
                "tool": tool_call_data["name"],
                "arguments": tool_call_data.get("arguments", {}),
            }

            cleaned_reply = _strip_markdown(reply.replace(original_call, "").strip())

            logger.info(f"Parsed JSON tool call: {parsed_call}")
            logger.info(f"Remaining reply text after tool call removal: '{cleaned_reply}'")
            return parsed_call, cleaned_reply

    except Exception:
        pass

    logger.info("Failed to parse as both dict and JSON")
    return None, reply
