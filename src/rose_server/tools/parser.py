import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Precompiled regex patterns
CODE_BLOCK_PATTERN = re.compile(r"```(?:xml)?\s*(.*?)\s*```", re.DOTALL)
XML_PATTERN_FULL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
TOOL_PATTERN = re.compile(r"<tool>(.*?)</tool>\s*<args>(.*?)</args>", re.DOTALL)


def _strip_markdown(text: str) -> str:
    """Strip markdown code blocks if present."""
    if "```" in text:
        match = CODE_BLOCK_PATTERN.search(text)
        if match:
            logger.info("Stripped markdown code blocks from tool call")
            return match.group(1)
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
    """Parse XML tool call from LLM response and return (tool_call, cleaned_reply).

    Expected XML format: <tool_call><tool>name</tool><args>...</args></tool_call>

    Args:
        reply: The LLM response containing potential XML tool calls
        available_tools: List of available tools for parameter validation
    Returns:
        Tuple of (tool_call_dict, cleaned_reply) where tool_call_dict contains
        the parsed tool information in a format compatible with the runs API
    """
    working_reply = _strip_markdown(reply)

    # Try to find a match in either format
    match = TOOL_PATTERN.search(working_reply)
    original_xml = match.group(0) if match else None

    if not match:
        # Check for wrapped format for backwards compatibility
        wrapped_match = XML_PATTERN_FULL.search(working_reply)
        if wrapped_match:
            inner_match = TOOL_PATTERN.search(wrapped_match.group(1))
            if inner_match:
                match = inner_match
                original_xml = wrapped_match.group(0)

    if not match:
        return None, reply

    tool_name = match.group(1).strip()
    args_content = match.group(2).strip()

    # Wrap in standard format for parsing
    xml_content = f"<tool_call><tool>{tool_name}</tool><args>{args_content}</args></tool_call>"
    logger.info(f"Found tool call: {tool_name}")

    # Check if there are multiple tool calls
    all_matches = XML_PATTERN_FULL.findall(working_reply)
    if len(all_matches) > 1:
        logger.warning(f"Found {len(all_matches)} tool calls in response - only processing the first one for safety")

    try:
        root = ET.fromstring(xml_content)
        tool_name = root.find("tool")
        args_element = root.find("args")

        if tool_name is None or args_element is None or tool_name.text is None:
            return None, reply

        args_dict = _parse_args(args_element, tool_name.text)

        parsed_call = {
            "tool": tool_name.text,
            "arguments": args_dict,
        }

        cleaned_reply = _strip_markdown(reply.replace(original_xml or "", "").strip())

        logger.info(f"Parsed XML tool call: {parsed_call}")
        logger.info(f"Remaining reply text after XML removal: '{cleaned_reply}'")
        return parsed_call, cleaned_reply
    except Exception as e:
        logger.warning(f"Failed to parse XML tool call: {e}")
        return None, reply
