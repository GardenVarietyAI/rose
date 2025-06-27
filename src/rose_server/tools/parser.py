import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Precompiled regex patterns
CODE_BLOCK_PATTERN = re.compile(r"```(?:xml)?\s*(.*?)\s*```", re.DOTALL)
XML_PATTERN_FULL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


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
    working_reply = reply
    if "```" in working_reply:
        match = CODE_BLOCK_PATTERN.search(working_reply)
        if match:
            working_reply = match.group(1)
            logger.info("Stripped markdown code blocks from tool call")
    # Look for tool calls in the simplified format: <tool>name</tool><args>...</args>
    tool_pattern = re.compile(r"<tool>(.*?)</tool>\s*<args>(.*?)</args>", re.DOTALL)
    match = tool_pattern.search(working_reply)

    if not match:
        # Also check for wrapped format for backwards compatibility
        wrapped_match = XML_PATTERN_FULL.search(working_reply)
        if wrapped_match:
            # Extract and re-parse the wrapped content
            inner_content = wrapped_match.group(1)
            inner_match = tool_pattern.search(inner_content)
            if inner_match:
                match = inner_match
                original_xml = wrapped_match.group(0)
            else:
                return None, reply
        else:
            return None, reply
    else:
        original_xml = match.group(0)

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
        if tool_name is None or args_element is None:
            return None, reply
        args_dict: Dict[str, Any] = {}
        if tool_name.text == "shell":
            commands: List[str] = []
            for cmd in args_element.findall("command"):
                if cmd.text:
                    commands.extend(cmd.text.split())
            if commands:
                args_dict["command"] = commands
            for child in args_element:
                if child.tag != "command" and child.text:
                    args_dict[child.tag] = child.text
        else:
            for child in args_element:
                if child.text:
                    args_dict[child.tag] = child.text
        parsed_call = {"tool": tool_name.text, "arguments": args_dict}
        cleaned_reply = reply.replace(original_xml, "").strip()
        if cleaned_reply.startswith("```xml") and cleaned_reply.endswith("```"):
            cleaned_reply = cleaned_reply[6:-3].strip()
        elif cleaned_reply.startswith("```") and cleaned_reply.endswith("```"):
            cleaned_reply = cleaned_reply[3:-3].strip()
        logger.info(f"Parsed XML tool call: {parsed_call}")
        logger.info(f"Remaining reply text after XML removal: '{cleaned_reply}'")
        return parsed_call, cleaned_reply
    except Exception as e:
        logger.warning(f"Failed to parse XML tool call: {e}")
        return None, reply
