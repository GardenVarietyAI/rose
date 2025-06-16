"""XML tool call parser for LLM responses."""
import logging
import re
import xml.etree.ElementTree as ET
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

def parse_xml_tool_call(reply: str, available_tools: Optional[list] = None) -> Tuple[Optional[Dict], str]:
    """Parse XML tool call from LLM response and return (tool_call, cleaned_reply).

    Supports two XML formats:
    1. Standard: <tool_call><tool>name</tool><args>...</args></tool_call>
    2. Qwen: <tool>name</tool><args>...</args>
    Args:
        reply: The LLM response containing potential XML tool calls
        available_tools: List of available tools for parameter validation
    Returns:
        Tuple of (tool_call_dict, cleaned_reply) where tool_call_dict contains
        the parsed tool information in a format compatible with the runs API
    """
    working_reply = reply
    if "```" in working_reply:
        code_block_pattern = r"```(?:xml)?\s*(.*?)\s*```"
        match = re.search(code_block_pattern, working_reply, re.DOTALL)
        if match:
            working_reply = match.group(1)
            logger.info("Stripped markdown code blocks from tool call")
    xml_pattern_full = r"<tool_call>(.*?)</tool_call>"
    tool_pattern = r"<tool>(.*?)</tool>"
    args_pattern = r"<args>(.*?)</args>"
    match = re.search(xml_pattern_full, working_reply, re.DOTALL)
    xml_content = None
    original_xml = None
    if match:
        xml_content = f"<tool_call>{match.group(1)}</tool_call>"
        original_xml = xml_content
        logger.info("Found standard <tool_call> format")
    else:
        tool_match = re.search(tool_pattern, working_reply, re.DOTALL)
        if tool_match:
            tool_end = tool_match.end()
            remaining = working_reply[tool_end:]
            args_match = re.search(args_pattern, remaining, re.DOTALL)
            if args_match and args_match.start() < 50:
                tool_name = tool_match.group(1).strip()
                args_content = args_match.group(1).strip()
                xml_content = f"<tool_call><tool>{tool_name}</tool><args>{args_content}</args></tool_call>"
                logger.info(f"Found Qwen format (no wrapper), converted to tool_call: {tool_name}")
                tool_start = tool_match.start()
                args_end = tool_end + args_match.end()
                original_xml = working_reply[tool_start:args_end]
            else:
                return None, reply
        else:
            return None, reply
    if not xml_content:
        return None, reply
    all_matches_full = re.findall(xml_pattern_full, working_reply, re.DOTALL)
    all_matches_partial = len(re.findall(tool_pattern, working_reply, re.DOTALL))
    total_matches = len(all_matches_full) + all_matches_partial
    if total_matches > 1:
        logger.warning(f"Found {total_matches} tool calls in response - only processing the first one for safety")
    try:
        root = ET.fromstring(xml_content)
        tool_name = root.find("tool")
        args_element = root.find("args")
        if tool_name is None or args_element is None:
            return None, reply
        args_dict = {}
        if tool_name.text == "shell":
            commands = []
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