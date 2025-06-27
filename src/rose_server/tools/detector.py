"""Streaming XML tool call detector."""

import logging
from typing import Any, Dict, Optional, Tuple

from .parser import parse_xml_tool_call

logger = logging.getLogger(__name__)


class StreamingXMLDetector:
    """State machine to detect XML tool calls in streaming text.

    This detector buffers streaming tokens and watches for XML tool call patterns,
    emitting text and tool calls as they are detected.
    """

    def __init__(self):
        self.buffer = ""
        # Remove unused state variables

    def process_token(self, token: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Process a streaming token and return (text_to_emit, tool_call_if_complete).

        Args:
            token: The new token from the stream
        Returns:
            Tuple of (text_to_emit, tool_call) where text_to_emit is any text
            that should be shown to the user and tool_call is a parsed tool call
            if one was completed
        """
        if not token:
            return None, None

        self.buffer += token

        # Check for complete tool call - look for <tool> followed by </args>
        tool_start = self.buffer.find("<tool>")
        if tool_start != -1:
            # Look for the closing </args> tag which ends a tool call
            args_end = self.buffer.find("</args>", tool_start)
            if args_end != -1:
                logger.info("XML detector found complete tool call!")
                args_end += 7  # Include closing tag
                text_before = self.buffer[:tool_start]
                tool_xml = self.buffer[tool_start:args_end]
                self.buffer = self.buffer[args_end:]

                # Strip common markdown wrappers
                stripped = text_before.strip()
                if stripped in ("```xml", "```") or stripped.startswith("```xml\n") or stripped == "```\n":
                    text_before = ""
                    logger.debug("Stripped markdown wrapper before tool call")

                logger.info(f"Parsing tool XML: {repr(tool_xml)}")
                tool_call, _ = parse_xml_tool_call(tool_xml)
                return (text_before if text_before else None), tool_call
            else:
                # We have <tool> but no </args> yet, keep buffering
                return None, None

        # If buffer could be start of a tool call, keep buffering
        # Check if buffer ends with a partial tag that could become <tool> or <args>
        for partial in ["<", "<t", "<to", "<too", "<tool", "<a", "<ar", "<arg", "<args"]:
            if self.buffer.endswith(partial):
                return None, None

        # If no tool pattern and no partial tags, emit the buffer
        text = self.buffer
        self.buffer = ""
        return text, None

    def flush(self) -> Optional[str]:
        """Flush any remaining buffer content.

        Returns:
            Any remaining text in the buffer
        """
        if self.buffer:
            text = self.buffer
            self.buffer = ""
            return text
        return None
