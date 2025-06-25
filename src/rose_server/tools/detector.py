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
        # Check for complete tool call
        start_idx = self.buffer.find("<tool_call>")
        if start_idx != -1:
            end_idx = self.buffer.find("</tool_call>", start_idx)
            if end_idx != -1:
                logger.info("XML detector found complete tool call!")
                end_idx += 12  # Include closing tag
                text_before = self.buffer[:start_idx]
                tool_xml = self.buffer[start_idx:end_idx]
                self.buffer = self.buffer[end_idx:]
                # Strip common markdown wrappers
                stripped = text_before.strip()
                if stripped in ("```xml", "```") or stripped.startswith("```xml\n") or stripped == "```\n":
                    text_before = ""
                    logger.debug("Stripped markdown wrapper before tool call")
                logger.info(f"Parsing tool XML: {repr(tool_xml)}")
                tool_call, _ = parse_xml_tool_call(tool_xml)
                return (text_before if text_before else None), tool_call
        # Check if we're building up a potential tool call tag
        # More efficient to check once for '<' and then check prefixes
        if self.buffer.endswith("<"):
            return None, None
        if self.buffer.endswith(
            ("<t", "<to", "<too", "<tool", "<tool_", "<tool_c", "<tool_ca", "<tool_cal", "<tool_call")
        ):
            return None, None
        # No tool call pattern detected, emit buffer content
        if self.buffer:
            text = self.buffer
            self.buffer = ""
            return text, None
        return None, None

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
