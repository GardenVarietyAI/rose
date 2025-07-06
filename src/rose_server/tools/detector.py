"""Streaming XML tool call detector."""

import logging
from typing import Any, Dict, Optional, Tuple

from rose_server.tools.parser import parse_xml_tool_call

logger = logging.getLogger(__name__)


class StreamingXMLDetector:
    """State machine to detect XML tool calls in streaming text.

    This detector buffers streaming tokens and watches for XML tool call patterns,
    emitting text and tool calls as they are detected.
    """

    def __init__(self):
        self.buffer = ""
        self.max_buffer_size = 8192  # 8KB max buffer to prevent infinite buffering
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

        # Check buffer size limit to prevent infinite buffering
        if len(self.buffer) > self.max_buffer_size:
            logger.warning(f"Buffer exceeded max size ({self.max_buffer_size} bytes), flushing")
            text = self.buffer
            self.buffer = ""
            return text, None

        # Check for complete tool call - look for <tool_call> followed by </tool_call>
        tool_start = self.buffer.find("<tool_call>")
        if tool_start != -1:
            # Look for the closing </tool_call> tag which ends a tool call
            tool_end = self.buffer.find("</tool_call>", tool_start)
            if tool_end != -1:
                logger.info("XML detector found complete tool call!")
                tool_end += 12  # Include closing tag
                text_before = self.buffer[:tool_start]
                tool_xml = self.buffer[tool_start:tool_end]
                self.buffer = self.buffer[tool_end:]

                # Strip common markdown wrappers
                stripped = text_before.strip()
                if stripped in ("```xml", "```") or stripped.startswith("```xml\n") or stripped == "```\n":
                    text_before = ""
                    logger.debug("Stripped markdown wrapper before tool call")

                logger.info(f"Parsing tool XML: {repr(tool_xml)}")
                tool_call, _ = parse_xml_tool_call(tool_xml)
                return (text_before if text_before else None), tool_call
            else:
                # We have <tool_call> but no </tool_call> yet, keep buffering
                return None, None

        # If buffer could be start of a tool call, keep buffering
        # Check if buffer ends with a partial tag that could become <tool_call>
        for partial in ["<", "<t", "<to", "<too", "<tool", "<tool_", "<tool_c", "<tool_ca", "<tool_cal", "<tool_call"]:
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
