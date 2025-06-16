"""Streaming XML tool call detector."""

import logging
from typing import Dict, Optional, Tuple

from .parser import parse_xml_tool_call

logger = logging.getLogger(__name__)


class StreamingXMLDetector:
    """State machine to detect XML tool calls in streaming text.

    This detector buffers streaming tokens and watches for XML tool call patterns,
    emitting text and tool calls as they are detected.
    """

    def __init__(self):
        self.buffer = ""
        self.in_tool_call = False
        self.partial_tag = ""

    def process_token(self, token: str) -> Tuple[Optional[str], Optional[Dict]]:
        """Process a streaming token and return (text_to_emit, tool_call_if_complete).

        Args:
            token: The new token from the stream
        Returns:
            Tuple of (text_to_emit, tool_call) where text_to_emit is any text
            that should be shown to the user and tool_call is a parsed tool call
            if one was completed
        """
        if token is None:
            return None, None
        self.buffer += token
        if "<tool_call>" in self.buffer and "</tool_call>" in self.buffer:
            logger.info("XML detector found complete tool call!")
            start_idx = self.buffer.find("<tool_call>")
            end_idx = self.buffer.find("</tool_call>") + 12
            text_before = self.buffer[:start_idx]
            tool_xml = self.buffer[start_idx:end_idx]
            self.buffer = self.buffer[end_idx:]
            if text_before.strip() in ["```xml", "```", "```xml\n", "```\n"]:
                text_before = ""
                logger.debug("Stripped markdown wrapper before tool call")
            logger.info(f"Parsing tool XML: {repr(tool_xml)}")
            tool_call, _ = parse_xml_tool_call(tool_xml)
            if text_before:
                return text_before, tool_call
            else:
                return None, tool_call
        if (
            "<tool>" in self.buffer
            and "</tool>" in self.buffer
            and "<args>" in self.buffer
            and "</args>" in self.buffer
        ):
            tool_start = self.buffer.find("<tool>")
            tool_end = self.buffer.find("</tool>") + 7
            args_start = self.buffer.find("<args>", tool_end)
            args_end = self.buffer.find("</args>", args_start) + 7 if args_start >= 0 else -1
            if args_end > 0 and args_start - tool_end < 50:
                text_before = self.buffer[:tool_start]
                tool_xml = self.buffer[tool_start:args_end]
                self.buffer = self.buffer[args_end:]
                if text_before.strip() in ["```xml", "```", "```xml\n", "```\n"]:
                    text_before = ""
                    logger.debug("Stripped markdown wrapper before Qwen tool call")
                tool_call, _ = parse_xml_tool_call(tool_xml)
                if text_before:
                    return text_before, tool_call
                else:
                    return None, tool_call
        if self.buffer.endswith("<"):
            return None, None
        for partial in [
            "<t",
            "<to",
            "<too",
            "<tool",
            "<tool>",
            "<tool_",
            "<tool_c",
            "<tool_ca",
            "<tool_cal",
            "<tool_call",
        ]:
            if self.buffer.endswith(partial):
                return None, None
        if "<tool>" in self.buffer:
            tool_idx = self.buffer.find("<tool>")
            if tool_idx > 0:
                text_before = self.buffer[:tool_idx]
                self.buffer = self.buffer[tool_idx:]
                return text_before, None
            else:
                return None, None
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
