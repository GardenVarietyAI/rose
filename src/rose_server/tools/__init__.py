"""Unified tools system"""

from rose_server.tools.detector import StreamingXMLDetector
from rose_server.tools.formatter import format_function_output, format_tools_for_prompt
from rose_server.tools.parser import parse_tool_call

__all__ = [
    "parse_tool_call",
    "format_tools_for_prompt",
    "format_function_output",
    "StreamingXMLDetector",
]
