"""Unified tools system"""

from rose_server.tools.detector import StreamingXMLDetector
from rose_server.tools.formatter import format_function_output, format_tools_for_prompt, validate_tools
from rose_server.tools.handlers import handle_file_search_tool_call, intercept_file_search_tool_call
from rose_server.tools.parser import parse_xml_tool_call
from rose_server.tools.toolbox import BUILTIN_TOOLS, Tool, ToolCall, ToolFunction, ToolOutput

__all__ = [
    "Tool",
    "ToolCall",
    "ToolOutput",
    "ToolFunction",
    "BUILTIN_TOOLS",
    "parse_xml_tool_call",
    "format_tools_for_prompt",
    "format_function_output",
    "validate_tools",
    "StreamingXMLDetector",
    "handle_file_search_tool_call",
    "intercept_file_search_tool_call",
]
