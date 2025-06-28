"""Unified tools system"""

from .detector import StreamingXMLDetector
from .formatter import format_function_output, format_tools_for_prompt, validate_tools
from .handlers import handle_file_search_tool_call, intercept_file_search_tool_call
from .parser import parse_xml_tool_call
from .toolbox import BUILTIN_TOOLS, Tool, ToolCall, ToolFunction, ToolOutput

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
