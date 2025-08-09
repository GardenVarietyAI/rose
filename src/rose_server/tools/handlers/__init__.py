"""Built-in tool handlers."""

from rose_server.tools.handlers.file_search import intercept_file_search_tool_call

__all__ = [
    "intercept_file_search_tool_call",
]
