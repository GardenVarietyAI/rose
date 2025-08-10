"""Built-in file search tool handler."""

import json
import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


async def intercept_file_search_tool_call(tool_call: Dict[str, Any], assistant_id: str) -> Optional[Tuple[str, str]]:
    """Intercept and handle file search tool calls internally.

    Args:
        tool_call: The parsed tool call dict with 'tool' and 'arguments' keys
        assistant_id: The assistant ID for scoping file search
    Returns:
        Tuple of (tool_name, result_text) if this was a file search tool,
        None if it should be passed through to the client
    """
    tool_name = tool_call.get("tool", "")
    if tool_name != "file_search":
        return None

    try:
        args = tool_call.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)

        query = args.get("query", "")
        if not query:
            return (tool_name, "Error: No query provided for file search")

        logger.warning("File search tool called but ChromaDB support has been removed")
        result = "File search is currently unavailable. ChromaDB support has been removed."

        logger.info(f"File search for assistant {assistant_id} with query: {query}")
        return (tool_name, result)

    except Exception as e:
        logger.error(f"Error handling file search tool call: {str(e)}")
        return (tool_name, f"Error searching documents: {str(e)}")
