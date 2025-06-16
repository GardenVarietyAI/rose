"""Handle retrieval tool calls for assistants."""
import json
import logging
from typing import Dict, Optional, Tuple
from rose_server.tools import handle_retrieval_tool_call
logger = logging.getLogger(__name__)

async def intercept_retrieval_tool_call(tool_call: Dict[str, any], assistant_id: str) -> Optional[Tuple[str, str]]:
    """
    Intercept and handle retrieval tool calls internally.

    Args:
        tool_call: The parsed tool call dict with 'tool' and 'args' keys
        assistant_id: The assistant ID to search documents for
    Returns:
        Tuple of (tool_name, result_text) if this was a retrieval tool,
        None if it should be passed through to the client
    """
    tool_name = tool_call.get("tool", "")
    if tool_name != "search_documents":
        return None
    try:
        args = tool_call.get("args", {})
        if isinstance(args, str):
            args = json.loads(args)
        query = args.get("query", "")
        if not query:
            return ("search_documents", "Error: No query provided for document search")
        result = await handle_retrieval_tool_call(assistant_id=assistant_id, query=query)
        logger.info(f"Retrieval search for assistant {assistant_id} with query: {query}")
        return ("search_documents", result)
    except Exception as e:
        logger.error(f"Error handling retrieval tool call: {str(e)}")
        return ("search_documents", f"Error searching documents: {str(e)}")