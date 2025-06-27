"""Web search tool handler using DuckDuckGo."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

# Limit results for performance and token usage
MAX_SEARCH_RESULTS = 5
MAX_RESULT_LENGTH = 500  # Characters per result snippet


async def handle_web_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> List[Dict[str, Any]]:
    """Perform a web search using DuckDuckGo.

    Args:
        query: The search query
        max_results: Maximum number of results to return

    Returns:
        List of search results with title, url, and body
    """
    try:

        def _search() -> List[Dict[str, Any]]:
            results: List[Dict[str, Any]] = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    result = {
                        "title": r.get("title", ""),
                        "url": r.get("link", ""),
                        "snippet": r.get("body", "")[:MAX_RESULT_LENGTH],
                    }
                    results.append(result)
                    if len(results) >= max_results:
                        break
            return results

        # Run in default executor (for I/O operations)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, _search)

        return results
    except Exception as e:
        logger.error(f"Error performing web search: {str(e)}")
        # Fallback: return a message about the error
        return [
            {
                "title": "Search Error",
                "url": "",
                "snippet": f"Unable to perform web search at this time. Error: {str(e)}",
            }
        ]


async def intercept_web_search_tool_call(
    tool_call: Dict[str, Any], assistant_id: Optional[str] = None
) -> Optional[Tuple[str, str]]:
    """Intercept and handle web search tool calls.

    Args:
        tool_call: The parsed tool call dict with 'tool' and 'arguments' keys
        assistant_id: Optional assistant ID (not used for web search)

    Returns:
        Tuple of (tool_name, result_text) if this was a web search tool,
        None if it should be passed through to the client
    """
    tool_name = tool_call.get("tool", "")
    if tool_name != "web_search":
        return None

    try:
        args = tool_call.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)

        query = args.get("query", "")
        if not query:
            return ("web_search", "Error: No query provided for web search")

        # Perform the search
        results = await handle_web_search(query)

        if not results:
            return ("web_search", "No search results found.")

        # Format results for the model
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(f"[Result {i}] {result['title']}\nURL: {result['url']}\n{result['snippet']}\n")

        result_text = "\n".join(formatted_results)
        logger.info(f"Web search for query: {query} returned {len(results)} results")

        return ("web_search", result_text)

    except Exception as e:
        logger.error(f"Error handling web search tool call: {str(e)}")
        return ("web_search", f"Error performing web search: {str(e)}")
