"""Built-in file search tool handler."""

import json
import logging
from typing import Any, Dict, Optional, Tuple

from rose_server.embeddings.embedding import generate_embeddings
from rose_server.vector_stores.chroma import Chroma

logger = logging.getLogger(__name__)


async def handle_file_search_tool_call(chroma: Chroma, assistant_id: str, query: str) -> str:
    """Handle the built-in file search tool call.

    Args:
        chroma: The shared Chroma instance
        assistant_id: The assistant ID to search documents for
        query: The search query
    Returns:
        Formatted search results as a string
    """
    collection_name = f"assistant_{assistant_id}_docs"
    try:
        embedding_response = generate_embeddings(
            model="text-embedding-ada-002",
            input=[query],
        )
        if not embedding_response["data"]:
            return "Error generating query embedding."
        query_embedding = embedding_response["data"][0]["embedding"]
        results = chroma.query_vectors(
            collection_name=collection_name,
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas", "distances"],
        )
        if not results or not results.get("documents") or not results["documents"][0]:
            return "No relevant documents found for the query."
        formatted_results = []
        documents = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]
        for i, (doc, score) in enumerate(zip(documents, distances)):
            metadata = metadatas[i] if i < len(metadatas) else {}
            filename = metadata.get("filename", "Unknown")
            formatted_results.append(f"[Document {i + 1} - {filename} (relevance: {1 - score:.2f})]:\n{doc}\n")
        return "\n".join(formatted_results)
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        return f"Error searching documents: {str(e)}"


async def intercept_file_search_tool_call(
    chroma: Chroma, tool_call: Dict[str, Any], assistant_id: str
) -> Optional[Tuple[str, str]]:
    """Intercept and handle file search tool calls internally.

    Args:
        chroma: The shared Chroma instance
        tool_call: The parsed tool call dict with 'tool' and 'arguments' keys
        assistant_id: The assistant ID to search documents for
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
            return ("file_search", "Error: No query provided for file search")
        result = await handle_file_search_tool_call(chroma=chroma, assistant_id=assistant_id, query=query)
        logger.info(f"File search for assistant {assistant_id} with query: {query}")
        return (tool_name, result)
    except Exception as e:
        logger.error(f"Error handling file search tool call: {str(e)}")
        return (tool_name, f"Error searching documents: {str(e)}")
