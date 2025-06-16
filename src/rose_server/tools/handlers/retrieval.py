"""Built-in retrieval tool handler."""

import json
import logging
from typing import Dict, Optional, Tuple

from rose_server import vector
from rose_server.embeddings.embedding import generate_embeddings

logger = logging.getLogger(__name__)


async def handle_retrieval_tool_call(assistant_id: str, query: str) -> str:
    """Handle the built-in retrieval tool call.

    Args:
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
        results = vector.query_vectors(
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


async def intercept_retrieval_tool_call(tool_call: Dict[str, any], assistant_id: str) -> Optional[Tuple[str, str]]:
    """Intercept and handle retrieval tool calls internally.

    Args:
        tool_call: The parsed tool call dict with 'tool' and 'arguments' keys
        assistant_id: The assistant ID to search documents for
    Returns:
        Tuple of (tool_name, result_text) if this was a retrieval tool,
        None if it should be passed through to the client
    """
    tool_name = tool_call.get("tool", "")
    if tool_name != "search_documents":
        return None
    try:
        args = tool_call.get("arguments", {})
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
