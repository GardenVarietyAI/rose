"""Built-in file search tool handler."""

import json
import logging
from typing import Any, Dict, Optional, Tuple

from rose_server.embeddings.embedding import generate_embeddings_async
from rose_server.vector_stores.chroma import Chroma

logger = logging.getLogger(__name__)


async def intercept_file_search_tool_call(
    chroma: Chroma, tool_call: Dict[str, Any], collection_name: str
) -> Optional[Tuple[str, str]]:
    """Intercept and handle file search tool calls internally.

    Args:
        chroma: The shared Chroma instance
        tool_call: The parsed tool call dict with 'tool' and 'arguments' keys
        collection_name: The collection to search documents for
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

        embedding_response = await generate_embeddings_async(texts=query)
        if not embedding_response["data"]:
            return (tool_name, "Error generating query embedding.")

        query_embedding = embedding_response["data"][0]["embedding"]
        results = chroma.query_vectors(
            collection_name=collection_name,
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])
        distances = results.get("distances", [[]])
        metadatas = results.get("metadatas", [[]])

        if documents and documents[0]:
            formatted_results = []
            for i, (doc, score) in enumerate(zip(documents[0], distances[0])):
                metadata = metadatas[0][i] if i < len(metadatas[0]) else {}
                filename = metadata.get("filename")
                if not filename:
                    result = "No filename given to file search tool."
                relevance = max(0.0, min(1.0, 1 - score))  # Clamp to [0,1]
                formatted_results.append(f"[Document {i + 1} - {filename} (relevance: {relevance:.2f})]:\n{doc}\n")
            result = "\n".join(formatted_results)
        else:
            result = "No relevant documents found for the query."

        logger.info(f"File search for assistant {collection_name} with query: {query}")
        return (tool_name, result)

    except Exception as e:
        logger.error(f"Error handling file search tool call: {str(e)}")
        return (tool_name, f"Error searching documents: {str(e)}")
