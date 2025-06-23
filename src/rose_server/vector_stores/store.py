"""Vector operations within stores."""

import logging
from typing import Any, Dict, List

from rose_server.schemas.vector_stores import (
    Vector,
    VectorSearch,
    VectorSearchResult,
)
from rose_server.vector import ChromaDBManager

logger = logging.getLogger(__name__)


class VectorStoreNotFoundError(RuntimeError):
    pass


async def delete_vectors(vector: ChromaDBManager, vid: str, ids: List[str]) -> Dict[str, Any]:
    if vid not in vector.list_collections():
        raise VectorStoreNotFoundError(vid)

    vector.client.get_collection(vid).delete(ids=ids)
    logger.info("Deleted %d vectors from %s", len(ids), vid)

    return {"object": "list", "data": [{"id": i, "object": "vector.deleted", "deleted": True} for i in ids]}


async def search_vectors(vector: ChromaDBManager, vid: str, s: VectorSearch) -> VectorSearchResult:
    if vid not in vector.list_collections():
        raise VectorStoreNotFoundError(vid)

    col = vector.client.get_collection(vid)

    # Use ChromaDB's built-in embedding for text queries
    if isinstance(s.query, str):
        res = col.query(
            query_texts=[s.query],
            n_results=s.max_num_results,
            where=s.filters,
            include=["metadatas", "embeddings", "distances"],
        )
        # Estimate usage for API compatibility
        usage = {"prompt_tokens": len(s.query.split()), "total_tokens": len(s.query.split())}
    else:
        # Direct vector query
        res = col.query(
            query_embeddings=[s.query],
            n_results=s.max_num_results,
            where=s.filters,
            include=["metadatas", "embeddings", "distances"],
        )
        usage = {"prompt_tokens": 0, "total_tokens": 0}

    out = []
    for i, vec_id in enumerate(res["ids"][0]):
        meta = (res.get("metadatas") or [[]])[0][i] or {}
        vec = Vector(id=vec_id, metadata=meta)
        if s.include_values and "embeddings" in res:
            vec.values = res["embeddings"][0][i]
        out.append(vec)

    return VectorSearchResult(data=out, usage=usage)
