import logging
import time
import uuid
from typing import Any, Dict, List

from chromadb.utils import embedding_functions

from rose_server.schemas.vector_stores import (
    Vector,
    VectorSearch,
    VectorSearchResult,
    VectorStoreCreate,
    VectorStoreList,
    VectorStoreMetadata,
    VectorStoreUpdate,
)
from rose_server.vector import vector

logger = logging.getLogger(__name__)
_META_EXCLUDE = {"display_name", "dimensions", "created_at"}


class VectorStoreNotFoundError(RuntimeError):
    pass


async def list_vector_stores() -> VectorStoreList:
    stores = []
    for name in vector.list_collections():
        try:
            meta = vector.get_collection_info(name).get("metadata", {})
            stores.append(
                VectorStoreMetadata(
                    id=name,
                    name=meta.get("display_name", name),
                    dimensions=meta.get("dimensions", 0),
                    metadata=meta,
                    created_at=int(meta.get("created_at", time.time())),
                )
            )
        except Exception as exc:
            logger.warning("Skip collection %s: %s", name, exc)

    return VectorStoreList(data=stores)


async def create_vector_store(p: VectorStoreCreate) -> VectorStoreMetadata:
    vid = f"vs_{uuid.uuid4().hex}"
    meta_in = p.metadata or {}

    meta = {
        **meta_in,
        "display_name": p.name,
        "created_at": int(time.time()),
    }
    # Create collection with ChromaDB's default embedding function
    vector.get_or_create_collection(
        vid, metadata=meta, embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )
    logger.info("Created vector store %s (%s)", p.name, vid)
    public_meta = {k: v for k, v in meta.items() if k not in _META_EXCLUDE}

    # ChromaDB's default model uses 384 dimensions
    return VectorStoreMetadata(id=vid, name=p.name, dimensions=384, metadata=public_meta, created_at=meta["created_at"])


async def get_vector_store(vid: str) -> VectorStoreMetadata:
    if vid not in vector.list_collections():
        raise VectorStoreNotFoundError(vid)

    col = vector.client.get_collection(vid)
    meta = col.metadata or {}

    return VectorStoreMetadata(
        id=vid,
        name=meta.get("display_name", vid),
        dimensions=meta.get("dimensions", 0),
        metadata={k: v for k, v in meta.items() if k not in _META_EXCLUDE},
        created_at=int(meta.get("created_at", time.time())),
    )


async def update_vector_store(vid: str, p: VectorStoreUpdate) -> VectorStoreMetadata:
    if vid not in vector.list_collections():
        raise VectorStoreNotFoundError(vid)

    col = vector.client.get_collection(vid)
    meta = dict(col.metadata or {})

    if p.name:
        meta["display_name"] = p.name

    if p.metadata:
        meta.update(p.metadata)

    col.modify(metadata=meta)
    logger.info("Updated vector store %s", vid)

    return VectorStoreMetadata(
        id=vid,
        name=meta.get("display_name", vid),
        dimensions=meta.get("dimensions", 0),
        metadata={k: v for k, v in meta.items() if k not in _META_EXCLUDE},
        created_at=int(meta.get("created_at", time.time())),
    )


async def delete_vector_store(vid: str) -> Dict[str, Any]:
    if vid not in vector.list_collections():
        raise VectorStoreNotFoundError(vid)

    vector.delete_collection(vid)
    logger.info("Deleted vector store %s", vid)

    return {"id": vid, "object": "vector_store.deleted", "deleted": True}


async def delete_vectors(vid: str, ids: List[str]) -> Dict[str, Any]:
    if vid not in vector.list_collections():
        raise VectorStoreNotFoundError(vid)

    vector.client.get_collection(vid).delete(ids=ids)
    logger.info("Deleted %d vectors from %s", len(ids), vid)

    return {
        "object": "list",
        "data": [
            {
                "id": i,
                "object": "vector.deleted",
                "deleted": True,
            }
            for i in ids
        ],
    }


async def search_vectors(vid: str, s: VectorSearch) -> VectorSearchResult:
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
