"""API router for vector stores endpoints."""

import logging
import time
import uuid
from typing import Any, Dict, List

from chromadb.utils import embedding_functions
from fastapi import APIRouter, Body, HTTPException, Path

from rose_server.schemas.vector_stores import (
    VectorSearch,
    VectorSearchResult,
    VectorStoreCreate,
    VectorStoreList,
    VectorStoreMetadata,
    VectorStoreUpdate,
)
from rose_server.vector_stores.deps import VectorManager
from rose_server.vector_stores.store import (
    VectorStoreNotFoundError,
    delete_vectors as delete_vectors_from_store,
    search_vectors,
)

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)
_META_EXCLUDE = {"display_name", "dimensions", "created_at"}


@router.get("/vector_stores")
async def index(vector: VectorManager) -> VectorStoreList:
    """List all vector stores."""
    try:
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
    except Exception as e:
        logger.error(f"Error listing vector stores: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing vector stores: {str(e)}")


@router.post("/vector_stores")
async def create(vector: VectorManager, request: VectorStoreCreate = Body(...)) -> VectorStoreMetadata:
    """Create a new vector store."""
    try:
        vid = f"vs_{uuid.uuid4().hex}"
        meta_in = request.metadata or {}

        meta = {
            **meta_in,
            "display_name": request.name,
            "created_at": int(time.time()),
        }
        # Create collection with ChromaDB's default embedding function
        vector.get_or_create_collection(
            vid, metadata=meta, embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
        logger.info("Created vector store %s (%s)", request.name, vid)
        public_meta = {k: v for k, v in meta.items() if k not in _META_EXCLUDE}

        # ChromaDB's default model uses 384 dimensions
        return VectorStoreMetadata(
            id=vid, name=request.name, dimensions=384, metadata=public_meta, created_at=meta["created_at"]
        )
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating vector store: {str(e)}")


@router.get("/vector_stores/{vector_store_id}")
async def get(
    vector: VectorManager, vector_store_id: str = Path(..., description="The ID of the vector store")
) -> VectorStoreMetadata:
    """Get a vector store by ID."""
    try:
        if vector_store_id not in vector.list_collections():
            raise HTTPException(status_code=404, detail=f"Vector store {vector_store_id} not found")

        col = vector.client.get_collection(vector_store_id)
        meta = col.metadata or {}

        return VectorStoreMetadata(
            id=vector_store_id,
            name=meta.get("display_name", vector_store_id),
            dimensions=meta.get("dimensions", 0),
            metadata={k: v for k, v in meta.items() if k not in _META_EXCLUDE},
            created_at=int(meta.get("created_at", time.time())),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting vector store: {str(e)}")


@router.post("/vector_stores/{vector_store_id}")
async def update(
    vector: VectorManager,
    vector_store_id: str = Path(..., description="The ID of the vector store"),
    request: VectorStoreUpdate = Body(...),
) -> VectorStoreMetadata:
    """Update a vector store."""
    try:
        if vector_store_id not in vector.list_collections():
            raise HTTPException(status_code=404, detail=f"Vector store {vector_store_id} not found")

        col = vector.client.get_collection(vector_store_id)
        meta = dict(col.metadata or {})

        if request.name:
            meta["display_name"] = request.name

        if request.metadata:
            meta.update(request.metadata)

        col.modify(metadata=meta)
        logger.info("Updated vector store %s", vector_store_id)

        return VectorStoreMetadata(
            id=vector_store_id,
            name=meta.get("display_name", vector_store_id),
            dimensions=meta.get("dimensions", 0),
            metadata={k: v for k, v in meta.items() if k not in _META_EXCLUDE},
            created_at=int(meta.get("created_at", time.time())),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating vector store: {str(e)}")


@router.delete("/vector_stores/{vector_store_id}")
async def delete(
    vector: VectorManager, vector_store_id: str = Path(..., description="The ID of the vector store")
) -> Dict[str, Any]:
    """Delete a vector store."""
    try:
        if vector_store_id not in vector.list_collections():
            raise HTTPException(status_code=404, detail=f"Vector store {vector_store_id} not found")

        vector.delete_collection(vector_store_id)
        logger.info("Deleted vector store %s", vector_store_id)

        return {"id": vector_store_id, "object": "vector_store.deleted", "deleted": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting vector store: {str(e)}")


@router.post("/vector_stores/{vector_store_id}/vectors/delete")
async def delete_vectors(
    vector: VectorManager,
    vector_store_id: str = Path(..., description="The ID of the vector store"),
    ids: List[str] = Body(..., embed=True),
) -> Dict[str, Any]:
    """Delete vectors from a vector store."""
    try:
        result = await delete_vectors_from_store(vector, vector_store_id, ids)
        return result
    except VectorStoreNotFoundError as e:
        logger.error(f"Vector store not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting vectors: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting vectors: {str(e)}")


@router.post("/vector_stores/{vector_store_id}/search")
async def search_vector_store(
    vector: VectorManager,
    vector_store_id: str = Path(..., description="The ID of the vector store"),
    request: VectorSearch = Body(...),
) -> VectorSearchResult:
    """Search for vectors in a vector store (OpenAI-compatible)."""
    try:
        result = await search_vectors(vector, vector_store_id, request)
        return result
    except VectorStoreNotFoundError as e:
        logger.error(f"Vector store not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error searching vectors: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching vectors: {str(e)}")
