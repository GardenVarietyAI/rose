"""API router for vector stores endpoints."""

import logging
import time
from typing import Any, Dict, List

from fastapi import APIRouter, Body, HTTPException, Path, Request

from rose_server.config.settings import settings
from rose_server.embeddings.embedding import get_tokenizer
from rose_server.schemas.vector_stores import (
    VectorSearch,
    VectorSearchChunk,
    VectorSearchResult,
    VectorSearchUsage,
    VectorStoreCreate,
    VectorStoreList,
    VectorStoreMetadata,
    VectorStoreUpdate,
)
from rose_server.vector_stores.deps import VectorManager
from rose_server.vector_stores.files.router import router as files_router
from rose_server.vector_stores.store import (
    create_vector_store,
    get_vector_store,
    list_vector_stores,
    search_vector_store,
)

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)

router.include_router(files_router, prefix="/vector_stores/{vector_store_id}/files", tags=["vector_store_files"])

_META_EXCLUDE = {"display_name", "dimensions", "created_at"}
_INTERNAL_FIELDS = frozenset(["file_id", "filename", "total_chunks", "start_index", "end_index", "decode_errors"])


class VectorStoreNotFoundError(RuntimeError):
    pass


@router.get("/vector_stores")
async def index() -> VectorStoreList:
    """List all vector stores."""
    try:
        stores = await list_vector_stores()
        return VectorStoreList(
            data=[
                VectorStoreMetadata(
                    id=store.id,
                    name=store.name,
                    dimensions=store.dimensions,
                    metadata=store.meta or {},
                    created_at=store.created_at,
                )
                for store in stores
            ]
        )
    except Exception as e:
        logger.error(f"Error listing vector stores: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing vector stores: {str(e)}")


@router.post("/vector_stores")
async def create(request: VectorStoreCreate = Body(...)) -> VectorStoreMetadata:
    """Create a new vector store."""
    try:
        vector_store = await create_vector_store(request.name)
        logger.info("Created vector store %s (%s)", request.name, vector_store.id)

        return VectorStoreMetadata(
            id=vector_store.id,
            name=vector_store.name,
            dimensions=vector_store.dimensions,
            metadata=vector_store.meta or {},
            created_at=vector_store.created_at,
        )
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating vector store: {str(e)}")


@router.get("/vector_stores/{vector_store_id}")
async def get(vector_store_id: str = Path(..., description="The ID of the vector store")) -> VectorStoreMetadata:
    """Get a vector store by ID."""
    try:
        vector_store = await get_vector_store(vector_store_id)
        if not vector_store:
            raise HTTPException(status_code=404, detail=f"Vector store {vector_store_id} not found")

        return VectorStoreMetadata(
            id=vector_store.id,
            name=vector_store.name,
            dimensions=vector_store.dimensions,
            metadata=vector_store.meta or {},
            created_at=vector_store.created_at,
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
        if vector_store_id not in vector.list_collections():
            raise VectorStoreNotFoundError(vector_store_id)

        vector.client.get_collection(vector_store_id).delete(ids=ids)
        logger.info("Deleted %d vectors from %s", len(ids), vector_store_id)

        return {"object": "list", "data": [{"id": i, "object": "vector.deleted", "deleted": True} for i in ids]}
    except VectorStoreNotFoundError as e:
        logger.error(f"Vector store not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting vectors: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting vectors: {str(e)}")


@router.post("/vector_stores/{vector_store_id}/search")
async def search_store(
    http_request: Request,
    vector_store_id: str = Path(..., description="The ID of the vector store"),
    request: VectorSearch = Body(...),
) -> VectorSearchResult:
    """Search for vectors in a vector store (OpenAI-compatible)."""
    try:
        # Validate filters (not supported yet)
        if request.filters:
            raise HTTPException(
                status_code=400, detail="Filters are not supported yet. Coming soon in a future release."
            )

        # Check if vector store exists
        vector_store = await get_vector_store(vector_store_id)
        if not vector_store:
            raise HTTPException(status_code=404, detail=f"Vector store {vector_store_id} not found")

        # Calculate token usage for the query
        if isinstance(request.query, str):
            tokenizer = get_tokenizer(settings.default_embedding_model)
            prompt_tokens = len(tokenizer.encode(request.query))
            documents = await search_vector_store(vector_store_id, request.query, request.max_num_results + 1)
        elif isinstance(request.query, list):
            prompt_tokens = 1  # Vector queries use minimal tokens
            documents = await search_vector_store(vector_store_id, request.query, request.max_num_results + 1)
        else:
            prompt_tokens = 0
            documents = []

        # Convert documents to API response format
        search_chunks = []
        for doc in documents:
            # Extract file info from metadata
            meta = doc.document.meta or {}

            # Create attributes from metadata (excluding our internal fields)
            attributes = {k: v for k, v in meta.items() if k not in _INTERNAL_FIELDS}

            chunk = VectorSearchChunk(
                file_id=meta.get("file_id", ""),
                filename=meta.get("filename", ""),
                score=doc.score,  # Already converted to similarity (1 - distance)
                attributes=attributes,
                content=[{"type": "text", "text": doc.document.content}],
            )
            search_chunks.append(chunk)

        # Determine query string for response
        query_str = request.query if isinstance(request.query, str) else "[vector query]"

        # Calculate pagination fields
        first_id = documents[0].document.id if documents else None
        last_id = documents[-1].document.id if documents else None
        has_more = len(documents) > request.max_num_results
        # Trim results to requested limit
        documents = documents[: request.max_num_results]
        search_chunks = search_chunks[: request.max_num_results]

        # Generate next_page URL if there are more results
        next_page = None
        if has_more and last_id:
            base_url = str(http_request.base_url).rstrip("/")
            next_page = (
                f"{base_url}{router.prefix}/vector_stores/{vector_store_id}/search"
                f"?after={last_id}&limit={request.max_num_results}"
            )

        return VectorSearchResult(
            search_query=query_str,
            data=search_chunks,
            first_id=first_id,
            last_id=last_id,
            has_more=has_more,
            next_page=next_page,
            usage=VectorSearchUsage(
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens + len(search_chunks),  # Include processing overhead per result
            ),
        )
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Invalid search parameters: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid search parameters: {str(e)}")
    except Exception as e:
        logger.error(f"Error searching vectors: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching vectors: {str(e)}")
