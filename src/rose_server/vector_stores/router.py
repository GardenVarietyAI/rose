"""API router for vector stores endpoints."""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException, Path

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
from rose_server.vector_stores.files.router import router as files_router
from rose_server.vector_stores.files.store import (
    FileNotFoundError as StoreFileNotFoundError,
    VectorStoreNotFoundError as StoreVectorStoreNotFoundError,
    add_file_to_vector_store,
)
from rose_server.vector_stores.store import (
    create_vector_store,
    delete_vector_store,
    get_vector_store,
    list_vector_stores,
    search_vector_store,
    update_vector_store,
)

router = APIRouter(prefix="/v1/vector_stores")
logger = logging.getLogger(__name__)

router.include_router(files_router)

_INTERNAL_FIELDS = frozenset(["file_id", "filename", "total_chunks", "start_index", "end_index", "decode_errors"])


class VectorStoreNotFoundError(RuntimeError):
    pass


@router.get("")
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


@router.post("")
async def create(request: VectorStoreCreate = Body(...)) -> VectorStoreMetadata:
    """Create a new vector store."""
    try:
        vector_store = await create_vector_store(request.name)
        logger.info(f"Created vector store {request.name} ({vector_store.id})")

        # TODO: batch this operation
        if request.file_ids:
            for file_id in request.file_ids:
                try:
                    await add_file_to_vector_store(vector_store_id=vector_store.id, file_id=file_id)
                    logger.info(f"Added file {file_id} to vector store {request.name} ({vector_store.id})")
                except (StoreVectorStoreNotFoundError, StoreFileNotFoundError) as e:
                    raise HTTPException(status_code=404, detail=str(e))
                except Exception as e:
                    raise HTTPException(status_code=422, detail=f"Failed to process file {file_id}: {str(e)}")

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
        logger.error(f"Error creating vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating vector store: {str(e)}")


@router.get("/{vector_store_id}")
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


@router.post("/{vector_store_id}")
async def update(vector_store_id: str = Path(...), request: VectorStoreUpdate = Body(...)) -> VectorStoreMetadata:
    """Update a vector store."""
    try:
        vector_store = await update_vector_store(
            vector_store_id=vector_store_id,
            name=request.name,
            metadata=request.metadata,
        )

        if not vector_store:
            raise HTTPException(status_code=404, detail="VectorStore not found")

        logger.info(f"Updated vector store {vector_store_id}")

        return VectorStoreMetadata(
            id=vector_store_id,
            name=vector_store.name,
            dimensions=vector_store.dimensions,
            metadata=vector_store.meta,
            created_at=vector_store.created_at,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating vector store: {str(e)}")


@router.delete("/{vector_store_id}")
async def delete(
    vector_store_id: str = Path(..., description="The ID of the vector store"),  # noqa: ARG001
) -> Dict[str, Any]:
    """Delete a vector store."""
    success = await delete_vector_store(vector_store_id)
    if not success:
        raise HTTPException(status_code=404, detail="VectorStore not found")
    return {"id": vector_store_id, "object": "vector_store.deleted", "deleted": True}


@router.post("/{vector_store_id}/search")
async def search_store(vector_store_id: str = Path(...), request: VectorSearch = Body(...)) -> VectorSearchResult:
    """Search for vectors in a vector store (OpenAI-compatible)."""
    try:
        if request.filters:
            raise HTTPException(status_code=400, detail="Filters are not supported yet.")

        vector_store = await get_vector_store(vector_store_id)
        if not vector_store:
            raise HTTPException(status_code=404, detail=f"Vector store {vector_store_id} not found")

        # Calculate token usage for the query
        if isinstance(request.query, str):
            tokenizer = get_tokenizer(settings.default_embedding_model)
            prompt_tokens = len(tokenizer.encode(request.query).ids)
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
                similarity=doc.score,  # Already converted to similarity (1 - distance)
                attributes=attributes,
                content=[{"type": "text", "text": doc.document.content}],
            )
            search_chunks.append(chunk)

        # Determine query string for response
        query_str = request.query if isinstance(request.query, str) else "[vector query]"

        # Calculate pagination fields
        has_more = len(documents) > request.max_num_results
        # Trim both lists to requested limit
        documents = documents[: request.max_num_results]
        search_chunks = search_chunks[: request.max_num_results]
        first_id = documents[0].document.id if documents else None
        last_id = documents[-1].document.id if documents else None

        return VectorSearchResult(
            search_query=query_str,
            data=search_chunks,
            first_id=first_id,
            last_id=last_id,
            has_more=has_more,
            next_page=None,
            usage=VectorSearchUsage(
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens + len(search_chunks),
            ),
        )
    except HTTPException:
        raise
    except ValueError:
        logger.exception("Invalid search parameters")
        raise HTTPException(status_code=400, detail="Invalid search parameters")
    except Exception:
        logger.exception("Error searching vectors")
        raise HTTPException(status_code=500, detail="Error searching vectors")
