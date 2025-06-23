"""API router for vector stores endpoints."""

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Body, HTTPException, Path

from rose_server.schemas.vector_stores import (
    VectorSearch,
    VectorSearchResult,
    VectorStoreCreate,
    VectorStoreList,
    VectorStoreMetadata,
    VectorStoreUpdate,
)
from rose_server.vector_stores.store import (
    create_vector_store,
    delete_vector_store,
    delete_vectors as delete_vectors_from_store,
    get_vector_store,
    list_vector_stores,
    search_vectors,
    update_vector_store,
)

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)


@router.get("/vector_stores")
async def index() -> VectorStoreList:
    """List all vector stores."""
    try:
        result = await list_vector_stores()
        return result
    except Exception as e:
        logger.error(f"Error listing vector stores: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing vector stores: {str(e)}")


@router.post("/vector_stores")
async def create(request: VectorStoreCreate = Body(...)) -> VectorStoreMetadata:
    """Create a new vector store."""
    try:
        result = await create_vector_store(request)
        return result
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating vector store: {str(e)}")


@router.get("/vector_stores/{vector_store_id}")
async def get(vector_store_id: str = Path(..., description="The ID of the vector store")) -> VectorStoreMetadata:
    """Get a vector store by ID."""
    try:
        result = await get_vector_store(vector_store_id)
        return result
    except ValueError as e:
        logger.error(f"Vector store not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting vector store: {str(e)}")


@router.post("/vector_stores/{vector_store_id}")
async def update(
    vector_store_id: str = Path(..., description="The ID of the vector store"),
    request: VectorStoreUpdate = Body(...),
) -> VectorStoreMetadata:
    """Update a vector store."""
    try:
        result = await update_vector_store(vector_store_id, request)
        return result
    except ValueError as e:
        logger.error(f"Vector store not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating vector store: {str(e)}")


@router.delete("/vector_stores/{vector_store_id}")
async def delete(vector_store_id: str = Path(..., description="The ID of the vector store")) -> Dict[str, Any]:
    """Delete a vector store."""
    try:
        result = await delete_vector_store(vector_store_id)
        return result
    except ValueError as e:
        logger.error(f"Vector store not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting vector store: {str(e)}")


@router.post("/vector_stores/{vector_store_id}/vectors/delete")
async def delete_vectors(
    vector_store_id: str = Path(..., description="The ID of the vector store"),
    ids: List[str] = Body(..., embed=True),
) -> Dict[str, Any]:
    """Delete vectors from a vector store."""
    try:
        result = await delete_vectors_from_store(vector_store_id, ids)
        return result
    except ValueError as e:
        logger.error(f"Error deleting vectors: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting vectors: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting vectors: {str(e)}")


@router.post("/vector_stores/{vector_store_id}/search")
async def search_vector_store(
    vector_store_id: str = Path(..., description="The ID of the vector store"),
    request: VectorSearch = Body(...),
) -> VectorSearchResult:
    """Search for vectors in a vector store (OpenAI-compatible)."""
    try:
        result = await search_vectors(vector_store_id, request)
        return result
    except ValueError as e:
        logger.error(f"Error searching vectors: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error searching vectors: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching vectors: {str(e)}")
