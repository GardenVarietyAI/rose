"""API router for vector stores endpoints."""
import logging
from typing import List

from fastapi import APIRouter, Body, HTTPException, Path

from rose_server.services import get_vector_store_store
from rose_server.vector_stores.models import (
    VectorBatch,
    VectorSearch,
    VectorStoreCreate,
    VectorStoreUpdate,
)

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)
@router.get("/vector_stores")

async def list_vector_stores():
    """List all vector stores."""
    try:
        manager = get_vector_store_store()
        result = await manager.list_vector_stores()
        return result
    except Exception as e:
        logger.error(f"Error listing vector stores: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing vector stores: {str(e)}")
@router.post("/vector_stores")

async def create_vector_store(request: VectorStoreCreate = Body(...)):
    """Create a new vector store."""
    try:
        manager = get_vector_store_store()
        result = await manager.create_vector_store(request)
        return result
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating vector store: {str(e)}")
@router.get("/vector_stores/{vector_store_id}")

async def get_vector_store(vector_store_id: str = Path(..., description="The ID of the vector store")):
    """Get a vector store by ID."""
    try:
        manager = get_vector_store_store()
        result = await manager.get_vector_store(vector_store_id)
        return result
    except ValueError as e:
        logger.error(f"Vector store not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting vector store: {str(e)}")
@router.post("/vector_stores/{vector_store_id}")

async def update_vector_store(
    vector_store_id: str = Path(..., description="The ID of the vector store"),
    request: VectorStoreUpdate = Body(...),
):
    """Update a vector store."""
    try:
        manager = get_vector_store_store()
        result = await manager.update_vector_store(vector_store_id, request)
        return result
    except ValueError as e:
        logger.error(f"Vector store not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating vector store: {str(e)}")
@router.delete("/vector_stores/{vector_store_id}")

async def delete_vector_store(vector_store_id: str = Path(..., description="The ID of the vector store")):
    """Delete a vector store."""
    try:
        manager = get_vector_store_store()
        result = await manager.delete_vector_store(vector_store_id)
        return result
    except ValueError as e:
        logger.error(f"Vector store not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting vector store: {str(e)}")
@router.post("/vector_stores/{vector_store_id}/vectors")

async def add_vectors(
    vector_store_id: str = Path(..., description="The ID of the vector store"),
    request: VectorBatch = Body(...),
):
    """Add vectors to a vector store."""
    try:
        manager = get_vector_store_store()
        result = await manager.add_vectors(vector_store_id, request)
        return result
    except ValueError as e:
        logger.error(f"Error adding vectors: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error adding vectors: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding vectors: {str(e)}")
@router.post("/vector_stores/{vector_store_id}/vectors/delete")

async def delete_vectors(
    vector_store_id: str = Path(..., description="The ID of the vector store"),
    ids: List[str] = Body(..., embed=True),
):
    """Delete vectors from a vector store."""
    try:
        manager = get_vector_store_store()
        result = await manager.delete_vectors(vector_store_id, ids)
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
):
    """Search for vectors in a vector store (OpenAI-compatible)."""
    try:
        manager = get_vector_store_store()
        result = await manager.search_vectors(vector_store_id, request)
        return result
    except ValueError as e:
        logger.error(f"Error searching vectors: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error searching vectors: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching vectors: {str(e)}")