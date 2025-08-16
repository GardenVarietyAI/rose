"""API router for vector store files endpoints."""

import logging

from fastapi import APIRouter, Body, HTTPException, Path

from rose_server.schemas.vector_stores import VectorStoreFile, VectorStoreFileCreate
from rose_server.vector_stores.files.store import add_file_to_vector_store

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("", response_model=VectorStoreFile)
async def create(
    vector_store_id: str = Path(..., description="The ID of the vector store"),
    request: VectorStoreFileCreate = Body(...),
) -> VectorStoreFile:
    """Add a file to a vector store."""
    try:
        vector_store_file = await add_file_to_vector_store(vector_store_id, request.file_id)
        logger.info("Added file %s to vector store %s", request.file_id, vector_store_id)

        return VectorStoreFile(
            id=vector_store_file.id,
            vector_store_id=vector_store_file.vector_store_id,
            status=vector_store_file.status,
            created_at=vector_store_file.created_at,
        )
    except Exception as e:
        logger.error(f"Error adding file to vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding file to vector store: {str(e)}")
