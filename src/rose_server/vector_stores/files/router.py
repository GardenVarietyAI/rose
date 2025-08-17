"""API router for vector store files endpoints."""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException, Path, Query

from rose_server.schemas.vector_stores import VectorStoreFile, VectorStoreFileCreate, VectorStoreFileList
from rose_server.vector_stores.files.store import (
    add_file_to_vector_store,
    delete_file_from_vector_store,
    list_vector_store_files,
)

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
        logger.info(f"Added file {request.file_id} to vector store {vector_store_id}")

        return VectorStoreFile(
            id=vector_store_file.id,
            vector_store_id=vector_store_file.vector_store_id,
            status=vector_store_file.status,
            created_at=vector_store_file.created_at,
        )
    except Exception as e:
        logger.error(f"Error adding file to vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding file to vector store: {str(e)}")


@router.get("", response_model=VectorStoreFileList)
async def list_files(
    vector_store_id: str = Path(..., description="The ID of the vector store"),
    limit: int = Query(20, ge=1, le=100, description="Max number of files to return"),
    order: str = Query("desc", description="Order by created_at (asc or desc)"),
    after: str = Query(None, description="File ID to start pagination after"),
    before: str = Query(None, description="File ID to end pagination before"),
) -> VectorStoreFileList:
    """List files in a vector store."""
    try:
        files = await list_vector_store_files(vector_store_id, limit, order, after, before)
        logger.info(f"Listed {len(files)} files for vector store {vector_store_id}")

        return VectorStoreFileList(
            data=[
                VectorStoreFile(
                    id=f.id,
                    vector_store_id=f.vector_store_id,
                    status=f.status,
                    created_at=f.created_at,
                    last_error=f.last_error,
                )
                for f in files
            ]
        )

    except Exception as e:
        logger.error(f"Error listing vector store files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing vector store files: {str(e)}")


@router.delete("/{file_id}")
async def delete_file(
    vector_store_id: str = Path(..., description="The ID of the vector store"),
    file_id: str = Path(..., description="The ID of the file to delete"),
) -> Dict[str, Any]:
    """Remove a file from a vector store."""
    try:
        await delete_file_from_vector_store(vector_store_id, file_id)
        logger.info(f"Deleted file {file_id} from vector store {vector_store_id}")

        return {"id": file_id, "object": "vector_store.file.deleted", "deleted": True}
    except Exception as e:
        logger.error(f"Error deleting file from vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting file from vector store: {str(e)}")
