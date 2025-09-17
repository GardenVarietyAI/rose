"""API router for vector store files endpoints."""

import asyncio
import logging
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException, Path, Query, Request

from rose_server.config.settings import settings
from rose_server.database import get_session
from rose_server.entities.files import UploadedFile
from rose_server.schemas.vector_stores import VectorStoreFile, VectorStoreFileCreate, VectorStoreFileList
from rose_server.vector_stores.files.store import (
    ChunkingError,
    EmptyFileError,
    FileNotFoundError,
    VectorStoreNotFoundError,
    add_file_to_vector_store,
    create_chunks,
    decode_file_content,
    list_vector_store_files,
    remove_file_from_vector_store,
)

router = APIRouter(prefix="/{vector_store_id}/files", tags=["vector_store_files"])
logger = logging.getLogger(__name__)


@router.post("", response_model=VectorStoreFile)
async def create(
    req: Request,
    vector_store_id: str = Path(..., description="The ID of the vector store"),
    request: VectorStoreFileCreate = Body(...),
) -> VectorStoreFile:
    """Add a file to a vector store."""
    if not req.app.state.embedding_model or not req.app.state.embedding_tokenizer:
        raise HTTPException(status_code=500, detail="Embedding model not initialized")

    try:
        # Load the uploaded file
        async with get_session(read_only=True) as session:
            uploaded_file = await session.get(UploadedFile, request.file_id)
            if not uploaded_file:
                raise FileNotFoundError(f"Uploaded file {request.file_id} not found")

        # Decode file content (pure function)
        text_content, decode_errors = decode_file_content(uploaded_file.content, uploaded_file.filename)

        # Create chunks (pure function)
        chunks = create_chunks(
            text_content, req.app.state.embedding_tokenizer, settings.default_chunk_size, settings.default_chunk_overlap
        )

        if not chunks:
            raise ChunkingError(f"No chunks generated from file {request.file_id}")

        # Generate embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = await asyncio.to_thread(lambda: list(req.app.state.embedding_model.embed(texts)))

        # Store with pre-computed embeddings
        vector_store_file = await add_file_to_vector_store(
            vector_store_id, request.file_id, embeddings, chunks, decode_errors
        )
        logger.info("Added file %s to vector store %s", request.file_id, vector_store_id)

        return VectorStoreFile(
            id=vector_store_file.id,
            vector_store_id=vector_store_file.vector_store_id,
            status=vector_store_file.status,
            created_at=vector_store_file.created_at,
        )
    except (VectorStoreNotFoundError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (EmptyFileError, ChunkingError) as e:
        raise HTTPException(status_code=422, detail=str(e))
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
        logger.info("Listed %d files for vector store %s", len(files), vector_store_id)

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

    except VectorStoreNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error listing vector store files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing vector store files: {str(e)}")


@router.delete("/{file_id}")
async def delete_file(
    vector_store_id: str = Path(..., description="The ID of the vector store"),
    file_id: str = Path(..., description="The ID of the file to remove from vector store"),
) -> Dict[str, Any]:
    """Remove a file from a vector store. The file itself remains in storage."""
    try:
        deleted = await remove_file_from_vector_store(vector_store_id, file_id)
        if deleted:
            logger.info("Deleted file %s from vector store %s", file_id, vector_store_id)
        return {"id": file_id, "object": "vector_store.file.deleted", "deleted": True}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting file from vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting file from vector store: {str(e)}")
