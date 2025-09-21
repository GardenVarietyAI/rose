"""API router for vector store files endpoints."""

import logging
from typing import Any, Dict

from chonkie import TokenChunker
from fastapi import APIRouter, Body, HTTPException, Path, Query, Request

from rose_server.config.settings import settings
from rose_server.embeddings.service import generate_embeddings
from rose_server.schemas.vector_stores import VectorStoreFile, VectorStoreFileCreate, VectorStoreFileList
from rose_server.vector_stores.files.service import EmptyFileError, decode_file_content
from rose_server.vector_stores.files.store import (
    ChunkingError,
    FileNotFoundError,
    VectorStoreNotFoundError,
    get_uploaded_file,
    list_vector_store_files,
    remove_file_from_vector_store,
    store_file_chunks_with_embeddings,
)

router = APIRouter(prefix="/{vector_store_id}/files", tags=["vector_store_files"])
logger = logging.getLogger(__name__)


@router.post("", response_model=VectorStoreFile)
async def create(
    req: Request,
    vector_store_id: str = Path(..., description="The ID of the vector store"),
    request: VectorStoreFileCreate = Body(...),
) -> VectorStoreFile:
    if not req.app.state.embedding_model or not req.app.state.embedding_tokenizer:
        raise HTTPException(status_code=500, detail="Embedding model not initialized")

    try:
        uploaded_file = await get_uploaded_file(request.file_id)
        text_content, decode_errors = decode_file_content(uploaded_file.content, uploaded_file.filename)
        chunker = TokenChunker(
            chunk_size=settings.default_chunk_size,
            chunk_overlap=settings.default_chunk_overlap,
            tokenizer=req.app.state.embedding_tokenizer,
        )
        chunks = chunker.chunk(text_content)

        if not chunks:
            raise ChunkingError(f"No chunks generated from file {request.file_id}")

        texts = [chunk.text for chunk in chunks]
        embeddings, _ = await generate_embeddings(texts, req.app.state.embedding_model)
        vector_store_file = await store_file_chunks_with_embeddings(
            vector_store_id, request.file_id, chunks, embeddings, decode_errors
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
