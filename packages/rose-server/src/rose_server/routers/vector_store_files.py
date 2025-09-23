import logging
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, Path, Query, Request
from rose_server.entities.files import UploadedFile
from rose_server.entities.vector_stores import (
    Document,
    VectorStore,
    VectorStoreFile as VectorStoreFileEntity,
)
from rose_server.routers.vector_stores import _process_vector_store_files
from rose_server.schemas.vector_stores import VectorStoreFile, VectorStoreFileCreate, VectorStoreFileList
from rose_server.services.vector_store_documents import prepare_embedding_deletion_params
from rose_server.services.vector_store_files import EmptyFileError
from sqlalchemy import (
    delete as sql_delete,
    desc,
    text,
)
from sqlalchemy.dialects.sqlite import insert
from sqlmodel import col, select


class VectorStoreNotFoundError(ValueError):
    """Vector store does not exist."""


class FileNotFoundError(ValueError):
    """File does not exist."""


router = APIRouter(prefix="/v1/vector_stores/{vector_store_id}/files", tags=["vector_store_files"])
logger = logging.getLogger(__name__)


@router.post("", response_model=VectorStoreFile)
async def create(
    req: Request,
    background_tasks: BackgroundTasks,
    vector_store_id: str = Path(..., description="The ID of the vector store"),
    request: VectorStoreFileCreate = Body(...),
) -> VectorStoreFile:
    if not req.app.state.embedding_model or not req.app.state.embedding_tokenizer:
        raise HTTPException(status_code=500, detail="Embedding model not initialized")

    try:
        async with req.app.state.get_db_session(read_only=True) as session:
            vector_store = await session.get(VectorStore, vector_store_id)
            if not vector_store:
                raise VectorStoreNotFoundError(f"Vector store {vector_store_id} not found")

            uploaded_file = await session.get(UploadedFile, request.file_id)
            if not uploaded_file:
                raise FileNotFoundError(f"Uploaded file {request.file_id} not found")

        async with req.app.state.get_db_session() as session:
            await session.execute(
                insert(VectorStoreFileEntity)
                .values(vector_store_id=vector_store_id, file_id=request.file_id)
                .on_conflict_do_nothing(
                    index_elements=[
                        VectorStoreFileEntity.vector_store_id,
                        VectorStoreFileEntity.file_id,
                    ]
                )
            )

            vector_store_file = await session.scalar(
                select(VectorStoreFileEntity).where(
                    VectorStoreFileEntity.vector_store_id == vector_store_id,
                    VectorStoreFileEntity.file_id == request.file_id,
                )
            )

            await session.commit()

        background_tasks.add_task(_process_vector_store_files, req.app, vector_store_id, [request.file_id])

        logger.info("Scheduled file %s for processing in vector store %s", request.file_id, vector_store_id)

        if not vector_store_file:
            raise HTTPException(status_code=500, detail="Failed to create vector store file")
        return VectorStoreFile(
            id=vector_store_file.id,
            vector_store_id=vector_store_file.vector_store_id,
            status=vector_store_file.status,
            created_at=vector_store_file.created_at,
        )
    except (VectorStoreNotFoundError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except EmptyFileError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error adding file to vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding file to vector store: {str(e)}")


@router.get("", response_model=VectorStoreFileList)
async def list_files(
    req: Request,
    vector_store_id: str = Path(..., description="The ID of the vector store"),
    limit: int = Query(20, ge=1, le=100, description="Max number of files to return"),
    order: str = Query("desc", description="Order by created_at (asc or desc)"),
    after: str = Query(None, description="File ID to start pagination after"),
    before: str = Query(None, description="File ID to end pagination before"),
) -> VectorStoreFileList:
    try:
        async with req.app.state.get_db_session() as session:
            query = select(VectorStoreFileEntity).where(VectorStoreFileEntity.vector_store_id == vector_store_id)

            if after:
                query = query.where(VectorStoreFileEntity.id > after)
            if before:
                query = query.where(VectorStoreFileEntity.id < before)

            if order == "desc":
                query = query.order_by(desc(VectorStoreFileEntity.created_at))  # type: ignore[arg-type]
            else:
                query = query.order_by(VectorStoreFileEntity.created_at)  # type: ignore[arg-type]

            query = query.limit(limit + 1)

            result = await session.execute(query)
            files = list(result.scalars().all())

            has_more = len(files) > limit
            if has_more:
                files = files[:limit]
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
    req: Request,
    vector_store_id: str = Path(..., description="The ID of the vector store"),
    file_id: str = Path(..., description="The ID of the file to remove from vector store"),
) -> Dict[str, Any]:
    """Remove a file from a vector store. The file itself remains in storage."""
    try:
        async with req.app.state.get_db_session() as session:
            vsf = await session.scalar(
                select(VectorStoreFileEntity).where(
                    VectorStoreFileEntity.vector_store_id == vector_store_id,
                    VectorStoreFileEntity.file_id == file_id,
                )
            )

            if not vsf:
                deleted = False
            else:
                doc_ids_result = await session.scalars(
                    select(Document.id).where(
                        col(Document.id).like(f"{file_id}#%"),
                        Document.vector_store_id == vector_store_id,
                    )
                )
                doc_ids = list(doc_ids_result)

                if doc_ids:
                    placeholders, params = prepare_embedding_deletion_params(doc_ids)
                    await session.execute(text(f"DELETE FROM vec0 WHERE document_id IN ({placeholders})"), params)

                    await session.execute(sql_delete(Document).where(col(Document.id).in_(doc_ids)))

                await session.delete(vsf)
                await session.commit()
                deleted = True

        if deleted:
            logger.info("Deleted file %s from vector store %s", file_id, vector_store_id)
        return {"id": file_id, "object": "vector_store.file.deleted", "deleted": True}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting file from vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting file from vector store: {str(e)}")
