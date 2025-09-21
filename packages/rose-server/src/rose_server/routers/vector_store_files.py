import logging
from typing import Any, Dict

from chonkie import TokenChunker
from fastapi import APIRouter, Body, HTTPException, Path, Query, Request
from rose_server.entities.files import UploadedFile
from rose_server.entities.vector_stores import (
    Document,
    VectorStore,
    VectorStoreFile as VectorStoreFileEntity,
)
from rose_server.schemas.vector_stores import VectorStoreFile, VectorStoreFileCreate, VectorStoreFileList
from rose_server.services.vector_store_documents import (
    prepare_documents_and_embeddings,
    prepare_embedding_deletion_params,
)
from rose_server.services.vector_store_files import EmptyFileError, decode_file_content
from sqlalchemy import (
    delete as sql_delete,
    desc,
    text,
    update as sql_update,
)
from sqlalchemy.dialects.sqlite import insert
from sqlmodel import col, select


class VectorStoreNotFoundError(ValueError):
    """Vector store does not exist."""


class FileNotFoundError(ValueError):
    """File does not exist."""


class ChunkingError(ValueError):
    """Failed to generate chunks from file."""


router = APIRouter(prefix="/v1/vector_stores/{vector_store_id}/files", tags=["vector_store_files"])
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
        # Inline get_uploaded_file
        async with req.app.state.get_db_session(read_only=True) as session:
            uploaded_file = await session.get(UploadedFile, request.file_id)
            if not uploaded_file:
                raise FileNotFoundError(f"Uploaded file {request.file_id} not found")
        text_content, decode_errors = decode_file_content(uploaded_file.content, uploaded_file.filename)
        chunker = TokenChunker(
            chunk_size=req.app.state.settings.default_chunk_size,
            chunk_overlap=req.app.state.settings.default_chunk_overlap,
            tokenizer=req.app.state.embedding_tokenizer,
        )
        chunks = chunker.chunk(text_content)

        if not chunks:
            raise ChunkingError(f"No chunks generated from file {request.file_id}")

        texts = [chunk.text for chunk in chunks]
        embeddings, _ = await req.app.state.embedding_model.encode_batch(texts)

        async with req.app.state.get_db_session() as session:
            vector_store = await session.get(VectorStore, vector_store_id)
            if not vector_store:
                raise VectorStoreNotFoundError(f"Vector store {vector_store_id} not found")

            uploaded_file_check = await session.get(UploadedFile, request.file_id)
            if not uploaded_file_check:
                raise FileNotFoundError(f"Uploaded file {request.file_id} not found")

            await session.execute(
                insert(VectorStoreFileEntity)
                .values(vector_store_id=vector_store_id, file_id=request.file_id)
                .on_conflict_do_nothing(
                    index_elements=[VectorStoreFileEntity.vector_store_id, VectorStoreFileEntity.file_id]
                )
            )

            vector_store_file = await session.scalar(
                select(VectorStoreFileEntity).where(
                    VectorStoreFileEntity.vector_store_id == vector_store_id,
                    VectorStoreFileEntity.file_id == request.file_id,
                )
            )

            if vector_store_file and vector_store_file.status != "in_progress":
                pass  # don't double-ingest
            else:
                try:
                    documents, embedding_data, created_at = prepare_documents_and_embeddings(
                        uploaded_file, vector_store_id, chunks, embeddings, decode_errors
                    )

                    await session.execute(sql_delete(Document).where(col(Document.id).like(f"{request.file_id}#%")))

                    for doc in documents:
                        session.add(doc)

                    # Batch insert embeddings using sqlite-vec
                    await session.execute(
                        text("INSERT OR REPLACE INTO vec0 (document_id, embedding) VALUES (:doc_id, :embedding)"),
                        embedding_data,
                    )

                    await session.commit()

                    if vector_store_file:
                        vector_store_file.status = "completed"
                        vector_store_file.last_error = None
                    await session.commit()

                    await session.execute(
                        sql_update(VectorStore).where(VectorStore.id == vector_store_id).values(last_used_at=created_at)  # type: ignore[arg-type]
                    )
                    await session.commit()

                except Exception as e:
                    logger.error(f"Failed to add file {request.file_id} to vector store {vector_store_id}: {str(e)}")
                    if vector_store_file:
                        vector_store_file.status = "failed"
                        vector_store_file.last_error = {"error": str(e)}
                    await session.commit()
                    raise

        logger.info("Added file %s to vector store %s", request.file_id, vector_store_id)

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
    except (EmptyFileError, ChunkingError) as e:
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
