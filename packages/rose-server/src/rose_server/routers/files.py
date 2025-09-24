import hashlib
import logging
from typing import Any, Literal, Optional, Sequence

import numpy as np
from chonkie import TokenChunker
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from openai.types import FileDeleted, FileObject
from rose_server.entities.file_chunks import FileChunk
from rose_server.entities.files import UploadedFile
from rose_server.schemas.files import FileListResponse
from rose_server.services.vector_store_files import decode_file_content
from sqlalchemy import col, delete, desc
from sqlmodel import select

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1")


async def process_file_chunks(app: Any, file_id: str) -> None:
    """Background task to chunk and embed uploaded files."""
    try:
        async with app.state.get_db_session() as session:
            # Get the file
            uploaded_file = await session.get(UploadedFile, file_id)
            if not uploaded_file:
                logger.error(f"File {file_id} not found for processing")
                return

        # Decode and chunk the content
        text_content, decode_errors = decode_file_content(uploaded_file.content, uploaded_file.filename)

        chunker = TokenChunker(
            chunk_size=app.state.settings.default_chunk_size,
            chunk_overlap=app.state.settings.default_chunk_overlap,
            tokenizer=app.state.embedding_tokenizer,
        )
        chunks = chunker.chunk(text_content)

        if not chunks:
            async with app.state.get_db_session() as session:
                uploaded_file = await session.get(UploadedFile, file_id)
                if uploaded_file:
                    uploaded_file.status = "processed"
                    await session.commit()
            logger.warning(f"No chunks generated from file {file_id}")
            return

        # Generate embeddings for all chunks
        texts = [chunk.text for chunk in chunks]
        embeddings, _ = await app.state.embedding_model.encode_batch(texts)

        # Build all chunks upfront
        file_chunks = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            content_hash = hashlib.sha256(chunk.text.encode()).hexdigest()
            file_chunk = FileChunk(
                content_hash=content_hash,
                file_id=file_id,
                chunk_index=i,
                content=chunk.text,
                embedding=np.array(embedding, dtype=np.float32).tobytes(),
                meta={
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index,
                    "decode_errors": decode_errors,
                },
            )
            file_chunks.append(file_chunk)

        # Batch insert all chunks
        async with app.state.get_db_session() as session:
            # Get existing hashes to avoid duplicates
            existing_hashes = set()
            if file_chunks:
                hashes = [c.content_hash for c in file_chunks]
                result = await session.execute(
                    select(FileChunk.content_hash).where(col(FileChunk.content_hash).in_(hashes))
                )
                existing_hashes = {h for (h,) in result.fetchall()}

            # Only add new chunks
            new_chunks = [c for c in file_chunks if c.content_hash not in existing_hashes]
            if new_chunks:
                session.add_all(new_chunks)

            # Update file status
            uploaded_file = await session.get(UploadedFile, file_id)
            if uploaded_file:
                uploaded_file.status = "processed"

            await session.commit()
            logger.info(f"Added {len(new_chunks)} new chunks for file {file_id}")

    except Exception as e:
        logger.error(f"Failed to process file {file_id}: {e}")
        async with app.state.get_db_session() as session:
            uploaded_file = await session.get(UploadedFile, file_id)
            if uploaded_file:
                uploaded_file.status = "error"
                await session.commit()


@router.post("/files")
async def create(
    req: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    purpose: Literal["assistants", "batch", "fine-tune", "vision", "user_data"] = Form(...),
) -> FileObject:
    try:
        content = await file.read()

        file_size = file.size if file.size is not None else len(content)
        filename = file.filename if file.filename else "unknown"

        uploaded_file = UploadedFile(
            object="file",
            bytes=file_size,
            filename=filename,
            purpose=purpose,
            status="processing",
            content=content,
        )

        async with req.app.state.get_db_session() as session:
            session.add(uploaded_file)
            await session.commit()
            await session.refresh(uploaded_file)
            logger.info(f"Created file {uploaded_file.id} with BLOB content, filename {uploaded_file.filename}")

        # Trigger background processing
        background_tasks.add_task(process_file_chunks, req.app, uploaded_file.id)

        # Return response without binary content to avoid serialization issues
        return FileObject(
            id=uploaded_file.id,
            object="file",
            bytes=uploaded_file.bytes,
            created_at=uploaded_file.created_at,
            filename=uploaded_file.filename,
            purpose=uploaded_file.purpose,  # type: ignore
            status=uploaded_file.status if uploaded_file.status else "processed",  # type: ignore
            expires_at=uploaded_file.expires_at,
            status_details=uploaded_file.status_details,
        )
    except Exception as e:
        logger.error(f"File upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files")
async def index(
    req: Request,
    purpose: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    after: Optional[str] = Query(None),
) -> FileListResponse:
    async with req.app.state.get_db_session(read_only=True) as session:
        query = select(UploadedFile)

        if purpose:
            query = query.where(UploadedFile.purpose == purpose)

        query = query.order_by(desc(UploadedFile.created_at))  # type: ignore[arg-type]

        if after:
            after_result = await session.execute(select(UploadedFile.created_at).where(UploadedFile.id == after))
            after_created_at = after_result.scalar_one_or_none()
            if after_created_at:
                query = query.where(UploadedFile.created_at < after_created_at)

        query = query.limit(limit)
        result = await session.execute(query)
        uploaded_files: Sequence[UploadedFile] = result.scalars().all()

    return FileListResponse(
        data=[
            FileObject(
                id=f.id,
                object="file",
                bytes=f.bytes,
                created_at=f.created_at,
                filename=f.filename,
                purpose=f.purpose,  # type: ignore
                status=f.status if f.status else "processed",  # type: ignore
                expires_at=f.expires_at,
                status_details=f.status_details,
            )
            for f in uploaded_files
        ],
        has_more=len(uploaded_files) == limit,
    )


@router.get("/files/{file_id}")
async def get(req: Request, file_id: str) -> FileObject:
    async with req.app.state.get_db_session(read_only=True) as session:
        result = await session.execute(select(UploadedFile).where(UploadedFile.id == file_id))
        uploaded_file = result.scalar_one_or_none()

    if not uploaded_file:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")

    return FileObject(
        id=uploaded_file.id,
        object="file",
        bytes=uploaded_file.bytes,
        created_at=uploaded_file.created_at,
        filename=uploaded_file.filename,
        purpose=uploaded_file.purpose,
        status=uploaded_file.status if uploaded_file.status else "processed",
        expires_at=uploaded_file.expires_at,
        status_details=uploaded_file.status_details,
    )


@router.get("/files/{file_id}/content")
async def get_content(req: Request, file_id: str) -> Response:
    async with req.app.state.get_db_session(read_only=True) as session:
        result = await session.execute(select(UploadedFile).where(UploadedFile.id == file_id))
        file_obj = result.scalar_one_or_none()

        if not file_obj:
            raise HTTPException(status_code=404, detail=f"File {file_id} not found")

        if file_obj.content is None:
            raise HTTPException(status_code=404, detail=f"File {file_id} content not found")

    return Response(
        content=file_obj.content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{file_obj.filename}"'},
    )


@router.delete("/files/{file_id}")
async def remove(req: Request, file_id: str) -> FileDeleted:
    async with req.app.state.get_db_session() as session:
        delete_stmt = delete(UploadedFile).where(UploadedFile.id == file_id)  # type: ignore[arg-type]
        result = await session.execute(delete_stmt)

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail=f"File {file_id} not found")

        await session.commit()
        logger.info(f"Deleted file {file_id}")

    return FileDeleted(id=file_id, object="file", deleted=True)
