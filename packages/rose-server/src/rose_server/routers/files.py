import hashlib
import logging
from typing import Any, Literal, Optional, Sequence

import numpy as np
from chonkie import TokenChunker
from fastapi import APIRouter, File, Form, HTTPException, Path, Query, Request, UploadFile
from fastapi.responses import Response
from openai.types import FileDeleted, FileObject
from rose_server.entities.file_chunks import FileChunk
from rose_server.entities.files import UploadedFile
from rose_server.schemas.files import FileListResponse
from sqlalchemy import delete, desc
from sqlmodel import (
    col,
    select,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1")


async def process_file_chunks(app: Any, file_id: str) -> None:
    MAX_EMBEDDING_BATCH_SIZE = 10

    async with app.state.get_db_session() as session:
        uploaded_file = await session.get(UploadedFile, file_id)
        if not uploaded_file:
            logger.error(f"File {file_id} not found")
            return

        try:
            text_content = uploaded_file.content.decode("utf-8")
            decode_errors = False
        except UnicodeDecodeError:
            text_content = uploaded_file.content.decode("utf-8", errors="replace")
            decode_errors = True

        try:
            chunker = TokenChunker(
                chunk_size=app.state.settings.default_chunk_size,
                chunk_overlap=app.state.settings.default_chunk_overlap,
                tokenizer=app.state.embedding_tokenizer,
            )
            chunks = chunker.chunk(text_content)

            if not chunks:
                raise ValueError(f"No chunks generated from file {file_id}")

            texts = [chunk.text for chunk in chunks]

            embeddings = []
            for i in range(0, len(texts), MAX_EMBEDDING_BATCH_SIZE):
                batch_texts = texts[i : i + MAX_EMBEDDING_BATCH_SIZE]
                batch_embeddings, _ = await app.state.embedding_model.encode_batch(batch_texts)
                embeddings.extend(batch_embeddings)

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

            session.add_all(file_chunks)
            uploaded_file.status = "processed"
            await session.commit()
            logger.info(f"Processed file {file_id} with {len(file_chunks)} chunks")

        except Exception as e:
            logger.error(f"Failed to process file {file_id}: {e}")
            uploaded_file.status = "error"
            await session.commit()


@router.get("/files/{file_id}")
async def retrieve(
    req: Request,
    file_id: str = Path(..., description="The ID of the file to retrieve"),
) -> FileObject:
    async with req.app.state.get_db_session() as session:
        uploaded_file = await session.get(UploadedFile, file_id)
        if not uploaded_file:
            raise HTTPException(status_code=404, detail=f"File {file_id} not found")

        return FileObject(
            id=uploaded_file.id,
            object=uploaded_file.object,
            bytes=uploaded_file.bytes,
            created_at=uploaded_file.created_at,
            filename=uploaded_file.filename,
            purpose=uploaded_file.purpose,
            status=uploaded_file.status,
        )


@router.post("/files")
async def create(
    req: Request,
    file: UploadFile = File(...),
    purpose: Literal["assistants", "batch", "fine-tune", "vision", "user_data"] = Form(...),
) -> FileObject:
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    CHUNK_SIZE = 1024 * 1024  # 1MB chunks

    if file.content_type and not file.content_type.startswith("text/"):
        raise HTTPException(status_code=415, detail=f"Unsupported media type: {file.content_type}.")

    try:
        chunks = []
        total_size = 0

        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break

            total_size += len(chunk)
            if total_size > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail=("File too large."))

            chunks.append(chunk)

        if total_size == 0:
            raise HTTPException(status_code=400, detail="File is empty")

        content = b"".join(chunks)
        file_size = total_size
        filename = file.filename if file.filename else "unknown"

        uploaded_file = UploadedFile(
            object="file",
            bytes=file_size,
            filename=filename,
            purpose=purpose,
            status="uploaded",
            content=content,
        )

        async with req.app.state.get_db_session() as session:
            session.add(uploaded_file)
            await session.commit()
            await session.refresh(uploaded_file)
            logger.info(f"Created file {uploaded_file.id} with BLOB content, filename {uploaded_file.filename}")

            await req.app.state.file_processing_queue.put(uploaded_file.id)

            # Return response without binary content to avoid serialization issues
            return FileObject(
                id=uploaded_file.id,
                object="file",
                bytes=uploaded_file.bytes,
                created_at=uploaded_file.created_at,
                filename=uploaded_file.filename,
                purpose=uploaded_file.purpose,  # type: ignore
                status="uploaded",
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

        query = query.order_by(desc(col(UploadedFile.created_at)))

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
