import logging
from typing import Literal, Optional, Sequence

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from openai.types import FileDeleted, FileObject
from rose_server.entities.files import UploadedFile
from rose_server.schemas.files import FileListResponse
from sqlalchemy import delete, desc
from sqlmodel import select

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1")


@router.post("/files")
async def create(
    req: Request,
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
            status="processed",
            content=content,
        )

        async with req.app.state.get_db_session() as session:
            session.add(uploaded_file)
            await session.commit()
            await session.refresh(uploaded_file)
            logger.info(f"Created file {uploaded_file.id} with BLOB content, filename {uploaded_file.filename}")

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

    async with req.app.state.get_db_session(read_only=True) as session:
        result = await session.execute(select(UploadedFile.content).where(UploadedFile.id == file_id))
        content = result.scalar_one_or_none()

    if content is None:
        raise HTTPException(status_code=404, detail=f"File {file_id} content not found")

    return Response(
        content=content,
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
