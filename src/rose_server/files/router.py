"""File management API endpoints."""

import logging
from typing import Literal, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from openai.types import FileDeleted, FileObject

from rose_server.files.store import create_file, delete_file, get_file, get_file_content, list_files
from rose_server.schemas.files import FileListResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1")


@router.post("/files")
async def create(
    file: UploadFile = File(...),
    purpose: Literal["assistants", "batch", "fine-tune", "vision", "user_data"] = Form(...),
) -> FileObject:
    """Upload a file."""
    try:
        content = await file.read()
        file_size = file.size or len(content)

        uploaded_file = await create_file(file_size=file_size, purpose=purpose, filename=file.filename, content=content)

        # Return response without binary content to avoid serialization issues
        return FileObject(
            id=uploaded_file.id,
            object=uploaded_file.object,
            bytes=uploaded_file.bytes,
            created_at=uploaded_file.created_at,
            filename=uploaded_file.filename,
            purpose=uploaded_file.purpose,
            status=uploaded_file.status,
            expires_at=uploaded_file.expires_at,
            status_details=uploaded_file.status_details,
        )
    except Exception as e:
        logger.error(f"File upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files")
async def index(
    purpose: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    after: Optional[str] = Query(None),
) -> FileListResponse:
    """List files."""
    uploaded_files = await list_files(purpose=purpose, limit=limit, after=after)
    return FileListResponse(
        data=[
            FileObject(
                id=f.id,
                object=f.object,
                bytes=f.bytes,
                created_at=f.created_at,
                filename=f.filename,
                purpose=f.purpose,
                status=f.status,
                expires_at=f.expires_at,
                status_details=f.status_details,
            )
            for f in uploaded_files
        ],
        has_more=len(uploaded_files) == limit,
    )


@router.get("/files/{file_id}")
async def get(file_id: str) -> FileObject:
    """Get file metadata."""
    uploaded_file = await get_file(file_id)
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
        expires_at=uploaded_file.expires_at,
        status_details=uploaded_file.status_details,
    )


@router.get("/files/{file_id}/content")
async def get_content(file_id: str) -> Response:
    """Get file content."""
    file_obj = await get_file(file_id)
    if not file_obj:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")

    content = await get_file_content(file_id)
    if content is None:
        raise HTTPException(status_code=404, detail=f"File {file_id} content not found")

    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{file_obj.filename}"'},
    )


@router.delete("/files/{file_id}")
async def remove(file_id: str) -> FileDeleted:
    """Delete a file."""
    try:
        deleted = await delete_file(file_id)
        return FileDeleted(id=file_id, object="file", deleted=deleted)
    except Exception:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")
