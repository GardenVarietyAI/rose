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
        file_obj = await create_file(file=file.file, purpose=purpose, filename=file.filename)
        if file_obj is None:
            raise HTTPException(status_code=500, detail="Failed to create file")
        if not isinstance(file_obj, FileObject):
            raise HTTPException(status_code=500, detail="Invalid file object returned")
        return file_obj
    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files")
async def index(
    purpose: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    after: Optional[str] = Query(None),
) -> FileListResponse:
    """List files."""
    files = await list_files(purpose=purpose, limit=limit, after=after)
    return FileListResponse(data=files, has_more=len(files) == limit)


@router.get("/files/{file_id}")
async def get(file_id: str) -> FileObject:
    """Get file metadata."""
    file_obj = await get_file(file_id)
    if not file_obj:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")
    if not isinstance(file_obj, FileObject):
        raise HTTPException(status_code=500, detail="Internal error: file object has invalid type")
    return file_obj


@router.get("/files/{file_id}/content")
async def get_content(file_id: str) -> Response:
    """Get file content."""
    content = await get_file_content(file_id)
    if content is None:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")
    file_obj = await get_file(file_id)
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
        if not isinstance(deleted, FileDeleted):
            raise HTTPException(status_code=500, detail="Internal error: invalid FileDeleted object")
        return deleted
    except Exception:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")
