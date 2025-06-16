"""File management API endpoints."""
import logging
from typing import Literal, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from openai.types import FileDeleted, FileObject

from ..schemas.files import FileListResponse
from ..services import get_file_store

logger = logging.getLogger(__name__)
router = APIRouter()
@router.post("/v1/files")

async def create_file(
    file: UploadFile = File(...),
    purpose: Literal[
        "assistants", "assistants_output", "batch", "batch_output", "fine-tune", "fine-tune-results", "vision"
    ] = Form(...),
) -> FileObject:
    """Upload a file that can be used across various endpoints."""
    try:
        if file.size and file.size > 1_000_000_000:
            raise HTTPException(status_code=413, detail="File size exceeds 1GB limit")
        file_obj = await get_file_store().create_file(file=file.file, purpose=purpose, filename=file.filename)
        if purpose in ["fine-tune", "fine-tune-results"]:
            valid, error = await get_file_store().validate_jsonl(file_obj.id)
            if not valid:
                await get_file_store().delete_file(file_obj.id)
                raise HTTPException(status_code=400, detail=f"Invalid JSONL file: {error}")
            else:
                await get_file_store().update_file_status(file_obj.id, "processed", "File validated successfully")
                file_obj = get_file_store().get_file(file_obj.id)
        else:
            await get_file_store().update_file_status(file_obj.id, "processed")
            file_obj = get_file_store().get_file(file_obj.id)
        return file_obj
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/v1/files")

async def list_files(
    purpose: Optional[str] = Query(None), limit: int = Query(20, ge=1, le=100), after: Optional[str] = Query(None)
) -> FileListResponse:
    """List files."""
    files = get_file_store().list_files(purpose=purpose, limit=limit, after=after)
    return FileListResponse(data=files, has_more=len(files) == limit)
@router.get("/v1/files/{file_id}")

async def retrieve_file(file_id: str) -> FileObject:
    """Get file metadata."""
    file_obj = get_file_store().get_file(file_id)
    if not file_obj:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")
    return file_obj
@router.get("/v1/files/{file_id}/content")

async def retrieve_file_content(file_id: str):
    """Get file content."""
    content = await get_file_store().get_file_content(file_id)
    if content is None:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")
    file_obj = get_file_store().get_file(file_id)
    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{file_obj.filename}"'},
    )
@router.delete("/v1/files/{file_id}")

async def delete_file(file_id: str) -> FileDeleted:
    """Delete a file."""
    result = await get_file_store().delete_file(file_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")
    return result