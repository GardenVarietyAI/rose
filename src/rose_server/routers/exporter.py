import tempfile
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from htpy.starlette import HtpyResponse
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import FileResponse

from rose_server.dependencies import get_readonly_db_session
from rose_server.schemas.exporter import ExportRequest, ExportResponse
from rose_server.services.exporter import (
    build_conversations,
    query_assistant_messages,
    query_thread_ids,
    query_user_messages,
    split_dataset,
    write_jsonl,
)
from rose_server.services.lenses import list_lenses_messages
from rose_server.views.pages.exporter import render_export_page

router = APIRouter(prefix="/v1", tags=["exporter"])


def _export_dir(export_id: str) -> Path:
    try:
        uuid.UUID(export_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Export not found") from exc

    return Path(tempfile.gettempdir()) / "rose_exports" / export_id


@router.get("/export", response_model=None)
async def export_page(
    session: AsyncSession = Depends(get_readonly_db_session),
) -> HtpyResponse:
    lenses = await list_lenses_messages(session)
    return HtpyResponse(render_export_page(lenses=lenses))


@router.post("/export/training", response_model=ExportResponse)
async def create_export(
    request: ExportRequest,
    session: AsyncSession = Depends(get_readonly_db_session),
) -> ExportResponse:
    export_id = str(uuid.uuid4())
    created_at = int(time.time())

    thread_ids = await query_thread_ids(
        session,
        request.filters.lens_id,
        request.filters.accepted_only,
    )

    if not thread_ids:
        return ExportResponse(
            export_id=export_id,
            total_conversations=0,
            train_count=0,
            valid_count=0,
            created_at=created_at,
        )

    user_messages = await query_user_messages(session, thread_ids)
    assistant_messages = await query_assistant_messages(
        session,
        thread_ids,
        request.filters.lens_id,
        request.filters.accepted_only,
    )

    conversations = await build_conversations(
        user_messages,
        assistant_messages,
        request.filters.lens_id,
        session,
    )

    train, valid = split_dataset(conversations, request.split_ratio)

    export_dir = _export_dir(export_id)
    write_jsonl(train, export_dir / "train.jsonl")
    write_jsonl(valid, export_dir / "valid.jsonl")

    return ExportResponse(
        export_id=export_id,
        total_conversations=len(conversations),
        train_count=len(train),
        valid_count=len(valid),
        created_at=created_at,
    )


@router.get("/export/training/{export_id}/train.jsonl", response_model=None)
async def download_train(export_id: str) -> FileResponse:
    export_dir = _export_dir(export_id)
    filepath = export_dir / "train.jsonl"
    if not filepath.is_file():
        raise HTTPException(status_code=404, detail="Export file not found")

    return FileResponse(
        path=str(filepath),
        media_type="application/x-ndjson",
        filename="train.jsonl",
    )


@router.get("/export/training/{export_id}/valid.jsonl", response_model=None)
async def download_valid(export_id: str) -> FileResponse:
    export_dir = _export_dir(export_id)
    filepath = export_dir / "valid.jsonl"
    if not filepath.is_file():
        raise HTTPException(status_code=404, detail="Export file not found")

    return FileResponse(
        path=str(filepath),
        media_type="application/x-ndjson",
        filename="valid.jsonl",
    )
