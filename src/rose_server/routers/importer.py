import time
from typing import Any

from fastapi import APIRouter, Depends
from htpy.starlette import HtpyResponse
from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from rose_server.dependencies import get_db_session
from rose_server.models.messages import Message
from rose_server.views.pages.importer import render_import

router = APIRouter(prefix="/v1", tags=["import"])


@router.get("/import")
async def import_page() -> Any:
    return HtpyResponse(render_import())


class ImportMessage(BaseModel):
    uuid: str
    thread_id: str
    role: str
    content: str | None
    model: str | None
    created_at: int
    meta: dict[str, Any] | None

    model_config = ConfigDict(extra="ignore")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got '{v}'")
        return v

    @field_validator("created_at")
    @classmethod
    def validate_timestamp(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"created_at must be non-negative, got {v}")
        return v


class ImportRequest(BaseModel):
    messages: list[ImportMessage]


@router.post("/import/messages")
async def import_messages(
    request: ImportRequest,
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    imported_count = 0

    for msg in request.messages:
        meta = msg.meta.copy() if msg.meta else {}
        meta["imported_id"] = msg.uuid
        meta["imported_source"] = "claude_code_jsonl"
        meta["imported_at"] = int(time.time())

        message = Message(
            thread_id=msg.thread_id,
            role=msg.role,
            content=msg.content,
            model=msg.model,
            created_at=msg.created_at,
            meta=meta,
        )
        session.add(message)
        imported_count += 1

    await session.commit()
    return {"imported": imported_count}
