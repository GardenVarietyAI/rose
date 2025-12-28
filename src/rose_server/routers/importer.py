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
    import_source: str
    messages: list[ImportMessage]

    model_config = ConfigDict(extra="ignore")

    @field_validator("import_source")
    @classmethod
    def validate_import_source(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("import_source cannot be empty")
        if len(normalized) > 64:
            raise ValueError("import_source too long")
        if not normalized.replace("_", "").replace("-", "").isalnum():
            raise ValueError("import_source must be alphanumeric, underscore, or hyphen")
        return normalized


@router.post("/import/messages")
async def import_messages(request: ImportRequest, session: AsyncSession = Depends(get_db_session)) -> dict[str, Any]:
    imported_at = int(time.time())
    messages = []

    for external in request.messages:
        # Merge client-provided meta with server-added import fields.
        meta = external.meta.copy() if external.meta else {}
        meta["imported_id"] = external.uuid
        meta["imported_source"] = request.import_source
        meta["imported_at"] = imported_at

        messages.append(
            Message(
                thread_id=external.thread_id,
                role=external.role,
                content=external.content,
                model=external.model,
                created_at=external.created_at,
                meta=meta,
            )
        )

    session.add_all(messages)
    return {"imported": len(messages)}
