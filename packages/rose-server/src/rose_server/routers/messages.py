import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from htpy.starlette import HtpyResponse
from pydantic import BaseModel
from rose_server.dependencies import get_db_session
from rose_server.models.messages import Message
from rose_server.views.components.response_message import response_message
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select, update

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["messages"])


class UpdateMessageRequest(BaseModel):
    accepted: bool


class UpdateMessageResponse(BaseModel):
    status: str
    message_uuid: str
    accepted: bool


@router.get("/messages/{message_uuid}/fragment", response_model=None)
async def get_message_fragment(
    message_uuid: str,
    session: AsyncSession = Depends(get_db_session),
) -> HtpyResponse:
    stmt = select(Message).where(col(Message.uuid) == message_uuid).limit(1)
    result = await session.execute(stmt)
    message = result.scalar_one_or_none()

    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    return HtpyResponse(
        response_message(
            uuid=message.uuid,
            dom_id=f"msg-{message.uuid}",
            role=message.role,
            model=message.model,
            content=message.content or "",
            created_at=message.created_at,
            accepted=bool(message.accepted_at),
        )
    )


@router.patch("/messages/{message_uuid}", response_model=UpdateMessageResponse)
async def update_message(
    message_uuid: str,
    body: UpdateMessageRequest,
    session: AsyncSession = Depends(get_db_session),
) -> UpdateMessageResponse:
    accepted_at = int(time.time()) if body.accepted else None
    statement = (
        update(Message)
        .where(col(Message.uuid) == message_uuid)
        .where(col(Message.role) == "assistant")
        .values(accepted_at=accepted_at)
    )
    result = await session.execute(statement)

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Message not found")

    return UpdateMessageResponse(
        status="updated",
        message_uuid=message_uuid,
        accepted=body.accepted,
    )
