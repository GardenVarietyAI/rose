import logging
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from rose_server.models.messages import Message
from sqlmodel import col, update

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["messages"])


class UpdateMessageRequest(BaseModel):
    accepted: bool


class UpdateMessageResponse(BaseModel):
    status: str
    message_uuid: str
    accepted: bool


@router.patch("/messages/{message_uuid}", response_model=UpdateMessageResponse)
async def update_message(
    request: Request,
    message_uuid: str,
    body: UpdateMessageRequest,
) -> UpdateMessageResponse:
    async with request.app.state.get_db_session() as session:
        accepted_at_value = int(time.time()) if body.accepted else None
        statement = (
            update(Message)
            .where(col(Message.uuid) == message_uuid)
            .where(col(Message.role) == "assistant")
            .values(accepted_at=accepted_at_value)
        )
        result = await session.execute(statement)

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Message not found")

    return UpdateMessageResponse(
        status="updated",
        message_uuid=message_uuid,
        accepted=body.accepted,
    )
