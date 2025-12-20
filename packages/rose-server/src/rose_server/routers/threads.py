import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from htpy.starlette import HtpyResponse
from pydantic import BaseModel
from rose_server.dependencies import get_readonly_db_session
from rose_server.models.messages import Message
from rose_server.views.pages.thread import render_thread
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["threads"])


class ThreadResponse(BaseModel):
    thread_id: str
    prompt: Optional[Message]
    responses: list[Message]


@router.get("/threads/{thread_id}", response_model=None)
async def get_thread(
    request: Request,
    thread_id: str,
    session: AsyncSession = Depends(get_readonly_db_session),
) -> Any:
    prompt_stmt = (
        select(Message)
        .where(col(Message.thread_id) == thread_id)
        .where(col(Message.role) == "user")
        .order_by(col(Message.created_at))
        .limit(1)
    )
    prompt_result = await session.execute(prompt_stmt)
    prompt = prompt_result.scalar_one_or_none()

    responses_stmt = (
        select(Message)
        .where(col(Message.thread_id) == thread_id)
        .where(col(Message.role) == "assistant")
        .order_by(
            col(Message.accepted_at).desc().nulls_last(),
            col(Message.created_at).desc(),
        )
    )
    responses_result = await session.execute(responses_stmt)
    responses = list(responses_result.scalars().all())

    if not prompt and not responses:
        raise HTTPException(status_code=404, detail="Thread not found")

    response_data = ThreadResponse(
        thread_id=thread_id,
        prompt=prompt,
        responses=responses,
    )

    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return HtpyResponse(
            render_thread(
                thread_id=thread_id,
                prompt=prompt,
                responses=responses,
            )
        )

    return response_data
