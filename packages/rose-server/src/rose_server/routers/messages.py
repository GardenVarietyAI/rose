import logging
import time
from typing import Any, Literal

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from htpy.starlette import HtpyResponse
from pydantic import BaseModel
from rose_server.dependencies import get_db_session, get_llama_client, get_settings
from rose_server.models.messages import Message
from rose_server.services import jobs
from rose_server.services.llama import normalize_model_name, serialize_message_content
from rose_server.settings import Settings
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


class CreateMessageRequest(BaseModel):
    thread_id: str
    role: Literal["assistant"] = "assistant"
    content: Any | None = None
    model: str | None = None
    meta: dict[str, Any] | None = None
    generate_assistant: bool = False


class CreateMessageResponse(BaseModel):
    thread_id: str
    message_uuid: str
    job_uuid: str | None = None


@router.post("/messages", response_model=CreateMessageResponse)
async def create_message(
    body: CreateMessageRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
    llama_client: httpx.AsyncClient = Depends(get_llama_client),
    session: AsyncSession = Depends(get_db_session),
) -> CreateMessageResponse:
    requested_model = body.model.strip() if body.model else None
    model_used, _model_path = normalize_model_name(settings.llama_model_path or requested_model or "default")

    if body.generate_assistant and body.content is not None:
        raise HTTPException(status_code=400, detail="Cannot provide content when generate_assistant=true")

    if not body.generate_assistant and body.content is None:
        raise HTTPException(status_code=400, detail="Message content cannot be empty")

    thread_id = body.thread_id

    exists_stmt = select(Message.id).where(col(Message.thread_id) == thread_id).limit(1)
    exists_result = await session.execute(exists_stmt)
    if exists_result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    if body.generate_assistant:
        prompt_stmt = (
            select(Message)
            .where(col(Message.thread_id) == thread_id)
            .where(col(Message.role) == "user")
            .order_by(col(Message.created_at))
            .limit(1)
        )
        prompt_result = await session.execute(prompt_stmt)
        prompt = prompt_result.scalar_one_or_none()
        if prompt is None or not (prompt.content or "").strip():
            raise HTTPException(status_code=404, detail="Thread not found")

        job_message = await jobs.create_generate_assistant_job(
            session,
            thread_id=thread_id,
            user_message_uuid=prompt.uuid,
            model=model_used,
        )
        background_tasks.add_task(
            jobs.run_generate_assistant_job,
            thread_id=thread_id,
            job_uuid=job_message.uuid,
            model_used=model_used,
            requested_model=requested_model,
            messages=[{"role": "user", "content": prompt.content or ""}],
            llama_client=llama_client,
            bind=session.bind,
        )

        return CreateMessageResponse(thread_id=thread_id, message_uuid=job_message.uuid, job_uuid=job_message.uuid)

    content = serialize_message_content(body.content)
    if content is None or not content.strip():
        raise HTTPException(status_code=400, detail="Message content cannot be empty")

    assistant_message = Message(
        thread_id=thread_id,
        role="assistant",
        content=content,
        model=model_used,
        meta=body.meta,
    )
    session.add(assistant_message)

    return CreateMessageResponse(thread_id=thread_id, message_uuid=assistant_message.uuid)


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
