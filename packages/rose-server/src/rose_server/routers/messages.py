import logging
import time
from typing import Any, Literal

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from htpy.starlette import HtpyResponse
from pydantic import BaseModel
from rose_server.dependencies import get_db_session, get_llama_client, get_readonly_db_session, get_settings
from rose_server.models.messages import Message
from rose_server.routers.lenses import get_lens_message
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
    lens_id: str | None = None


class CreateMessageResponse(BaseModel):
    thread_id: str
    message_uuid: str
    job_uuid: str | None = None


class RevisionMessage(BaseModel):
    uuid: str
    thread_id: str | None
    role: str
    content: str | None
    reasoning: str | None
    model: str | None
    meta: dict[str, Any] | None
    created_at: int
    accepted_at: int | None


class ListRevisionsResponse(BaseModel):
    root_message_id: str
    latest_message_uuid: str
    messages: list[RevisionMessage]


class CreateRevisionRequest(BaseModel):
    content: Any


class CreateRevisionResponse(BaseModel):
    root_message_id: str
    message_uuid: str
    latest_message_uuid: str


def _effective_root_message_id(message: Message) -> str:
    return message.root_message_id or message.uuid


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
            .where(col(Message.deleted_at).is_(None))
            .order_by(col(Message.created_at).desc(), col(Message.id).desc())
            .limit(1)
        )
        prompt_result = await session.execute(prompt_stmt)
        prompt = prompt_result.scalar_one_or_none()
        if prompt is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        if prompt.content is None or not prompt.content.strip():
            raise HTTPException(status_code=400, detail="Prompt content cannot be empty")

        generation_messages: list[dict[str, Any]] = []
        if body.lens_id:
            lens_message = await get_lens_message(session, body.lens_id)
            if lens_message is None:
                raise HTTPException(status_code=400, detail="Unknown lens")
            lens_prompt = lens_message.content
            if lens_prompt is None:
                raise HTTPException(status_code=400, detail="Lens missing content")
            meta = lens_message.meta
            if meta is None:
                raise HTTPException(status_code=400, detail="Lens missing meta")
            generation_messages.append({"role": "system", "content": lens_prompt})
        generation_messages.append({"role": "user", "content": prompt.content})

        job_message = await jobs.create_generate_assistant_job(
            session,
            thread_id=thread_id,
            user_message_uuid=prompt.uuid,
            model=model_used,
            lens_id=body.lens_id,
        )
        background_tasks.add_task(
            jobs.run_generate_assistant_job,
            thread_id=thread_id,
            job_uuid=job_message.uuid,
            model_used=model_used,
            requested_model=requested_model,
            messages=generation_messages,
            lens_id=body.lens_id,
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


@router.get("/messages/{message_uuid}/revisions", response_model=ListRevisionsResponse)
async def list_message_revisions(
    message_uuid: str,
    session: AsyncSession = Depends(get_readonly_db_session),
) -> ListRevisionsResponse:
    message_result = await session.execute(
        select(Message).where(col(Message.uuid) == message_uuid).where(col(Message.deleted_at).is_(None)).limit(1)
    )
    message = message_result.scalar_one_or_none()
    if message is None:
        raise HTTPException(status_code=404, detail="Message not found")

    root_message_id = _effective_root_message_id(message)
    revisions_result = await session.execute(
        select(Message)
        .where(col(Message.deleted_at).is_(None))
        .where((col(Message.uuid) == root_message_id) | (col(Message.root_message_id) == root_message_id))
        .order_by(col(Message.created_at).desc(), col(Message.id).desc())
    )
    revisions = list(revisions_result.scalars().all())
    if not revisions:
        raise HTTPException(status_code=404, detail="Message not found")

    latest_message_uuid = revisions[0].uuid
    return ListRevisionsResponse(
        root_message_id=root_message_id,
        latest_message_uuid=latest_message_uuid,
        messages=[
            RevisionMessage(
                uuid=revision.uuid,
                thread_id=revision.thread_id,
                role=revision.role,
                content=revision.content,
                reasoning=revision.reasoning,
                model=revision.model,
                meta=revision.meta,
                created_at=revision.created_at,
                accepted_at=revision.accepted_at,
            )
            for revision in revisions
        ],
    )


@router.post("/messages/{message_uuid}/revisions", response_model=CreateRevisionResponse)
async def create_message_revision(
    message_uuid: str,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> CreateRevisionResponse:
    message_result = await session.execute(
        select(Message).where(col(Message.uuid) == message_uuid).where(col(Message.deleted_at).is_(None)).limit(1)
    )
    message = message_result.scalar_one_or_none()
    if message is None:
        raise HTTPException(status_code=404, detail="Message not found")

    if message.role != "user":
        raise HTTPException(status_code=400, detail="Only user messages can be revised")

    payload = await request.json()
    body = CreateRevisionRequest.model_validate(payload)
    content = serialize_message_content(body.content)
    if content is None or not content.strip():
        raise HTTPException(status_code=400, detail="Message content cannot be empty")

    root_message_id = _effective_root_message_id(message)
    meta: dict[str, Any] = dict(message.meta or {})
    meta["root_message_id"] = root_message_id
    meta["parent_message_id"] = message.uuid

    revision = Message(
        thread_id=message.thread_id,
        role=message.role,
        content=content,
        reasoning=message.reasoning,
        model=message.model,
        meta=meta,
        accepted_at=message.accepted_at,
    )
    session.add(revision)
    await session.flush()

    return CreateRevisionResponse(
        root_message_id=root_message_id,
        message_uuid=revision.uuid,
        latest_message_uuid=revision.uuid,
    )


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
