import asyncio
import json
import logging
import uuid
from typing import Any, Literal

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from htpy.starlette import HtpyResponse
from pydantic import BaseModel, ConfigDict, field_validator
from rose_server.dependencies import get_db_session, get_llama_client, get_readonly_db_session, get_settings
from rose_server.models.messages import Message
from rose_server.models.search_events import SearchEvent
from rose_server.routers.lenses import get_lens_message, list_lens_options
from rose_server.services import jobs
from rose_server.services.llama import normalize_model_name, serialize_message_content
from rose_server.settings import Settings
from rose_server.views.pages.thread_activity import render_thread_activity
from rose_server.views.pages.thread_messages import render_thread_messages
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select
from sse_starlette.event import ServerSentEvent
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["threads"])


class ThreadResponse(BaseModel):
    thread_id: str
    prompt: Message | None
    responses: list[Message]


class CreateThreadRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    thread_id: str | None = None
    model: str | None = None
    messages: list[dict[str, Any]]
    lens_id: str | None = None

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not value:
            raise ValueError("Messages must contain at least one entry")
        return value


class CreateThreadResponse(BaseModel):
    thread_id: str
    message_uuid: str
    job_uuid: str


async def _generate_assistant_response(
    *,
    thread_id: str,
    job_uuid: str,
    model_used: str,
    requested_model: str | None,
    messages: list[dict[str, Any]],
    lens_id: str | None,
    llama_client: httpx.AsyncClient,
    bind: Any,
) -> None:
    await jobs.run_generate_assistant_job(
        thread_id=thread_id,
        job_uuid=job_uuid,
        model_used=model_used,
        requested_model=requested_model,
        messages=messages,
        lens_id=lens_id,
        llama_client=llama_client,
        bind=bind,
    )


@router.post("/threads", response_model=CreateThreadResponse)
async def create_thread_message(
    body: CreateThreadRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
    llama_client: httpx.AsyncClient = Depends(get_llama_client),
    session: AsyncSession = Depends(get_db_session),
) -> CreateThreadResponse:
    if len(body.messages) != 1 or body.messages[0].get("role") != "user":
        raise HTTPException(status_code=400, detail="Must contain exactly 1 user message")

    thread_id = str(uuid.uuid4())

    requested_model = body.model.strip() if body.model else "default"
    model_used, _model_path = normalize_model_name(settings.llama_model_path or requested_model)

    content = serialize_message_content(body.messages[0].get("content"))
    if content is None or not str(content).strip():
        raise HTTPException(status_code=400, detail="Message content cannot be empty")

    message = Message(thread_id=thread_id, role="user", content=content, model=model_used)
    session.add(message)
    session.add(SearchEvent(event_type="ask", search_mode="llm", query=content, result_count=0, thread_id=thread_id))

    lens_id = body.lens_id.strip() if body.lens_id else None
    lens_message = await get_lens_message(session, lens_id) if lens_id else None
    if lens_id and lens_message is None:
        raise HTTPException(status_code=400, detail="Unknown lens")
    if lens_message is None:
        lens_prompt = None
    else:
        meta = lens_message.meta
        if meta is None:
            raise HTTPException(status_code=400, detail="Lens missing meta")
        lens_prompt = lens_message.content
        if lens_prompt is None:
            raise HTTPException(status_code=400, detail="Lens missing content")

    generation_messages: list[dict[str, Any]] = []
    if lens_prompt is not None:
        generation_messages.append({"role": "system", "content": lens_prompt})
    generation_messages.append({"role": "user", "content": content})

    job_message = await jobs.create_generate_assistant_job(
        session,
        thread_id=thread_id,
        user_message_uuid=message.uuid,
        model=model_used,
        lens_id=lens_id,
    )

    background_tasks.add_task(
        _generate_assistant_response,
        thread_id=thread_id,
        job_uuid=job_message.uuid,
        model_used=model_used,
        requested_model=requested_model,
        messages=generation_messages,
        lens_id=lens_id,
        llama_client=llama_client,
        bind=session.bind,
    )

    return CreateThreadResponse(thread_id=thread_id, message_uuid=message.uuid, job_uuid=job_message.uuid)


@router.get("/threads/{thread_id}", response_model=None)
async def get_thread(
    request: Request,
    thread_id: str,
    session: AsyncSession = Depends(get_readonly_db_session),
) -> Any:
    prompt_result = await session.execute(
        select(Message)
        .where(col(Message.thread_id) == thread_id)
        .where(col(Message.role) == "user")
        .order_by(col(Message.created_at))
        .limit(1)
    )
    prompt = prompt_result.scalar_one_or_none()

    responses_result = await session.execute(
        select(Message)
        .where(col(Message.thread_id) == thread_id)
        .where(col(Message.role) == "assistant")
        .order_by(
            col(Message.accepted_at).desc().nulls_last(),
            col(Message.created_at).desc(),
        )
    )
    responses = list(responses_result.scalars().all())

    if not prompt and not responses:
        raise HTTPException(status_code=404, detail="Thread not found")

    response_data = ThreadResponse(
        thread_id=thread_id,
        prompt=prompt,
        responses=responses,
    )

    if "text/html" in request.headers.get("accept", ""):
        lenses = await list_lens_options(session)
        selected_lens_id: str | None = request.query_params.get("lens_id")
        if not selected_lens_id:
            selected_result = await session.execute(
                select(Message.lens_id)
                .where(col(Message.thread_id) == thread_id)
                .where(col(Message.lens_id).is_not(None))
                .order_by(col(Message.created_at).desc(), col(Message.id).desc())
                .limit(1)
            )
            selected_lens_id = selected_result.scalar_one_or_none()

        return HtpyResponse(
            render_thread_messages(
                thread_id=thread_id,
                prompt=prompt,
                responses=responses,
                lenses=lenses,
                selected_lens_id=selected_lens_id,
            )
        )

    return response_data


@router.get("/threads/{thread_id}/activity", response_model=None)
async def get_thread_activity(
    request: Request,
    thread_id: str,
    session: AsyncSession = Depends(get_readonly_db_session),
) -> Any:
    prompt_result = await session.execute(
        select(Message)
        .where(col(Message.thread_id) == thread_id)
        .where(col(Message.role) == "user")
        .order_by(col(Message.created_at))
        .limit(1)
    )
    prompt = prompt_result.scalar_one_or_none()

    system_result = await session.execute(
        select(Message)
        .where(col(Message.thread_id) == thread_id)
        .where(col(Message.role) == "system")
        .order_by(col(Message.created_at), col(Message.id))
    )
    system_messages = list(system_result.scalars().all())

    if not prompt and not system_messages:
        raise HTTPException(status_code=404, detail="Thread not found")

    if "text/html" in request.headers.get("accept", ""):
        return HtpyResponse(
            render_thread_activity(
                thread_id=thread_id,
                prompt=prompt,
                system_messages=system_messages,
            )
        )

    return {"thread_id": thread_id, "system_messages": system_messages}


@router.get("/threads/{thread_id}/events", response_model=None)
async def thread_events(
    request: Request,
    thread_id: str,
    role: Literal["assistant", "user", "system"] = Query("assistant"),
    after_uuid: str | None = Query(None),
) -> EventSourceResponse:
    async def gen() -> Any:
        after_id: int | None = None
        if after_uuid:
            async with request.app.state.get_db_session(read_only=True) as session:
                after_result = await session.execute(
                    select(Message.id)
                    .where(col(Message.thread_id) == thread_id)
                    .where(col(Message.role) == role)
                    .where(col(Message.uuid) == after_uuid)
                    .limit(1)
                )
                after_id = after_result.scalar_one_or_none()

        while True:
            if await request.is_disconnected():
                return

            async with request.app.state.get_db_session(read_only=True) as session:
                stmt = (
                    select(Message)
                    .where(col(Message.thread_id) == thread_id)
                    .where(col(Message.role) == role)
                    .order_by(col(Message.created_at).desc(), col(Message.id).desc())
                    .limit(1)
                )
                if after_id is not None:
                    stmt = (
                        select(Message)
                        .where(col(Message.thread_id) == thread_id)
                        .where(col(Message.role) == role)
                        .where(col(Message.id) > after_id)
                        .order_by(col(Message.id))
                        .limit(1)
                    )
                result = await session.execute(stmt)
                message = result.scalar_one_or_none()
                if message:
                    payload: dict[str, Any] = {"uuid": message.uuid}
                    if role == "system":
                        payload["meta"] = message.meta
                    yield ServerSentEvent(event=role, data=json.dumps(payload))
                    return

                exists_result = await session.execute(
                    select(Message).where(col(Message.thread_id) == thread_id).limit(1)
                )
                if exists_result.scalar_one_or_none() is None:
                    yield ServerSentEvent(event="error", data=json.dumps({"detail": "Thread not found"}))
                    return

            await asyncio.sleep(0.5)

    return EventSourceResponse(
        gen(),
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
