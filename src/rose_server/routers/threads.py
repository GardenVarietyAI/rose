import asyncio
import json
import logging
import uuid
from typing import Any, Literal

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from htpy.starlette import HtpyResponse
from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select
from sse_starlette.event import ServerSentEvent
from sse_starlette.sse import EventSourceResponse

from rose_server.dependencies import get_db_session, get_llama_client, get_readonly_db_session, get_settings
from rose_server.models.job_events import JobEvent
from rose_server.models.messages import Message
from rose_server.models.search_events import SearchEvent
from rose_server.routers.lenses import list_lens_options
from rose_server.services import assistant, jobs
from rose_server.services.llama import resolve_model, serialize_message_content
from rose_server.settings import Settings
from rose_server.views.pages.thread_activity import render_thread_activity
from rose_server.views.pages.thread_messages import render_thread_messages

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["threads"])


class JobEventResponse(BaseModel):
    uuid: str
    event_type: str
    job_id: str
    status: str
    created_at: int
    attempt: int
    error: str | None


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

    requested_model = body.model.strip() if body.model else None
    model_used, _model_path = resolve_model(settings.llama_model_path, requested_model)

    content = serialize_message_content(body.messages[0].get("content"))
    if content is None or not str(content).strip():
        raise HTTPException(status_code=400, detail="Message content cannot be empty")

    message = Message(thread_id=thread_id, role="user", content=content, model=model_used)
    session.add(message)
    session.add(SearchEvent(event_type="ask", search_mode="llm", query=content, result_count=0, thread_id=thread_id))

    lens_id = body.lens_id.strip() if body.lens_id else None

    job_id, generation_messages, lens_at_name = await assistant.prepare_and_generate_assistant(
        session,
        user_message=message,
        lens_id=lens_id,
    )

    background_tasks.add_task(
        jobs.run_generate_assistant_job,
        thread_id=thread_id,
        job_id=job_id,
        model_used=model_used,
        requested_model=requested_model,
        messages=generation_messages,
        lens_id=lens_id,
        lens_at_name=lens_at_name,
        llama_client=llama_client,
        bind=session.bind,
    )

    return CreateThreadResponse(thread_id=thread_id, message_uuid=message.uuid, job_uuid=job_id)


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
        .where(col(Message.deleted_at).is_(None))
        .order_by(col(Message.created_at).desc(), col(Message.id).desc())
        .limit(1)
    )
    prompt = prompt_result.scalar_one_or_none()

    responses_sql = """
        WITH ranked AS (
            SELECT
                id,
                row_number() OVER (
                    PARTITION BY COALESCE(root_message_id, uuid)
                    ORDER BY created_at DESC, id DESC
                ) AS rn
            FROM messages
            WHERE thread_id = :thread_id
              AND role = 'assistant'
              AND deleted_at IS NULL
        )
        SELECT m.*
        FROM messages m
        JOIN ranked r ON r.id = m.id
        WHERE r.rn = 1
        ORDER BY m.accepted_at DESC NULLS LAST, m.created_at DESC, m.id DESC
    """
    responses_result = await session.execute(
        select(Message).from_statement(text(responses_sql)), {"thread_id": thread_id}
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
    job_events_result = await session.execute(
        select(JobEvent)
        .where(col(JobEvent.thread_id) == thread_id)
        .order_by(col(JobEvent.created_at).desc(), col(JobEvent.id).desc())
        .limit(10)
    )
    job_events = list(job_events_result.scalars().all())

    if not job_events:
        thread_exists_result = await session.execute(
            select(Message).where(col(Message.thread_id) == thread_id).limit(1)
        )
        if thread_exists_result.scalar_one_or_none() is None:
            raise HTTPException(status_code=404, detail="Thread not found")

    if "text/html" in request.headers.get("accept", ""):
        return HtpyResponse(
            render_thread_activity(
                thread_id=thread_id,
                job_events=job_events,
            )
        )

    return {
        "thread_id": thread_id,
        "job_events": [
            JobEventResponse(
                uuid=je.uuid,
                event_type=je.event_type,
                job_id=je.job_id,
                status=je.status,
                created_at=je.created_at,
                attempt=je.attempt,
                error=je.error,
            )
            for je in job_events
        ],
    }


@router.get("/threads/{thread_id}/stream", response_model=None)
async def thread_stream(
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
