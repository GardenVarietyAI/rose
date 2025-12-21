import asyncio
import json
import logging
import pathlib
import uuid
from typing import Any, Literal

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from htpy.starlette import HtpyResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from rose_server.dependencies import get_db_session, get_llama_client, get_readonly_db_session, get_settings
from rose_server.models.messages import Message
from rose_server.models.search_events import SearchEvent
from rose_server.services import jobs
from rose_server.settings import Settings
from rose_server.views.pages.thread import render_thread
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


class CompletionMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    content: Any


class CompletionChoice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: CompletionMessage
    finish_reason: Any = None


class CompletionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    choices: list[CompletionChoice] = Field(default_factory=list)
    model: str | None = None
    usage: dict[str, Any] | None = None


def _normalize_model_name(model: str | None) -> str:
    if not model or model == "default":
        return "default"
    if ("/" in model or "\\" in model) and model.lower().endswith(".gguf"):
        name = pathlib.PurePath(model).name
        return name or model
    return model


def _serialize_message_content(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content)
    except (TypeError, ValueError):
        return str(content)


async def _request_chat_completion(
    llama_client: httpx.AsyncClient,
    payload: dict[str, Any],
) -> CompletionResponse | None:
    try:
        completion_response = await llama_client.post("chat/completions", json=payload)
        completion_response.raise_for_status()
    except httpx.HTTPStatusError as e:
        upstream_response = e.response
        logger.warning("LLM server error (%s): %s", upstream_response.status_code, upstream_response.text)
        return None
    except httpx.RequestError:
        logger.exception("LLM server unavailable")
        return None

    try:
        response_json = completion_response.json()
    except ValueError:
        logger.warning("LLM server returned invalid JSON: %s", completion_response.text)
        return None

    try:
        completion = CompletionResponse.model_validate(response_json)
    except ValidationError:
        logger.warning("Invalid completion response format: %r", response_json)
        return None

    if not completion.choices:
        logger.warning("Completion missing choices: %r", response_json)
        return None

    return completion


async def _generate_assistant_response(
    *,
    thread_id: str,
    job_uuid: str,
    model_used: str,
    requested_model: str | None,
    messages: list[dict[str, Any]],
    llama_client: httpx.AsyncClient,
    bind: Any,
) -> None:
    async with AsyncSession(bind=bind, expire_on_commit=False) as session:
        job_message = await jobs.mark_running(session, job_uuid)
        if job_message is None:
            logger.warning("Job message missing for uuid=%s", job_uuid)
            return

        await session.commit()

        payload: dict[str, Any] = {"messages": messages, "stream": False}
        if requested_model and requested_model != "default":
            payload["model"] = requested_model

        completion = await _request_chat_completion(llama_client, payload)
        if completion is None:
            await jobs.mark_failed(session, job_uuid, error="LLM completion failed")
            await session.commit()
            return

        choice = completion.choices[0]
        assistant_content = _serialize_message_content(choice.message.content)
        if assistant_content is None:
            logger.warning("Completion missing assistant content: %r", choice)
            await jobs.mark_failed(session, job_uuid, error="Completion missing assistant content")
            await session.commit()
            return

        assistant_message = Message(
            thread_id=thread_id,
            role="assistant",
            content=assistant_content,
            model=model_used,
            meta={
                "completion_id": completion.id,
                "finish_reason": choice.finish_reason,
            },
        )
        session.add(assistant_message)
        await session.flush()

        await jobs.mark_succeeded(session, job_uuid, assistant_uuid=assistant_message.uuid)
        await session.commit()


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
    model_used = _normalize_model_name(settings.llama_model_path or requested_model)

    content = _serialize_message_content(body.messages[0].get("content"))
    if content is None or not str(content).strip():
        raise HTTPException(status_code=400, detail="Message content cannot be empty")

    message = Message(thread_id=thread_id, role="user", content=content, model=model_used)
    session.add(message)
    session.add(SearchEvent(event_type="ask", search_mode="llm", query=content, result_count=0, thread_id=thread_id))

    job_message = jobs.create_generate_assistant_job(
        thread_id=thread_id,
        for_message_uuid=message.uuid,
        model=model_used,
    )
    session.add(job_message)

    await session.commit()

    background_tasks.add_task(
        _generate_assistant_response,
        thread_id=thread_id,
        job_uuid=job_message.uuid,
        model_used=model_used,
        requested_model=requested_model,
        messages=body.messages,
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

    if "text/html" in request.headers.get("accept", ""):
        return HtpyResponse(
            render_thread(
                thread_id=thread_id,
                prompt=prompt,
                responses=responses,
            )
        )

    return response_data


@router.get("/threads/{thread_id}/events", response_model=None)
async def thread_events(
    request: Request,
    thread_id: str,
    role: Literal["assistant", "user", "system"] = Query("assistant"),
) -> EventSourceResponse:
    async def gen() -> Any:
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
                result = await session.execute(stmt)
                message = result.scalar_one_or_none()
                if message:
                    payload: dict[str, Any] = {"uuid": message.uuid}
                    if role == "system":
                        payload["meta"] = message.meta
                    yield ServerSentEvent(event=role, data=json.dumps(payload))
                    return

                exists_stmt = select(Message).where(col(Message.thread_id) == thread_id).limit(1)
                exists_result = await session.execute(exists_stmt)
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
