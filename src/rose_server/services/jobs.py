from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from rose_server.models.messages import Message
from rose_server.services.llama import (
    LlamaError,
    parse_completion_response,
    request_chat_completion_json,
    serialize_message_content,
)

JOB_OBJECT = "job"
JOB_NAME_GENERATE_ASSISTANT = "generate_assistant"

JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_SUCCEEDED = "succeeded"
JOB_STATUS_FAILED = "failed"


async def create_generate_assistant_job(
    session: AsyncSession,
    *,
    thread_id: str,
    user_message_uuid: str,
    model: str,
    lens_id: str | None = None,
) -> Message:
    message = Message(
        thread_id=thread_id,
        role="system",
        content="Generating response",
        model=model,
        meta={
            "object": JOB_OBJECT,
            "job_name": JOB_NAME_GENERATE_ASSISTANT,
            "status": JOB_STATUS_QUEUED,
            "user_message_uuid": user_message_uuid,
            "attempt": 0,
            **({"lens_id": lens_id} if lens_id else {}),
        },
    )
    session.add(message)
    await session.flush()
    return message


async def get(session: AsyncSession, job_uuid: str) -> Message | None:
    stmt = select(Message).where(col(Message.uuid) == job_uuid).limit(1)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def mark_running(session: AsyncSession, job_uuid: str) -> Message | None:
    message = await get(session, job_uuid)
    if message is None:
        return None
    _update_meta(message, status=JOB_STATUS_RUNNING)
    return message


async def mark_succeeded(session: AsyncSession, job_uuid: str, *, assistant_message_uuid: str) -> Message | None:
    message = await get(session, job_uuid)
    if message is None:
        return None
    _update_meta(message, status=JOB_STATUS_SUCCEEDED, assistant_message_uuid=assistant_message_uuid)
    return message


async def mark_failed(
    session: AsyncSession,
    job_uuid: str,
    *,
    error: str,
    content: str = "Generation failed.",
) -> Message | None:
    message = await get(session, job_uuid)
    if message is None:
        return None
    _update_meta(message, status=JOB_STATUS_FAILED, error=error)
    message.content = content
    return message


def _update_meta(message: Message, **updates: Any) -> None:
    meta = dict(message.meta or {})
    meta.update(updates)
    message.meta = meta


async def run_generate_assistant_job(
    *,
    thread_id: str,
    job_uuid: str,
    model_used: str,
    requested_model: str | None,
    messages: list[dict[str, Any]],
    lens_id: str | None = None,
    llama_client: httpx.AsyncClient,
    bind: Any,
) -> str | None:
    async with AsyncSession(bind=bind, expire_on_commit=False) as session:
        job_message = await mark_running(session, job_uuid)
        if job_message is None:
            return None

        await session.commit()

        payload: dict[str, Any] = {"messages": messages, "stream": False}
        if requested_model and requested_model != "default":
            payload["model"] = requested_model

        try:
            response_json = await request_chat_completion_json(llama_client, payload)
            completion = parse_completion_response(response_json)
        except LlamaError as e:
            await mark_failed(session, job_uuid, error=str(e))
            await session.commit()
            return None

        choice = completion.choices[0]
        assistant_content = serialize_message_content(choice.message.content)
        if assistant_content is None or not str(assistant_content).strip():
            await mark_failed(session, job_uuid, error="Completion missing assistant content")
            await session.commit()
            return None

        assistant_message = Message(
            thread_id=thread_id,
            role="assistant",
            content=assistant_content,
            model=model_used,
            meta={
                "completion_id": completion.id,
                "finish_reason": choice.finish_reason,
                **({"lens_id": lens_id} if lens_id else {}),
            },
        )
        session.add(assistant_message)
        await session.flush()

        await mark_succeeded(session, job_uuid, assistant_message_uuid=assistant_message.uuid)
        await session.commit()
        return assistant_message.uuid
