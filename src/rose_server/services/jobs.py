from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from rose_server.models.job_events import JobEvent
from rose_server.models.messages import Message
from rose_server.services.llama import (
    LlamaError,
    parse_completion_response,
    request_chat_completion_json,
    serialize_message_content,
)

JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_SUCCEEDED = "succeeded"
JOB_STATUS_FAILED = "failed"


async def create_generate_assistant_job(
    session: AsyncSession,
    *,
    user_message_uuid: str,
    thread_id: str,
) -> str:
    job_event = JobEvent(
        job_id=user_message_uuid,
        thread_id=thread_id,
        status=JOB_STATUS_QUEUED,
        attempt=0,
    )
    session.add(job_event)
    await session.flush()
    return user_message_uuid


async def create_job_event(
    session: AsyncSession,
    *,
    job_id: str,
    thread_id: str,
    status: str,
    attempt: int = 0,
    error: str | None = None,
) -> JobEvent:
    job_event = JobEvent(
        job_id=job_id,
        thread_id=thread_id,
        status=status,
        attempt=attempt,
        error=error,
    )
    session.add(job_event)
    await session.flush()
    return job_event


async def get_latest_job_event(
    session: AsyncSession,
    *,
    job_id: str,
) -> JobEvent | None:
    result = await session.execute(
        select(JobEvent)
        .where(col(JobEvent.job_id) == job_id)
        .order_by(col(JobEvent.created_at).desc(), col(JobEvent.id).desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def run_generate_assistant_job(
    *,
    thread_id: str,
    job_id: str,
    model_used: str,
    requested_model: str | None,
    messages: list[dict[str, Any]],
    lens_id: str | None = None,
    lens_at_name: str | None = None,
    factsheet_ids: list[str] | None = None,
    llama_client: httpx.AsyncClient,
    bind: Any,
) -> str | None:
    async with AsyncSession(bind=bind, expire_on_commit=False) as session:
        await create_job_event(session, job_id=job_id, thread_id=thread_id, status=JOB_STATUS_RUNNING)
        await session.commit()

        payload: dict[str, Any] = {"messages": messages, "stream": False}
        if requested_model and requested_model != "default":
            payload["model"] = requested_model

        try:
            response_json = await request_chat_completion_json(llama_client, payload)
            completion = parse_completion_response(response_json)
        except LlamaError as e:
            await create_job_event(session, job_id=job_id, thread_id=thread_id, status=JOB_STATUS_FAILED, error=str(e))
            await session.commit()
            return None

        choice = completion.choices[0]
        assistant_content = serialize_message_content(choice.message.content)
        if assistant_content is None or not str(assistant_content).strip():
            await create_job_event(
                session,
                job_id=job_id,
                thread_id=thread_id,
                status=JOB_STATUS_FAILED,
                error="Completion missing assistant content",
            )
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
                **({"lens_at_name": lens_at_name} if lens_at_name else {}),
                **({"factsheet_ids": factsheet_ids} if factsheet_ids else {}),
            },
        )
        session.add(assistant_message)
        await session.flush()

        await create_job_event(session, job_id=job_id, thread_id=thread_id, status=JOB_STATUS_SUCCEEDED)
        await session.commit()
        return assistant_message.uuid
