from typing import Any

from rose_server.models.messages import Message
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

JOB_OBJECT = "job"
JOB_NAME_GENERATE_ASSISTANT = "generate_assistant"

JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_SUCCEEDED = "succeeded"
JOB_STATUS_FAILED = "failed"


def create_generate_assistant_job(*, thread_id: str, user_message_uuid: str, model: str) -> Message:
    return Message(
        thread_id=thread_id,
        role="system",
        content="Generating response…",
        model=model,
        meta={
            "object": JOB_OBJECT,
            "job_name": JOB_NAME_GENERATE_ASSISTANT,
            "status": JOB_STATUS_QUEUED,
            "user_message_uuid": user_message_uuid,
            "attempt": 0,
        },
    )


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
