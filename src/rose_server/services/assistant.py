from typing import Any

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from rose_server.models.message_types import LensMessage
from rose_server.models.messages import Message
from rose_server.routers.lenses import get_lens_message
from rose_server.services import jobs


async def prepare_and_generate_assistant(
    session: AsyncSession,
    *,
    thread_id: str,
    user_message: Message,
    lens_id: str | None,
    model_used: str,
) -> tuple[Message, list[dict[str, Any]], str | None]:
    lens_message = await get_lens_message(session, lens_id) if lens_id else None
    if lens_id and lens_message is None:
        raise HTTPException(status_code=400, detail="Unknown lens")

    lens_at_name: str | None = None
    lens_prompt: str | None = None

    if lens_message is not None:
        try:
            lens = LensMessage(message=lens_message)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail="Lens missing meta") from e
        lens_prompt = lens.message.content
        if lens_prompt is None:
            raise HTTPException(status_code=400, detail="Lens missing content")
        lens_at_name = lens.at_name

    generation_messages: list[dict[str, Any]] = []
    if lens_prompt is not None:
        generation_messages.append({"role": "system", "content": lens_prompt})
    generation_messages.append({"role": "user", "content": user_message.content})

    job_message = await jobs.create_generate_assistant_job(
        session,
        thread_id=thread_id,
        user_message_uuid=user_message.uuid,
        model=model_used,
        lens_id=lens_id,
        lens_at_name=lens_at_name,
    )

    return job_message, generation_messages, lens_at_name
