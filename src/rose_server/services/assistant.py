from typing import Any

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from rose_server.models.message_types import FactsheetMessage, LensMessage
from rose_server.models.messages import Message
from rose_server.routers.lenses import get_lens_message
from rose_server.services import jobs
from rose_server.services.factsheets import get_factsheet_message


async def prepare_and_generate_assistant(
    session: AsyncSession,
    *,
    user_message: Message,
    lens_id: str | None,
    factsheet_ids: list[str] | None,
) -> tuple[str, list[dict[str, Any]], str | None, list[str]]:
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

    normalized_factsheet_ids: list[str] = []
    if factsheet_ids:
        seen: set[str] = set()
        for factsheet_id in factsheet_ids:
            if not factsheet_id or factsheet_id in seen:
                continue
            normalized_factsheet_ids.append(factsheet_id)
            seen.add(factsheet_id)

    resolved_factsheet_ids: list[str] = []
    factsheet_system_messages: list[dict[str, Any]] = []
    for factsheet_id in normalized_factsheet_ids:
        factsheet_message = await get_factsheet_message(session, factsheet_id)
        if factsheet_message is None:
            continue
        try:
            factsheet = FactsheetMessage(message=factsheet_message)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail="Factsheet missing meta") from e
        factsheet_body = factsheet.message.content
        if factsheet_body is None:
            raise HTTPException(status_code=400, detail="Factsheet missing content")

        resolved_factsheet_ids.append(factsheet.factsheet_id)
        factsheet_system_messages.append(
            {
                "role": "system",
                "content": f"Factsheet: {factsheet.title} (#{factsheet.tag})\n\n{factsheet_body}",
            }
        )

    if user_message.thread_id is None:
        raise HTTPException(status_code=400, detail="User message missing thread_id")

    generation_messages: list[dict[str, Any]] = []
    if lens_prompt is not None:
        generation_messages.append({"role": "system", "content": lens_prompt})
    generation_messages.extend(factsheet_system_messages)
    generation_messages.append({"role": "user", "content": user_message.content})

    job_uuid = await jobs.create_generate_assistant_job(
        session,
        user_message_uuid=user_message.uuid,
        thread_id=user_message.thread_id,
    )

    return job_uuid, generation_messages, lens_at_name, resolved_factsheet_ids
