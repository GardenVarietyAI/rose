import logging
import uuid

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from rose_server.dependencies import get_db_session, get_llama_client, get_settings
from rose_server.models.messages import Message
from rose_server.models.search_events import SearchEvent
from rose_server.services import assistant, jobs
from rose_server.services.llama import resolve_model, serialize_message_content
from rose_server.settings import Settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["ask"])


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    content: str
    thread_id: str | None = None
    lens_id: str | None = None
    model: str | None = None

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Content cannot be empty")
        return value


class AskResponse(BaseModel):
    thread_id: str
    user_message_id: str
    job_message_id: str
    assistant_message_id: str | None = None
    status: str


@router.post("/ask", response_model=AskResponse)
async def ask(
    body: AskRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
    llama_client: httpx.AsyncClient = Depends(get_llama_client),
    session: AsyncSession = Depends(get_db_session),
) -> AskResponse:
    thread_id = body.thread_id or str(uuid.uuid4())

    requested_model = body.model.strip() if body.model else None
    model_used, _model_path = resolve_model(settings.llama_model_path, requested_model)

    content = serialize_message_content(body.content)
    if content is None or not str(content).strip():
        raise HTTPException(status_code=400, detail="Message content cannot be empty")

    user_message = Message(thread_id=thread_id, role="user", content=content, model=model_used)
    session.add(user_message)
    session.add(SearchEvent(event_type="ask", search_mode="llm", query=content, result_count=0, thread_id=thread_id))

    lens_id = body.lens_id.strip() if body.lens_id else None

    job_message, generation_messages, _lens_at_name = await assistant.prepare_and_generate_assistant(
        session,
        thread_id=thread_id,
        user_message=user_message,
        lens_id=lens_id,
        model_used=model_used,
    )

    background_tasks.add_task(
        jobs.run_generate_assistant_job,
        thread_id=thread_id,
        job_uuid=job_message.uuid,
        model_used=model_used,
        requested_model=requested_model,
        messages=generation_messages,
        lens_id=lens_id,
        llama_client=llama_client,
        bind=session.bind,
    )

    return AskResponse(
        thread_id=thread_id,
        user_message_id=user_message.uuid,
        job_message_id=job_message.uuid,
        assistant_message_id=None,
        status="generating",
    )
