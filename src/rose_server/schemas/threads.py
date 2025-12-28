from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from rose_server.models.messages import Message


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


class ThreadListItem(BaseModel):
    thread_id: str
    first_message_content: str | None
    first_message_role: str
    created_at: int
    last_activity_at: int
    has_assistant_response: bool
    import_source: str | None


class ThreadListResponse(BaseModel):
    threads: list[ThreadListItem]
    total: int
    page: int
    limit: int
