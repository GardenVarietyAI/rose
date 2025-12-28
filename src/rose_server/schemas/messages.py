from typing import Any, Literal

from pydantic import BaseModel

from rose_server.models.messages import Message


class UpdateMessageRequest(BaseModel):
    accepted: bool


class UpdateMessageResponse(BaseModel):
    status: str
    message_uuid: str
    accepted: bool


class CreateMessageRequest(BaseModel):
    thread_id: str
    role: Literal["assistant"] = "assistant"
    content: Any | None = None
    model: str | None = None
    meta: dict[str, Any] | None = None
    generate_assistant: bool = False
    lens_id: str | None = None


class CreateMessageResponse(BaseModel):
    thread_id: str
    message_uuid: str
    job_uuid: str | None = None


class RevisionMessage(BaseModel):
    uuid: str
    thread_id: str | None
    role: str
    content: str | None
    reasoning: str | None
    model: str | None
    meta: dict[str, Any] | None
    created_at: int
    accepted_at: int | None


class ListRevisionsResponse(BaseModel):
    root_message_id: str
    latest_message_uuid: str
    messages: list[RevisionMessage]


class CreateRevisionRequest(BaseModel):
    content: Any


class CreateRevisionResponse(BaseModel):
    root_message_id: str
    message_uuid: str
    latest_message_uuid: str


class ListMessagesResponse(BaseModel):
    messages: list[Message]
