from typing import Any, Literal

from pydantic import BaseModel


class UpdateMessageRequest(BaseModel):
    accepted: bool


class CreateMessageRequest(BaseModel):
    thread_id: str
    role: Literal["assistant"] = "assistant"
    content: Any | None = None
    model: str | None = None
    meta: dict[str, Any] | None = None
    generate_assistant: bool = False
    lens_id: str | None = None


class CreateRevisionRequest(BaseModel):
    content: Any
