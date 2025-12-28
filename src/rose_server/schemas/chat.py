from typing import Any, Union

from pydantic import BaseModel, field_validator


class ChatRequest(BaseModel):
    thread_id: str | None = None
    model: str | None = None
    messages: list[dict[str, Any]]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    stop: Union[str, list[str]] | None = None
    seed: int | None = None
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Union[str, dict[str, Any]] | None = None
    stream: bool = False

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not value:
            raise ValueError("messages must contain at least one entry")
        return value
