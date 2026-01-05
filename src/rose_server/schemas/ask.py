from pydantic import BaseModel, ConfigDict, field_validator

from rose_server.schemas.query import QueryRequest


class AskRequest(QueryRequest):
    model_config = ConfigDict(extra="forbid")

    content: str
    thread_id: str | None = None
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
    job_id: str
    assistant_message_id: str | None = None
    status: str
