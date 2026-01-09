from pydantic import BaseModel, Field


class ExportFilter(BaseModel):
    accepted_only: bool
    lens_id: str | None
    thread_ids: list[str] | None


class ExportRequest(BaseModel):
    filters: ExportFilter
    split_ratio: float = Field(ge=0.5, le=0.99)


class ChatMessage(BaseModel):
    role: str
    content: str


class Conversation(BaseModel):
    messages: list[ChatMessage]


class ExportResponse(BaseModel):
    export_id: str
    total_conversations: int
    train_count: int
    valid_count: int
    created_at: int
