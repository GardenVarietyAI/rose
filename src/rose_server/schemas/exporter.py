from pydantic import BaseModel, Field


class ExportFilter(BaseModel):
    accepted_only: bool
    lens_id: str | None
    thread_ids: list[str] | None


class ExportRequest(BaseModel):
    filters: ExportFilter


class ChatMessage(BaseModel):
    role: str
    content: str


class Conversation(BaseModel):
    messages: list[ChatMessage]


class ExportResponse(BaseModel):
    export_id: str
    total_conversations: int
    created_at: int
