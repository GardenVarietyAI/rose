from pydantic import BaseModel


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
