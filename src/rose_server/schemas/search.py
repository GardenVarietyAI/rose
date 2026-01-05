from pydantic import BaseModel

from rose_server.schemas.query import QueryRequest


class SearchHit(BaseModel):
    thread_id: str
    score: float
    user_message_id: str
    user_message_text: str
    user_message_excerpt: str | None
    user_message_created_at: int
    assistant_message_id: str
    assistant_message_text: str
    assistant_message_excerpt: str
    assistant_message_created_at: int
    assistant_message_model: str | None
    accepted: bool
    matched_role: str
    matched_message_id: str


class SearchResponse(BaseModel):
    index: str
    query: str
    hits: list[SearchHit]


class SearchRequest(QueryRequest):
    pass
