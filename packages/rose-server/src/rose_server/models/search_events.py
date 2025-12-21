import time
import uuid as uuid_module

from sqlalchemy import Index
from sqlmodel import Field, SQLModel


class SearchEvent(SQLModel, table=True):
    __tablename__ = "search_events"
    __table_args__ = (Index("ix_search_events_type_created", "event_type", "created_at"),)

    id: int | None = Field(default=None, primary_key=True)
    uuid: str = Field(default_factory=lambda: str(uuid_module.uuid4()), index=True, unique=True)
    event_type: str = Field(index=True)
    search_mode: str = Field(index=True)
    query: str
    original_query: str | None = Field(default=None)
    result_count: int = Field(index=True)
    thread_id: str | None = Field(default=None, index=True)
    created_at: int = Field(default_factory=lambda: int(time.time()), index=True)
