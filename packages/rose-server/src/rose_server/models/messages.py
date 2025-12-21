"""Message database entity."""

import time
import uuid as uuid_module
from typing import Any, Dict

from sqlalchemy import JSON, Column, Computed, Index, String
from sqlmodel import Field, SQLModel


class Message(SQLModel, table=True):
    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_completion_id", "completion_id"),)

    id: int | None = Field(default=None, primary_key=True)
    uuid: str = Field(default_factory=lambda: str(uuid_module.uuid4()), index=True, unique=True)
    thread_id: str = Field(index=True)
    role: str
    content: str | None = Field(default=None)
    reasoning: str | None = Field(default=None)
    model: str
    meta: Dict[str, Any] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    created_at: int = Field(default_factory=lambda: int(time.time()))
    accepted_at: int | None = Field(default=None, index=True)

    completion_id: str | None = Field(
        default=None,
        sa_column=Column(String, Computed("json_extract(meta, '$.completion_id')"), index=False),
    )
