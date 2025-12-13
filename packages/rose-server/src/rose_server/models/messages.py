"""Message database entity."""

import time
import uuid as uuid_module
from typing import Any, Dict, Optional

from sqlalchemy import JSON, Column, Computed, Index, String
from sqlmodel import Field, SQLModel


class Message(SQLModel, table=True):
    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_completion_id", "completion_id"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    uuid: str = Field(default_factory=lambda: str(uuid_module.uuid4()), index=True, unique=True)
    thread_id: str = Field(index=True)
    role: str
    content: str
    reasoning: Optional[str] = Field(default=None)
    model: str
    meta: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)
    created_at: int = Field(default_factory=lambda: int(time.time()))

    completion_id: Optional[str] = Field(
        default=None,
        sa_column=Column(String, Computed("json_extract(meta, '$.completion_id')"), index=False),
    )
