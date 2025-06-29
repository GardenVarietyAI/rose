"""Thread and message database entities."""

import time
import uuid
from typing import Any, Dict, Optional

from sqlalchemy import JSON, Index
from sqlmodel import Field, SQLModel


class Thread(SQLModel, table=True):
    """Thread model for database storage."""

    __tablename__: str = "threads"
    id: str = Field(primary_key=True, default_factory=lambda: f"thread_{uuid.uuid4().hex[:16]}")
    object: str = Field(default="thread")
    created_at: int = Field(default_factory=lambda: int(time.time()))
    tool_resources: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)
    meta: Dict[str, Any] = Field(default_factory=dict, sa_type=JSON)
    __table_args__ = (Index("idx_threads_created", "created_at"),)


class MessageMetadata(SQLModel, table=True):
    """Additional metadata for messages."""

    __tablename__: str = "message_metadata"
    id: int = Field(primary_key=True)
    message_id: str = Field(foreign_key="messages.id")
    key: str
    value: str
    __table_args__ = (
        Index("idx_message_metadata_message", "message_id"),
        Index("idx_message_metadata_key", "key"),
    )
