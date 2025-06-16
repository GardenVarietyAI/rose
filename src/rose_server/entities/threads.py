"""Thread and message database entities."""
import time
from typing import Any, Dict, List, Optional
from sqlalchemy import JSON, Index
from sqlmodel import Field, SQLModel

class Thread(SQLModel, table=True):
    """Thread model for database storage."""

    __tablename__ = "threads"
    id: str = Field(primary_key=True)
    object: str = Field(default="thread")
    created_at: int = Field(default_factory=lambda: int(time.time()))
    tool_resources: Dict[str, Any] = Field(default_factory=dict, sa_type=JSON)
    meta: Dict[str, Any] = Field(default_factory=dict, sa_type=JSON)
    __table_args__ = (Index("idx_threads_created", "created_at"),)

class Message(SQLModel, table=True):
    """Message model for database storage."""

    __tablename__ = "messages"
    id: str = Field(primary_key=True)
    object: str = Field(default="thread.message")
    created_at: int = Field(default_factory=lambda: int(time.time()))
    thread_id: Optional[str] = Field(default=None, foreign_key="threads.id")
    status: str = Field(default="completed")
    incomplete_details: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)
    incomplete_at: Optional[int] = None
    completed_at: Optional[int] = Field(default_factory=lambda: int(time.time()))
    role: str
    content: List[Dict[str, Any]] = Field(sa_type=JSON)
    assistant_id: Optional[str] = None
    run_id: Optional[str] = None
    attachments: List[Dict[str, Any]] = Field(default_factory=list, sa_type=JSON)
    meta: Dict[str, Any] = Field(default_factory=dict, sa_type=JSON)
    __table_args__ = (
        Index("idx_messages_thread", "thread_id"),
        Index("idx_messages_created", "created_at"),
        Index("idx_messages_role", "role"),
    )

class MessageMetadata(SQLModel, table=True):
    """Additional metadata for messages."""

    __tablename__ = "message_metadata"
    id: int = Field(primary_key=True)
    message_id: str = Field(foreign_key="messages.id")
    key: str
    value: str
    __table_args__ = (
        Index("idx_message_metadata_message", "message_id"),
        Index("idx_message_metadata_key", "key"),
    )