"""Message database entity."""

import time
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Index
from sqlmodel import Field, SQLModel


class Message(SQLModel, table=True):
    """Message model for database storage."""

    __tablename__ = "messages"

    id: str = Field(primary_key=True, default_factory=lambda: f"msg_{uuid.uuid4().hex[:16]}")
    object: str = Field(default="thread.message")
    created_at: int = Field(default_factory=lambda: int(time.time()))
    thread_id: Optional[str] = Field(default=None, foreign_key="threads.id")
    role: str
    content: List[Dict[str, Any]] = Field(sa_type=JSON)
    assistant_id: Optional[str] = None
    run_id: Optional[str] = None
    attachments: List[Dict[str, Any]] = Field(default_factory=list, sa_type=JSON)
    meta: Dict[str, Any] = Field(default_factory=dict, sa_type=JSON)
    response_chain_id: Optional[str] = Field(default=None, index=True)
    status: str = Field(default="completed")
    incomplete_details: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)
    incomplete_at: Optional[int] = None
    completed_at: Optional[int] = Field(default_factory=lambda: int(time.time()))

    __table_args__ = (
        Index("idx_messages_thread", "thread_id"),
        Index("idx_messages_created", "created_at"),
        Index("idx_messages_role", "role"),
        Index("idx_messages_response_chain", "response_chain_id"),
    )

    @classmethod
    def generate_chain_id(cls) -> str:
        """Generate a unique conversation chain ID."""
        return f"chain_{uuid.uuid4().hex[:16]}"
