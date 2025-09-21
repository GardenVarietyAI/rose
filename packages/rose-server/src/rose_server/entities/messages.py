"""Message database entity."""

import time
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON
from sqlmodel import Field, SQLModel


class Message(SQLModel, table=True):
    """Message model for database storage."""

    __tablename__ = "messages"

    id: str = Field(primary_key=True, default_factory=lambda: f"msg_{uuid.uuid4().hex[:16]}")
    object: str = Field(default="thread.message")
    role: str
    content: List[Dict[str, Any]] = Field(sa_type=JSON)
    attachments: Optional[List[Dict[str, Any]]] = Field(default=None, sa_type=JSON)
    meta: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)
    status: Optional[str] = Field(default=None)
    created_at: int = Field(default_factory=lambda: int(time.time()))
    response_chain_id: Optional[str] = Field(default=None, index=True)
