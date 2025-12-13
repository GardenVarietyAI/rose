"""Message database entity."""

import time
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON
from sqlmodel import Field, SQLModel


class Message(SQLModel, table=True):
    __tablename__ = "messages"

    id: str = Field(primary_key=True, default_factory=lambda: f"msg_{uuid.uuid4().hex[:16]}")
    object: str = Field(default="thread.message")
    thread_id: str = Field(index=True)
    role: str
    content: List[Dict[str, Any]] = Field(sa_type=JSON)
    attachments: Optional[List[Dict[str, Any]]] = Field(default=None, sa_type=JSON)
    meta: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)
    status: Optional[str] = Field(default=None)
    created_at: int = Field(default_factory=lambda: int(time.time()))
    updated_at: int = Field(default_factory=lambda: int(time.time()))

    # Thread metadata (only populated on first message)
    thread_title: Optional[str] = Field(default=None)
