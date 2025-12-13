"""Message database entity."""

import time
import uuid
from typing import Any, Dict, Optional

from sqlalchemy import JSON
from sqlmodel import Field, SQLModel


class Message(SQLModel, table=True):
    __tablename__ = "messages"

    id: str = Field(primary_key=True, default_factory=lambda: f"msg_{uuid.uuid4().hex[:16]}")
    thread_id: str = Field(index=True)
    role: str
    content: str
    model: str
    meta: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)
    created_at: int = Field(default_factory=lambda: int(time.time()))
