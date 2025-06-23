"""Assistant database entities."""

import time
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Index
from sqlmodel import Field, SQLModel


class Assistant(SQLModel, table=True):
    """Assistant model for database storage."""

    __tablename__ = "assistants"
    id: str = Field(primary_key=True)
    object: str = Field(default="assistant")
    created_at: int = Field(default_factory=lambda: int(time.time()))
    name: Optional[str] = None
    description: Optional[str] = None
    model: str
    instructions: Optional[str] = None
    tools: List[Dict[str, Any]] = Field(default_factory=list, sa_type=JSON)
    tool_resources: Dict[str, Any] = Field(default_factory=dict, sa_type=JSON)
    meta: Dict[str, Any] = Field(default_factory=dict, sa_type=JSON)
    temperature: Optional[float] = Field(default=1.0)
    top_p: Optional[float] = Field(default=1.0)
    response_format: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)
    __table_args__ = (
        Index("idx_assistants_created", "created_at"),
        Index("idx_assistants_model", "model"),
    )
