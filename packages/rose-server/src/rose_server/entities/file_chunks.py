"""File chunks entity models."""

import time
from typing import Any, Dict, Optional

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class FileChunk(SQLModel, table=True):
    __tablename__ = "file_chunks"

    content_hash: str = Field(primary_key=True)
    file_id: str = Field(foreign_key="files.id", index=True)
    chunk_index: int
    content: str
    embedding: Optional[bytes] = Field(default=None)
    meta: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    created_at: int = Field(default_factory=lambda: int(time.time()))
