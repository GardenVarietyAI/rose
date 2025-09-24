import time
import uuid
from typing import Any, Dict, Optional

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class FileChunk(SQLModel, table=True):
    __tablename__ = "file_chunks"

    id: str = Field(primary_key=True, default_factory=lambda: f"fc_{uuid.uuid4().hex}")
    content_hash: str = Field(index=True)
    file_id: str = Field(foreign_key="files.id", index=True)
    chunk_index: int
    content: str
    embedding: Optional[bytes] = Field(default=None)
    meta: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    created_at: int = Field(default_factory=lambda: int(time.time()))
