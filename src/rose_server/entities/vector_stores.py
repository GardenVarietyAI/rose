"""Vector store entity models."""

import uuid
from typing import Any, Dict, Optional

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class VectorStore(SQLModel, table=True):
    """Vector store entity for storing vector store metadata."""

    __tablename__ = "vector_stores"

    id: str = Field(primary_key=True, default_factory=lambda: f"vs_{uuid.uuid4().hex[:24]}")
    object: str = Field(default="vector_store")
    name: str
    dimensions: int
    created_at: int
    last_used_at: Optional[int] = Field(default=None)
    meta: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))


class Document(SQLModel, table=True):
    """Document entity for storing chunked documents."""

    __tablename__ = "documents"

    id: str = Field(primary_key=True, default_factory=lambda: f"doc_{uuid.uuid4().hex[:24]}")
    vector_store_id: str = Field(foreign_key="vector_stores.id")
    chunk_index: int
    content: str
    meta: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    created_at: int


class DocumentSearchResult(SQLModel):
    """Document search result with similarity score."""
    
    document: Document
    score: float  # 1 - distance, higher is more similar