"""Vector store entity models."""

import time
import uuid
from typing import Any, Dict, Optional, Union

from sqlalchemy import JSON, Column, Index, UniqueConstraint
from sqlmodel import Field, SQLModel


class VectorStore(SQLModel, table=True):
    """Vector store entity for storing vector store metadata."""

    __tablename__ = "vector_stores"

    id: str = Field(primary_key=True, default_factory=lambda: f"vs_{uuid.uuid4().hex[:24]}")
    object: str = Field(default="vector_store")
    name: str
    dimensions: int
    created_at: int = Field(default_factory=lambda: int(time.time()))
    last_used_at: Optional[int] = Field(default=None)
    meta: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))


class Document(SQLModel, table=True):
    """Document entity for storing chunked documents."""

    __tablename__ = "documents"

    id: str = Field(primary_key=True, default_factory=lambda: f"doc_{uuid.uuid4().hex[:24]}")
    vector_store_id: str = Field(foreign_key="vector_stores.id")
    file_id: str = Field(index=True)
    chunk_index: int
    content: str
    content_hash: Optional[str] = Field(default=None, index=True)  # SHA256 hash of content
    meta: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    created_at: int = Field(default_factory=lambda: int(time.time()))

    __table_args__ = (
        Index("idx_doc_vs_file", "vector_store_id", "file_id"),
        Index("idx_doc_vs_created", "vector_store_id", "created_at"),
    )


class VectorStoreFile(SQLModel, table=True):
    """Vector store file entity for tracking file processing status."""

    __tablename__ = "vector_store_files"

    id: str = Field(primary_key=True, default_factory=lambda: f"vsf_{uuid.uuid4().hex[:24]}")
    object: str = Field(default="vector_store.file")
    vector_store_id: str = Field(foreign_key="vector_stores.id")
    file_id: str = Field(foreign_key="files.id")
    status: str = Field(default="in_progress")
    created_at: int = Field(default_factory=lambda: int(time.time()))
    last_error: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    attributes: Optional[Dict[str, Union[str, float, bool]]] = Field(default=None, sa_column=Column(JSON))

    __table_args__ = (
        UniqueConstraint("vector_store_id", "file_id", name="idx_vector_store_files_unique"),
        Index("idx_vsf_lookup", "vector_store_id", "file_id"),
        Index("idx_vsf_status", "vector_store_id", "status"),
    )


class DocumentSearchResult(SQLModel):
    """Document search result with distance score."""

    document: Document
    score: float  # Raw distance from sqlite-vec
