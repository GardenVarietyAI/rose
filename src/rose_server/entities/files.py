"""File entity models."""

import uuid
from typing import Any, Optional

from sqlalchemy import LargeBinary
from sqlmodel import Column, Field, SQLModel


class UploadedFile(SQLModel, table=True):
    """Upload entity for storing file metadata."""

    __tablename__ = "files"

    id: str = Field(
        primary_key=True, default_factory=lambda: f"file-{uuid.uuid4().hex[:24]}", description="File identifier"
    )
    object: str = Field(default="file", description="Object type, always 'file'")
    bytes: int = Field(description="Size of the file in bytes")
    created_at: int = Field(description="Unix timestamp when file was created")
    expires_at: Optional[int] = Field(default=None, description="Unix timestamp when file expires")
    filename: str = Field(description="Name of the file")
    purpose: str = Field(description="assistants, assistants_output, batch, batch_output, fine-tune, fine-tune-results")
    status: Optional[str] = Field(default="processed", description="Deprecated. Status: uploaded, processed, or error")
    status_details: Optional[str] = Field(default=None, description="Deprecated. Details on validation errors")
    storage_path: str = Field(description="Internal storage path relative to uploads directory")
    content: Any = Field(default=None, sa_column=Column(LargeBinary), description="File content stored as BLOB")
