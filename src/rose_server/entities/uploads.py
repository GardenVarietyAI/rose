"""File entity models."""

from typing import Optional

from sqlmodel import Field, SQLModel


class Upload(SQLModel, table=True):
    """Upload entity for storing file metadata."""

    __tablename__ = "uploads"

    id: str = Field(primary_key=True, description="File identifier")
    object: str = Field(default="file", description="Object type, always 'file'")
    bytes: int = Field(description="Size of the file in bytes")
    created_at: int = Field(description="Unix timestamp when file was created")
    expires_at: Optional[int] = Field(default=None, description="Unix timestamp when file expires")
    filename: str = Field(description="Name of the file")
    purpose: str = Field(
        description="Intended purpose: assistants, assistants_output, batch, batch_output, fine-tune, "
        "fine-tune-results, vision"
    )
    status: Optional[str] = Field(default="processed", description="Deprecated. Status: uploaded, processed, or error")
    status_details: Optional[str] = Field(default=None, description="Deprecated. Details on validation errors")

    # Additional fields for internal use
    storage_path: str = Field(description="Internal storage path relative to uploads directory")
