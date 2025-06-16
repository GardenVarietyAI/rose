"""Job queue entities."""

from typing import Dict, Optional

from sqlalchemy import JSON, Index
from sqlmodel import Field, SQLModel


class Job(SQLModel, table=True):
    """Generic job for background processing."""

    __tablename__ = "jobs"
    id: int = Field(primary_key=True)
    type: str
    status: str
    payload: Dict = Field(sa_type=JSON)
    result: Optional[Dict] = Field(default=None, sa_type=JSON)
    error: Optional[str] = None
    created_at: int
    started_at: Optional[int] = None
    completed_at: Optional[int] = None
    attempts: int = Field(default=0)
    max_attempts: int = Field(default=3)
    __table_args__ = (
        Index("idx_jobs_status", "status"),
        Index("idx_jobs_type_status", "type", "status"),
    )
