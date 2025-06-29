"""Run step database entity."""

import time
import uuid
from typing import Any, Dict, Optional

from sqlalchemy import JSON, Index
from sqlmodel import Field, SQLModel


class RunStep(SQLModel, table=True):
    """Run step model for database storage."""

    __tablename__: str = "run_steps"
    id: str = Field(primary_key=True, default_factory=lambda: f"step_{uuid.uuid4().hex[:16]}")
    object: str = Field(default="thread.run.step")
    created_at: int = Field(default_factory=lambda: int(time.time()))
    run_id: str = Field(foreign_key="runs.id")
    assistant_id: str
    thread_id: str

    # Step type and details
    type: str = Field(description="Type of run step")  # message_creation or tool_calls
    step_details: Dict[str, Any] = Field(sa_type=JSON)

    # Status fields
    status: str = Field(default="in_progress")  # in_progress, completed, failed, cancelled, expired
    cancelled_at: Optional[int] = None
    completed_at: Optional[int] = None
    expired_at: Optional[int] = None
    failed_at: Optional[int] = None

    # Error and usage
    last_error: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)
    usage: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)

    __table_args__ = (
        Index("idx_run_steps_run", "run_id"),
        Index("idx_run_steps_created", "created_at"),
        Index("idx_run_steps_status", "status"),
    )
