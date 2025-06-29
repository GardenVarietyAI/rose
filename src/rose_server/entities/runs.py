"""Run database entity."""

import time
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Index
from sqlmodel import Field, SQLModel


class Run(SQLModel, table=True):
    """Run model for database storage."""

    __tablename__: str = "runs"
    id: str = Field(primary_key=True, default_factory=lambda: f"run_{uuid.uuid4().hex[:16]}")
    object: str = Field(default="thread.run")
    created_at: int = Field(default_factory=lambda: int(time.time()))
    thread_id: str = Field(foreign_key="threads.id")
    assistant_id: str = Field(foreign_key="assistants.id")

    # Status fields
    status: str = Field(default="queued")  # queued, in_progress, completed, failed, cancelled, expired
    started_at: Optional[int] = None
    expires_at: Optional[int] = None
    failed_at: Optional[int] = None
    completed_at: Optional[int] = None
    cancelled_at: Optional[int] = None

    # Configuration
    model: str
    instructions: Optional[str] = None
    tools: List[Dict[str, Any]] = Field(default_factory=list, sa_type=JSON)
    meta: Dict[str, Any] = Field(default_factory=dict, sa_type=JSON)

    # Model parameters
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_prompt_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None

    # Tool configuration
    tool_choice: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)
    parallel_tool_calls: bool = Field(default=True)

    # Response configuration
    response_format: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)
    truncation_strategy: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)

    # Results
    last_error: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)
    incomplete_details: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)
    usage: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)

    # Required actions (for tool calls)
    required_action: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)

    __table_args__ = (
        Index("idx_runs_thread", "thread_id"),
        Index("idx_runs_assistant", "assistant_id"),
        Index("idx_runs_status", "status"),
        Index("idx_runs_created", "created_at"),
    )
