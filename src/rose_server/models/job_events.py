import time
import uuid

from sqlalchemy import Index
from sqlmodel import Field, SQLModel


class JobEvent(SQLModel, table=True):
    __tablename__ = "job_events"
    __table_args__ = (
        Index("ix_job_events_event_type", "event_type"),
        Index("ix_job_events_job_id", "job_id"),
        Index("ix_job_events_thread_id", "thread_id"),
        Index("ix_job_events_created_at", "created_at"),
        Index("ix_job_events_status", "status"),
        Index("ix_job_events_job_id_created_at", "job_id", "created_at"),
    )

    id: int | None = Field(default=None, primary_key=True)
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()), index=True, unique=True)
    event_type: str = Field(default="job", index=True)
    job_id: str = Field(index=True)
    thread_id: str = Field(index=True)
    created_at: int = Field(default_factory=lambda: int(time.time()))
    status: str
    attempt: int = Field(default=0)
    error: str | None = Field(default=None)
