import time
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Index
from sqlmodel import Field, SQLModel


class FineTuningJob(SQLModel, table=True):
    __tablename__ = "fine_tuning_jobs"

    id: str = Field(primary_key=True, default_factory=lambda: f"ftjob-{uuid.uuid4().hex[:24]}")
    created_at: int = Field(default_factory=lambda: int(time.time()))
    finished_at: Optional[int] = None
    model: str
    fine_tuned_model: Optional[str] = None
    organization_id: str = Field(default="org-local")
    result_files: List[str] = Field(default_factory=list, sa_type=JSON)
    status: str
    validation_file: Optional[str] = None
    training_file: str
    error: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)
    seed: int = Field(default=42)
    trained_tokens: Optional[int] = None
    meta: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)
    started_at: Optional[int] = None
    suffix: Optional[str] = None
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, sa_type=JSON)
    method: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)
    trainer: str = Field(default="huggingface")  # "huggingface" or "torchtune"
    training_metrics: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)

    __table_args__ = (
        Index("idx_ft_jobs_status", "status"),
        Index("idx_ft_jobs_created", "created_at"),
    )


class FineTuningEvent(SQLModel, table=True):
    __tablename__ = "fine_tuning_events"

    id: str = Field(primary_key=True, default_factory=lambda: f"ftevent-{uuid.uuid4().hex[:24]}")
    object: str = Field(default="fine_tuning.job.event")
    created_at: int = Field(default_factory=lambda: int(time.time()))
    level: str
    message: str
    data: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)
    job_id: str = Field(foreign_key="fine_tuning_jobs.id")

    __table_args__ = (
        Index("idx_events_job_id", "job_id"),
        Index("idx_events_created", "created_at"),
    )
