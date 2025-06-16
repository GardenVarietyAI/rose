"""Fine-tuning database entities."""

import time
from typing import Dict, List, Optional

from openai.types.fine_tuning import (
    FineTuningJob as OpenAIFineTuningJob,
)
from openai.types.fine_tuning import (
    FineTuningJobEvent as OpenAIFineTuningJobEvent,
)
from sqlalchemy import JSON, Index
from sqlmodel import Field, SQLModel


class FineTuningJob(SQLModel, table=True):
    """Fine-tuning job model - OpenAI compatible."""

    __tablename__ = "fine_tuning_jobs"
    id: str = Field(primary_key=True)
    created_at: int
    finished_at: Optional[int] = None
    model: str
    fine_tuned_model: Optional[str] = None
    organization_id: str = Field(default="org-local")
    result_files: List[str] = Field(default_factory=list, sa_type=JSON)
    status: str
    validation_file: Optional[str] = None
    training_file: str
    error: Optional[Dict] = Field(default=None, sa_type=JSON)
    seed: int = Field(default=42)
    trained_tokens: Optional[int] = None
    meta: Optional[Dict] = Field(default=None, sa_type=JSON)
    started_at: Optional[int] = None
    suffix: Optional[str] = None
    hyperparameters: Dict = Field(default_factory=dict, sa_type=JSON)
    method: Optional[Dict] = Field(default=None, sa_type=JSON)
    __table_args__ = (
        Index("idx_ft_jobs_status", "status"),
        Index("idx_ft_jobs_created", "created_at"),
    )

    def to_openai(self) -> OpenAIFineTuningJob:
        """Convert to OpenAI format using JSON columns."""
        data = self.model_dump()
        data["object"] = "fine_tuning.job"
        if "hyperparameters" not in data or data["hyperparameters"] is None:
            data["hyperparameters"] = {}
        data["metadata"] = data.pop("meta", None)
        internal_fields = ["started_at"]
        for field in internal_fields:
            data.pop(field, None)
        return OpenAIFineTuningJob(**data)


class FineTuningEvent(SQLModel, table=True):
    """Fine-tuning event model - OpenAI compatible."""

    __tablename__ = "fine_tuning_events"
    id: str = Field(primary_key=True)
    object: str = Field(default="fine_tuning.job.event")
    created_at: int = Field(default_factory=lambda: int(time.time()))
    level: str
    message: str
    data: Optional[Dict] = Field(default=None, sa_type=JSON)
    job_id: str = Field(foreign_key="fine_tuning_jobs.id")
    __table_args__ = (
        Index("idx_events_job_id", "job_id"),
        Index("idx_events_created", "created_at"),
    )

    def to_openai(self) -> OpenAIFineTuningJobEvent:
        """Convert to OpenAI FineTuningJobEvent."""
        return OpenAIFineTuningJobEvent(**self.model_dump())
