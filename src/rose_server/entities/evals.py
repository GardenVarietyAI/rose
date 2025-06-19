"""Evaluation database entities - OpenAI compatible."""

import time
from typing import Dict, List, Optional

from sqlalchemy import JSON, Index
from sqlmodel import Field, SQLModel


class Eval(SQLModel, table=True):
    """Evaluation definition - what test to run."""

    __tablename__ = "evals"
    id: str = Field(primary_key=True)
    object: str = Field(default="eval", description="Object type for OpenAI compatibility")
    name: Optional[str] = Field(default=None, index=True, description="Evaluation name")
    data_source_config: Dict = Field(sa_type=JSON, description="Data source configuration")
    testing_criteria: List[Dict] = Field(sa_type=JSON, description="Testing criteria/graders")
    created_at: int = Field(default_factory=lambda: int(time.time()))
    meta: Optional[Dict] = Field(default=None, sa_type=JSON)
    __table_args__ = (
        Index("idx_evals_name", "name"),
        Index("idx_evals_created", "created_at"),
    )

    def model_dump(self, *args, **kwargs):
        base = super().model_dump(*args, **kwargs)
        base["metadata"] = self.meta
        base.pop("meta", None)  # Remove internal field
        return base

    def dict(self, *args, **kwargs):
        """Legacy method for backwards compatibility."""
        return self.model_dump(*args, **kwargs)


class EvalRun(SQLModel, table=True):
    """Evaluation execution run - results of running an eval."""

    __tablename__ = "eval_runs"
    id: str = Field(primary_key=True)
    object: str = Field(default="eval.run", description="Object type for OpenAI compatibility")
    eval_id: str = Field(foreign_key="evals.id", index=True, description="Reference to eval definition")
    name: str = Field(description="Run name/description")
    model: str = Field(index=True, description="Model being evaluated")
    status: str = Field(default="queued", index=True, description="queued, running, completed, failed, cancelled")
    created_at: int = Field(default_factory=lambda: int(time.time()))
    started_at: Optional[int] = None
    completed_at: Optional[int] = None
    data_source: Optional[Dict] = Field(default=None, sa_type=JSON, description="Run-specific data source config")
    results: Optional[Dict] = Field(default=None, sa_type=JSON, description="Evaluation results")
    result_counts: Optional[Dict] = Field(default=None, sa_type=JSON, description="OpenAI-style result counts")
    report_url: Optional[str] = Field(default=None, description="URL to view eval report")
    meta: Optional[Dict] = Field(default=None, sa_type=JSON)
    error_message: Optional[str] = None
    __table_args__ = (
        Index("idx_eval_runs_eval", "eval_id"),
        Index("idx_eval_runs_model", "model"),
        Index("idx_eval_runs_status", "status"),
        Index("idx_eval_runs_created", "created_at"),
    )

    def model_dump(self, *args, **kwargs):
        base = super().model_dump(*args, **kwargs)
        base["metadata"] = self.meta

        # Map error field
        if self.error_message:
            base["error"] = self.error_message

        # Remove internal fields
        base.pop("meta", None)
        base.pop("error_message", None)
        return base

    def dict(self, *args, **kwargs):
        """Legacy method for backwards compatibility."""
        return self.model_dump(*args, **kwargs)


class EvalSample(SQLModel, table=True):
    """Individual evaluation sample - for detailed analysis."""

    __tablename__ = "eval_samples"
    id: str = Field(primary_key=True)
    object: str = Field(default="eval.sample", description="Object type for OpenAI compatibility")
    eval_run_id: str = Field(foreign_key="eval_runs.id")
    sample_index: int = Field(description="Index within the eval run")
    input: str = Field(description="Input prompt/question")
    ideal: str = Field(description="Ideal/expected output")
    completion: str = Field(description="Model's completion/output")
    score: float = Field(description="Sample score (0.0 to 1.0)")
    passed: bool = Field(description="Whether sample passed threshold")
    response_time: Optional[float] = None
    tokens_used: Optional[int] = None
    meta: Optional[Dict] = Field(default=None, sa_type=JSON)
    created_at: int = Field(default_factory=lambda: int(time.time()))
    __table_args__ = (
        Index("idx_eval_samples_run", "eval_run_id"),
        Index("idx_eval_samples_score", "score"),
        Index("idx_eval_samples_passed", "passed"),
    )

    def model_dump(self, *args, **kwargs):
        base = super().model_dump(*args, **kwargs)
        base["metadata"] = self.meta
        base.pop("meta", None)
        return base

    def dict(self, *args, **kwargs):
        """Legacy method for backwards compatibility."""
        return self.model_dump(*args, **kwargs)
