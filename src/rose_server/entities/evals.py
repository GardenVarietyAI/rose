"""Evaluation database entities - OpenAI compatible."""
import time
from typing import Dict, List, Optional
from sqlalchemy import JSON, Index
from sqlmodel import Field, SQLModel

class Eval(SQLModel, table=True):
    """Evaluation definition - what test to run."""

    __tablename__ = "evals"
    id: str = Field(primary_key=True)
    name: str = Field(index=True, description="Evaluation name (gsm8k, humaneval, etc.)")
    description: Optional[str] = Field(default=None, description="Evaluation description")
    data_source_config: Dict = Field(sa_type=JSON, description="Data source configuration")
    testing_criteria: List[Dict] = Field(sa_type=JSON, description="Grading criteria")
    created_at: int = Field(default_factory=lambda: int(time.time()))
    meta: Optional[Dict] = Field(default=None, sa_type=JSON)
    __table_args__ = (
        Index("idx_evals_name", "name"),
        Index("idx_evals_created", "created_at"),
    )

    def to_openai_format(self) -> Dict:
        """Convert to OpenAI-compatible format."""
        return {
            "id": self.id,
            "object": "eval",
            "name": self.name,
            "description": self.description,
            "data_source_config": self.data_source_config,
            "testing_criteria": self.testing_criteria,
            "created_at": self.created_at,
            "metadata": self.meta,
        }

class EvalRun(SQLModel, table=True):
    """Evaluation execution run - results of running an eval."""

    __tablename__ = "eval_runs"
    id: str = Field(primary_key=True)
    eval_id: str = Field(foreign_key="evals.id", index=True, description="Reference to eval definition")
    name: str = Field(description="Run name/description")
    model: str = Field(index=True, description="Model being evaluated")
    status: str = Field(default="queued", index=True, description="queued, running, completed, failed, cancelled")
    created_at: int = Field(default_factory=lambda: int(time.time()))
    started_at: Optional[int] = None
    completed_at: Optional[int] = None
    data_source: Optional[Dict] = Field(default=None, sa_type=JSON, description="Run-specific data source")
    results: Optional[Dict] = Field(default=None, sa_type=JSON, description="Evaluation results")
    meta: Optional[Dict] = Field(default=None, sa_type=JSON)
    total_samples: int = Field(default=0)
    completed_samples: int = Field(default=0)
    failed_samples: int = Field(default=0)
    avg_response_time: Optional[float] = None
    total_tokens: Optional[int] = None
    error_message: Optional[str] = None
    __table_args__ = (
        Index("idx_eval_runs_eval", "eval_id"),
        Index("idx_eval_runs_model", "model"),
        Index("idx_eval_runs_status", "status"),
        Index("idx_eval_runs_created", "created_at"),
    )

    def to_openai_format(self) -> Dict:
        """Convert to OpenAI-compatible format."""
        return {
            "id": self.id,
            "object": "eval.run",
            "eval_id": self.eval_id,
            "name": self.name,
            "model": self.model,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "data_source": self.data_source,
            "results": self.results,
            "metadata": self.meta,
        }

class EvalSample(SQLModel, table=True):
    """Individual evaluation sample - for detailed analysis."""

    __tablename__ = "eval_samples"
    id: str = Field(primary_key=True)
    eval_run_id: str = Field(foreign_key="eval_runs.id")
    sample_index: int = Field(description="Index within the eval run")
    input: str = Field(description="Input prompt/question")
    expected_output: str = Field(description="Expected answer/output")
    actual_output: str = Field(description="Model's actual output")
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