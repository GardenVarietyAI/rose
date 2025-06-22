"""Job queue schemas."""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel


class JobStatus(str, Enum):
    """Job status states."""

    QUEUED = "queued"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


class JobType(str, Enum):
    """Job types."""

    TRAINING = "training"
    EVAL = "eval"
    BATCH_INFERENCE = "batch_inference"


class JobUpdateRequest(BaseModel):
    """Job update request."""

    status: str
    result: Optional[Dict[str, Any]] = None


class JobResponse(BaseModel):
    """Job response model."""

    id: int
    type: str
    status: str
    payload: Dict
    created_at: int
    started_at: Optional[int] = None
    completed_at: Optional[int] = None
    error: Optional[str] = None
