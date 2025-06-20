"""Job queue schemas."""

from enum import Enum
from typing import Dict, Optional

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
    result: Optional[Dict] = None
