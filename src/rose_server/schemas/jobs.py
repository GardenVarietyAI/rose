"""Job queue schemas."""
from enum import Enum


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