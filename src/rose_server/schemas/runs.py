"""Schema definitions for runs and run steps."""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class RunStepType(str, Enum):
    """Types of run steps."""

    MESSAGE_CREATION = "message_creation"
    TOOL_CALLS = "tool_calls"


class RunStep(BaseModel):
    """Represents a step in a run execution."""

    id: str = Field(description="Unique identifier for the step")
    object: str = Field(default="thread.run.step", description="Object type")
    created_at: int = Field(description="Unix timestamp when the step was created")
    run_id: str = Field(description="ID of the run this step belongs to")
    assistant_id: str = Field(description="ID of the assistant")
    thread_id: str = Field(description="ID of the thread")
    type: RunStepType = Field(description="Type of run step")
    status: str = Field(description="Status of the step: in_progress, completed, failed, cancelled")
    step_details: Dict[str, Any] = Field(description="Details specific to the step type")
    last_error: Optional[Dict[str, Any]] = Field(default=None, description="Last error if failed")
    completed_at: Optional[int] = Field(default=None, description="When the step completed")
    failed_at: Optional[int] = Field(default=None, description="When the step failed")
    cancelled_at: Optional[int] = Field(default=None, description="When the step was cancelled")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Step metadata")
    usage: Optional[Dict[str, Any]] = Field(default=None, description="Usage for this step")
