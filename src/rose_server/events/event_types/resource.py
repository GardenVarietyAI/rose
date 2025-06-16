"""Resource management events for model loading and GPU coordination."""
from typing import Literal, Optional
from pydantic import Field
from .base import LLMEvent

class ModelLoaded(LLMEvent):
    """Fired when a model is loaded into memory."""

    model_path: Optional[str] = Field(None, description="Path to model files")
    device: str = Field(..., description="Device where model was loaded (cuda, cpu)")
    memory_usage: Optional[float] = Field(None, ge=0, description="Memory usage in GB")
    load_time: Optional[float] = Field(None, ge=0, description="Time taken to load in seconds")

class ModelUnloaded(LLMEvent):
    """Fired when a model is unloaded from memory."""

    reason: Literal["idle_timeout", "memory_pressure", "explicit_request"] = Field(
        ..., description="Reason for unloading"
    )
    memory_freed: Optional[float] = Field(None, ge=0, description="Memory freed in GB")

class ResourceAcquired(LLMEvent):
    """Fired when a computational resource is acquired."""

    resource_type: Literal["inference", "training", "gpu", "cpu"] = Field(..., description="Type of resource acquired")
    available_slots: int = Field(..., ge=0, description="Remaining available resource slots")
    queue_time: Optional[float] = Field(None, ge=0, description="Time spent waiting in queue")

class ResourceReleased(LLMEvent):
    """Fired when a computational resource is released."""

    resource_type: Literal["inference", "training", "gpu", "cpu"] = Field(..., description="Type of resource released")
    available_slots: int = Field(..., ge=0, description="Total available resource slots after release")
    usage_time: Optional[float] = Field(None, ge=0, description="Time resource was in use")