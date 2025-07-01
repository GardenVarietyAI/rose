"""Event-native LLM system built with Pydantic for clean serialization."""

from rose_server.events.generator import EventGenerator

from .event_types.base import LLMEvent
from .event_types.generation import (
    InputTokensCounted,
    ResponseCompleted,
    ResponseStarted,
    TokenGenerated,
    ToolCallCompleted,
    ToolCallResult,
    ToolCallStarted,
)
from .event_types.resource import ModelLoaded, ModelUnloaded, ResourceAcquired, ResourceReleased
from .event_types.training import (
    CheckpointSaved,
    TrainingCompleted,
    TrainingError,
    TrainingStarted,
    TrainingStepCompleted,
)

__all__ = [
    "LLMEvent",
    "ResponseStarted",
    "InputTokensCounted",
    "TokenGenerated",
    "ToolCallStarted",
    "ToolCallCompleted",
    "ToolCallResult",
    "ResponseCompleted",
    "TrainingStarted",
    "TrainingStepCompleted",
    "CheckpointSaved",
    "TrainingCompleted",
    "TrainingError",
    "ModelLoaded",
    "ModelUnloaded",
    "ResourceAcquired",
    "ResourceReleased",
    "EventGenerator",
]
