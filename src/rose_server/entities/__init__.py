"""SQLModel database entities for persistent storage.
This module contains all SQLModel models used for database tables.
API schemas are in the schemas/ module.
"""
from .assistants import (
    Assistant,
    AssistantTool,
)
from .evals import (
    Eval,
    EvalRun,
    EvalSample,
)
from .fine_tuning import (
    FineTuningEvent,
    FineTuningJob,
)
from .threads import (
    Message,
    MessageMetadata,
    Thread,
)

__all__ = [
    "FineTuningJob",
    "FineTuningEvent",
    "Assistant",
    "AssistantTool",
    "Thread",
    "Message",
    "MessageMetadata",
    "Eval",
    "EvalRun",
    "EvalSample",
]