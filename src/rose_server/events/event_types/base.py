"""Base event class using Pydantic for clean serialization."""

import time
import uuid
from abc import ABC
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class LLMEvent(BaseModel, ABC):
    """Base class for all LLM events using Pydantic.

    Every event in the system extends this base class, providing:
    - Unique identification
    - Timestamp for ordering and debugging
    - Model context for multi-model scenarios
    - Automatic serialization via Pydantic
    - Type validation and documentation
    """

    id: str = Field(default_factory=lambda: f"event_{uuid.uuid4().hex[:8]}")
    timestamp: float = Field(default_factory=time.time)
    model_name: str = Field(..., description="Name/ID of the model that generated this event")
    event_type: str = Field(default="", description="Type of event for easy filtering")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def __init__(self, **data):
        """Initialize and set event_type to class name if not provided."""
        super().__init__(**data)
        if not self.event_type:
            self.event_type = self.__class__.__name__

    @classmethod
    def create(cls, model_name: str, **kwargs) -> "LLMEvent":
        """Convenience factory method for creating events."""
        return cls(model_name=model_name, **kwargs)

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        use_enum_values = True
