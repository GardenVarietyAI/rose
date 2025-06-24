"""Thread API schemas."""

from typing import Any, Dict, List, Optional

from openai.types.beta import Thread as OpenAIThread
from pydantic import BaseModel, Field


class ThreadResponse(OpenAIThread):
    """Response model for thread operations."""

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias="meta",
        serialization_alias="metadata",
        description="Set of key-value pairs for metadata",
    )


class ThreadCreateRequest(BaseModel):
    """Request to create a new thread."""

    messages: Optional[List[Dict[str, Any]]] = Field(default=None, description="Initial messages for the thread")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Set of key-value pairs for metadata")
