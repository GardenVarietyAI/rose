"""Streaming response schemas for chat completions and responses API."""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class StreamingChoice(BaseModel):
    """Choice object for streaming responses."""

    index: int = Field(description="Index of the choice")
    delta: Dict[str, Any] = Field(description="Delta containing the incremental content")
    finish_reason: Optional[Literal["stop", "length", "content_filter", "tool_calls"]] = Field(
        default=None, description="Reason the model stopped generating"
    )

class StreamingResponse(BaseModel):
    """OpenAI-compatible streaming response chunk."""

    id: str = Field(description="Unique identifier for the chat completion")
    object: str = Field(default="chat.completion.chunk", description="Object type")
    created: int = Field(description="Unix timestamp when the completion was created")
    model: str = Field(description="Model used for the completion")
    choices: List[StreamingChoice] = Field(description="List of completion choices")
    usage: Optional[Dict[str, int]] = Field(default=None, description="Usage statistics")
__all__ = ["StreamingChoice", "StreamingResponse"]