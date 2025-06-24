"""Message API schemas."""

from typing import Any, Dict, List, Literal, Union

from openai.types.beta.threads import Message as OpenAIMessage
from pydantic import BaseModel, Field


class MessageResponse(OpenAIMessage):
    """Response model for message operations."""

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias="meta",
        serialization_alias="metadata",
        description="Set of key-value pairs for metadata",
    )


class MessageCreateRequest(BaseModel):
    """Request to create a new message in a thread."""

    role: Literal["user", "assistant"] = Field(description="Role of the message sender")
    content: Union[str, List[Dict[str, Any]]] = Field(description="Content of the message")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Set of key-value pairs for metadata")
