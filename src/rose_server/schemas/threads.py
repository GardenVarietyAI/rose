"""Thread API schemas."""

from typing import Any, Dict, List, Literal, Optional, Union

from openai.types.beta import Thread as OpenAIThread
from openai.types.beta.threads import Message as OpenAIMessage
from pydantic import BaseModel, Field


class ThreadMessage(OpenAIMessage):
    """Extends OpenAI's Message type for our thread messages."""

    pass


class Thread(OpenAIThread):
    """Extends OpenAI's Thread type for our threads."""

    pass


class ThreadCreateRequest(BaseModel):
    """Request to create a new thread."""

    messages: Optional[List[Dict[str, Any]]] = Field(default=None, description="Initial messages for the thread")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Set of key-value pairs for metadata")


class CreateMessageRequest(BaseModel):
    """Request to create a new message in a thread."""

    role: Literal["user", "assistant"] = Field(description="Role of the message sender")
    content: Union[str, List[Dict[str, Any]]] = Field(description="Content of the message")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Set of key-value pairs for metadata")
