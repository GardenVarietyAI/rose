"""Chat message schemas."""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "function", "developer"] = Field(
        description="Role of the message sender"
    )
    content: Optional[Union[str, List[Dict[str, Any]]]] = Field(
        default=None, description="Content of the message (text, images, etc.)"
    )
    name: Optional[str] = Field(default=None, description="Name of the message sender")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(default=None, description="Tool calls made by the assistant")
    function_call: Optional[Dict[str, Any]] = Field(default=None, description="Deprecated: Use tool_calls instead")
    tool_call_id: Optional[str] = Field(default=None, description="ID of the tool call this message is responding to")
    refusal: Optional[str] = Field(default=None, description="Refusal message from the assistant")
    audio: Optional[Dict[str, Any]] = Field(default=None, description="Audio content from the assistant")
