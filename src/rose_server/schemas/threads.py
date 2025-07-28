"""Thread API schemas."""

from typing import Any, Dict, List, Optional, Union

from openai.types.beta import Thread as OpenAIThread
from openai.types.beta.assistant_tool import AssistantTool
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


class ThreadAndRunCreateRequest(BaseModel):
    assistant_id: str = Field(description="ID of the assistant to use")
    thread: Dict[str, Any] = Field(default={}, description="Thread params")
    model: Optional[str] = Field(default=None, description="Override the model")
    instructions: Optional[str] = Field(default=None, description="Override the instructions")
    additional_instructions: Optional[str] = Field(default=None, description="Additional instructions")
    additional_messages: Optional[List[Dict[str, Any]]] = Field(default=None, description="Additional messages")
    tools: Optional[List[AssistantTool]] = Field(default=None, description="Override tools")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Set of key-value pairs for metadata")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    max_prompt_tokens: Optional[int] = Field(default=None, description="Maximum tokens for the prompt")
    max_completion_tokens: Optional[int] = Field(default=None, description="Maximum tokens for the completion")
    truncation_strategy: Optional[Dict[str, Any]] = Field(default=None, description="Truncation strategy")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(default=None, description="Tool choice strategy")
    parallel_tool_calls: Optional[bool] = Field(default=None, description="Whether to enable parallel tool calls")
    response_format: Optional[Dict[str, Any]] = Field(default=None, description="Response format specification")
    stream: Optional[bool] = Field(default=None, description="Whether to stream the response")
