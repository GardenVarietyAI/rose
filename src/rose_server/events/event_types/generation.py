"""Generation-related events for text completion and tool calls."""

import json
import uuid
from typing import Any, Dict, Literal, Optional

from pydantic import Field, validator

from .base import LLMEvent


class ResponseStarted(LLMEvent):
    """Fired when a response generation begins.

    This is the first event in any completion, providing context
    for all subsequent events in the generation stream.
    """

    response_id: str = Field(default_factory=lambda: f"resp_{uuid.uuid4().hex[:16]}")
    input_tokens: int = Field(..., ge=0, description="Number of input tokens")
    max_tokens: Optional[int] = Field(None, ge=1, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature")


class InputTokensCounted(LLMEvent):
    """Fired when input tokens have been counted.

    This event is sent after tokenization but before generation starts.
    It provides the accurate input token count.
    """

    response_id: str = Field(..., description="ID of the response")
    input_tokens: int = Field(..., ge=0, description="Actual number of input tokens")


class TokenGenerated(LLMEvent):
    """Fired for each token generated during completion.

    This is the core streaming event that represents incremental
    text generation. Multiple tokens form the complete response.
    """

    token: str = Field(..., description="The generated token text")
    token_id: int = Field(..., description="Token ID from tokenizer vocabulary")
    position: int = Field(..., ge=0, description="Position in the generated sequence")
    logprob: Optional[float] = Field(None, description="Log probability of this token")

    @validator("position")
    def validate_position(cls, v):
        if v < 0:
            raise ValueError("Token position must be non-negative")
        return v


class ToolCallStarted(LLMEvent):
    """Fired when a tool/function call begins parsing.

    Indicates the model has started generating a structured function call.
    The arguments may be streamed in subsequent events.
    """

    function_name: str = Field(..., description="Name of the function being called")
    call_id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:16]}")
    arguments_so_far: str = Field(default="", description="Partial arguments parsed so far")


class ToolCallCompleted(LLMEvent):
    """Fired when a tool/function call is fully parsed.

    Contains the complete function call with all arguments.
    This triggers actual tool execution in the system.
    """

    function_name: str = Field(..., description="Name of the function being called")
    arguments: str = Field(..., description="Complete function arguments as JSON string")
    call_id: str = Field(..., description="Unique identifier for this function call")
    parsed_arguments: Optional[Dict[str, Any]] = Field(None, description="Parsed JSON arguments")

    def __init__(self, **data):
        """Initialize and parse JSON arguments if not already provided."""
        super().__init__(**data)
        if self.parsed_arguments is None and self.arguments:
            try:
                self.parsed_arguments = json.loads(self.arguments)
            except json.JSONDecodeError:
                self.parsed_arguments = {"raw": self.arguments}


class ToolCallResult(LLMEvent):
    """Fired when a tool execution completes.

    Contains the result of tool execution, which may be fed back
    into the model for further processing.
    """

    call_id: str = Field(..., description="ID of the function call that was executed")
    function_name: str = Field(..., description="Name of the function that was executed")
    result: Any = Field(..., description="Result returned by the function")
    success: bool = Field(..., description="Whether the function execution succeeded")
    error_message: Optional[str] = Field(None, description="Error message if execution failed")
    execution_time: Optional[float] = Field(None, ge=0, description="Time taken to execute in seconds")


class ResponseCompleted(LLMEvent):
    """Fired when response generation is complete.

    This is the final event in a generation stream, providing
    summary statistics and completion status.
    """

    response_id: str = Field(..., description="ID of the completed response")
    total_tokens: int = Field(..., ge=0, description="Total tokens in the response")
    finish_reason: Literal["stop", "length", "tool_calls", "error", "cancelled"] = Field(
        ..., description="Reason why generation finished"
    )
    output_tokens: Optional[int] = Field(None, ge=0, description="Number of output tokens generated")
    completion_time: Optional[float] = Field(None, ge=0, description="Time taken for completion in seconds")
