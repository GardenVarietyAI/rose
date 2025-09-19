"""Generation-related events for text completion and tool calls."""

import json
import time
import uuid
from typing import Any, Dict, Literal, Optional, Type

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LLMEvent(BaseModel):
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
    metadata: Optional[Dict[str, Any]] = Field(default={})

    def __init__(self, **data: Any) -> None:
        """Initialize and set event_type to class name if not provided."""
        super().__init__(**data)
        if not self.event_type:
            self.event_type = self.__class__.__name__

    @classmethod
    def create(cls: Type["LLMEvent"], model_name: str, **kwargs: Any) -> "LLMEvent":
        """Convenience factory method for creating events."""
        return cls(model_name=model_name, **kwargs)

    model_config = ConfigDict(arbitrary_types_allowed=True, use_enum_values=True)


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

    @field_validator("position")
    @classmethod
    def validate_position(cls: LLMEvent, v: int) -> int:
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


class ToolCallArgument(LLMEvent):
    """Fired for incremental tool call argument streaming.

    Streams partial JSON arguments as they're generated.
    """

    call_id: str = Field(..., description="ID of the tool call")
    argument_delta: str = Field(..., description="Incremental argument text")


class ToolCallCompleted(LLMEvent):
    """Fired when a tool/function call is fully parsed.

    Contains the complete function call with all arguments.
    This triggers actual tool execution in the system.
    """

    function_name: str = Field(..., description="Name of the function being called")
    arguments: str = Field(..., description="Complete function arguments as JSON string")
    call_id: str = Field(..., description="Unique identifier for this function call")
    parsed_arguments: Optional[Dict[str, Any]] = Field(None, description="Parsed JSON arguments")

    def __init__(self: LLMEvent, **data: Any) -> None:
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
    finish_reason: Literal["stop", "length", "tool_calls", "error", "cancelled", "timeout"] = Field(
        ..., description="Reason why generation finished"
    )
    input_tokens: int = Field(0, ge=0, description="Number of input tokens")
    output_tokens: Optional[int] = Field(None, ge=0, description="Number of output tokens generated")
    completion_time: Optional[float] = Field(None, ge=0, description="Time taken for completion in seconds")
