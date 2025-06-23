"""Schema definitions for runs and run steps."""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from openai.types.beta.assistant_tool import AssistantTool
from pydantic import BaseModel, Field


class RunStepType(str, Enum):
    """Types of run steps."""

    MESSAGE_CREATION = "message_creation"
    TOOL_CALLS = "tool_calls"


class RunStep(BaseModel):
    """Represents a step in a run execution."""

    id: str = Field(description="Unique identifier for the step")
    object: str = Field(default="thread.run.step", description="Object type")
    created_at: int = Field(description="Unix timestamp when the step was created")
    run_id: str = Field(description="ID of the run this step belongs to")
    assistant_id: str = Field(description="ID of the assistant")
    thread_id: str = Field(description="ID of the thread")
    type: RunStepType = Field(description="Type of run step")
    status: str = Field(description="Status of the step: in_progress, completed, failed, cancelled")
    step_details: Dict[str, Any] = Field(description="Details specific to the step type")
    last_error: Optional[Dict[str, Any]] = Field(default=None, description="Last error if failed")
    completed_at: Optional[int] = Field(default=None, description="When the step completed")
    failed_at: Optional[int] = Field(default=None, description="When the step failed")
    cancelled_at: Optional[int] = Field(default=None, description="When the step was cancelled")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Step metadata")
    usage: Optional[Dict[str, Any]] = Field(default=None, description="Usage for this step")


class RunResponse(BaseModel):
    id: str = Field(description="Unique identifier for the run")
    object: str = Field(default="thread.run", description="Object type")
    created_at: int = Field(description="Unix timestamp when the run was created")
    thread_id: str = Field(description="ID of the thread this run belongs to")
    assistant_id: str = Field(description="ID of the assistant used for this run")
    status: Literal[
        "queued",
        "in_progress",
        "requires_action",
        "cancelling",
        "cancelled",
        "failed",
        "completed",
        "incomplete",
        "expired",
    ] = Field(description="Status of the run")
    required_action: Optional[Dict[str, Any]] = Field(default=None, description="Details on action required from user")
    last_error: Optional[Dict[str, Any]] = Field(default=None, description="Last error encountered during run")
    expires_at: Optional[int] = Field(default=None, description="Unix timestamp when the run expires")
    started_at: Optional[int] = Field(default=None, description="Unix timestamp when the run started")
    cancelled_at: Optional[int] = Field(default=None, description="Unix timestamp when the run was cancelled")
    failed_at: Optional[int] = Field(default=None, description="Unix timestamp when the run failed")
    completed_at: Optional[int] = Field(default=None, description="Unix timestamp when the run completed")
    incomplete_details: Optional[Dict[str, Any]] = Field(default=None, description="Details on why run is incomplete")
    model: str = Field(description="Model used for the run")
    instructions: Optional[str] = Field(default=None, description="Instructions used for the run")
    tools: List[AssistantTool] = Field(default_factory=list, description="Tools used for the run")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Set of key-value pairs for metadata")
    usage: Optional[Dict[str, Any]] = Field(default=None, description="Usage statistics for the run")
    temperature: Optional[float] = Field(default=None, description="Sampling temperature used")
    top_p: Optional[float] = Field(default=None, description="Nucleus sampling parameter used")
    max_prompt_tokens: Optional[int] = Field(default=None, description="Maximum tokens for the prompt")
    max_completion_tokens: Optional[int] = Field(default=None, description="Maximum tokens for the completion")
    truncation_strategy: Optional[Dict[str, Any]] = Field(default=None, description="Truncation strategy")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(default=None, description="Tool choice strategy")
    parallel_tool_calls: bool = Field(default=True, description="Whether to enable parallel tool calls")
    response_format: Optional[Dict[str, Any]] = Field(default=None, description="Response format specification")


class CreateRunRequest(BaseModel):
    assistant_id: str = Field(description="ID of the assistant to use")
    model: Optional[str] = Field(default=None, description="Override the model")
    instructions: Optional[str] = Field(default=None, description="Override the instructions")
    additional_instructions: Optional[str] = Field(default=None, description="Additional instructions")
    additional_messages: Optional[List[Dict[str, Any]]] = Field(default=None, description="Additional messages")
    tools: Optional[List[AssistantTool]] = Field(default=None, description="Override tools")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Set of key-value pairs for metadata")
    temperature: Optional[float] = Field(default=None, description="Sampling temperature")
    top_p: Optional[float] = Field(default=None, description="Nucleus sampling parameter")
    max_prompt_tokens: Optional[int] = Field(default=None, description="Maximum tokens for the prompt")
    max_completion_tokens: Optional[int] = Field(default=None, description="Maximum tokens for the completion")
    truncation_strategy: Optional[Dict[str, Any]] = Field(default=None, description="Truncation strategy")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(default=None, description="Tool choice strategy")
    parallel_tool_calls: Optional[bool] = Field(default=None, description="Whether to enable parallel tool calls")
    response_format: Optional[Dict[str, Any]] = Field(default=None, description="Response format specification")
    stream: Optional[bool] = Field(default=False, description="Whether to stream the response")
