from typing import Any, Dict, List, Optional

from openai.types.beta.assistant_tool import AssistantTool
from pydantic import BaseModel, Field


class AssistantResponse(BaseModel):
    id: str = Field(description="Unique identifier for the assistant")
    object: str = Field(default="assistant", description="Object type")
    created_at: int = Field(description="Unix timestamp when the assistant was created")
    name: Optional[str] = Field(default=None, description="Name of the assistant")
    description: Optional[str] = Field(default=None, description="Description of the assistant")
    model: str = Field(description="Model used by the assistant")
    instructions: Optional[str] = Field(default=None, description="System instructions for the assistant")
    tools: List[AssistantTool] = Field(default_factory=list, description="Tools enabled for the assistant")
    tool_resources: Dict[str, Any] = Field(default_factory=dict, description="Tool resources configuration")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias="meta",
        serialization_alias="metadata",
        description="Set of key-value pairs for metadata",
    )
    temperature: Optional[float] = Field(default=1.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=1.0, description="Nucleus sampling parameter")
    response_format: Optional[Dict[str, Any]] = Field(default=None, description="Response format specification")


class AssistantCreateRequest(BaseModel):
    model: str = Field(description="Model to use for the assistant")
    name: Optional[str] = Field(default=None, description="Name of the assistant")
    description: Optional[str] = Field(default=None, description="Description of the assistant")
    instructions: Optional[str] = Field(default=None, description="System instructions for the assistant")
    tools: List[AssistantTool] = Field(default_factory=list, description="Tools for the assistant")
    tool_resources: Dict[str, Any] = Field(default_factory=dict, description="Tool resources configuration")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Set of key-value pairs for metadata")
    temperature: Optional[float] = Field(default=1.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=1.0, description="Nucleus sampling parameter")
    response_format: Optional[Dict[str, Any]] = Field(default=None, description="Response format specification")


class AssistantUpdateRequest(BaseModel):
    model: Optional[str] = Field(default=None, description="Model to use for the assistant")
    name: Optional[str] = Field(default=None, description="Name of the assistant")
    description: Optional[str] = Field(default=None, description="Description of the assistant")
    instructions: Optional[str] = Field(default=None, description="System instructions for the assistant")
    tools: Optional[List[AssistantTool]] = Field(default=None, description="Tools for the assistant")
    tool_resources: Optional[Dict[str, Any]] = Field(default=None, description="Tool resources configuration")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Set of key-value pairs for metadata")
    temperature: Optional[float] = Field(default=None, description="Sampling temperature")
    top_p: Optional[float] = Field(default=None, description="Nucleus sampling parameter")
    response_format: Optional[Dict[str, Any]] = Field(default=None, description="Response format specification")
