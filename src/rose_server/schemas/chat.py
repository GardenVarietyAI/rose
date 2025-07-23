"""Chat completions API schemas."""

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


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(description="Chat messages in the conversation")
    model: str = Field(default="Qwen--Qwen2.5-1.5B-Instruct", description="Model to use for generation")
    mode: Optional[str] = Field(default=None, description="Deprecated: Use model instead")
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    stream: Optional[bool] = Field(default=False, description="Stream response")
    tools: Optional[List[Dict[str, Any]]] = Field(default=None, description="Available tools")
    tool_choice: Optional[str] = Field(default="auto", description="Tool selection mode")
    return_tool_calls: Optional[bool] = Field(default=True, description="Return tool calls in response")
    logprobs: Optional[bool] = Field(default=None, description="Whether to return log probabilities")
    top_logprobs: Optional[int] = Field(
        default=None, ge=0, le=5, description="Number of most likely tokens to return at each position"
    )
    seed: Optional[int] = Field(default=None, ge=0, description="Seed for deterministic generation")


class ChatResponse(BaseModel):
    reply: str = Field(description="Generated response from the LLM")
    model: str = Field(description="Model used for generation")
    prompt_tokens: Optional[int] = Field(default=None, description="Number of tokens in the prompt")
    completion_tokens: Optional[int] = Field(default=None, description="Number of tokens in the completion")
    total_tokens: Optional[int] = Field(default=None, description="Total number of tokens used")
