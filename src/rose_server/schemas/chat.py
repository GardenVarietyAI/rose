"""Chat completions API schemas."""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "tool"] = Field(description="Role of the message sender")
    content: Union[str, List[Dict[str, Any]]] = Field(description="Content of the message")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(description="Chat messages in the conversation")
    mode: str = Field(description="Model to use for generation")


class ChatResponse(BaseModel):
    reply: str = Field(description="Generated response from the LLM")
    model: str = Field(description="Model used for generation")
    prompt_tokens: Optional[int] = Field(default=None, description="Number of tokens in the prompt")
    completion_tokens: Optional[int] = Field(default=None, description="Number of tokens in the completion")
    total_tokens: Optional[int] = Field(default=None, description="Total number of tokens used")


class OpenAIRequestMessage(BaseModel):
    role: str
    content: str


class OpenAIRequest(BaseModel):
    model: str = "qwen-coder"
    messages: List[OpenAIRequestMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[str] = "auto"
    return_tool_calls: Optional[bool] = True
