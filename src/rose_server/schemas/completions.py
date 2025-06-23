"""OpenAI-compatible completions API schemas."""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    prompt: Union[str, List[str]] = Field(..., description="The prompt(s) to generate completions for")
    suffix: Optional[str] = Field(None, description="The suffix that comes after a completion of inserted text")
    max_tokens: Optional[int] = Field(16, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(1.0, description="Sampling temperature between 0 and 2")
    top_p: Optional[float] = Field(1.0, description="Nucleus sampling parameter")
    n: Optional[int] = Field(1, description="Number of completions to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream back partial progress")
    logprobs: Optional[int] = Field(None, description="Include log probabilities on the most likely tokens")
    echo: Optional[bool] = Field(False, description="Echo back the prompt in addition to the completion")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Sequences where the API will stop generating")
    presence_penalty: Optional[float] = Field(
        0.0, description="Penalize new tokens based on their presence in the text so far"
    )
    frequency_penalty: Optional[float] = Field(
        0.0, description="Penalize new tokens based on their frequency in the text so far"
    )
    best_of: Optional[int] = Field(1, description="Generate best_of completions and return the best")
    logit_bias: Optional[Dict[str, float]] = Field(
        None, description="Modify the likelihood of specified tokens appearing"
    )
    user: Optional[str] = Field(None, description="Unique identifier representing the end-user")


class CompletionChoice(BaseModel):
    text: str = Field(..., description="The generated text")
    index: int = Field(..., description="The index of the choice")
    logprobs: Optional[Dict[str, Any]] = Field(None, description="Log probabilities of the output tokens")
    finish_reason: Optional[str] = Field(None, description="The reason the model stopped generating")


class CompletionUsage(BaseModel):
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total number of tokens used")


class CompletionResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field("text_completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of when the completion was created")
    model: str = Field(..., description="The model used for the completion")
    choices: List[CompletionChoice] = Field(..., description="List of completion choices")
    usage: Optional[CompletionUsage] = Field(None, description="Usage statistics for the request")
    system_fingerprint: Optional[str] = Field(None, description="System fingerprint")


class CompletionChunk(BaseModel):
    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field("text_completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of when the completion was created")
    choices: List[CompletionChoice] = Field(..., description="List of completion choices")
    model: str = Field(..., description="The model used for the completion")
    system_fingerprint: Optional[str] = Field(None, description="System fingerprint")
