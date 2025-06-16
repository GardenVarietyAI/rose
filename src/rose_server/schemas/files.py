"""File API schemas."""
from typing import List, Optional

from openai.types import FileObject
from pydantic import BaseModel, Field, validator


class TrainingData(BaseModel):
    """Validation model for fine-tuning training data - OpenAI compatible formats only."""

    messages: Optional[List[dict]] = None
    input: Optional[dict] = None
    preferred_output: Optional[List[dict]] = None
    non_preferred_output: Optional[List[dict]] = None
    prompt: Optional[str] = None
    completion: Optional[str] = None
    @validator("messages")

    def validate_messages(cls, v):
        if v is not None and not v:
            raise ValueError("messages cannot be empty")
        return v
    @validator("input")

    def validate_input(cls, v):
        if v is not None:
            if "messages" not in v:
                raise ValueError("input must contain messages field")
            if not v["messages"]:
                raise ValueError("input.messages cannot be empty")
        return v
    @validator("preferred_output", "non_preferred_output")

    def validate_outputs(cls, v):
        if v is not None and not v:
            raise ValueError("preferred_output and non_preferred_output cannot be empty")
        return v

    def __init__(self, **data):
        super().__init__(**data)
        has_messages = self.messages is not None and len(self.messages) > 0
        has_dpo_format = all(
            [self.input is not None, self.preferred_output is not None, self.non_preferred_output is not None]
        )
        has_completion_format = self.prompt is not None and self.completion is not None
        if not (has_messages or has_dpo_format or has_completion_format):
            raise ValueError(
                "Training data must be in one of these OpenAI formats:\n"
                "1. Supervised fine-tuning: {'messages': [{'role': 'user', 'content': '...'}, ...]}\n"
                "2. DPO preference format: {'input': {'messages': [...]}, 'preferred_output': [...], 'non_preferred_output': [...]}\n"
                "3. Legacy completion format: {'prompt': '...', 'completion': '...'}"
            )

class FileListResponse(BaseModel):
    """Response for listing files."""

    object: str = Field(default="list", description="Object type")
    data: List[FileObject] = Field(default_factory=list, description="List of files")
    has_more: bool = Field(default=False, description="Whether there are more files")