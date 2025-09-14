"""Language model entities for database storage."""

import time
from typing import List, Optional

from sqlmodel import JSON, Column, Field, SQLModel


class LanguageModel(SQLModel, table=True):
    __tablename__ = "models"

    id: str = Field(primary_key=True)
    model_name: str = Field(index=True)  # HuggingFace model name
    path: Optional[str] = Field(default=None)  # Local path for fine-tuned models
    kind: Optional[str] = Field(default=None)
    is_fine_tuned: bool = Field(default=False, index=True)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.9)
    timeout: Optional[int] = Field(default=None)  # Timeout in seconds
    quantization: Optional[str] = Field(default=None)  # Quantization type: "int8", etc.
    lora_target_modules: Optional[List[str]] = Field(sa_column=Column(JSON), default=None)
    owned_by: str = Field(default="organization-owner")
    parent: Optional[str] = Field(default=None)
    permissions: Optional[List[str]] = Field(sa_column=Column(JSON), default=None)
    created_at: int = Field(default_factory=lambda: int(time.time()))
