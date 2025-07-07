"""Schemas for model-related operations."""

from typing import List, Optional

from pydantic import BaseModel


class ModelCreateRequest(BaseModel):
    """Request schema for creating a new model."""

    model_name: str  # HuggingFace model name
    name: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 0.9
    memory_gb: float = 2.0
    timeout: Optional[int] = None
    lora_target_modules: Optional[List[str]] = None
