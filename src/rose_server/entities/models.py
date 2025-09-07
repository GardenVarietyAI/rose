"""Language model entities for database storage."""

import json
import time
from typing import List, Optional

from sqlmodel import Field, SQLModel


class LanguageModel(SQLModel, table=True):
    __tablename__ = "models"

    id: str = Field(primary_key=True)
    name: Optional[str] = Field(default=None)
    model_name: str = Field(index=True)  # HuggingFace model name
    model_type: str = Field(default="huggingface")
    path: Optional[str] = Field(default=None)  # Local path for fine-tuned models
    is_fine_tuned: bool = Field(default=False, index=True)

    # Model parameters
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.9)
    timeout: Optional[int] = Field(default=None)  # Timeout in seconds
    quantization: Optional[str] = Field(default=None)  # Quantization type: "int8", etc.

    # LoRA configuration (stored as JSON)
    lora_target_modules: Optional[str] = Field(default=None)  # JSON array

    # OpenAI API compatibility fields
    owned_by: str = Field(default="organization-owner")
    parent: Optional[str] = Field(default=None)  # Parent model (None for base, base_model for fine-tuned)
    permissions: Optional[str] = Field(default="[]")  # JSON array of permissions

    # Metadata
    created_at: int = Field(default_factory=lambda: int(time.time()))

    def get_lora_modules(self) -> Optional[List[str]]:
        """Get LoRA target modules as a list."""
        if self.lora_target_modules:
            return json.loads(self.lora_target_modules)  # type: ignore[no-any-return]
        return None

    def set_lora_modules(self, modules: Optional[List[str]]) -> None:
        """Set LoRA target modules from a list."""
        self.lora_target_modules = json.dumps(modules) if modules else None
