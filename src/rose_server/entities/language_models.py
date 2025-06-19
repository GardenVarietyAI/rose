"""Language model entities for database storage."""

import time
from typing import Optional

from sqlmodel import Field, SQLModel


class LanguageModel(SQLModel, table=True):
    __tablename__ = "language_models"

    id: str = Field(primary_key=True)
    name: Optional[str] = Field(default=None)
    path: Optional[str] = Field(default=None)
    base_model: Optional[str] = Field(default=None)
    hf_model_name: Optional[str] = Field(default=None)
    created_at: int = Field(default_factory=lambda: int(time.time()))
